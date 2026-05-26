"""Ingest a BEAM dataset into EventMemory.

Qdrant is used for the vector store; SegmentStore is the SQLAlchemy-backed
implementation against the URL given by the SQL_URL env var (Postgres or
SQLite both work).

One Qdrant collection + one SQLAlchemy segment-store partition per BEAM
conversation. Sessions are encoded one at a time so the segment store sees
chronologically consistent batches.
"""

import argparse
import asyncio
import logging
import os
import time

import boto3
import openai
from beam_event_adapter import conversation_to_events_by_session
from beam_models import BEAMConversation, load_beam_dataset
from dotenv import load_dotenv
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine


def _partition_key(conversation_id: str) -> str:
    return conversation_id.lower().replace("-", "_")


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", required=True, help="Path to BEAM JSON file")
    parser.add_argument(
        "--namespace",
        default="beam",
        help="Qdrant collection namespace",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Ingest only this conversation ID",
    )
    parser.add_argument(
        "--conversation-offset",
        type=int,
        default=None,
        help="Start index into the conversation list",
    )
    parser.add_argument(
        "--conversation-limit",
        type=int,
        default=None,
        help="Number of conversations to process",
    )
    parser.add_argument(
        "--conversation-concurrency",
        type=int,
        default=5,
        help="Max concurrent conversations",
    )
    parser.add_argument(
        "--max-text-chunk-length",
        type=int,
        default=500,
        help="Max code-point length for text chunks",
    )
    parser.add_argument(
        "--embedder-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )
    args = parser.parse_args()

    conversations = load_beam_dataset(args.data_path)
    if args.conversation_id:
        conversations = [
            c for c in conversations if c.conversation_id == args.conversation_id
        ]
        if not conversations:
            print(f"No conversation found with ID: {args.conversation_id}")
            return
    if args.conversation_offset is not None:
        offset = args.conversation_offset
        limit = args.conversation_limit or len(conversations)
        conversations = conversations[offset : offset + limit]
    print(f"{len(conversations)} total conversations")

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sql_url = os.getenv("SQL_URL")
    if not sql_url:
        raise ValueError("SQL_URL must be set for the SQLAlchemy segment store")
    engine = create_async_engine(sql_url, pool_size=200, max_overflow=200)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model=args.embedder_model,
            dimensions=1536,
            max_input_length=8192,
        )
    )

    region = "us-west-2"
    aws_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    reranker = AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=aws_client,
            region=region,
            model_id="cohere.rerank-v3-5:0",
        )
    )

    segmenter = TextSegmenter(max_chunk_length=args.max_text_chunk_length)
    deriver = WholeTextDeriver()

    schema = EventMemory.expected_vector_store_collection_schema()

    async def process_conversation(conversation: BEAMConversation):
        partition_key = _partition_key(conversation.conversation_id)

        await vector_store.delete_collection(
            namespace=args.namespace, name=partition_key
        )
        await segment_store.delete_partition(partition_key)

        collection = await vector_store.open_or_create_collection(
            namespace=args.namespace,
            name=partition_key,
            config=VectorStoreCollectionConfig(
                vector_dimensions=embedder.dimensions,
                similarity_metric=embedder.similarity_metric,
                indexed_properties_schema=schema,
            ),
        )
        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key,
            SegmentStorePartitionConfig(),
        )

        memory = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=segment_store_partition,
                segmenter=segmenter,
                deriver=deriver,
                embedder=embedder,
                reranker=reranker,
            )
        )

        for session_events in conversation_to_events_by_session(conversation):
            if not session_events:
                continue
            try:
                await memory.encode_events(session_events)
            except Exception as e:
                print(
                    f"Error ingesting conversation={partition_key}: "
                    f"{len(session_events)} events: {e}"
                )
                raise

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

    start_time = time.monotonic()
    print(f"Ingestion started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        semaphore = asyncio.Semaphore(args.conversation_concurrency)
        tasks = [async_with(semaphore, process_conversation(c)) for c in conversations]
        await asyncio.gather(*tasks)
    finally:
        elapsed = time.monotonic() - start_time
        print(
            f"Ingestion finished at {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(elapsed: {elapsed:.1f}s)"
        )

    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
