"""Ingest a BEAM dataset into EventMemory.

One Qdrant collection + one SQLAlchemy segment-store partition per BEAM
conversation (isolation_unit = "conversation", matching both official BEAM and
Vectorize).

Sessions within a conversation are encoded one at a time so eviction /
batch-predecessor logic sees chronologically consistent batches.
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
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

NAMESPACE = "beam"

EMBEDDER_OPENAI = "openai"
EMBEDDER_ST = "sentence-transformer"
EMBEDDERS: tuple[str, ...] = (EMBEDDER_OPENAI, EMBEDDER_ST)

DEFAULT_ST_MODEL = "BAAI/bge-large-en-v1.5"


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", required=True, help="Path to BEAM JSON file")
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
        help="Max concurrent conversations (default: 5)",
    )
    parser.add_argument(
        "--derive-sentences",
        action="store_true",
        help="Derive sentence-level derivatives from content",
    )
    parser.add_argument(
        "--eviction-similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold for cluster eviction (None disables eviction)",
    )
    parser.add_argument(
        "--eviction-search-limit",
        type=int,
        default=20,
        help="Max similar vectors to retrieve per derivative for eviction",
    )
    parser.add_argument(
        "--eviction-target-size",
        type=int,
        default=15,
        help="Target cluster size before eviction kicks in",
    )
    parser.add_argument(
        "--embedder",
        default=EMBEDDER_OPENAI,
        choices=list(EMBEDDERS),
        help=(
            "Embedding backend (default: openai). 'sentence-transformer' uses "
            "a local SentenceTransformer model — matches LIGHT's open-source "
            "stack (BAAI/bge-large-en-v1.5 by default)."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        default=None,
        help=(
            "Model name for the chosen embedder. Defaults: openai → "
            f"text-embedding-3-small, sentence-transformer → {DEFAULT_ST_MODEL}."
        ),
    )
    parser.add_argument(
        "--namespace",
        default=NAMESPACE,
        help=(
            f"Qdrant collection namespace (default: {NAMESPACE}). Switching "
            "embedders creates incompatible vector spaces — use a distinct "
            "namespace (e.g. 'beam-bge') to avoid mixing ingested runs."
        ),
    )
    args = parser.parse_args()
    namespace = args.namespace

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

    engine = create_async_engine(os.getenv("SQL_URL"), pool_size=200, max_overflow=200)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    openai_client: openai.AsyncOpenAI | None = None
    if args.embedder == EMBEDDER_OPENAI:
        openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embedder = OpenAIEmbedder(
            OpenAIEmbedderParams(
                client=openai_client,
                model=args.embedder_model or "text-embedding-3-small",
                dimensions=1536,
                max_input_length=8192,
            )
        )
    else:
        from memmachine_server.common.embedder.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
            SentenceTransformerEmbedderParams,
        )
        from sentence_transformers import SentenceTransformer

        st_model_name = args.embedder_model or DEFAULT_ST_MODEL
        st_model = SentenceTransformer(st_model_name)
        embedder = SentenceTransformerEmbedder(
            SentenceTransformerEmbedderParams(
                model_name=st_model_name,
                sentence_transformer=st_model,
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

    async def get_or_create_collection(partition_key: str):
        return await vector_store.open_or_create_collection(
            namespace=NAMESPACE,
            name=partition_key,
            config=VectorStoreCollectionConfig(
                vector_dimensions=embedder.dimensions,
                similarity_metric=embedder.similarity_metric,
                properties_schema=EventMemory.expected_vector_store_collection_schema(),
            ),
        )

    eviction_enabled = args.eviction_similarity_threshold is not None

    async def process_conversation(conversation: BEAMConversation):
        partition_key = conversation.conversation_id

        # Drop any stale partial data before re-ingesting.
        await vector_store.delete_collection(namespace=namespace, name=partition_key)
        await segment_store.delete_partition(partition_key)

        collection = await get_or_create_collection(partition_key)
        segment_store_partition = await segment_store.open_or_create_partition(
            partition_key
        )

        memory = EventMemory(
            EventMemoryParams(
                vector_store_collection=collection,
                segment_store_partition=segment_store_partition,
                embedder=embedder,
                reranker=reranker,
                derive_sentences=args.derive_sentences,
                max_text_chunk_length=500,
                eviction_similarity_threshold=args.eviction_similarity_threshold,
                eviction_search_limit=args.eviction_search_limit,
                eviction_target_size=args.eviction_target_size,
                serialize_encode=eviction_enabled,
            )
        )

        # Process sessions sequentially for deterministic eviction order.
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
    if openai_client is not None:
        await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
