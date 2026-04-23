import argparse
import asyncio
import os
import time
from uuid import uuid4

import boto3
import openai
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.data_types import VectorStoreCollectionConfig
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    Event,
    MessageContext,
    Text,
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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--question-id",
        default=None,
        help="Ingest only this question ID",
    )
    parser.add_argument(
        "--question-offset",
        type=int,
        default=None,
        help="Start index into the question list (for sharding)",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Number of questions to process (for sharding)",
    )
    parser.add_argument(
        "--question-concurrency",
        type=int,
        default=20,
        help="Max concurrent questions",
    )
    parser.add_argument(
        "--session-concurrency",
        type=int,
        default=3,
        help="Max concurrent sessions per question",
    )
    parser.add_argument(
        "--skip-startup",
        action="store_true",
        help="Skip DB/collection startup (tables already created)",
    )
    parser.add_argument(
        "--random-embedder",
        action="store_true",
        help="Use random vectors instead of OpenAI embeddings (for benchmarking)",
    )
    parser.add_argument(
        "--noop-vector-store",
        action="store_true",
        help="Use no-op vector store (for benchmarking)",
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
    args = parser.parse_args()

    data_path = args.data_path

    all_questions = load_longmemeval_dataset(data_path)
    if args.question_id:
        all_questions = [q for q in all_questions if q.question_id == args.question_id]
        if not all_questions:
            print(f"No question found with ID: {args.question_id}")
            return
    if hasattr(args, "question_offset") and args.question_offset is not None:
        offset = args.question_offset
        limit = args.question_limit or len(all_questions)
        all_questions = all_questions[offset : offset + limit]
    num_questions = len(all_questions)
    print(f"{num_questions} total questions")

    if args.noop_vector_store:
        from memmachine_server.common.vector_store.noop_vector_store import (
            NoopVectorStore,
        )

        vector_store = NoopVectorStore()
        qdrant_client = None
    else:
        qdrant_client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            prefer_grpc=True,
            timeout=300,
            port=int(os.getenv("QDRANT_PORT", "6333")),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
        )
        vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    if not args.skip_startup:
        await vector_store.startup()

    engine = create_async_engine(os.getenv("SQL_URL"), pool_size=500, max_overflow=500)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    if not args.skip_startup:
        await segment_store.startup()

    if args.random_embedder:
        from memmachine_server.common.embedder.random_embedder import RandomEmbedder

        embedder = RandomEmbedder(dimensions=1536)
        openai_client = None
    else:
        openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embedder = OpenAIEmbedder(
            OpenAIEmbedderParams(
                client=openai_client,
                model="text-embedding-3-small",
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

    async def get_or_create_collection(namespace: str, name: str):
        return await vector_store.open_or_create_collection(
            namespace=namespace,
            name=name,
            config=VectorStoreCollectionConfig(
                vector_dimensions=embedder.dimensions,
                similarity_metric=embedder.similarity_metric,
                properties_schema=EventMemory.expected_vector_store_collection_schema(),
            ),
        )

    async def process_conversation(question: LongMemEvalItem):
        partition_key = question.question_id
        session_ids = list(question.session_id_map.keys())

        # Delete any existing data for this question so partial ingestions
        # are cleaned up before re-ingesting.
        await vector_store.delete_collection(
            namespace="longmemeval", name=question.question_id
        )
        await segment_store.delete_partition(partition_key)

        collection = await get_or_create_collection(
            namespace="longmemeval", name=question.question_id
        )

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
                serialize_encode=True,
            )
        )

        async def ingest_session(session_id: str):
            session = question.get_session(session_id)

            events = []
            for turn in session:
                events.append(
                    Event(
                        uuid=uuid4(),
                        timestamp=turn.timestamp,
                        body=Content(
                            context=MessageContext(
                                source="Assistant"
                                if turn.role == "assistant"
                                else "User"
                            ),
                            items=[Text(text=turn.content.strip())],
                        ),
                        properties={
                            "longmemeval_session_id": session_id,
                            "has_answer": turn.has_answer,
                            "turn_id": turn.index,
                        },
                    )
                )

            try:
                await memory.encode_events(events)
            except Exception as e:
                print(
                    f"Error ingesting question={partition_key} "
                    f"session={session_id} ({len(events)} events): {e}"
                )
                raise

        # Process sessions sequentially to ensure deterministic eviction order.
        for sid in session_ids:
            await ingest_session(sid)

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

    start_time = time.monotonic()
    print(f"Ingestion started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        semaphore = asyncio.Semaphore(args.question_concurrency)
        tasks = [
            async_with(semaphore, process_conversation(question))
            for question in all_questions
        ]
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
    if qdrant_client is not None:
        await qdrant_client.close()
    if openai_client is not None:
        await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
