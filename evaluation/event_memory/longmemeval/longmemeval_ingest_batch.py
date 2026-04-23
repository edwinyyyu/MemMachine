"""Batch ingestion with per-question retry.

Processes questions in batches (default 10). Each question is retried up to
--max-retries times on failure; the retry deletes any partial data first.

Usage:
    uv run longmemeval_ingest_batch.py --data-path data.json
    uv run longmemeval_ingest_batch.py --data-path data.json --batch-size 20 --max-retries 5
"""

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
        help="Start index into the question list",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Number of questions to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Questions per batch (default: 20)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per question on failure (default: 3)",
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

    all_questions = load_longmemeval_dataset(args.data_path)
    if args.question_id:
        all_questions = [q for q in all_questions if q.question_id == args.question_id]
        if not all_questions:
            print(f"No question found with ID: {args.question_id}")
            return
    if args.question_offset is not None:
        offset = args.question_offset
        limit = args.question_limit or len(all_questions)
        all_questions = all_questions[offset : offset + limit]

    print(f"{len(all_questions)} total questions, batch_size={args.batch_size}")

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    engine = create_async_engine(os.getenv("SQL_URL"), pool_size=400, max_overflow=400)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

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

    async def delete_question_data(question_id: str):
        await vector_store.delete_collection(namespace="longmemeval", name=question_id)
        await segment_store.delete_partition(question_id)

    async def ingest_question(question: LongMemEvalItem):
        partition_key = question.question_id

        await delete_question_data(partition_key)

        collection = await vector_store.open_or_create_collection(
            namespace="longmemeval",
            name=partition_key,
            config=VectorStoreCollectionConfig(
                vector_dimensions=embedder.dimensions,
                similarity_metric=embedder.similarity_metric,
                properties_schema=EventMemory.expected_vector_store_collection_schema(),
            ),
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
                serialize_encode=args.eviction_similarity_threshold is not None,
            )
        )

        # Process sessions sequentially for deterministic eviction order.
        for session_id in question.session_id_map:
            session = question.get_session(session_id)
            events = [
                Event(
                    uuid=uuid4(),
                    timestamp=turn.timestamp,
                    body=Content(
                        context=MessageContext(
                            source="Assistant" if turn.role == "assistant" else "User"
                        ),
                        items=[Text(text=turn.content.strip())],
                    ),
                    properties={
                        "longmemeval_session_id": session_id,
                        "has_answer": turn.has_answer,
                        "turn_id": turn.index,
                    },
                )
                for turn in session
            ]
            await memory.encode_events(events)

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

    async def ingest_with_retry(question: LongMemEvalItem) -> bool:
        for attempt in range(1, args.max_retries + 1):
            try:
                await ingest_question(question)
                return True
            except Exception as e:
                print(
                    f"  FAILED question={question.question_id} "
                    f"attempt={attempt}/{args.max_retries}: {e}"
                )
                if attempt < args.max_retries:
                    # Clean up partial data before retrying.
                    try:
                        await delete_question_data(question.question_id)
                    except Exception:
                        pass
        return False

    start_time = time.monotonic()
    print(f"Ingestion started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    succeeded = 0
    failed_questions: list[str] = []
    num_batches = (len(all_questions) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch = all_questions[batch_start : batch_start + args.batch_size]

        print(
            f"Batch {batch_idx + 1}/{num_batches} "
            f"(questions {batch_start}..{batch_start + len(batch) - 1})"
        )

        results = await asyncio.gather(
            *[ingest_with_retry(q) for q in batch],
        )

        for question, ok in zip(batch, results):
            if ok:
                succeeded += 1
            else:
                failed_questions.append(question.question_id)

    elapsed = time.monotonic() - start_time
    print(
        f"\nIngestion finished at {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(elapsed: {elapsed:.1f}s)"
    )
    print(f"Succeeded: {succeeded}/{len(all_questions)}")
    if failed_questions:
        print(f"Failed ({len(failed_questions)}): {failed_questions}")

    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
