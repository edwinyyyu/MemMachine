"""Search the EventMemory store for LongMemEval questions and dump QueryResult JSON."""

import argparse
import asyncio
import json
import logging
import os
import time

import boto3
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.sqlite_vector_store import (
    SQLiteVectorStore,
    SQLiteVectorStoreParams,
)
from memmachine_server.common.vector_store.vector_search_engine import (
    VectorSearchEngine,
)
from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import (
    USearchVectorSearchEngine,
)
from memmachine_server.episodic_memory.event_memory.data_types import FormatOptions
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
from openai import AsyncOpenAI
from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry


def _configure_sqlite_for_perf(engine: AsyncEngine) -> None:
    """Set WAL + synchronous=NORMAL for ingest/search throughput.

    NORMAL is safe under WAL but can lose committed transactions on OS
    crash / power loss (no corruption). Acceptable tradeoff for benchmarks.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(
        dbapi_connection: DBAPIConnection,
        _connection_record: ConnectionPoolEntry,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


# CLDR-styled timestamps on retrieved segments. "long" date / "short" time
# gives e.g. "May 7, 2026 at 7:00 PM" in en_US — verbose enough for the LLM
# to reason about temporal questions without the wordiness of full+long.
_FORMAT_OPTIONS = FormatOptions(date_style="full", time_style="short")


def _partition_key(question_id: str) -> str:
    return question_id.lower().replace("-", "_")


def _make_usearch_engine(
    num_dimensions: int, similarity_metric: SimilarityMetric
) -> VectorSearchEngine:
    return USearchVectorSearchEngine(
        num_dimensions=num_dimensions, similarity_metric=similarity_metric
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", required=True, help="Path to the source data file"
    )
    parser.add_argument(
        "--target-path", required=True, help="Path to the target data file"
    )
    parser.add_argument(
        "--segment-db",
        default="longmemeval_segments.db",
        help="SQLite path for the segment store",
    )
    parser.add_argument(
        "--vector-db",
        default="longmemeval_vectors.db",
        help="SQLite path for the vector store",
    )
    parser.add_argument(
        "--index-directory",
        default="longmemeval_indexes",
        help="Directory holding USearch index files",
    )
    parser.add_argument(
        "--vector-search-limit",
        type=int,
        default=100,
        help="Number of vectors to retrieve",
    )
    parser.add_argument(
        "--expand-context",
        type=int,
        default=0,
        help="Number of context segments to expand",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent questions",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable the reranker and rank by embedding similarity only",
    )
    parser.add_argument(
        "--bm25-fusion",
        choices=["none", "additive", "rrf", "rsf"],
        default="none",
        help="BM25 fusion mode over the vector-retrieved candidate pool: "
        "'none' (default), 'additive' (calibrated additive with weighted "
        "semantic + sigmoid(bm25)), 'rrf' (Reciprocal Rank Fusion, k=60), "
        "'rsf' (Relative Score Fusion, max-normalized weighted average).",
    )
    parser.add_argument(
        "--bm25-fusion-weight",
        type=float,
        default=0.5,
        help="BM25 channel weight in [0.0, 1.0] for 'additive' and 'rsf' "
        "modes; semantic weight is 1 - weight (default: 0.5).",
    )
    args = parser.parse_args()

    all_questions = load_longmemeval_dataset(args.data_path)

    segment_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.segment_db}",
        connect_args={"timeout": 30},
        pool_size=20,
        max_overflow=80,
    )
    _configure_sqlite_for_perf(segment_engine)
    segment_store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=segment_engine)
    )
    await segment_store.startup()

    vector_engine = create_async_engine(
        f"sqlite+aiosqlite:///{args.vector_db}",
        connect_args={"timeout": 30},
        pool_size=20,
        max_overflow=80,
    )
    _configure_sqlite_for_perf(vector_engine)
    vector_store = SQLiteVectorStore(
        SQLiteVectorStoreParams(
            sqlalchemy_engine=vector_engine,
            vector_search_engine_factory=_make_usearch_engine,
            index_directory=args.index_directory,
        )
    )
    await vector_store.startup()

    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    )

    reranker: Reranker | None
    if args.no_reranker:
        reranker = None
    else:
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

    segmenter = TextSegmenter()
    deriver = WholeTextDeriver()

    namespace = "longmemeval"

    async def process_question(question: LongMemEvalItem):
        partition_key = _partition_key(question.question_id)

        collection = await vector_store.open_collection(
            namespace=namespace, name=partition_key
        )
        if collection is None:
            print(f"No collection for question {question.question_id}; skipping.")
            return None
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

        memory_start = time.monotonic()
        search_query = f"User: {question.question}"
        query_result = await memory.query(
            query=search_query,
            vector_search_limit=args.vector_search_limit,
            expand_context=args.expand_context,
            format_options=_FORMAT_OPTIONS,
            bm25_fusion=args.bm25_fusion,
            bm25_fusion_weight=args.bm25_fusion_weight,
        )
        memory_latency = time.monotonic() - memory_start

        print(
            f"Question ID: {question.question_id}\n"
            f"Question: {question.question}\n"
            f"Question Type: {question.question_type}\n"
            f"Memory retrieval time: {memory_latency:.2f} seconds\n"
        )

        result = {
            "question_id": question.question_id,
            "question_date": question.question_date,
            "question": question.question,
            "answer": question.answer,
            "answer_turn_indices": question.answer_turn_indices,
            "question_type": question.question_type.value,
            "abstention": question.abstention_question,
            "memory_latency": memory_latency,
            "query_result": query_result.model_dump(mode="json"),
        }

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

        return result

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        async_with(semaphore, process_question(question)) for question in all_questions
    ]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]

    with open(args.target_path, "w") as f:
        json.dump(results, f, indent=4)

    await segment_store.shutdown()
    await vector_store.shutdown()
    await segment_engine.dispose()
    await vector_engine.dispose()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig()
    logging.getLogger(
        "memmachine_server.episodic_memory.event_memory.event_memory"
    ).setLevel(logging.DEBUG)
    asyncio.run(main())
