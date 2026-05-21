"""Ingest LongMemEval into EventMemory.

Uses SQLite for everything: SQLAlchemySegmentStore for the segment store,
and SQLiteVectorStore with the USearch search engine for the vector store.
One segment-store partition + one vector-store collection per question.
"""

import argparse
import asyncio
import os
import time
from pathlib import Path
from uuid import uuid4

import openai
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
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store import (
    VectorStoreCollectionConfig,
)
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
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.cue_worthiness_filtering_deriver import (
    CueWorthinessFilteringDeriver,
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
from memmachine_server.episodic_memory.event_memory.segmenter.rewrite_segmenter import (
    RewriteSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
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


def _ensure_index_directory(directory: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)


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
    parser.add_argument("--data-path", required=True, help="Path to the data file")
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
        help="Directory for persisting USearch index files",
    )
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
        "--max-text-chunk-length",
        type=int,
        default=500,
        help="Max code-point length for text chunks (TextSegmenter only)",
    )
    parser.add_argument(
        "--segmenter",
        choices=["text", "rewrite"],
        default="text",
        help="Segmenter type: 'text' (TextSegmenter) or 'rewrite' "
        "(RewriteSegmenter — deterministic split + LLM third-person "
        "rewrite). Default: text.",
    )
    parser.add_argument(
        "--segmenter-model",
        default="gpt-5.4-nano",
        help="OpenAI model for --segmenter rewrite.",
    )
    parser.add_argument(
        "--segmenter-reasoning",
        default="low",
        help="reasoning_effort for --segmenter rewrite.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent questions",
    )
    parser.add_argument(
        "--session-concurrency",
        type=int,
        default=50,
        help="Max concurrent sessions across all questions",
    )
    parser.add_argument(
        "--filter-cues",
        action="store_true",
        help="Wrap the deriver in a CueWorthinessFilteringDeriver to drop "
        "messages an LLM judges to be conversational plumbing before embedding.",
    )
    parser.add_argument(
        "--cue-filter-model",
        default="gpt-5.4-nano",
        help="OpenAI model for cue-worthiness classification.",
    )
    parser.add_argument(
        "--cue-filter-reasoning",
        default="low",
        help="reasoning_effort for the cue-worthiness model.",
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
    print(f"{len(all_questions)} total questions")

    _ensure_index_directory(args.index_directory)

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
        connect_args={"timeout": 600},
        pool_size=200,
        max_overflow=800,
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

    openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    match args.segmenter:
        case "rewrite":
            segmenter_lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            segmenter = RewriteSegmenter(language_model=segmenter_lm)
        case _:
            segmenter = TextSegmenter(max_chunk_length=args.max_text_chunk_length)
    deriver = WholeTextDeriver()
    if args.filter_cues:
        cue_lm = OpenAIResponsesLanguageModel(
            OpenAIResponsesLanguageModelParams(
                client=openai_client,
                model=args.cue_filter_model,
                reasoning_effort=args.cue_filter_reasoning,
            )
        )
        deriver = CueWorthinessFilteringDeriver(
            inner=deriver,
            language_model=cue_lm,
        )

    namespace = "longmemeval"
    schema = EventMemory.expected_vector_store_collection_schema()

    session_semaphore = asyncio.Semaphore(args.session_concurrency)

    async def process_conversation(question: LongMemEvalItem):
        partition_key = _partition_key(question.question_id)
        session_ids = list(question.session_id_map.keys())

        await vector_store.delete_collection(namespace=namespace, name=partition_key)
        await segment_store.delete_partition(partition_key)

        collection = await vector_store.open_or_create_collection(
            namespace=namespace,
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
            )
        )

        async def ingest_session(session_id: str):
            session = question.get_session(session_id)

            events: list[Event] = []
            for turn in session:
                if turn.timestamp is None:
                    raise ValueError(
                        f"Turn {turn.index} of session {session_id} has no timestamp"
                    )
                source = "Assistant" if turn.role == "assistant" else "User"
                events.append(
                    Event(
                        uuid=uuid4(),
                        timestamp=turn.timestamp,
                        context=ProducerContext(producer=source),
                        blocks=[TextBlock(text=turn.content.strip())],
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

        await asyncio.gather(
            *[async_with(session_semaphore, ingest_session(sid)) for sid in session_ids]
        )

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

    start_time = time.monotonic()
    print(f"Ingestion started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        semaphore = asyncio.Semaphore(args.concurrency)
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
    await segment_engine.dispose()
    await vector_engine.dispose()
    await openai_client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
