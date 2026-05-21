"""Ingest LoCoMo10 conversations into EventMemory.

Uses SQLite for everything: SQLAlchemySegmentStore for the segment store,
and SQLiteVecVectorStore (sqlite-vec) for the vector store. One segment-store
partition + one vector-store collection per conversation group.
"""

import argparse
import asyncio
import os

# Keep v2-fp and v3 imports for ablation reproducibility. The production
# rewriting segmenter is imported above from the package proper.
import sys as _sys
import time
from datetime import timedelta
from uuid import uuid4

import openai
from dotenv import load_dotenv
from length_routed_segmenter import LengthRoutedSegmenter
from locomo_models import (
    attachment_suffix,
    datetime_from_locomo_time,
    load_locomo_dataset,
)
from memmachine_server.common.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    SentenceTransformerEmbedderParams,
)
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    SurroundingEvent,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.llm_text_deriver import (
    LLMTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    SentenceTextDeriver,
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
from memmachine_server.episodic_memory.event_memory.segmenter.llm_text_segmenter import (
    LLMTextSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.rewrite_segmenter import (
    RewriteSegmenter as _RewriteSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from sentence_transformers import SentenceTransformer

_sys.path.insert(
    0,
    "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval/llm_pipeline_probe",
)
from probe_segmenter_rewrite_v2_fp import RewriteSegmenterFP as _RewriteSegmenterV2FP
from probe_segmenter_rewrite_v3 import RewriteSegmenter as _RewriteSegmenterV3
from probe_segmenter_rewrite_v4 import RewriteSegmenter as _RewriteSegmenterV4
from probe_segmenter_rewrite_v5 import RewriteSegmenter as _RewriteSegmenterV5
from probe_segmenter_rewrite_v6 import RewriteSegmenter as _RewriteSegmenterV6
from probe_segmenter_rewrite_v7 import RewriteSegmenter as _RewriteSegmenterV7
from probe_segmenter_rewrite_v8 import RewriteSegmenter as _RewriteSegmenterV8
from probe_segmenter_rewrite_v9 import RewriteSegmenter as _RewriteSegmenterV9
from probe_segmenter_rewrite_v10 import RewriteSegmenter as _RewriteSegmenterV10
from probe_segmenter_rewrite_v11 import RewriteSegmenter as _RewriteSegmenterV11
from probe_segmenter_rewrite_v12 import RewriteSegmenter as _RewriteSegmenterV12
from probe_segmenter_rewrite_v13 import RewriteSegmenter as _RewriteSegmenterV13
from probe_segmenter_rewrite_v14 import RewriteSegmenter as _RewriteSegmenterV14
from probe_segmenter_rewrite_v15 import RewriteSegmenter as _RewriteSegmenterV15
from probe_segmenter_rewrite_v16 import RewriteSegmenter as _RewriteSegmenterV16
from probe_segmenter_rewrite_v17 import RewriteSegmenter as _RewriteSegmenterV17
from probe_segmenter_rewrite_v18 import RewriteSegmenter as _RewriteSegmenterV18
from probe_segmenter_rewrite_v22 import RewriteSegmenter as _RewriteSegmenterV22
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


def _build_segmenter(args, openai_client):
    """Build the segmenter selected by --segmenter."""
    match args.segmenter:
        case "llm":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return LLMTextSegmenter(language_model=lm)
        case "llm-routed":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return LengthRoutedSegmenter(
                language_model=lm,
                threshold_chars=args.routed_threshold,
            )
        case "rewrite" | "rewrite-v2":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenter(language_model=lm)
        case "rewrite-v3":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV3(language_model=lm)
        case "rewrite-v2-fp":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV2FP(language_model=lm)
        case "rewrite-v4":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV4(language_model=lm)
        case "rewrite-v5":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV5(language_model=lm)
        case "rewrite-v6":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV6(language_model=lm)
        case "rewrite-v7":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV7(language_model=lm)
        case "rewrite-v8":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV8(language_model=lm)
        case "rewrite-v9":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV9(language_model=lm)
        case "rewrite-v10":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV10(language_model=lm)
        case "rewrite-v11":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV11(language_model=lm)
        case "rewrite-v12":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV12(language_model=lm)
        case "rewrite-v13":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV13(language_model=lm)
        case "rewrite-v14":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV14(language_model=lm)
        case "rewrite-v15":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV15(language_model=lm)
        case "rewrite-v16":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV16(language_model=lm)
        case "rewrite-v17":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV17(language_model=lm)
        case "rewrite-v18":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV18(language_model=lm)
        case "rewrite-v22":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.segmenter_model,
                    reasoning_effort=args.segmenter_reasoning,
                )
            )
            return _RewriteSegmenterV22(language_model=lm)
        case "text":
            return TextSegmenter(max_chunk_length=args.max_text_chunk_length)


def _build_deriver(args, openai_client):
    """Build the deriver selected by --deriver."""
    match args.deriver:
        case "llm":
            lm = OpenAIResponsesLanguageModel(
                OpenAIResponsesLanguageModelParams(
                    client=openai_client,
                    model=args.deriver_model,
                    reasoning_effort=args.deriver_reasoning,
                )
            )
            return LLMTextDeriver(language_model=lm)
        case "whole":
            return WholeTextDeriver()
        case "sentence":
            return SentenceTextDeriver()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument(
        "--segment-db",
        default="locomo_segments.db",
        help="SQLite path for the segment store",
    )
    parser.add_argument(
        "--vector-db",
        default="locomo_vectors.db",
        help="SQLite path for the sqlite-vec vector store",
    )
    parser.add_argument(
        "--group-index",
        type=int,
        default=None,
        help="Ingest only this group index",
    )
    parser.add_argument(
        "--max-text-chunk-length",
        type=int,
        default=500,
        help="Max code-point length for text chunks (TextSegmenter only)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent conversations",
    )
    parser.add_argument(
        "--segmenter",
        choices=[
            "text",
            "llm",
            "llm-routed",
            "rewrite",
            "rewrite-v2",
            "rewrite-v3",
            "rewrite-v2-fp",
            "rewrite-v4",
            "rewrite-v5",
            "rewrite-v6",
            "rewrite-v7",
            "rewrite-v8",
            "rewrite-v9",
            "rewrite-v10",
            "rewrite-v11",
            "rewrite-v12",
            "rewrite-v13",
            "rewrite-v14",
            "rewrite-v15",
            "rewrite-v16",
            "rewrite-v17",
            "rewrite-v18",
            "rewrite-v22",
        ],
        default="text",
        help="Segmenter type: 'text' (TextSegmenter, recursive splitter), "
        "'llm' (LLMTextSegmenter, v33 prompt), or 'llm-routed' (v47s for "
        "short inputs, v33 for long, threshold via --routed-threshold). "
        "Default: text.",
    )
    parser.add_argument(
        "--routed-threshold",
        type=int,
        default=200,
        help="Char-length threshold for --segmenter llm-routed (default: 200).",
    )
    parser.add_argument(
        "--segmenter-model",
        default="gpt-5.4-nano",
        help="OpenAI model for --segmenter llm (default: gpt-5.4-nano).",
    )
    parser.add_argument(
        "--segmenter-reasoning",
        default="low",
        help="reasoning_effort for --segmenter llm (default: low).",
    )
    parser.add_argument(
        "--deriver",
        choices=["sentence", "whole", "llm"],
        default="sentence",
        help="Deriver type: 'sentence' (one derivative per sentence), "
        "'whole' (one derivative per whole segment), "
        "or 'llm' (LLMTextDeriver, v65 prompt). Default: sentence.",
    )
    parser.add_argument(
        "--deriver-model",
        default="gpt-5-nano",
        help="OpenAI model for --deriver llm (default: gpt-5-nano).",
    )
    parser.add_argument(
        "--deriver-reasoning",
        default="low",
        help="reasoning_effort for --deriver llm (default: low).",
    )

    args = parser.parse_args()

    locomo_data = load_locomo_dataset(args.data_path)

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
    vector_store = SQLiteVecVectorStore(
        SQLiteVecVectorStoreParams(engine=vector_engine)
    )
    await vector_store.startup()

    openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    st_model = SentenceTransformer(
        "google/embeddinggemma-300m",
        token=os.getenv("HF_TOKEN"),
    )
    embedder = SentenceTransformerEmbedder(
        SentenceTransformerEmbedderParams(
            model_name="google/embeddinggemma-300m",
            sentence_transformer=st_model,
            max_input_length=2048,
            batch_size=32,
        )
    )

    segmenter = _build_segmenter(args, openai_client)
    deriver = _build_deriver(args, openai_client)

    namespace = "locomo"
    schema = EventMemory.expected_vector_store_collection_schema()

    async def process_conversation(idx: int, item: dict) -> None:
        if "conversation" not in item:
            return

        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        print(
            f"Processing conversation for group {idx} with speakers "
            f"{speaker_a} and {speaker_b}..."
        )

        partition_key = f"group_{idx}"

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

        session_idx = 0
        while True:
            session_idx += 1
            session_id = f"session_{session_idx}"
            if session_id not in conversation:
                break

            session = conversation[session_id]
            session_datetime = datetime_from_locomo_time(
                conversation[f"{session_id}_date_time"]
            )

            message_texts = [m["text"] + attachment_suffix(m) for m in session]
            neighbor_window = getattr(args, "neighbor_window", 2)
            neighbor_direction = getattr(args, "neighbor_direction", "both")
            events: list[Event] = []
            for message_index, message in enumerate(session):
                content = message_texts[message_index]
                before_window = (
                    neighbor_window if neighbor_direction in ("both", "before") else 0
                )
                after_window = (
                    neighbor_window if neighbor_direction in ("both", "after") else 0
                )
                lo = max(0, message_index - before_window)
                before = [
                    SurroundingEvent(
                        producer=session[j]["speaker"],
                        text=message_texts[j].strip(),
                    )
                    for j in range(lo, message_index)
                ]
                hi = min(len(session), message_index + 1 + after_window)
                after = [
                    SurroundingEvent(
                        producer=session[j]["speaker"],
                        text=message_texts[j].strip(),
                    )
                    for j in range(message_index + 1, hi)
                ]
                events.append(
                    Event(
                        uuid=uuid4(),
                        timestamp=session_datetime
                        + message_index * timedelta(seconds=1),
                        context=SurroundingEventsContext(
                            producer=message["speaker"],
                            before=before,
                            after=after,
                        ),
                        blocks=[TextBlock(text=content.strip())],
                        properties={
                            "locomo_session_id": session_id,
                            "dia_id": message.get("dia_id", ""),
                        },
                    )
                )

            try:
                await memory.encode_events(events)
            except Exception as e:
                print(
                    f"Error ingesting group={idx} session={session_id} "
                    f"({len(events)} events): {e}"
                )
                raise

        await segment_store.close_partition(segment_store_partition)
        await vector_store.close_collection(collection=collection)

    indices = (
        [args.group_index] if args.group_index is not None else range(len(locomo_data))
    )

    start_time = time.monotonic()
    print(f"Ingestion started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            async_with(semaphore, process_conversation(idx, locomo_data[idx]))
            for idx in indices
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
