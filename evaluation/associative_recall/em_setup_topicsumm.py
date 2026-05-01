"""EventMemory dual-view (topic-baked raw + turn-summary) ingestion for LoCoMo-30.

Goal 1: Combine Part A's two ingest augmentations on the same collection.

For each non-filler turn:
  - Event 1 (raw view):     MessageContext.source=<speaker>,
                            items=[Text(text="[topic: <topic>] <raw>")]
  - Event 2 (summary view): MessageContext.source=<speaker>,
                            items=[Text(text=<summary>)]
    * summary view has NO topic prefix -- the summary is already content-dense.
    * filler turns (empty/"<filler>" summary) fall back to raw-only.

Reuses caches:
  - cache/topicbake_llm_cache.json  (topics per turn)
  - cache/turnsumm_llm_cache.json   (summaries per turn)

New storage:
  - Qdrant collections: arc_em_lc30_topicsumm_v1_{26,30,41}
  - SQLite:             results/eventmemory_topicsumm.sqlite3
  - Namespace:          arc_em_locomo30_topicsumm

Run once:
    uv run python evaluation/associative_recall/em_setup_topicsumm.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import openai
from dotenv import load_dotenv
from em_setup_summ import (
    FILLER_MARKER,
    SUMMARY_MODEL,
    SummaryCache,
    _build_prev_context,
    _clean_summary,
    build_summary_prompt,
)
from em_setup_summ import (
    load_conversation_segments as load_summ_segments,
)

# Reuse the exact prompts / normalizers already used by em_setup_topic and
# em_setup_summ so cache keys line up and zero new LLM calls are needed.
from em_setup_topic import (
    TOPIC_MODEL,
    TopicCache,
    _build_topic_prompt,
    _normalize_topic,
)
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.utils import async_with
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
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

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"
CACHE_DIR = ASSOC_DIR / "cache"

TOPIC_LLM_CACHE = CACHE_DIR / "topicbake_llm_cache.json"
SUMMARY_LLM_CACHE = CACHE_DIR / "turnsumm_llm_cache.json"

# Qdrant collection naming: short prefix (<=32 bytes) for physical collection,
# verbose logical prefix kept in the manifest.
COLLECTION_PREFIX = "arc_em_lc30_topicsumm_v1"
LOGICAL_PREFIX = "arc_em_locomo30_topicsumm_v1"
NAMESPACE = "arc_em_locomo30_topicsumm"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]


def _conv_short(conv_id: str) -> str:
    return conv_id.rsplit("-", 1)[-1]


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f)["speakers"]


def _event_timestamp(base: datetime, turn_id: int) -> datetime:
    return base + timedelta(seconds=60 * turn_id)


def _build_topic_list_for_conv(
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
    topic_cache: TopicCache,
) -> dict[int, str]:
    """Lookup cached topic for each turn. Zero LLM calls (raises if miss)."""
    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"
    out: dict[int, str] = {}
    prev_text: str | None = None
    missing: list[int] = []
    for turn_id, role, text in segments:
        speaker = user_name if role == "user" else asst_name
        prompt = _build_topic_prompt(text, speaker, prev_text)
        raw = topic_cache.get(TOPIC_MODEL, prompt)
        if raw is None:
            missing.append(turn_id)
            out[turn_id] = "filler"
        else:
            out[turn_id] = _normalize_topic(raw)
        prev_text = text
    if missing:
        print(
            f"  [topic] WARNING: {len(missing)} turns missing from topic cache; "
            f"falling back to 'filler' (expected 0 if cache is fresh). "
            f"First few: {missing[:5]}"
        )
    return out


def _build_summary_list_for_conv(
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
    summary_cache: SummaryCache,
) -> dict[int, str]:
    """Lookup cached summary for each turn. Zero LLM calls (raises if miss)."""
    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"
    out: dict[int, str] = {}
    missing: list[int] = []
    for idx, (turn_id, role, text) in enumerate(segments):
        speaker = user_name if role == "user" else asst_name
        prev_ctx = _build_prev_context(segments, idx, speakers, n_prev=2)
        prompt = build_summary_prompt(text, speaker, prev_ctx)
        raw = summary_cache.get(SUMMARY_MODEL, prompt)
        if raw is None:
            missing.append(turn_id)
            out[turn_id] = FILLER_MARKER
        else:
            out[turn_id] = _clean_summary(raw)
    if missing:
        print(
            f"  [summary] WARNING: {len(missing)} turns missing from summary cache; "
            f"falling back to filler. First few: {missing[:5]}"
        )
    return out


async def ingest_conversation(
    conv_id: str,
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
    topics: dict[int, str],
    summaries: dict[int, str],
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
) -> dict:
    collection_name = f"{COLLECTION_PREFIX}_{_conv_short(conv_id)}"
    partition_key = collection_name

    await vector_store.delete_collection(namespace=NAMESPACE, name=collection_name)
    await segment_store.delete_partition(partition_key)

    collection = await vector_store.open_or_create_collection(
        namespace=NAMESPACE,
        name=collection_name,
        config=VectorStoreCollectionConfig(
            vector_dimensions=embedder.dimensions,
            similarity_metric=embedder.similarity_metric,
            properties_schema=EventMemory.expected_vector_store_collection_schema(),
        ),
    )
    partition = await segment_store.open_or_create_partition(partition_key)

    memory = EventMemory(
        EventMemoryParams(
            vector_store_collection=collection,
            segment_store_partition=partition,
            embedder=embedder,
            reranker=None,
            derive_sentences=False,
            max_text_chunk_length=500,
        )
    )

    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"
    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)

    events: list[Event] = []
    raw_count = 0
    summary_count = 0
    filler_summary_count = 0
    topic_filler_count = 0
    for turn_id, role, text in segments:
        source_name = user_name if role == "user" else asst_name
        ts = _event_timestamp(base_ts, turn_id)
        topic = topics.get(turn_id, "filler")
        summary = summaries.get(turn_id, FILLER_MARKER)
        if topic == "filler":
            topic_filler_count += 1

        # Raw view: speaker + topic-baked text.
        raw_text = f"[topic: {topic}] {text.strip()}"
        events.append(
            Event(
                uuid=uuid4(),
                timestamp=ts,
                body=Content(
                    context=MessageContext(source=source_name),
                    items=[Text(text=raw_text)],
                ),
                properties={
                    "arc_conversation_id": conv_id,
                    "turn_id": turn_id,
                    "role": role,
                    "topic": topic,
                    "view": "raw",
                },
            )
        )
        raw_count += 1

        # Summary view (no topic prefix): non-filler only.
        if summary and summary != FILLER_MARKER:
            events.append(
                Event(
                    uuid=uuid4(),
                    # Nudge timestamp by 1us for stable ordering inside the turn.
                    timestamp=ts + timedelta(microseconds=1),
                    body=Content(
                        context=MessageContext(source=source_name),
                        items=[Text(text=summary)],
                    ),
                    properties={
                        "arc_conversation_id": conv_id,
                        "turn_id": turn_id,
                        "role": role,
                        "topic": topic,
                        "view": "summary",
                    },
                )
            )
            summary_count += 1
        else:
            filler_summary_count += 1

    t0 = time.monotonic()
    await memory.encode_events(events)
    ingest_time = time.monotonic() - t0

    await segment_store.close_partition(partition)
    await vector_store.close_collection(collection=collection)

    return {
        "conversation_id": conv_id,
        "collection_name": collection_name,
        "logical_collection_name": f"{LOGICAL_PREFIX}_{conv_id}",
        "partition_key": partition_key,
        "namespace": NAMESPACE,
        "n_turns": len(segments),
        "n_raw_events": raw_count,
        "n_summary_events": summary_count,
        "n_filler_summary": filler_summary_count,
        "n_topic_filler": topic_filler_count,
        "ingest_time_s": round(ingest_time, 2),
        "user_name": user_name,
        "assistant_name": asst_name,
    }


async def main() -> None:
    print("[em_setup_topicsumm] loading data", flush=True)
    speakers_map = load_speaker_map()
    # Reuse em_setup_summ's loader (segments_extended_locomo_prefixed.npz).
    from em_setup_summ import SEGMENTS_FILE

    conv_segments = load_summ_segments(SEGMENTS_FILE)
    for cid in LOCOMO_CONV_IDS:
        print(f"  {cid}: {len(conv_segments[cid])} turns", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    topic_cache = TopicCache(TOPIC_LLM_CACHE)
    summary_cache = SummaryCache(SUMMARY_LLM_CACHE)

    # Per-conversation topic + summary maps (cache lookups only; zero new LLM).
    topics_by_conv: dict[str, dict[int, str]] = {}
    summaries_by_conv: dict[str, dict[int, str]] = {}
    for cid in LOCOMO_CONV_IDS:
        topics_by_conv[cid] = _build_topic_list_for_conv(
            conv_segments[cid], speakers_map[cid], topic_cache
        )
        summaries_by_conv[cid] = _build_summary_list_for_conv(
            conv_segments[cid], speakers_map[cid], summary_cache
        )

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sqlite_path = RESULTS_DIR / "eventmemory_topicsumm.sqlite3"
    sql_url_env = os.getenv("SQL_URL")
    sql_url = sql_url_env or f"sqlite+aiosqlite:///{sqlite_path}"
    try:
        if sql_url_env:
            engine = create_async_engine(sql_url, pool_size=50, max_overflow=50)
        else:
            engine = create_async_engine(sql_url)
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    except Exception as exc:
        print(f"[em_setup_topicsumm] SQL_URL {sql_url!r} not reachable: {exc!r}")
        sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
        print(f"[em_setup_topicsumm] falling back to SQLite at {sqlite_path}")
        engine = create_async_engine(sql_url)
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

    records = []
    try:
        sem = asyncio.Semaphore(3)
        tasks = [
            async_with(
                sem,
                ingest_conversation(
                    cid,
                    conv_segments[cid],
                    speakers_map[cid],
                    topics_by_conv[cid],
                    summaries_by_conv[cid],
                    vector_store,
                    segment_store,
                    embedder,
                ),
            )
            for cid in LOCOMO_CONV_IDS
        ]
        records = await asyncio.gather(*tasks)
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out_path = RESULTS_DIR / "eventmemory_topicsumm_collections.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "namespace": NAMESPACE,
                "prefix": COLLECTION_PREFIX,
                "logical_prefix": LOGICAL_PREFIX,
                "sql_url": sql_url,
                "max_text_chunk_length": 500,
                "derive_sentences": False,
                "topic_model": TOPIC_MODEL,
                "summary_model": SUMMARY_MODEL,
                "topic_cache": str(TOPIC_LLM_CACHE),
                "summary_cache": str(SUMMARY_LLM_CACHE),
                "conversations": records,
            },
            f,
            indent=2,
        )

    for r in records:
        print(
            f"[ingested-topicsumm] {r['conversation_id']}: "
            f"{r['n_raw_events']} raw + {r['n_summary_events']} summary "
            f"({r['n_topic_filler']} topic-filler, {r['n_filler_summary']} summ-filler) "
            f"in {r['ingest_time_s']}s -> {r['collection_name']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
