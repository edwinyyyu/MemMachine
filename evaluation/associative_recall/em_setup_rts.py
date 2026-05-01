"""EventMemory ingestion for LoCoMo-30 WITH REAL SESSION TIMESTAMPS.

Mirror of em_setup.py but:
  - Parses real `session_{i}_date_time` from evaluation/data/locomo10.json
    (using the exact parser mandated by the task).
  - Assigns each turn a timestamp = session_dt + timedelta(seconds=k) where
    k is the index of the turn WITHIN its session. This keeps timestamps
    strictly monotonic for EM.expand_context while still making
    cross-session queries resolvable to real calendar dates.
  - Writes to NEW Qdrant collections (prefix `arc_em_lc30_rts_v1_<short>`)
    and a NEW SQLite path (`results/eventmemory_locomo_rts.sqlite3`).
  - Stores session_date_time string and derived fields in metadata.

Run once:
    uv run python evaluation/associative_recall/em_setup_rts.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import openai
from dotenv import load_dotenv
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

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# New, disjoint prefix (max 32 bytes per Qdrant rules).
COLLECTION_PREFIX = "arc_em_lc30_rts_v1"
LOGICAL_PREFIX = "arc_em_locomo30_rts_v1"
NAMESPACE = "arc_em_locomo30_rts"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]
# Maps our conv_id -> index in locomo10.json
LOCOMO10_INDEX = {
    "locomo_conv-26": 0,
    "locomo_conv-30": 1,
    "locomo_conv-41": 2,
}
LOCOMO10_PATH = ROOT / "evaluation" / "data" / "locomo10.json"


def datetime_from_locomo_time(locomo_time_str: str) -> datetime:
    """User-mandated parser. Format: '4:00 pm on 30 April, 2023'."""
    return datetime.strptime(locomo_time_str, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=UTC
    )


def _conv_short(conv_id: str) -> str:
    return conv_id.rsplit("-", 1)[-1]


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f)["speakers"]


def load_conversation_segments(
    _npz_path_unused: Path,
) -> dict[str, list[tuple[int, str, str]]]:
    """Return dict conv_id -> sorted list of (turn_id, role, text).

    Instead of loading the heavy segments_extended.npz file, we synthesize
    the same structure from locomo10.json. Turn ids are 0..N-1 in session
    order (verified identical to segments_extended.npz). Role is derived
    from the speaker: the `user` role belongs to speaker_a, `assistant` to
    speaker_b — matches em_setup.py's assignment convention via
    conversation_two_speakers.json.
    """
    with open(LOCOMO10_PATH) as f:
        locomo_raw = json.load(f)

    out: dict[str, list[tuple[int, str, str]]] = {}
    for conv_id in LOCOMO_CONV_IDS:
        jidx = LOCOMO10_INDEX[conv_id]
        conv = locomo_raw[jidx]["conversation"]
        speaker_a = conv["speaker_a"]
        tid = 0
        segs: list[tuple[int, str, str]] = []
        for s in range(1, 1000):
            key = f"session_{s}"
            if key not in conv:
                break
            for msg in conv[key]:
                role = "user" if msg["speaker"] == speaker_a else "assistant"
                segs.append((tid, role, msg["text"]))
                tid += 1
        out[conv_id] = segs
    return out


def load_locomo10_turn_timestamps() -> dict[str, list[dict]]:
    """Return conv_id -> list indexed by turn_id of dicts:
        {session_idx, in_session_idx, session_date_time, speaker, dia_id}
    The flat list is in turn_id order (0..N-1 within each conv).
    """
    with open(LOCOMO10_PATH) as f:
        locomo_raw = json.load(f)

    out: dict[str, list[dict]] = {}
    for conv_id, jidx in LOCOMO10_INDEX.items():
        conv = locomo_raw[jidx]["conversation"]
        flat: list[dict] = []
        for s in range(1, 1000):
            key = f"session_{s}"
            if key not in conv:
                break
            dt_str = conv[f"session_{s}_date_time"]
            for k, msg in enumerate(conv[key]):
                flat.append(
                    {
                        "session_idx": s,
                        "in_session_idx": k,
                        "session_date_time": dt_str,
                        "speaker": msg["speaker"],
                        "dia_id": msg["dia_id"],
                    }
                )
        out[conv_id] = flat
    return out


async def ingest_conversation(
    conv_id: str,
    segments: list[tuple[int, str, str]],
    turn_meta: list[dict],
    speakers: dict[str, str],
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

    # Precompute per-session base datetimes so monotonic within-session
    # offsets don't collide across sessions. Different sessions already
    # differ by real hours/days; we add `in_session_idx` seconds only.
    events: list[Event] = []
    parse_samples: list[dict] = []
    for turn_id, role, text in segments:
        if turn_id >= len(turn_meta):
            raise RuntimeError(
                f"{conv_id}: turn_id {turn_id} beyond turn_meta len {len(turn_meta)}"
            )
        meta = turn_meta[turn_id]
        base_dt = datetime_from_locomo_time(meta["session_date_time"])
        ts = base_dt.replace(
            microsecond=meta["in_session_idx"]
        )  # use microsecond slot for within-session offset
        source_name = user_name if role == "user" else asst_name
        events.append(
            Event(
                uuid=uuid4(),
                timestamp=ts,
                body=Content(
                    context=MessageContext(source=source_name),
                    items=[Text(text=text.strip())],
                ),
                properties={
                    "arc_conversation_id": conv_id,
                    "turn_id": turn_id,
                    "role": role,
                    "session_idx": meta["session_idx"],
                    "in_session_idx": meta["in_session_idx"],
                    "session_date_time": meta["session_date_time"],
                    "dia_id": meta["dia_id"],
                },
            )
        )
        if len(parse_samples) < 5:
            parse_samples.append(
                {
                    "turn_id": turn_id,
                    "session_idx": meta["session_idx"],
                    "session_date_time": meta["session_date_time"],
                    "parsed_iso": base_dt.isoformat(),
                    "final_ts_iso": ts.isoformat(),
                    "speaker": meta["speaker"],
                    "text_preview": text.strip()[:60],
                }
            )

    t0 = time.monotonic()
    await memory.encode_events(events)
    ingest_time = time.monotonic() - t0

    # Derive per-session unique dates for the report
    unique_sessions: list[dict] = []
    seen = set()
    for meta in turn_meta:
        key = (meta["session_idx"], meta["session_date_time"])
        if key in seen:
            continue
        seen.add(key)
        parsed = datetime_from_locomo_time(meta["session_date_time"])
        unique_sessions.append(
            {
                "session_idx": meta["session_idx"],
                "session_date_time": meta["session_date_time"],
                "parsed_iso": parsed.isoformat(),
            }
        )

    await segment_store.close_partition(partition)
    await vector_store.close_collection(collection=collection)

    return {
        "conversation_id": conv_id,
        "collection_name": collection_name,
        "logical_collection_name": f"{LOGICAL_PREFIX}_{conv_id}",
        "partition_key": partition_key,
        "namespace": NAMESPACE,
        "n_events": len(events),
        "ingest_time_s": round(ingest_time, 2),
        "user_name": user_name,
        "assistant_name": asst_name,
        "n_sessions": len(unique_sessions),
        "first_session_dt": unique_sessions[0]["parsed_iso"]
        if unique_sessions
        else None,
        "last_session_dt": unique_sessions[-1]["parsed_iso"]
        if unique_sessions
        else None,
        "parse_samples": parse_samples,
        "sessions": unique_sessions,
    }


async def main() -> None:
    speakers_map = load_speaker_map()
    conv_segments = load_conversation_segments(DATA_DIR / "segments_extended.npz")
    turn_meta_all = load_locomo10_turn_timestamps()

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sqlite_path = RESULTS_DIR / "eventmemory_locomo_rts.sqlite3"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
        print(f"[em_setup_rts] SQL_URL {sql_url!r} not reachable: {exc!r}")
        sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
        print(f"[em_setup_rts] falling back to SQLite at {sqlite_path}")
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
        semaphore = asyncio.Semaphore(3)
        tasks = [
            async_with(
                semaphore,
                ingest_conversation(
                    conv_id,
                    conv_segments[conv_id],
                    turn_meta_all[conv_id],
                    speakers_map[conv_id],
                    vector_store,
                    segment_store,
                    embedder,
                ),
            )
            for conv_id in LOCOMO_CONV_IDS
        ]
        records = await asyncio.gather(*tasks)
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eventmemory_locomo_rts_collections.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "namespace": NAMESPACE,
                "prefix": COLLECTION_PREFIX,
                "logical_prefix": LOGICAL_PREFIX,
                "sql_url": sql_url,
                "max_text_chunk_length": 500,
                "derive_sentences": False,
                "timestamp_source": "locomo10.json session_{i}_date_time",
                "timestamp_format": "%I:%M %p on %d %B, %Y",
                "conversations": records,
            },
            f,
            indent=2,
            default=str,
        )

    # Write ingest log (markdown)
    md_lines: list[str] = [
        "# EventMemory LoCoMo real-timestamp ingest",
        "",
        "## Parser",
        "",
        "```",
        'datetime.strptime(s, "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)',
        "```",
        "",
        "## Per-conversation summary",
        "",
        "| Conversation | n_events | n_sessions | first_session | last_session | ingest_s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        md_lines.append(
            f"| {r['conversation_id']} | {r['n_events']} | {r['n_sessions']} | "
            f"{r['first_session_dt']} | {r['last_session_dt']} | "
            f"{r['ingest_time_s']} |"
        )
    md_lines += ["", "## Parse samples", ""]
    for r in records:
        md_lines.append(f"### {r['conversation_id']}")
        md_lines.append("")
        md_lines.append(
            "| turn_id | session_idx | session_date_time | parsed_iso | speaker | text |"
        )
        md_lines.append("| --- | --- | --- | --- | --- | --- |")
        for s in r["parse_samples"]:
            md_lines.append(
                f"| {s['turn_id']} | {s['session_idx']} | "
                f"{s['session_date_time']} | {s['parsed_iso']} | "
                f"{s['speaker']} | {s['text_preview']} |"
            )
        md_lines.append("")
    (RESULTS_DIR / "locomo_rts_ingest.md").write_text("\n".join(md_lines))

    for r in records:
        print(
            f"[ingested] {r['conversation_id']}: "
            f"{r['n_events']} events / {r['n_sessions']} sessions "
            f"in {r['ingest_time_s']}s "
            f"(user={r['user_name']}, asst={r['assistant_name']}) "
            f"-> {r['collection_name']}",
            flush=True,
        )
    print(f"Saved: {out_path}")
    print(f"Saved: {RESULTS_DIR / 'locomo_rts_ingest.md'}")


if __name__ == "__main__":
    asyncio.run(main())
