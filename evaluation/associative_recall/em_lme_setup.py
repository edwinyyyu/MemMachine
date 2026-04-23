"""EventMemory ingestion for LongMemEval-hard (90 questions).

Ingests each question's haystack_sessions into a dedicated EventMemory
collection keyed by question_id. Speaker is baked in via MessageContext
(User / Assistant). Timestamps come from haystack_dates (per session);
turns inside a session get monotonic sub-second offsets so EventMemory's
expand_context can walk adjacent turns.

Outputs:
  results/em_lme_hard_collections.json  (manifest)
  results/eventmemory_lme.sqlite3        (segment store; distinct from
                                          LoCoMo's eventmemory.sqlite3 to
                                          avoid collision with the other
                                          running agents)

Collection naming: arc_em_lmehard_v1_<question_id>

Run once:
    uv run python evaluation/associative_recall/em_lme_setup.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import openai
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

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


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(ROOT / ".env", override=False)

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"

LME_SRC = ROOT / "evaluation" / "data" / "longmemeval_s_cleaned.json"
HARD_QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_hard.json"

COLLECTION_PREFIX = "arc_em_lmehard_v1"
NAMESPACE = "arc_em_lmehard"
SQLITE_FILE = RESULTS_DIR / "eventmemory_lme.sqlite3"
COLLECTIONS_OUT = RESULTS_DIR / "em_lme_hard_collections.json"

USER_NAME = "User"
ASSISTANT_NAME = "Assistant"

# Parallel ingests of independent collections.
INGEST_CONCURRENCY = 3


def _parse_haystack_date(date_str: str) -> datetime:
    """Parse LME haystack_dates like '2023/05/20 (Sat) 00:31'."""
    clean = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
    dt = datetime.strptime(clean, "%Y/%m/%d %H:%M")
    return dt.replace(tzinfo=timezone.utc)


def _load_hard_questions() -> list[dict]:
    with open(HARD_QUESTIONS_JSON) as f:
        return json.load(f)


def _load_raw_lme_records(question_ids: set[str]) -> dict[str, dict]:
    """Load full LME records for just the hard-90 question ids."""
    with open(LME_SRC) as f:
        all_qs = json.load(f)
    return {q["question_id"]: q for q in all_qs if q["question_id"] in question_ids}


def _flatten_to_events(raw: dict) -> list[Event]:
    """Build Event objects from a raw LME question record.

    turn_id is the global index across sessions in the ORIGINAL haystack
    order — this matches `questions_longmemeval_hard.json`'s source_chat_ids
    which was computed that way.  Timestamp uses the parsed haystack_dates
    per session, with monotonic +1s offsets inside a session.  ~30/90
    questions have haystack_dates that are not already chronological, but
    we still iterate in the original order so turn_id == source_chat_id.
    EventMemory sorts events by timestamp internally for storage but
    properties.turn_id is what recall is scored against.
    """
    qid = raw["question_id"]
    events: list[Event] = []
    turn_idx = 0

    for sess_id, turns, date_str in zip(
        raw["haystack_session_ids"],
        raw["haystack_sessions"],
        raw["haystack_dates"],
        strict=True,
    ):
        base = _parse_haystack_date(date_str)
        for ti, turn in enumerate(turns):
            role = turn.get("role", "user")
            text = turn.get("content") or ""
            if not isinstance(text, str):
                text = str(text)
            source_name = USER_NAME if role == "user" else ASSISTANT_NAME
            ts = base + timedelta(seconds=ti)
            events.append(
                Event(
                    uuid=uuid4(),
                    timestamp=ts,
                    body=Content(
                        context=MessageContext(source=source_name),
                        items=[Text(text=text.strip())],
                    ),
                    properties={
                        "arc_question_id": qid,
                        "session_id": sess_id,
                        "turn_id": turn_idx,
                        "role": role,
                    },
                )
            )
            turn_idx += 1
    return events


async def ingest_question(
    raw: dict,
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
) -> dict:
    qid = raw["question_id"]
    collection_name = f"{COLLECTION_PREFIX}_{qid}"
    partition_key = collection_name

    # Wipe any stale data so reruns are idempotent.
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

    events = _flatten_to_events(raw)

    t0 = time.monotonic()
    await memory.encode_events(events)
    ingest_time = time.monotonic() - t0

    await segment_store.close_partition(partition)
    await vector_store.close_collection(collection=collection)

    return {
        "question_id": qid,
        "collection_name": collection_name,
        "partition_key": partition_key,
        "namespace": NAMESPACE,
        "n_events": len(events),
        "ingest_time_s": round(ingest_time, 2),
    }


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hard_qs = _load_hard_questions()
    hard_ids = {q["question_id"] for q in hard_qs}
    print(f"Loading raw LME records for {len(hard_ids)} hard questions ...", flush=True)
    raw_map = _load_raw_lme_records(hard_ids)
    missing = hard_ids - set(raw_map)
    if missing:
        raise RuntimeError(f"Missing raw records for: {sorted(missing)}")
    print(f"  got {len(raw_map)} raw records", flush=True)

    total_turns = sum(
        sum(len(s) for s in raw_map[qid]["haystack_sessions"]) for qid in hard_ids
    )
    print(f"Total haystack turns (events): {total_turns}", flush=True)

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    # Always use a dedicated SQLite file for LME-hard to avoid collision
    # with LoCoMo data in eventmemory.sqlite3 (other agents are using it).
    sql_url = f"sqlite+aiosqlite:///{SQLITE_FILE}"
    # Try Postgres first IF the user has SQL_URL set; the em_setup fallback
    # pattern shows Postgres auth often fails in this env, so we prefer
    # SQLite-from-start for isolation anyway.  Comment preserved for future
    # re-enable:
    # sql_url = os.getenv("SQL_URL") or f"sqlite+aiosqlite:///{SQLITE_FILE}"
    print(f"[em_lme_setup] using segment store: {sql_url}", flush=True)
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

    records: list[dict] = []
    try:
        semaphore = asyncio.Semaphore(INGEST_CONCURRENCY)
        tasks = [
            async_with(
                semaphore,
                ingest_question(
                    raw_map[q["question_id"]],
                    vector_store,
                    segment_store,
                    embedder,
                ),
            )
            for q in hard_qs
        ]
        t_all = time.monotonic()
        # Resolve in submission order so logs are stable, but via gather.
        records = await asyncio.gather(*tasks)
        total_elapsed = time.monotonic() - t_all
    finally:
        await segment_store.shutdown()
        await vector_store.shutdown()
        await engine.dispose()
        await qdrant_client.close()
        await openai_client.close()

    out = {
        "namespace": NAMESPACE,
        "prefix": COLLECTION_PREFIX,
        "sql_url": sql_url,
        "sqlite_file": str(SQLITE_FILE),
        "max_text_chunk_length": 500,
        "derive_sentences": False,
        "user_name": USER_NAME,
        "assistant_name": ASSISTANT_NAME,
        "n_questions": len(records),
        "n_events_total": sum(r["n_events"] for r in records),
        "ingest_total_s": round(total_elapsed, 1),
        "questions": records,
    }
    with open(COLLECTIONS_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {COLLECTIONS_OUT}", flush=True)
    print(
        f"Done: n_questions={out['n_questions']} n_events={out['n_events_total']} "
        f"in {out['ingest_total_s']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
