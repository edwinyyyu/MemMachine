"""EventMemory ingest with dual-view (raw + summary) turn indexing.

For each non-filler turn in LoCoMo-26/30/41:
  1. Generate a 1-sentence third-person summary via gpt-5-mini (cached)
  2. Ingest TWO events per turn, sharing the same turn_id + timestamp:
       - raw view: MessageContext.source = <real speaker>, items=[Text(text=raw)]
       - summary view: MessageContext.source = <real speaker>,
         items=[Text(text=summary)], with metadata flag "view":"summary"
     Non-filler turns with an empty/"<filler>" summary fall back to raw-only.

Collections & store:
  - Qdrant collection prefix: arc_em_lc30_summ_v1_{26,30,41}
  - SQLite: results/eventmemory_summ.sqlite3
  - Namespace: arc_em_locomo30_summ

Summary cache: cache/turnsumm_llm_cache.json (dedicated).

Run once:
    uv run python evaluation/associative_recall/em_setup_summ.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
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

DATA_DIR = Path(__file__).resolve().parent / "data"
# Use the locomo-only subset for ~100x faster loading than segments_extended.npz
# (9MB vs 3.7GB). Texts are identical (verified sample-equivalent to the full
# file) — conversation_ids/turn_ids/roles/texts are intact.
SEGMENTS_FILE = DATA_DIR / "segments_extended_locomo_prefixed.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
COLLECTION_PREFIX = "arc_em_lc30_summ_v1"
LOGICAL_PREFIX = "arc_em_locomo30_summ_v1"
NAMESPACE = "arc_em_locomo30_summ"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]

SUMMARY_MODEL = "gpt-5-mini"
SUMMARY_CACHE_FILE = CACHE_DIR / "turnsumm_llm_cache.json"

SUMMARY_PROMPT = """\
Summarize this conversation turn as a single declarative sentence using \
third-person reference. Extract the key fact or assertion.

Examples:
- "Yeah, I went yesterday, it was really powerful." (preceding: "Did you go to the LGBTQ support group?") -> "Caroline attended the LGBTQ support group on the day before this message."
- "About 16 weeks, starting next Monday." (preceding: "How long is Project Phoenix?") -> "Project Phoenix is estimated at 16 weeks starting the Monday after this message."
- "Haha yeah." -> "<filler>" (if no specific content)

Turn: {turn_text}
Speaker: {speaker}
Preceding context:
{prev_context}

Output: one declarative sentence OR exactly "<filler>" if no extractable content. Output only the sentence, nothing else."""


FILLER_MARKER = "<filler>"


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class SummaryCache:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        self._pending: dict[str, str] = {}

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(_sha(model, prompt))

    def put(self, model: str, prompt: str, value: str) -> None:
        key = _sha(model, prompt)
        self._cache[key] = value
        self._pending[key] = value

    def save(self) -> None:
        if not self._pending:
            return
        existing: dict[str, str] = {}
        if self._path.exists():
            try:
                with open(self._path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(self._pending)
        tmp = self._path.with_suffix(".json.tmp")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(existing, f)
        tmp.replace(self._path)
        self._pending.clear()


def _conv_short(conv_id: str) -> str:
    return conv_id.rsplit("-", 1)[-1]


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f)["speakers"]


def load_conversation_segments(
    npz_path: Path,
) -> dict[str, list[tuple[int, str, str]]]:
    d = np.load(npz_path, allow_pickle=True)
    out: dict[str, list[tuple[int, str, str]]] = {}
    for i in range(len(d["texts"])):
        cid = str(d["conversation_ids"][i])
        if cid not in LOCOMO_CONV_IDS:
            continue
        out.setdefault(cid, []).append(
            (int(d["turn_ids"][i]), str(d["roles"][i]), str(d["texts"][i]))
        )
    for cid in out:
        out[cid].sort(key=lambda t: t[0])
    return out


def _event_timestamp(base: datetime, turn_id: int) -> datetime:
    return base + timedelta(seconds=60 * turn_id)


def _build_prev_context(
    segments: list[tuple[int, str, str]],
    idx: int,
    speakers: dict[str, str],
    n_prev: int = 2,
) -> str:
    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"
    lines: list[str] = []
    for j in range(max(0, idx - n_prev), idx):
        _tid, role, text = segments[j]
        sp = user_name if role == "user" else asst_name
        lines.append(f"{sp}: {text.strip()}")
    if not lines:
        return "(none)"
    return "\n".join(lines)


def build_summary_prompt(
    turn_text: str, speaker: str, prev_context: str
) -> str:
    return SUMMARY_PROMPT.format(
        turn_text=turn_text.strip(),
        speaker=speaker,
        prev_context=prev_context,
    )


def _clean_summary(raw: str) -> str:
    """Strip quotes/whitespace; collapse to single line. Detect filler."""
    t = raw.strip().strip("'").strip('"').strip()
    if not t:
        return FILLER_MARKER
    # If the model returned the filler marker, normalize.
    if FILLER_MARKER in t:
        return FILLER_MARKER
    # Single line.
    t = t.split("\n")[0].strip()
    if not t:
        return FILLER_MARKER
    return t


async def generate_summaries(
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
    openai_client,
    cache: SummaryCache,
    *,
    concurrency: int = 10,
) -> dict[int, str]:
    """Return {turn_id: summary_or_filler}. Cached by prompt hash."""
    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"

    prompts: list[tuple[int, str]] = []
    for idx, (tid, role, text) in enumerate(segments):
        speaker = user_name if role == "user" else asst_name
        prev_ctx = _build_prev_context(segments, idx, speakers, n_prev=2)
        prompt = build_summary_prompt(text, speaker, prev_ctx)
        prompts.append((tid, prompt))

    result: dict[int, str] = {}
    sem = asyncio.Semaphore(concurrency)

    async def one(tid: int, prompt: str) -> None:
        async with sem:
            cached = cache.get(SUMMARY_MODEL, prompt)
            if cached is not None:
                result[tid] = _clean_summary(cached)
                return
            for attempt in range(3):
                try:
                    resp = await openai_client.chat.completions.create(
                        model=SUMMARY_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = resp.choices[0].message.content or ""
                    cache.put(SUMMARY_MODEL, prompt, text)
                    result[tid] = _clean_summary(text)
                    return
                except Exception as e:
                    if attempt == 2:
                        print(f"  [summary fail] turn {tid}: {e!r}")
                        result[tid] = FILLER_MARKER
                        return
                    await asyncio.sleep(1.0 + attempt)

    tasks = [one(tid, prompt) for tid, prompt in prompts]
    # Periodic save every N completions.
    done_counter = {"n": 0}

    total = len(tasks)

    async def wrapped(t):
        await t
        done_counter["n"] += 1
        n = done_counter["n"]
        if n % 50 == 0 or n == total:
            print(f"    summary progress {n}/{total}", flush=True)
            cache.save()

    await asyncio.gather(*[wrapped(t) for t in tasks])
    cache.save()
    return result


async def ingest_conversation(
    conv_id: str,
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
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
    summ_count = 0
    filler_count = 0
    for turn_id, role, text in segments:
        source_name = user_name if role == "user" else asst_name
        ts = _event_timestamp(base_ts, turn_id)

        # Raw view — always add.
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
                    "view": "raw",
                },
            )
        )
        raw_count += 1

        # Summary view — only if non-filler.
        summ = summaries.get(turn_id, FILLER_MARKER)
        if summ and summ != FILLER_MARKER:
            events.append(
                Event(
                    uuid=uuid4(),
                    # Nudge timestamp by 1 microsecond so ordering is stable
                    # but still inside the turn's time slot.
                    timestamp=ts + timedelta(microseconds=1),
                    body=Content(
                        context=MessageContext(source=source_name),
                        items=[Text(text=summ)],
                    ),
                    properties={
                        "arc_conversation_id": conv_id,
                        "turn_id": turn_id,
                        "role": role,
                        "view": "summary",
                    },
                )
            )
            summ_count += 1
        else:
            filler_count += 1

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
        "n_summary_events": summ_count,
        "n_filler": filler_count,
        "ingest_time_s": round(ingest_time, 2),
        "user_name": user_name,
        "assistant_name": asst_name,
    }


async def main() -> None:
    print("[em_setup_summ] loading speaker map + segments", flush=True)
    speakers_map = load_speaker_map()
    conv_segments = load_conversation_segments(SEGMENTS_FILE)
    for cid in LOCOMO_CONV_IDS:
        print(f"  {cid}: {len(conv_segments[cid])} turns", flush=True)

    # Qdrant
    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    # SQL (prefer Postgres, fall back to dedicated SQLite)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sqlite_path = RESULTS_DIR / "eventmemory_summ.sqlite3"
    sql_url = os.getenv("SQL_URL") or f"sqlite+aiosqlite:///{sqlite_path}"
    try:
        engine = create_async_engine(sql_url, pool_size=50, max_overflow=50)
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    except Exception as exc:
        print(f"[em_setup_summ] SQL_URL {sql_url!r} not reachable: {exc!r}")
        sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
        print(f"[em_setup_summ] falling back to SQLite at {sqlite_path}")
        engine = create_async_engine(sql_url)
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    # OpenAI
    openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
            max_input_length=8192,
        )
    )

    # Generate all summaries across all conversations.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = SummaryCache(SUMMARY_CACHE_FILE)

    all_summaries: dict[str, dict[int, str]] = {}
    t_summ = time.monotonic()
    for conv_id in LOCOMO_CONV_IDS:
        print(f"[summarize] {conv_id} ({len(conv_segments[conv_id])} turns)")
        summ = await generate_summaries(
            conv_segments[conv_id],
            speakers_map[conv_id],
            openai_client,
            cache,
            concurrency=20,
        )
        all_summaries[conv_id] = summ
        n_filler = sum(1 for v in summ.values() if v == FILLER_MARKER)
        print(f"  -> {len(summ)} summaries, {n_filler} filler")
    t_summ = time.monotonic() - t_summ
    print(f"[summarize] total: {t_summ:.1f}s")

    # Persist summaries for downstream inspection.
    summ_out = RESULTS_DIR / "turn_summaries.json"
    with open(summ_out, "w") as f:
        json.dump(
            {
                cid: {str(tid): s for tid, s in all_summaries[cid].items()}
                for cid in LOCOMO_CONV_IDS
            },
            f,
            indent=2,
        )
    print(f"Saved: {summ_out}")

    # Ingest
    records = []
    try:
        semaphore = asyncio.Semaphore(3)
        tasks = [
            async_with(
                semaphore,
                ingest_conversation(
                    conv_id,
                    conv_segments[conv_id],
                    speakers_map[conv_id],
                    all_summaries[conv_id],
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

    out_path = RESULTS_DIR / "eventmemory_summ_collections.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "namespace": NAMESPACE,
                "prefix": COLLECTION_PREFIX,
                "logical_prefix": LOGICAL_PREFIX,
                "sql_url": sql_url,
                "max_text_chunk_length": 500,
                "derive_sentences": False,
                "summary_model": SUMMARY_MODEL,
                "summary_cache": str(SUMMARY_CACHE_FILE),
                "summaries_file": str(summ_out),
                "conversations": records,
            },
            f,
            indent=2,
        )

    for r in records:
        print(
            f"[ingested] {r['conversation_id']}: "
            f"{r['n_raw_events']} raw + {r['n_summary_events']} summary "
            f"({r['n_filler']} filler) in {r['ingest_time_s']}s "
            f"-> {r['collection_name']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
