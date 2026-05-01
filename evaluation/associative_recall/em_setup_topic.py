"""EventMemory ingestion for LoCoMo-30 with TOPIC-BAKING at ingestion.

Mirror of em_setup.py but:
  - For every turn, extract a short topic phrase via gpt-5-mini (cached to
    `cache/topicbake_llm_cache.json`).
  - Prepend `[topic: <topic>] ` to the turn text before passing it to the
    Text primitive. Speaker is still baked via MessageContext.source.
    Final embedded text: "<speaker>: [topic: <topic>] <original>".
  - Writes to NEW Qdrant collections (prefix `arc_em_lc30_topic_v1_<short>`)
    and a NEW SQLite path (`results/eventmemory_topic.sqlite3`).
  - Stores topic in per-turn properties for audit.

Run once:
    uv run python evaluation/associative_recall/em_setup_topic.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
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
CACHE_DIR = Path(__file__).resolve().parent / "cache"
TOPIC_LLM_CACHE = CACHE_DIR / "topicbake_llm_cache.json"

# Avoid loading the 3.7GB segments_extended.npz (contains embeddings we
# don't need). Read turns from locomo10.json instead, like em_setup_rts.py.
LOCOMO10_PATH = ROOT / "evaluation" / "data" / "locomo10.json"
LOCOMO10_INDEX = {
    "locomo_conv-26": 0,
    "locomo_conv-30": 1,
    "locomo_conv-41": 2,
}

# Qdrant collection names must match [a-z0-9_]+ and be <=32 bytes. Use a short
# prefix; record the logical prefix separately.
COLLECTION_PREFIX = "arc_em_lc30_topic_v1"
LOGICAL_PREFIX = "arc_em_locomo30_topic_v1"
NAMESPACE = "arc_em_locomo30_topic"

LOCOMO_CONV_IDS = ["locomo_conv-26", "locomo_conv-30", "locomo_conv-41"]

TOPIC_MODEL = "gpt-5-mini"

TOPIC_PROMPT_TEMPLATE = """\
Classify the topic of this conversation turn with a short descriptive phrase (2-6 words). The topic should be a natural phrase a human would use to label what this turn is about. If the turn is conversational filler ("Haha", "yeah") with no clear topic, output "filler".

Examples:
- "I went to the LGBTQ support group yesterday" -> "LGBTQ support group visit"
- "Project Phoenix timeline is 16 weeks" -> "Project Phoenix timeline"
- "Yeah, that sounds good" -> "filler"

Turn: {turn_text}
Speaker: {speaker}
Preceding context: {prev_turn}

Output: one line, just the topic phrase."""


def _conv_short(conv_id: str) -> str:
    return conv_id.rsplit("-", 1)[-1]


def _sha(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


class TopicCache:
    """Simple JSON-backed cache keyed by sha(model, prompt)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                with open(path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
        self._dirty = False

    def get(self, model: str, prompt: str) -> str | None:
        return self._cache.get(_sha(model, prompt))

    def put(self, model: str, prompt: str, value: str) -> None:
        self._cache[_sha(model, prompt)] = value
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        tmp.replace(self._path)
        self._dirty = False


def load_speaker_map() -> dict[str, dict[str, str]]:
    with open(RESULTS_DIR / "conversation_two_speakers.json") as f:
        return json.load(f)["speakers"]


def load_conversation_segments(
    _unused_npz_path: Path,
) -> dict[str, list[tuple[int, str, str]]]:
    """Load turns from locomo10.json (tiny) -- avoids the 3.7GB npz which
    contains pre-computed embeddings we don't need. Turn-id convention
    matches segments_extended.npz: 0..N-1 in session order.
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


def _event_timestamp(base: datetime, turn_id: int) -> datetime:
    return base + timedelta(seconds=60 * turn_id)


def _truncate(s: str, max_chars: int = 400) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _build_topic_prompt(turn_text: str, speaker: str, prev_turn: str | None) -> str:
    prev_display = (
        _truncate(prev_turn) if prev_turn else "(none; this is the first turn)"
    )
    return TOPIC_PROMPT_TEMPLATE.format(
        turn_text=_truncate(turn_text),
        speaker=speaker,
        prev_turn=prev_display,
    )


def _normalize_topic(raw: str) -> str:
    """Pick the best single-line topic phrase from the LLM response."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return "filler"
    candidate = lines[0]
    # Strip quotes / trailing punctuation
    candidate = candidate.strip('"').strip("'").strip()
    # If gpt-5-mini sometimes responds "Topic: X", strip leading label.
    for prefix in ("Topic:", "topic:", "Output:", "output:", "-", "*"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip().strip('"').strip("'").strip()
    # Cap absurd lengths
    if len(candidate) > 80:
        candidate = candidate[:80].rstrip()
    return candidate or "filler"


async def _extract_topic(
    openai_client,
    turn_text: str,
    speaker: str,
    prev_turn: str | None,
    cache: TopicCache,
    sem: asyncio.Semaphore,
) -> tuple[str, bool]:
    """Returns (topic, cache_hit)."""
    prompt = _build_topic_prompt(turn_text, speaker, prev_turn)
    cached = cache.get(TOPIC_MODEL, prompt)
    if cached is not None:
        return _normalize_topic(cached), True
    async with sem:
        # Double-check cache while holding semaphore, in case a concurrent
        # caller populated it for the exact same prompt.
        cached = cache.get(TOPIC_MODEL, prompt)
        if cached is not None:
            return _normalize_topic(cached), True
        resp = await openai_client.chat.completions.create(
            model=TOPIC_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
    cache.put(TOPIC_MODEL, prompt, raw)
    return _normalize_topic(raw), False


async def extract_topics_for_conversation(
    conv_id: str,
    segments: list[tuple[int, str, str]],
    speakers: dict[str, str],
    openai_client,
    cache: TopicCache,
    *,
    llm_sem: asyncio.Semaphore,
) -> list[dict]:
    """Extract topic per turn. Returns list indexed by position of
    {turn_id, role, text, speaker, topic, cache_hit}. Saves cache periodically.
    """
    user_name = speakers.get("user") or "User"
    asst_name = speakers.get("assistant") or "Assistant"

    turn_meta: list[dict] = []
    # Build all prompts first (prev_turn needs previous text).
    tasks = []
    prev_text: str | None = None
    for turn_id, role, text in segments:
        speaker = user_name if role == "user" else asst_name
        tasks.append((turn_id, role, text, speaker, prev_text))
        prev_text = text

    # Kick off extractions in parallel (semaphore limits concurrency).
    async def _run(idx, turn_id, role, text, speaker, prev_text):
        topic, hit = await _extract_topic(
            openai_client, text, speaker, prev_text, cache, llm_sem
        )
        return idx, {
            "turn_id": turn_id,
            "role": role,
            "text": text,
            "speaker": speaker,
            "topic": topic,
            "cache_hit": hit,
        }

    coros = [
        _run(i, turn_id, role, text, speaker, prev_text)
        for i, (turn_id, role, text, speaker, prev_text) in enumerate(tasks)
    ]

    turn_meta = [None] * len(coros)
    done_count = 0
    # Process in chunks so we can checkpoint the cache periodically.
    chunk = 64
    for start in range(0, len(coros), chunk):
        batch = coros[start : start + chunk]
        results = await asyncio.gather(*batch)
        for idx, meta in results:
            turn_meta[idx] = meta
        done_count += len(results)
        cache.save()
        print(
            f"[topic] {conv_id}: {done_count}/{len(coros)} turns extracted",
            flush=True,
        )
    return turn_meta


async def ingest_conversation(
    conv_id: str,
    turn_meta: list[dict],
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

    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    events: list[Event] = []
    filler_count = 0
    for meta in turn_meta:
        turn_id = meta["turn_id"]
        topic = meta["topic"]
        text = meta["text"].strip()
        baked_text = f"[topic: {topic}] {text}"
        if topic == "filler":
            filler_count += 1
        events.append(
            Event(
                uuid=uuid4(),
                timestamp=_event_timestamp(base_ts, turn_id),
                body=Content(
                    context=MessageContext(source=meta["speaker"]),
                    items=[Text(text=baked_text)],
                ),
                properties={
                    "arc_conversation_id": conv_id,
                    "turn_id": turn_id,
                    "role": meta["role"],
                    "topic": topic,
                },
            )
        )

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
        "n_events": len(events),
        "filler_count": filler_count,
        "ingest_time_s": round(ingest_time, 2),
    }


async def main() -> None:
    speakers_map = load_speaker_map()
    conv_segments = load_conversation_segments(DATA_DIR / "segments_extended.npz")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    sqlite_path = RESULTS_DIR / "eventmemory_topic.sqlite3"
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
        print(f"[em_setup_topic] SQL_URL {sql_url!r} not reachable: {exc!r}")
        sql_url = f"sqlite+aiosqlite:///{sqlite_path}"
        print(f"[em_setup_topic] falling back to SQLite at {sqlite_path}")
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

    cache = TopicCache(TOPIC_LLM_CACHE)
    # Allow up to 16 concurrent topic extractions.
    llm_sem = asyncio.Semaphore(16)

    records = []
    all_turn_meta: dict[str, list[dict]] = {}
    try:
        # Step 1: extract topics per conversation (sequentially per conv so
        # logs are readable; concurrency inside a conv is controlled by
        # llm_sem).
        t0 = time.monotonic()
        for conv_id in LOCOMO_CONV_IDS:
            all_turn_meta[conv_id] = await extract_topics_for_conversation(
                conv_id,
                conv_segments[conv_id],
                speakers_map[conv_id],
                openai_client,
                cache,
                llm_sem=llm_sem,
            )
        topic_time = time.monotonic() - t0
        cache.save()
        n_total_calls = sum(
            1 for v in all_turn_meta.values() for m in v if not m["cache_hit"]
        )
        n_total_turns = sum(len(v) for v in all_turn_meta.values())
        print(
            f"[topic] extracted {n_total_turns} topics "
            f"({n_total_calls} live LLM calls, "
            f"{n_total_turns - n_total_calls} cached) "
            f"in {topic_time:.1f}s"
        )

        # Step 2: ingest each conversation with topic-baked text.
        sem = asyncio.Semaphore(3)
        tasks = [
            async_with(
                sem,
                ingest_conversation(
                    conv_id,
                    all_turn_meta[conv_id],
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

    # Write per-turn topic audit file.
    turns_out = []
    for conv_id in LOCOMO_CONV_IDS:
        for meta in all_turn_meta[conv_id]:
            turns_out.append(
                {
                    "conversation_id": conv_id,
                    "turn_id": meta["turn_id"],
                    "role": meta["role"],
                    "speaker": meta["speaker"],
                    "topic": meta["topic"],
                    "text_preview": meta["text"].strip()[:140],
                }
            )
    out_turns = RESULTS_DIR / "topic_baking_turns.json"
    with open(out_turns, "w") as f:
        json.dump({"turns": turns_out}, f, indent=2)

    out_path = RESULTS_DIR / "eventmemory_topic_collections.json"
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
                "topic_cache": str(TOPIC_LLM_CACHE),
                "conversations": records,
            },
            f,
            indent=2,
        )

    for r in records:
        print(
            f"[ingested-topic] {r['conversation_id']}: "
            f"{r['n_events']} events ({r['filler_count']} filler) "
            f"in {r['ingest_time_s']}s -> {r['collection_name']}"
        )
    print(f"Saved: {out_turns}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
