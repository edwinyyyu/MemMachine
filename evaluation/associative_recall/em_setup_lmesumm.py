"""EventMemory dual-view (raw + turn-summary) ingestion for LongMemEval-hard POC.

Goal 2 POC: Port turn_summary to LME. POC runs on 10 questions / category =
30 questions total (not 90). Full 90-question run follows only if the POC
cracks R@50 over em_v2f_lme_mixed_7030 (0.863).

For each turn in haystack_sessions:
  - Event 1 (raw view):     MessageContext.source=<User|Assistant>,
                            items=[Text(text=<raw>)]
  - Event 2 (summary view): MessageContext.source=<User|Assistant>,
                            items=[Text(text=<summary>)]
    * filler summaries fall back to raw-only.

Dedicated cache: cache/summlme_llm_cache.json (POC summaries only).

New storage:
  - Qdrant collections: arc_em_lmehard_summ_v1_<question_id>
  - SQLite:             results/eventmemory_lmesumm.sqlite3
  - Namespace:          arc_em_lmehard_summ

POC selection: first 10 question_ids per category sorted lex. Overridable
via env var LMESUMM_POC_SIZE (default 10).

Run once:
    uv run python evaluation/associative_recall/em_setup_lmesumm.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
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

ASSOC_DIR = Path(__file__).resolve().parent
DATA_DIR = ASSOC_DIR / "data"
RESULTS_DIR = ASSOC_DIR / "results"
CACHE_DIR = ASSOC_DIR / "cache"

LME_SRC = ROOT / "evaluation" / "data" / "longmemeval_s_cleaned.json"
HARD_QUESTIONS_JSON = DATA_DIR / "questions_longmemeval_hard.json"

COLLECTION_PREFIX = "arc_em_lmehard_summ_v1"
NAMESPACE = "arc_em_lmehard_summ"
SQLITE_FILE = RESULTS_DIR / "eventmemory_lmesumm.sqlite3"
COLLECTIONS_OUT = RESULTS_DIR / "eventmemory_lmesumm_collections.json"

USER_NAME = "User"
ASSISTANT_NAME = "Assistant"

SUMMARY_MODEL = "gpt-5-mini"
SUMMARY_CACHE_FILE = CACHE_DIR / "summlme_llm_cache.json"

# Concurrency.
SUMMARY_CONCURRENCY = 20
INGEST_CONCURRENCY = 3

# POC: 10 question_ids per category, sorted lex.
POC_PER_CATEGORY = int(os.getenv("LMESUMM_POC_SIZE", "10"))

# LME summary prompt -- dense, third-person, preserves key fact/decision.
# Mirrors em_setup_summ.SUMMARY_PROMPT but for User/Assistant roles rather
# than named speakers.
SUMMARY_PROMPT = """\
Summarize this chat turn as a single declarative third-person sentence. \
Extract the key fact, decision, preference, event, or assertion.

Examples:
- "I've decided to start running marathons next year" (role=user) -> "The user decided to start running marathons next year."
- "Based on what you said, it sounds like you're leaning toward Python." (role=assistant) -> "The assistant observed that the user is leaning toward Python."
- "Thanks, that's helpful!" -> "<filler>"

Turn: {turn_text}
Role: {role}
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


def _clean_summary(raw: str) -> str:
    t = raw.strip().strip("'").strip('"').strip()
    if not t:
        return FILLER_MARKER
    if FILLER_MARKER in t:
        return FILLER_MARKER
    t = t.split("\n")[0].strip()
    if not t:
        return FILLER_MARKER
    return t


def _truncate(s: str, max_chars: int = 1500) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _build_prev_context(
    flat_turns: list[tuple[str, str]],
    idx: int,
    n_prev: int = 2,
) -> str:
    """flat_turns: list of (role, text) across all sessions in original order."""
    lines: list[str] = []
    for j in range(max(0, idx - n_prev), idx):
        role, text = flat_turns[j]
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {_truncate(text, 400)}")
    if not lines:
        return "(none)"
    return "\n".join(lines)


def build_summary_prompt(turn_text: str, role: str, prev_context: str) -> str:
    return SUMMARY_PROMPT.format(
        turn_text=_truncate(turn_text),
        role=role,
        prev_context=prev_context,
    )


def _parse_haystack_date(date_str: str) -> datetime:
    """Parse LME haystack_dates like '2023/05/20 (Sat) 00:31'."""
    clean = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
    dt = datetime.strptime(clean, "%Y/%m/%d %H:%M")
    return dt.replace(tzinfo=timezone.utc)


def _load_hard_questions() -> list[dict]:
    with open(HARD_QUESTIONS_JSON) as f:
        return json.load(f)


def _load_raw_lme_records(question_ids: set[str]) -> dict[str, dict]:
    with open(LME_SRC) as f:
        all_qs = json.load(f)
    return {q["question_id"]: q for q in all_qs if q["question_id"] in question_ids}


def _select_poc_ids(hard_qs: list[dict], per_cat: int) -> list[str]:
    by_cat: dict[str, list[str]] = {}
    for q in hard_qs:
        by_cat.setdefault(q["category"], []).append(q["question_id"])
    selected: list[str] = []
    for cat in sorted(by_cat):
        selected.extend(sorted(by_cat[cat])[:per_cat])
    return selected


def _flatten_turns(raw: dict) -> list[dict]:
    """Return ordered list of
      {turn_id, session_id, role, text, timestamp, flat_idx}
    One turn per haystack turn across sessions, preserving original order.
    """
    turns: list[dict] = []
    turn_idx = 0
    for sess_id, sess_turns, date_str in zip(
        raw["haystack_session_ids"],
        raw["haystack_sessions"],
        raw["haystack_dates"],
        strict=True,
    ):
        base = _parse_haystack_date(date_str)
        for ti, turn in enumerate(sess_turns):
            role = turn.get("role", "user")
            text = turn.get("content") or ""
            if not isinstance(text, str):
                text = str(text)
            turns.append(
                {
                    "turn_id": turn_idx,
                    "session_id": sess_id,
                    "role": role,
                    "text": text,
                    "timestamp": base + timedelta(seconds=ti),
                }
            )
            turn_idx += 1
    return turns


async def generate_summaries_for_question(
    turns: list[dict],
    openai_client,
    cache: SummaryCache,
    *,
    concurrency: int = SUMMARY_CONCURRENCY,
) -> dict[int, str]:
    flat = [(t["role"], t["text"]) for t in turns]
    tasks_args: list[tuple[int, str]] = []
    for idx, t in enumerate(turns):
        prev_ctx = _build_prev_context(flat, idx, n_prev=2)
        prompt = build_summary_prompt(t["text"], t["role"], prev_ctx)
        tasks_args.append((t["turn_id"], prompt))

    result: dict[int, str] = {}
    sem = asyncio.Semaphore(concurrency)

    async def one(turn_id: int, prompt: str) -> None:
        async with sem:
            cached = cache.get(SUMMARY_MODEL, prompt)
            if cached is not None:
                result[turn_id] = _clean_summary(cached)
                return
            for attempt in range(3):
                try:
                    resp = await openai_client.chat.completions.create(
                        model=SUMMARY_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = resp.choices[0].message.content or ""
                    cache.put(SUMMARY_MODEL, prompt, text)
                    result[turn_id] = _clean_summary(text)
                    return
                except Exception as e:
                    if attempt == 2:
                        print(f"  [summary fail] turn {turn_id}: {e!r}")
                        result[turn_id] = FILLER_MARKER
                        return
                    await asyncio.sleep(1.0 + attempt)

    await asyncio.gather(*(one(tid, p) for tid, p in tasks_args))
    cache.save()
    return result


async def ingest_question(
    raw: dict,
    summaries: dict[int, str],
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    embedder: OpenAIEmbedder,
) -> dict:
    qid = raw["question_id"]
    collection_name = f"{COLLECTION_PREFIX}_{qid}"
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

    turns = _flatten_turns(raw)
    events: list[Event] = []
    raw_count = 0
    summ_count = 0
    filler_count = 0
    for t in turns:
        role = t["role"]
        source_name = USER_NAME if role == "user" else ASSISTANT_NAME
        ts = t["timestamp"]
        text = t["text"]

        # Raw view.
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
                    "session_id": t["session_id"],
                    "turn_id": t["turn_id"],
                    "role": role,
                    "view": "raw",
                },
            )
        )
        raw_count += 1

        # Summary view (non-filler only).
        summ = summaries.get(t["turn_id"], FILLER_MARKER)
        if summ and summ != FILLER_MARKER:
            events.append(
                Event(
                    uuid=uuid4(),
                    # +1us so timestamp is monotonic within the turn slot.
                    timestamp=ts + timedelta(microseconds=1),
                    body=Content(
                        context=MessageContext(source=source_name),
                        items=[Text(text=summ)],
                    ),
                    properties={
                        "arc_question_id": qid,
                        "session_id": t["session_id"],
                        "turn_id": t["turn_id"],
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
        "question_id": qid,
        "collection_name": collection_name,
        "partition_key": partition_key,
        "namespace": NAMESPACE,
        "n_turns": len(turns),
        "n_raw_events": raw_count,
        "n_summary_events": summ_count,
        "n_filler": filler_count,
        "ingest_time_s": round(ingest_time, 2),
    }


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hard_qs = _load_hard_questions()
    poc_ids = _select_poc_ids(hard_qs, POC_PER_CATEGORY)
    print(
        f"[em_setup_lmesumm] POC={len(poc_ids)} questions "
        f"({POC_PER_CATEGORY}/category)",
        flush=True,
    )

    raw_map = _load_raw_lme_records(set(poc_ids))
    missing = set(poc_ids) - set(raw_map)
    if missing:
        raise RuntimeError(f"Missing raw records for: {sorted(missing)}")

    # Pre-pass: flatten turns to count events, estimate budget.
    turns_by_q: dict[str, list[dict]] = {}
    for qid in poc_ids:
        turns_by_q[qid] = _flatten_turns(raw_map[qid])
    total_events = sum(len(v) for v in turns_by_q.values())
    print(f"  total POC turns: {total_events}", flush=True)

    # Check how many are already cached to narrow live LLM count.
    cache = SummaryCache(SUMMARY_CACHE_FILE)
    # Compute prompts and cache-hit count.
    n_cached = 0
    n_to_call = 0
    for qid, turns in turns_by_q.items():
        flat = [(t["role"], t["text"]) for t in turns]
        for i, t in enumerate(turns):
            prev = _build_prev_context(flat, i, n_prev=2)
            prompt = build_summary_prompt(t["text"], t["role"], prev)
            if cache.get(SUMMARY_MODEL, prompt) is not None:
                n_cached += 1
            else:
                n_to_call += 1
    print(
        f"  summary cache: {n_cached} hits / {n_to_call} new calls needed",
        flush=True,
    )
    # Safety budget check.  Realistic gpt-5-mini cost for these prompts
    # (~250 tokens in, ~30 tokens out) is ~$0.00012/call -> $1.80 for 14K
    # calls.  The old $0.003/call bound is extremely pessimistic; use a
    # realistic-but-safe $0.0005/call bound so the 30-question POC the plan
    # budgets for ($42 WORST-case) can run.  The plan's hard cap is $50;
    # we abort if the REALISTIC estimate exceeds $30 (~60K calls).
    realistic_per_call = 0.0005
    realistic = n_to_call * realistic_per_call
    pessimistic = n_to_call * 0.003
    print(
        f"  projected summary cost: realistic (${realistic_per_call}/call) "
        f"${realistic:.2f}  |  pessimistic ($0.003/call) ${pessimistic:.2f}"
    )
    if realistic > 30.0:
        print(
            f"[em_setup_lmesumm] ABORT: realistic projection ${realistic:.2f} "
            f"> $30 cap. Re-run with smaller LMESUMM_POC_SIZE.",
            flush=True,
        )
        return

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    # Dedicated SQLite for LME-summ.
    sql_url = f"sqlite+aiosqlite:///{SQLITE_FILE}"
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

    # Phase 1: generate summaries per question (concurrent within a Q).
    summaries_by_q: dict[str, dict[int, str]] = {}
    try:
        t_summ = time.monotonic()
        for i, qid in enumerate(poc_ids):
            print(
                f"[summarize {i + 1}/{len(poc_ids)}] {qid} "
                f"({len(turns_by_q[qid])} turns)",
                flush=True,
            )
            summ = await generate_summaries_for_question(
                turns_by_q[qid],
                openai_client,
                cache,
            )
            n_filler = sum(1 for v in summ.values() if v == FILLER_MARKER)
            print(f"  -> {len(summ)} summaries, {n_filler} filler", flush=True)
            summaries_by_q[qid] = summ
            # Save cache between questions for resumability.
            cache.save()
        t_summ = time.monotonic() - t_summ
        print(f"[summarize] total: {t_summ:.1f}s", flush=True)

        # Persist per-question summaries for audit.
        summ_out = RESULTS_DIR / "lmesumm_summaries.json"
        with open(summ_out, "w") as f:
            json.dump(
                {
                    qid: {str(tid): s for tid, s in summaries_by_q[qid].items()}
                    for qid in poc_ids
                },
                f,
                indent=2,
            )
        print(f"Saved: {summ_out}", flush=True)

        # Phase 2: ingest.
        semaphore = asyncio.Semaphore(INGEST_CONCURRENCY)
        ingest_tasks = [
            async_with(
                semaphore,
                ingest_question(
                    raw_map[qid],
                    summaries_by_q[qid],
                    vector_store,
                    segment_store,
                    embedder,
                ),
            )
            for qid in poc_ids
        ]
        t_ing = time.monotonic()
        records = await asyncio.gather(*ingest_tasks)
        t_ing = time.monotonic() - t_ing
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
        "summary_model": SUMMARY_MODEL,
        "summary_cache": str(SUMMARY_CACHE_FILE),
        "summaries_file": str(RESULTS_DIR / "lmesumm_summaries.json"),
        "poc_size_per_cat": POC_PER_CATEGORY,
        "poc_question_ids": poc_ids,
        "n_questions": len(records),
        "n_raw_events_total": sum(r["n_raw_events"] for r in records),
        "n_summary_events_total": sum(r["n_summary_events"] for r in records),
        "ingest_total_s": round(t_ing, 1),
        "questions": records,
    }
    with open(COLLECTIONS_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {COLLECTIONS_OUT}", flush=True)
    print(
        f"Done: n_questions={out['n_questions']} "
        f"raw={out['n_raw_events_total']} summ={out['n_summary_events_total']} "
        f"ingest={out['ingest_total_s']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
