"""Polarity-aware retrieval.

Extends the base interval store with a per-expression ``polarity``
column. Provides three retrieval variants:

- ``raw``:             ignore polarity; legacy behavior.
- ``default``:         for positive-intent queries, filter out
                       non-affirmed matches at ranking time.
- ``polarity_routed``: LLM classifies the query intent
                       (affirmed|negation|agnostic) and routes ranking
                       accordingly:
                           affirmed  -> only affirmed doc hits count
                           negation  -> only negated doc hits count
                           agnostic  -> all hits count (like raw)

We implement the polarity filter on TOP of the base IntervalStore by
maintaining a side-table {expr_id: polarity} built at insert time. This
keeps the retrieval scorer identical to the base pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from eval import flatten_query_intervals
from openai import AsyncOpenAI
from schema import TimeExpression
from scorer import Interval, score_pair
from store import IntervalStore

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "polarity"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INTENT_CACHE_FILE = CACHE_DIR / "intent_cache.json"


# ---------------------------------------------------------------------------
# Store wrapper
# ---------------------------------------------------------------------------
class PolarityIntervalStore:
    """Wraps IntervalStore + polarity side-table.

    The inner SQLite schema is untouched; polarity is stored in an
    auxiliary column added via ALTER TABLE on the ``expressions`` table.
    """

    def __init__(self, path: str | Path) -> None:
        self.store = IntervalStore(path)
        self._ensure_polarity_column()

    def _ensure_polarity_column(self) -> None:
        cur = self.store.conn.execute("PRAGMA table_info(expressions)")
        cols = {row[1] for row in cur.fetchall()}
        if "polarity" not in cols:
            self.store.conn.execute(
                "ALTER TABLE expressions ADD COLUMN polarity TEXT "
                "NOT NULL DEFAULT 'affirmed'"
            )
            self.store.conn.commit()

    def reset(self) -> None:
        self.store.reset()
        self._ensure_polarity_column()

    def close(self) -> None:
        self.store.close()

    def insert_expression(
        self, doc_id: str, te: TimeExpression, polarity: str = "affirmed"
    ) -> int:
        expr_id = self.store.insert_expression(doc_id, te)
        self.store.conn.execute(
            "UPDATE expressions SET polarity = ? WHERE expr_id = ?",
            (polarity, expr_id),
        )
        self.store.conn.commit()
        return expr_id

    def polarities_by_expr_id(self) -> dict[int, str]:
        cur = self.store.conn.execute("SELECT expr_id, polarity FROM expressions")
        return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Retrieval variants
# ---------------------------------------------------------------------------
def _retrieve(
    store: PolarityIntervalStore,
    query_exprs: list[TimeExpression],
    allowed_polarities: set[str] | None,
) -> dict[str, float]:
    """Core scoring. If ``allowed_polarities`` is None, all polarities
    count. Otherwise, expressions whose polarity is not in the set
    contribute 0 score."""
    pol_by_expr = store.polarities_by_expr_id()
    q_ivs: list[Interval] = []
    for te in query_exprs:
        q_ivs.extend(flatten_query_intervals(te))

    out: dict[str, float] = defaultdict(float)
    for qi in q_ivs:
        rows = store.store.query_overlap(qi.earliest_us, qi.latest_us)
        best_per_doc: dict[str, float] = {}
        for expr_id, doc_id, e_us, l_us, b_us, gran in rows:
            if allowed_polarities is not None:
                pol = pol_by_expr.get(expr_id, "affirmed")
                if pol not in allowed_polarities:
                    continue
            s = Interval(
                earliest_us=e_us,
                latest_us=l_us,
                best_us=b_us,
                granularity=gran,
            )
            sc = score_pair(qi, s)
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
        for d, sc in best_per_doc.items():
            out[d] += sc
    return dict(out)


def retrieve_raw(
    store: PolarityIntervalStore, query_exprs: list[TimeExpression]
) -> dict[str, float]:
    return _retrieve(store, query_exprs, allowed_polarities=None)


def retrieve_default(
    store: PolarityIntervalStore, query_exprs: list[TimeExpression]
) -> dict[str, float]:
    """Affirmed-only: assume positive-intent queries unless told otherwise."""
    return _retrieve(store, query_exprs, allowed_polarities={"affirmed"})


def retrieve_routed(
    store: PolarityIntervalStore,
    query_exprs: list[TimeExpression],
    intent: str,
) -> dict[str, float]:
    """intent in {affirmed, negation, agnostic}."""
    if intent == "affirmed":
        return _retrieve(store, query_exprs, allowed_polarities={"affirmed"})
    if intent == "negation":
        return _retrieve(store, query_exprs, allowed_polarities={"negated"})
    # agnostic
    return _retrieve(store, query_exprs, allowed_polarities=None)


# ---------------------------------------------------------------------------
# Query-intent classifier (LLM)
# ---------------------------------------------------------------------------
INTENT_SYSTEM = """Classify the polarity intent of a retrieval query.

Labels:
- "affirmed" : query asks for events/facts that DID happen
  ("when did she attend?", "what happened on April 10?")
- "negation" : query explicitly asks for what did NOT happen
  ("what didn't happen last March?", "who did I fail to meet?")
- "agnostic" : query asks for any information about a time or topic,
  including both positive and negative mentions ("what was discussed
  about last March?", "any info on the launch")

Output JSON: {"intent": "affirmed" | "negation" | "agnostic"}.
"""

INTENT_SCHEMA: dict[str, Any] = {
    "name": "intent",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["affirmed", "negation", "agnostic"],
            }
        },
        "required": ["intent"],
    },
}


class IntentClassifier:
    def __init__(self, concurrency: int = 10) -> None:
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        self._cache: dict[str, str] = {}
        if INTENT_CACHE_FILE.exists():
            with INTENT_CACHE_FILE.open() as f:
                self._cache = json.load(f)
        self._new: dict[str, str] = {}
        self.usage: dict[str, int] = {"input": 0, "output": 0}

    def _save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if INTENT_CACHE_FILE.exists():
            with INTENT_CACHE_FILE.open() as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = INTENT_CACHE_FILE.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(existing, f)
        tmp.replace(INTENT_CACHE_FILE)
        self._new.clear()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(
            f"{MODEL}|{INTENT_SYSTEM[:40]}|{text}".encode()
        ).hexdigest()

    async def classify(self, text: str) -> str:
        k = self._key(text)
        if k in self._cache:
            return self._cache[k]
        user = f'Query: {text}\n\nReturn {{"intent": ...}} as JSON.'
        async with self.sem:
            resp = await self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": INTENT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=100,
                response_format={
                    "type": "json_schema",
                    "json_schema": INTENT_SCHEMA,
                },
            )
        usage = resp.usage
        if usage:
            self.usage["input"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(usage, "completion_tokens", 0) or 0
        content = resp.choices[0].message.content or "{}"
        try:
            intent = json.loads(content).get("intent", "affirmed")
        except json.JSONDecodeError:
            intent = "affirmed"
        if intent not in ("affirmed", "negation", "agnostic"):
            intent = "affirmed"
        self._cache[k] = intent
        self._new[k] = intent
        return intent

    def save(self) -> None:
        self._save()


async def classify_many(
    texts: list[str],
) -> tuple[dict[str, str], dict[str, int]]:
    clf = IntentClassifier()
    results = await asyncio.gather(*(clf.classify(t) for t in texts))
    clf.save()
    return {t: r for t, r in zip(texts, results)}, clf.usage
