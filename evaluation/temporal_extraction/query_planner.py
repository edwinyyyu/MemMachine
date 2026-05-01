"""LLM-based structured query planner.

Replaces the regex parsers (recency.has_recency_cue,
negation.parse_negation_query, T_open_ended_router_eval.has_open_ended_cue,
T_causal_eval.detect_causal) with a single gpt-5-mini call returning a
structured plan with all dimensions extracted in parallel.

Motivation: composition queries (T_composition.md) expose priority
conflicts and greedy stripping in the regex stack. An LLM can extract
all dimensions independently in one call.

Schema (QueryPlan, JSON-serializable):

    {
      "absolute_anchor": "<str or null>",        # explicit date/range
      "open_ended": {
          "side": "after"|"before"|"since"|"until",
          "anchor": "<the date or event>"
      } or null,
      "recency_intent": <bool>,                  # latest / most recent
      "earliest_intent": <bool>,                 # first / earliest
      "negation": {"excluded_phrase": "<str>"} or null,
      "causal": {
          "anchor_phrase": "<event noun phrase>",
          "direction": "after"|"before"
      } or null,
      "cyclical_intent": <bool>                  # every X / on Sundays etc.
    }

Caches by (model + prompt) hash to avoid re-paying for repeated queries
across benchmark runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

# Load OPENAI_API_KEY from the repo .env if not already in the env.
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


MODEL = "gpt-5-mini"
PER_CALL_TIMEOUT_S = 45.0
CONCURRENCY = 8

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


PLAN_PROMPT = """Analyze this temporal query and extract dimensions.

Query: {query}
Reference time: {ref_time}

Return ONLY a JSON object with these fields (use null when not applicable):
- absolute_anchor: the explicit date phrase that names a fixed range or instant
  (e.g. "Q4 2023", "March 2024", "2023", "January 9 2025"). Use null when the
  query has no concrete date phrase. For one-sided bounds (before/after/since/
  until <date>) DO NOT put the date here — put it under open_ended instead.
- open_ended: object {{"side": "after"|"before"|"since"|"until",
  "anchor": "<the date or event>"}}, or null. Fires for one-sided temporal
  bounds. "side" follows the cue word verbatim.
- recency_intent: true iff the user wants the MOST RECENT / LATEST instance.
  (e.g. "latest", "most recent", "recently", "current", "as of now")
- earliest_intent: true iff the user wants the FIRST / EARLIEST instance.
- negation: object {{"excluded_phrase": "<the temporal phrase being excluded>"}},
  or null. Examples: "in 2024 not in summer" -> {{"excluded_phrase": "summer"}}.
  "outside Q3 2023" -> {{"excluded_phrase": "Q3 2023"}}. Only the EXCLUDED date/
  range phrase, not the surrounding topic.
- causal: object {{"anchor_phrase": "<the named event noun phrase>",
  "direction": "after"|"before"}}, or null. Fires when the query is anchored
  to an EVENT (not a date), e.g. "after the migration", "before the kickoff",
  "since the redesign shipped". Do NOT fire when the anchor is an explicit
  date — that goes under open_ended. The direction follows the cue verbatim
  ("after"/"since"/"following" -> "after"; "before"/"until"/"prior to" ->
  "before").
- cyclical_intent: true iff the query is about a recurring pattern
  (e.g. "every Friday", "on Sundays", "in summer", "weekday mornings").

Crucial rules:
1. A query can fire MULTIPLE dimensions simultaneously. e.g.
   "What I did in Q3 2023 after the launch" fires absolute_anchor="Q3 2023"
   AND causal={{anchor_phrase: "the launch", direction: "after"}}.
2. "in <DATE> not in <PHRASE>" fires absolute_anchor=<DATE> AND
   negation={{excluded_phrase: <PHRASE>}}. Keep the FIRST anchor in
   absolute_anchor; only the post-cue phrase goes to negation.
3. open_ended is for DATE bounds; causal is for EVENT bounds. They are
   mutually exclusive unless the query has both a date bound AND an
   event anchor (rare).
4. A "latest X in <YEAR>" query fires both recency_intent=true AND
   absolute_anchor=<YEAR>.

Output ONLY the JSON object, no commentary, no fences.
"""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class QueryPlan:
    absolute_anchor: str | None = None
    open_ended: dict[str, str] | None = None
    recency_intent: bool = False
    earliest_intent: bool = False
    negation: dict[str, str] | None = None
    causal: dict[str, str] | None = None
    cyclical_intent: bool = False
    raw: str | None = field(default=None, repr=False)
    parse_error: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "absolute_anchor": self.absolute_anchor,
            "open_ended": self.open_ended,
            "recency_intent": bool(self.recency_intent),
            "earliest_intent": bool(self.earliest_intent),
            "negation": self.negation,
            "causal": self.causal,
            "cyclical_intent": bool(self.cyclical_intent),
        }

    @classmethod
    def from_obj(cls, d: dict[str, Any], raw: str = "") -> QueryPlan:
        return cls(
            absolute_anchor=_clean_str(d.get("absolute_anchor")),
            open_ended=_clean_open_ended(d.get("open_ended")),
            recency_intent=bool(d.get("recency_intent") or False),
            earliest_intent=bool(d.get("earliest_intent") or False),
            negation=_clean_negation(d.get("negation")),
            causal=_clean_causal(d.get("causal")),
            cyclical_intent=bool(d.get("cyclical_intent") or False),
            raw=raw,
        )


def _clean_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "null":
        return None
    return s


def _clean_open_ended(v: Any) -> dict[str, str] | None:
    if not isinstance(v, dict):
        return None
    side = _clean_str(v.get("side"))
    anchor = _clean_str(v.get("anchor"))
    if not side or not anchor:
        return None
    side_l = side.lower()
    if side_l not in ("after", "before", "since", "until"):
        return None
    return {"side": side_l, "anchor": anchor}


def _clean_negation(v: Any) -> dict[str, str] | None:
    if not isinstance(v, dict):
        return None
    excl = _clean_str(v.get("excluded_phrase"))
    if not excl:
        return None
    return {"excluded_phrase": excl}


def _clean_causal(v: Any) -> dict[str, str] | None:
    if not isinstance(v, dict):
        return None
    phrase = _clean_str(v.get("anchor_phrase"))
    direction = _clean_str(v.get("direction"))
    if not phrase or not direction:
        return None
    dir_l = direction.lower()
    if dir_l not in ("after", "before"):
        return None
    return {"anchor_phrase": phrase, "direction": dir_l}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache: dict[str, str] = {}
_cache_loaded = False
_cache_lock = asyncio.Lock()


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache_sync(cache: dict[str, str]) -> None:
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=0))
    tmp.replace(CACHE_FILE)


def _cache_key(query: str, ref_time: str) -> str:
    body = f"{MODEL}|{query}|{ref_time}"
    return hashlib.sha256(body.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Parse the LLM output (tolerant of fenced / commentary-wrapped JSON)
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json_obj(s: str) -> dict[str, Any] | None:
    if not s:
        return None
    # Strip code fences.
    cleaned = _FENCE_RE.sub("", s).strip()
    # Try direct parse.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fall back: greedy match on first {...} block.
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class QueryPlanner:
    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.calls = 0
        self.cache_hits = 0
        self.parse_failures = 0
        global _cache, _cache_loaded
        if not _cache_loaded:
            _cache = _load_cache()
            _cache_loaded = True

    async def _llm(self, prompt: str) -> str:
        async with self.sem:
            try:
                resp = await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.model,
                        input=prompt,
                        # gpt-5-mini reasoning tokens can consume ~1k; budget
                        # generously so the JSON output isn't truncated.
                        max_output_tokens=3072,
                    ),
                    timeout=PER_CALL_TIMEOUT_S,
                )
            except Exception:
                return ""
            return resp.output_text or ""

    async def plan(self, query: str, ref_time: str) -> QueryPlan:
        prompt = PLAN_PROMPT.format(query=query, ref_time=ref_time)
        k = _cache_key(query, ref_time)
        if k in _cache:
            self.cache_hits += 1
            raw = _cache[k]
        else:
            self.calls += 1
            raw = await self._llm(prompt)
            if raw:
                async with _cache_lock:
                    _cache[k] = raw
                    # Persist eagerly so concurrent runs share progress.
                    _save_cache_sync(_cache)
        obj = _extract_json_obj(raw)
        if obj is None:
            self.parse_failures += 1
            return QueryPlan(raw=raw, parse_error="json_parse_failed")
        try:
            return QueryPlan.from_obj(obj, raw=raw)
        except Exception as e:
            self.parse_failures += 1
            return QueryPlan(raw=raw, parse_error=str(e))

    async def plan_many(
        self, items: list[tuple[str, str, str]]
    ) -> dict[str, QueryPlan]:
        """items: [(qid, query_text, ref_time)]; returns {qid: plan}."""

        async def _one(qid: str, qtext: str, rt: str) -> tuple[str, QueryPlan]:
            p = await self.plan(qtext, rt)
            return qid, p

        tasks = [_one(qid, qt, rt) for qid, qt, rt in items]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "parse_failures": self.parse_failures,
            "total_queries": self.calls + self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.calls + self.cache_hits),
        }


# Convenience
async def plan_query(
    query: str, ref_time: str, planner: QueryPlanner | None = None
) -> QueryPlan:
    p = planner or QueryPlanner()
    return await p.plan(query, ref_time)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
async def _smoke():
    test_queries = [
        ("My latest budget review in Q2 2024", "2025-06-15T00:00:00Z"),
        (
            "What client meetings did I have in 2024 not in summer?",
            "2025-06-15T00:00:00Z",
        ),
        ("My most recent update after the migration", "2025-06-15T00:00:00Z"),
        ("What did I do in Q3 2023 after the launch?", "2025-06-15T00:00:00Z"),
        ("What did I do after 2020 but not in 2023?", "2025-06-15T00:00:00Z"),
    ]
    p = QueryPlanner()
    for q, rt in test_queries:
        plan = await p.plan(q, rt)
        print(f"\nQ: {q}")
        print(f"  plan: {json.dumps(plan.to_dict(), indent=2)}")
    print("\nstats:", p.stats())


if __name__ == "__main__":
    asyncio.run(_smoke())
