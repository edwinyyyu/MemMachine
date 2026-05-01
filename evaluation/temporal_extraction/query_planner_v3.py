"""query_planner_v3 — extends v2 with event-anchor + offset resolution.

The v2 planner skipped event/entity names (correctly — they don't have a
calendar resolution by themselves). But queries like:

   "Four days after Election Day 2020, what state did AP call for Biden?"
   "About two months after the iPhone launched in 2007, what price cut?"
   "Three weeks before the Berlin Wall fell, ..."

reference *known events* with offsets. The LLM has world knowledge of these
events' dates and can do offset arithmetic. v3 instructs the LLM to resolve
such expressions to absolute dates in-place, outputting the resolved date as
a normal phrase with direction "in".

Schema unchanged from v2 (Constraint{phrase, direction} + extremum). The
planner just produces resolved dates instead of literal event references.
For events the LLM doesn't know, it falls back to v2's behavior (skip the
event-name part of the phrase, retain any concrete date token if present).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from query_planner_v2 import (
    _PLAN_JSON_SCHEMA,
    CONCURRENCY,
    MODEL,
    PER_CALL_TIMEOUT_S,
    AsyncOpenAI,
    QueryPlan,
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_v3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"

PLAN_PROMPT_V3 = """For each TEMPORAL EXPRESSION in this query, classify its direction.

Query: {query}
Reference time: {ref_time}

A constraint has:
  - phrase: the resolved date/period text. Output a CALENDAR-CONCRETE phrase
    (e.g., "Q4 2023", "March 2024", "October 13 2020", "summer 2024"). The
    extractor downstream needs a phrase it can resolve to actual calendar
    coordinates.

    Sources of phrases:
      (a) Direct date phrases in the query — copy verbatim.
          "in Q4 2023"   -> phrase: "Q4 2023"
          "March 2024"   -> phrase: "March 2024"

      (b) Relative deictic phrases — resolve against `Reference time`.
          "last quarter" / "two weeks ago" / "yesterday" / "back in college":
          KEEP THESE AS-IS — the downstream extractor handles deictic
          resolution against ref_time. Use direction "in".

      (c) Event-anchor + offset expressions — resolve IN-PLACE if you know
          the event's date. The query gives you "N (days/weeks/months/years)
          (after/before) EVENT" or "the day EVENT happened" patterns:

          "four days after Election Day 2020"
            Election Day 2020 = November 3, 2020.  +4 days = November 7, 2020.
            -> phrase: "November 7, 2020", direction: "in"

          "two months after the iPhone launched in 2007"
            iPhone original launch = June 29, 2007.  +2 months ≈ August 29, 2007.
            -> phrase: "August 29, 2007", direction: "in"

          "three weeks before the Berlin Wall fell"
            Berlin Wall fell = November 9, 1989.  -3 weeks ≈ October 19, 1989.
            -> phrase: "October 19, 1989", direction: "in"

          "the day Kennedy was shot"
            JFK assassination = November 22, 1963.
            -> phrase: "November 22, 1963", direction: "in"

          You may use direction "in" with the resolved date because you've
          already done the offset arithmetic — the user is asking about
          events ON the resolved date (or close to it). Use "after" /
          "before" only when the query phrasing requires open-ended search
          AFTER the resolved date.

          FUZZY QUANTITIES — when the user says "about", "around",
          "approximately", "roughly", "a few", or similar, WIDEN the
          resolved phrase to a coarser-precision interval that captures the
          natural fuzziness. Prefer month-or-larger spans over precise days
          for fuzzy quantities.

          "about two months after the iPhone launched in 2007"
            ≈ late August through September 2007 (two-month-ish window)
            -> phrase: "August or September 2007", direction: "in"

          "around three weeks before the Berlin Wall fell"
            ≈ mid-to-late October 1989
            -> phrase: "October 1989", direction: "in"

          "a few days after Election Day 2020"
            ≈ November 4-9, 2020 (a few days = 2-7 days)
            -> phrase: "early-to-mid November 2020", direction: "in"

      (d) Event-anchor (no offset) — also resolve if you know the date.
          "the year iPhone launched"     -> phrase: "2007", direction: "in"
          "in the month JFK was killed"  -> phrase: "November 1963", direction: "in"

      (e) Event/entity names with NO known date — skip. Don't emit a
          constraint for them. The retrieval/rerank handles the topical match.
          "the launch", "the migration", "the meeting" (no qualifier)

  - direction: pick exactly one of:
      "in"     — the date phrase NAMES the time of interest (DEFAULT)
                 (cues: "in", "during", "of", "from <date>", or no cue)
                 ALSO for relative deictic phrases ("two weeks ago",
                 "back in college", "last quarter").
                 ALSO for resolved event-anchor + offset expressions —
                 you've already pinpointed the date, the user is asking
                 about that date.
      "after"  — matches strictly AFTER the resolved date
                 (cues: "after", "since", "post" — only when the user
                 wants OPEN-ENDED search after the date, not a specific
                 day after.)
      "before" — matches strictly BEFORE the resolved date
                 (cues: "before", "until", "prior to")
      "not_in" — matches OUTSIDE this date phrase
                 (cues: "not in", "outside", "excluding", "except")
    When in doubt, use "in".

Also extract:
  - extremum: set ONLY when the query asks the system to PICK the
    most-recent / oldest from MULTIPLE candidates the user knows
    exist. Set "latest" or "earliest", else null.

      "Most recent meeting in March 2024"     -> "latest"
      "What's my latest budget review"        -> "latest"
      "What was my earliest job"              -> "earliest"

    DO NOT set extremum when "first/last" describes a SPECIFIC event:
      "When did Marcus host his first dinner party?"   -> null
      "When did Aiden have his first child?"           -> null
    "his/her/my first X" almost always names a specific event.

Examples:

Query: "in Q4 2023"
{{"constraints":[{{"phrase":"Q4 2023","direction":"in"}}],"extremum":null}}

Query: "after 2020"
{{"constraints":[{{"phrase":"2020","direction":"after"}}],"extremum":null}}

Query: "Four days after Election Day 2020, what state did AP call?"
{{"constraints":[{{"phrase":"November 7, 2020","direction":"in"}}],"extremum":null}}

Query: "About two months after the iPhone launched in 2007, what price cut?"
{{"constraints":[{{"phrase":"August 29, 2007","direction":"in"}}],"extremum":null}}

Query: "in 2024 not in summer"
{{"constraints":[{{"phrase":"2024","direction":"in"}},{{"phrase":"summer","direction":"not_in"}}],"extremum":null}}

Query: "What movie was popular the year iPhone launched"
{{"constraints":[{{"phrase":"2007","direction":"in"}}],"extremum":null}}

Query: "latest budget review in Q2 2024"
{{"constraints":[{{"phrase":"Q2 2024","direction":"in"}}],"extremum":"latest"}}

Query: "what did I do recently"
{{"constraints":[],"extremum":"latest"}}
"""


def _cache_key(query: str, ref_time: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"|v3.1|")
    h.update(query.encode())
    h.update(b"|")
    h.update(ref_time.encode())
    return h.hexdigest()


class QueryPlannerV3:
    """Like QueryPlanner (v2) but with the v3 prompt that resolves event
    anchors + offsets in-place. Schema unchanged."""

    def __init__(self):
        self._client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._calls = 0
        self._cache_hits = 0
        self._parse_failures = 0
        self._total = 0
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not CACHE_FILE.exists():
            return {}
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}

    def _save_cache(self):
        try:
            CACHE_FILE.write_text(json.dumps(self._cache))
        except Exception:
            pass

    async def _plan_one(self, qid: str, query: str, ref_time: str) -> QueryPlan:
        self._total += 1
        key = _cache_key(query, ref_time)
        if key in self._cache:
            self._cache_hits += 1
            try:
                return QueryPlan.from_obj(
                    self._cache[key],
                    raw=json.dumps(self._cache[key]),
                )
            except Exception:
                pass

        prompt = PLAN_PROMPT_V3.format(query=query, ref_time=ref_time)
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "query_plan",
                            "strict": True,
                            "schema": _PLAN_JSON_SCHEMA,
                        }
                    },
                )
                self._calls += 1
                raw = resp.output_text
                obj = json.loads(raw)
                plan = QueryPlan.from_obj(obj, raw=raw)
                self._cache[key] = obj
                self._save_cache()
                return plan
            except Exception as e:
                self._parse_failures += 1
                return QueryPlan(parse_error=str(e), raw="")

    async def plan_many(self, items) -> dict[str, QueryPlan]:
        """items: iterable of (qid, query_text, ref_time_iso)."""
        items = list(items)
        coros = [self._plan_one(qid, q, rt) for qid, q, rt in items]
        plans = await asyncio.gather(*coros)
        return {qid: plan for (qid, _, _), plan in zip(items, plans)}

    def stats(self) -> dict:
        return {
            "model": MODEL,
            "total_queries": self._total,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total),
            "parse_failures": self._parse_failures,
        }
