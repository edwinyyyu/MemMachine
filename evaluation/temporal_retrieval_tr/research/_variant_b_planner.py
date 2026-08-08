"""Variant B planner: recurring queries → past 3 + future 1 as ONE target.

Design 2 co-design: a recurring query (pure recurring OR ambiguous
deictic-vs-recurring) emits a single multi-interval target with 4
intervals — the 3 most-recent past occurrences and the 1 upcoming
occurrence. The matching doc anchor (also past-3 + future-1 under
extractor v3.5) satisfies the target if it falls in ANY of those
intervals.

Single consistent rule for all recurring shapes — no layering of
"pure → empty, ambiguous → enumerate". Anaphora still gets empty
(date unknown, can't enumerate).

Principle: model recurrence without overcommitting falsehoods. A
4-instance window expresses the standing pattern without claiming
specific dates that haven't happened or didn't.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

from temporal_retrieval_tr.planner import (
    MODEL, PER_CALL_TIMEOUT_S, CONCURRENCY, Plan,
    _json_to_targets,
)
from temporal_retrieval_tr.research._no_anaphora_planner import (
    PROMPT as BASE_PROMPT,
    _PLAN_JSON_SCHEMA_NO_ANAPHORA,
)

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except Exception:
        pass


PROMPT_VERSION = "v5-recurring-enum-tod"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_variant_b"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


# Replace the EMPTY OUTPUT section with the unified Design 2 rule set.
_OLD_EMPTY_SECTION = """EMPTY OUTPUT
============
If the query has NO temporal scope at all (e.g., "how do I plan my
morning?", "lessons from the launch", "how did the migration go?") emit
{{"targets": []}}.

"What did I do recently" / "show me what happened lately" → deictic;
resolve to a recent window (e.g. last 60-90 days from REF_TIME) as a
target with extremum="latest"."""


_NEW_EMPTY_SECTION = """EMPTY OUTPUT
============
Emit empty targets — {{"targets": []}} — for queries with no usable
temporal anchor:

1. No temporal flavor at all:
   "how do I plan my morning?", "lessons from the launch",
   "how did the migration go?"

2. Anaphoric references (named event, date unknown):
   "since the v3 launch", "after the migration", "before the merger"
   The event's date isn't in the query; let semantic search find
   the doc by content.

3. "Most recently X" / "most recent X" with a specific subject:
   "When did I most recently cook tonkotsu ramen?"
   "my most recent rooftop gathering"
   The user is asking for the LATEST occurrence of a specific
   activity, not "things in the last 60-90 days". Emit empty
   targets with extremum="latest" so any-date docs about that
   activity remain eligible and recency tiebreak picks the latest.

"What did I do recently" / "show me what happened lately" (without a
specific subject) → deictic; resolve to a recent window (e.g. last
60-90 days from REF_TIME) as a target with extremum="latest".


RECURRING / HABITUAL PATTERNS — past 3 + future 1
=================================================
Queries about a recurring or habitual day-of-week, weekend, month,
quarter, etc. (pure recurring OR singular ambiguous between "this
one" and "the pattern") emit ONE target with FOUR intervals:

- The 3 most-recent past occurrences (relative to REF_TIME)
- The 1 next upcoming occurrence

This expresses the standing pattern without overcommitting to a
year of specific dates that haven't happened. The matching doc is
also anchored to a few recent past + 1 upcoming occurrence (same
shape), so they overlap reliably.

If REF_TIME itself falls ON an occurrence (e.g., REF_TIME is Thursday
and the query is "Thursdays"), count it as the most-recent past
occurrence.

# Per-occurrence within-day window

Each occurrence's lo/hi is determined by the time-of-day qualifier
in the query, using STANDARDIZED overlapping bands. Adjacent bands
overlap a few hours so a phrase real users would call by either
name still matches.

| Qualifier in query                 | Window (UTC, on that day)        |
| ---------------------------------- | -------------------------------- |
| "at HH:MM" / "at HHam/pm"          | [HH:00, HH+1:00)  — 1 hour       |
| "morning"                          | [03:00, 13:00)                   |
| "noon"                             | [10:00, 14:00)                   |
| "afternoon"                        | [11:00, 19:00)                   |
| "evening"                          | [16:00, 00:00 next day)                   |
| "night"                            | [19:00, 07:00 next day)          |
| no time qualifier                  | full day [00:00, 00:00 next day) |

When emitting intra-day windows use full ISO 8601 with "Z":
"2026-04-23T06:00:00Z". When the window IS a full day, you may emit
date-only "2026-04-23".

If multiple band qualifiers appear ("afternoon or evening"), use
the UNION (one interval per occurrence per band).

For monthly/quarterly patterns ("every March", "every Q4") use the
full-month or full-quarter span (no within-day window — the unit IS
the calendar block).

Triggers (single rule, no pure-vs-ambiguous distinction):
- Plural day-name: "what do I do on Thursdays?", "Fridays?", "my
  Wednesday meetings"
- Bare singular day-name without explicit deictic marker:
  "Saturday activities?", "Tuesday specials?"
- Bare weekend / month / quarter without year: "weekend events?",
  "what do I do in March?", "Q4 activities?"
- Possessive routine: "my morning workouts", "my Saturday routine"
- "Every X" pattern: "every Friday", "every month"

EXCEPTIONS (still emit bounded targets):
- Explicit deictic markers: "THIS Saturday", "NEXT Friday",
  "tomorrow", "yesterday" → bounded to the specific occurrence
  (still respecting the time-of-day qualifier if present).
- Specific calendar period with year: "March 2024", "Q4 2023" →
  bounded to that period.

EXAMPLES (assume REF_TIME = 2026-04-23 = Thursday):

Query: "What do I do on Thursdays?"  (no time qualifier — full day)
{{"targets":[{{"intervals":[
  {{"lo":"2026-04-23","hi":"2026-04-24"}},
  {{"lo":"2026-04-16","hi":"2026-04-17"}},
  {{"lo":"2026-04-09","hi":"2026-04-10"}},
  {{"lo":"2026-04-30","hi":"2026-05-01"}}
]}}],"extremum":null}}

Query: "What do I do Thursday mornings?"  (morning band per Thursday)
{{"targets":[{{"intervals":[
  {{"lo":"2026-04-23T03:00:00Z","hi":"2026-04-23T13:00:00Z"}},
  {{"lo":"2026-04-16T03:00:00Z","hi":"2026-04-16T13:00:00Z"}},
  {{"lo":"2026-04-09T03:00:00Z","hi":"2026-04-09T13:00:00Z"}},
  {{"lo":"2026-04-30T03:00:00Z","hi":"2026-04-30T13:00:00Z"}}
]}}],"extremum":null}}

Query: "Saturday activities?"  (no time qualifier — full day)
{{"targets":[{{"intervals":[
  {{"lo":"2026-04-18","hi":"2026-04-19"}},
  {{"lo":"2026-04-11","hi":"2026-04-12"}},
  {{"lo":"2026-04-04","hi":"2026-04-05"}},
  {{"lo":"2026-04-25","hi":"2026-04-26"}}
]}}],"extremum":null}}

Query: "What do I do in Q4?"
{{"targets":[{{"intervals":[
  {{"lo":"2025-10-01","hi":"2026-01-01"}},
  {{"lo":"2024-10-01","hi":"2025-01-01"}},
  {{"lo":"2023-10-01","hi":"2024-01-01"}},
  {{"lo":"2026-10-01","hi":"2027-01-01"}}
]}}],"extremum":null}}

Query: "What did I tell you yesterday morning?"  (deictic + morning band)
{{"targets":[{{"intervals":[{{"lo":"2026-04-22T03:00:00Z","hi":"2026-04-22T13:00:00Z"}}]}}],"extremum":null}}

Query: "What do I have THIS Saturday?"  (explicit deictic, full day)
{{"targets":[{{"intervals":[{{"lo":"2026-04-25","hi":"2026-04-26"}}]}}],"extremum":null}}"""


assert _OLD_EMPTY_SECTION in BASE_PROMPT, (
    "EMPTY OUTPUT section drifted; refresh _OLD_EMPTY_SECTION"
)
PROMPT = BASE_PROMPT.replace(_OLD_EMPTY_SECTION, _NEW_EMPTY_SECTION)


def _cache_key(query: str, ref_time: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"|")
    h.update(PROMPT_VERSION.encode())
    h.update(b"|")
    h.update(query.encode())
    h.update(b"|")
    h.update(ref_time.encode())
    return h.hexdigest()


class VariantBPlanner:
    """Single consistent recurring rule: past 3 + future 1 multi-interval."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._calls = 0
        self._cache_hits = 0
        self._parse_failures = 0
        self._total = 0
        self._cache_file = CACHE_FILE
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not self._cache_file.exists():
            return {}
        try:
            return json.loads(self._cache_file.read_text())
        except Exception:
            return {}

    def _save_cache(self) -> None:
        import fcntl
        with contextlib.suppress(Exception):
            lock_path = self._cache_file.with_suffix(self._cache_file.suffix + ".lock")
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    disk: dict = {}
                    if self._cache_file.exists():
                        try:
                            disk = json.loads(self._cache_file.read_text())
                        except Exception:
                            disk = {}
                    disk.update(self._cache)
                    self._cache = disk
                    tmp = self._cache_file.with_suffix(self._cache_file.suffix + ".tmp")
                    tmp.write_text(json.dumps(self._cache))
                    tmp.replace(self._cache_file)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    async def plan(self, query: str, ref_time: str) -> Plan:
        self._total += 1
        key = _cache_key(query, ref_time)
        if key in self._cache:
            self._cache_hits += 1
            obj = self._cache[key]
            return Plan(
                targets=_json_to_targets(obj.get("targets", [])),
                extremum=obj.get("extremum"),
                raw=json.dumps(obj),
            )

        prompt = PROMPT.format(query=query, ref_time=ref_time)
        format_config: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema",
            "name": "plan",
            "strict": True,
            "schema": _PLAN_JSON_SCHEMA_NO_ANAPHORA,
        }
        text_config: ResponseTextConfigParam = {"format": format_config}
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL, input=prompt, text=text_config,
                )
                self._calls += 1
                raw = resp.output_text
                obj = json.loads(raw)
                targets = _json_to_targets(obj.get("targets", []))
                extremum = obj.get("extremum")
                if extremum not in ("latest", "earliest"):
                    extremum = None
                self._cache[key] = obj
                self._save_cache()
                return Plan(targets=targets, extremum=extremum, raw=raw)
            except Exception as e:
                self._parse_failures += 1
                return Plan(parse_error=str(e), raw="")

    def stats(self) -> dict:
        return {
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "total_queries": self._total,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total),
            "parse_failures": self._parse_failures,
        }
