"""Planner v7: principles + output-shape demos only (no NL examples).

Iteration on v6-principles which lost axis -0.150 and lattice -0.050:
the principles alone didn't lock in the specific JSON output structure
for the recurring rule. v7 adds bare output-shape templates with
placeholders (YYYY-MM-DD, HH-strings) — NO natural-language query →
JSON example pairs.

The structural rules the LLM needs to derive:
- Recurring → ONE target with FOUR intervals
- Each interval per occurrence at a specific date
- TOD bands → intra-day windows with full ISO-with-Z

These are shown as shape templates, not as worked input→output cases.
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
    _json_to_targets, _PLAN_JSON_SCHEMA,
)

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except Exception:
        pass


PROMPT_VERSION = "v7-shape-demos"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_shape_demos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


PROMPT = """You translate a natural-language query into a TIME-RANGE PLAN.

A query describes WHAT MOMENTS IN TIME a matching document's date
anchor should fall inside. You describe that set as a list of
TARGETS. Each target is a SET of allowed moments expressed as one or
more half-open intervals [lo, hi). Null endpoints mean unbounded
(lo=null is -infinity; hi=null is +infinity).

A doc anchor scores higher when it satisfies MORE targets — mean of
per-target overlap.

# Key concepts

- An INTERVAL is half-open [lo, hi). hi is EXCLUSIVE.
- A TARGET is a SET of allowed moments. A doc anchor satisfies a
  target if it falls in ANY of the target's intervals.
- Multiple TARGETS = each target is a SEPARATE constraint. The doc
  is scored by how many it satisfies (graded coverage).

# One multi-interval target vs multiple targets

ONE target with multiple intervals when the query describes a
SINGLE allowed region that happens to have holes — the intervals
are interchangeable, any one satisfies.

MULTIPLE targets when the query lists SEPARATE periods the doc
should match independently — coverage matters, matching both >
matching one.

Litmus: if matching BOTH periods should rank higher than matching
one → emit multiple targets. If they're interchangeable → emit one
multi-interval target.

# Composition rules

Resolve at the LLM level — emit the composed result.

- "in X" → ONE target = [X.lo, X.hi)
- "after X" → ONE target = [X.hi, null)   (excludes X)
- "before X" → ONE target = [null, X.lo)
- "not in X" / "outside X" / "excluding X" → ONE target = complement
  of [X] (two intervals around X)
- "in A not in B" with B inside A → ONE target = A minus B (two
  intervals flanking B)
- "not in A or B" with A, B disjoint → ONE target = three intervals
- "in A and B" colloquial with A, B disjoint → TWO targets; do not
  intersect (intersection is empty)
- "in A or B" / "either A or B" → TWO targets
- "between A and B" → ONE target = [A.lo, B.hi)
- "since X" / "from X onwards" → ONE target = [X.lo, null)
- "until X" → ONE target = [null, X.hi)

# Verb-polarity rule

"not" / "didn't" / "did not" / "wasn't" attached to a VERB is EVENT
POLARITY, not temporal scoping. IGNORE it; emit the same plan as
for the affirmative verb. ONLY treat "not" as temporal scoping when
it attaches DIRECTLY to a temporal preposition.

# Empty output

Emit empty targets when the query has no usable temporal anchor:

1. No temporal flavor at all.
2. Anaphoric reference: the query names an event whose date is not
   in the query. Semantic search finds the doc by content.
3. "Most recently X" / "most recent X" with a specific subject:
   the query asks for the LATEST occurrence of a specific activity,
   not a recent-N-days window. Emit empty + extremum="latest".

A purely deictic "recently" / "lately" query without a specific
subject resolves to a recent window (e.g. last 60-90 days from
ref time) with extremum="latest".

# Recurring / habitual patterns — past 3 + future 1

A query about a recurring or habitual day-of-week, weekend, month,
or quarter (pure recurring OR singular ambiguous between "this one"
and "the pattern") emits ONE target with FOUR intervals: the 3
most-recent past occurrences + 1 next upcoming, relative to
REF_TIME. If REF_TIME falls ON an occurrence, count it as the most-
recent past.

Output shape (no time-of-day qualifier):
{{"targets":[{{"intervals":[
  {{"lo":"<most-recent-past-date>","hi":"<next-day>"}},
  {{"lo":"<2nd-past-date>","hi":"<next-day>"}},
  {{"lo":"<3rd-past-date>","hi":"<next-day>"}},
  {{"lo":"<next-upcoming-date>","hi":"<next-day>"}}
]}}],"extremum":null}}

Output shape (with TOD band — use ISO with "Z", values from the
bands table below):
{{"targets":[{{"intervals":[
  {{"lo":"<most-recent-past-date>T<band-lo>:00:00Z","hi":"<most-recent-past-date>T<band-hi>:00:00Z"}},
  {{"lo":"<2nd-past-date>T<band-lo>:00:00Z","hi":"<2nd-past-date>T<band-hi>:00:00Z"}},
  {{"lo":"<3rd-past-date>T<band-lo>:00:00Z","hi":"<3rd-past-date>T<band-hi>:00:00Z"}},
  {{"lo":"<next-upcoming-date>T<band-lo>:00:00Z","hi":"<next-upcoming-date>T<band-hi>:00:00Z"}}
]}}],"extremum":null}}

For monthly/quarterly recurring units, the same four-interval shape
applies with full-month or full-quarter spans (no within-day window).

Triggers (single rule, no pure-vs-ambiguous distinction):
- Plural weekday name, or after "on", or in a possessive routine.
- Bare singular weekday name with no "this"/"next"/"last" qualifier.
- Bare weekend / month-name / quarter with no year.
- Possessive routine.
- "Every X" pattern.

Exceptions (still emit bounded targets):
- Explicit deictic markers ("this", "next", "last", "tomorrow",
  "yesterday") → bound to the specific occurrence (respecting any
  time-of-day qualifier).
- Specific calendar period with year → bounded to that period.

# Time-of-day bands (standardized, overlapping at boundaries)

| Qualifier                     | <band-lo> – <band-hi>           |
| ----------------------------- | ------------------------------- |
| "at HH:MM" / "at HHam/pm"     | HH:00 – HH+1:00  (1 hour)       |
| "morning"                     | 03:00 – 13:00                   |
| "noon"                        | 10:00 – 14:00                   |
| "afternoon"                   | 11:00 – 19:00                   |
| "evening"                     | 16:00 – 00:00 next day          |
| "night"                       | 19:00 – 07:00 next day          |
| no qualifier                  | full day                        |

When the band spans into the next day, the interval's lo and hi are
on different calendar dates.

# Extremum

Set extremum ONLY when the query asks the system to PICK the most-
recent / oldest from MULTIPLE candidates the user knows exist.

DO NOT set extremum for:
- "just" used as recently-deictic — not "latest-of-many".
- "first" / "last" describing a SPECIFIC occurrence the user has in
  mind — the user is naming ONE event, not asking to pick from many.

# Deictic resolution

Resolve deictic phrases against REF_TIME:

- "this year" → [Jan 1 of ref year, Jan 1 of next year)
- "last year" → year before
- "this quarter" / "last quarter" / "next quarter" → corresponding
  calendar quarter
- "this month" / "last month" / "next month" → corresponding
  calendar month
- "yesterday" → 1-day interval before ref date
- "today" → ref date (1-day)
- "this week" / "last week" / "next week" → Mon-Sun week of ref
- "two weeks ago" / "three months ago" → resolve arithmetically

NOW PRODUCE THE PLAN FOR:

Query: {query}
Reference time: {ref_time}
"""


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


class ShapeDemosPlanner:
    """v7: principles + output-shape templates with placeholders, no NL examples."""

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
            "schema": _PLAN_JSON_SCHEMA,
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
