"""Principles-only planner: rewrite of v4-recurring-enum-tod with no
worked examples or bench-mirroring illustrations.

Rules and explicit mappings only:
- Definitions of interval / target / multi-target / extremum
- Composition rule operators (with abstract X)
- Verb-polarity rule
- Empty-output cases (3 numbered principles)
- Recurring rule (past 3 + future 1)
- Time-of-day bands (explicit mapping table)
- Deictic resolution mappings

The JSON schema is delivered via response_format (strict json_schema)
so the LLM has the output shape from the schema, not from examples.
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


PROMPT_VERSION = "v6-principles"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_principles"
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

Resolve at the LLM level — emit the composed result, not the
intermediate operator structure.

- "in X" → ONE target = [X.lo, X.hi)
- "after X" → ONE target = [X.hi, null)   (excludes X)
- "before X" → ONE target = [null, X.lo)
- "not in X" / "outside X" / "excluding X" → ONE target = complement
  of [X] (two intervals around X)
- "in A not in B" with B inside A → ONE target = A minus B (two
  intervals flanking B)
- "not in A or B" with A, B disjoint → ONE target = three intervals
- "in A and B" (colloquial, A and B disjoint dates) → TWO targets;
  do not intersect (their intersection is empty)
- "in A or B" / "either A or B" → TWO targets
- "between A and B" → ONE target = [A.lo, B.hi) (inclusive of both)
- "since X" / "starting X" / "from X onwards" → ONE target = [X.lo, null)
- "until X" → ONE target = [null, X.hi)

# Verb-polarity rule

"not" / "didn't" / "did not" / "wasn't" attached to a VERB is EVENT
POLARITY, not temporal scoping. IGNORE it; emit the same plan as
for the affirmative verb. ONLY treat "not" as temporal scoping when
it attaches DIRECTLY to a temporal preposition ("not in X", "not
during Y", "not before Z").

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
REF_TIME. If REF_TIME falls ON an occurrence, count it as the
most-recent past.

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

For monthly/quarterly recurring units use full-month or full-quarter
spans (no within-day window).

# Per-occurrence within-day window (standardized bands)

Each within-day interval uses one of these bands based on the time-
of-day qualifier in the query. Adjacent bands overlap a few hours so
a phrase real users would call by either name still matches.

| Qualifier                     | Window (UTC)                     |
| ----------------------------- | -------------------------------- |
| "at HH:MM" / "at HHam/pm"     | [HH:00, HH+1:00)  — 1 hour       |
| "morning"                     | [03:00, 13:00)                   |
| "noon"                        | [10:00, 14:00)                   |
| "afternoon"                   | [11:00, 19:00)                   |
| "evening"                     | [16:00, 00:00 next day)          |
| "night"                       | [19:00, 07:00 next day)          |
| no qualifier                  | full day [00:00, 00:00 next day) |

When emitting intra-day windows use full ISO 8601 with "Z". When
the window IS a full day, date-only is fine. If multiple band
qualifiers appear, use their union.

# Extremum

Set extremum ONLY when the query asks the system to PICK the most-
recent / oldest from MULTIPLE candidates the user knows exist.

DO NOT set extremum for:
- "just" used as recently-deictic ("I just had X", "we just shipped
  Y") — not "latest-of-many".
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


class PrinciplesPlanner:
    """v6: principles-only prompt rewrite. Same schema, no examples."""

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
