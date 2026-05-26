"""V7 direct planner — gpt-5-mini emits a flat list of TimeRange refs.

The planner DROPS the legacy intersect/after/before/disjoint relation
enum entirely. The LLM resolves dates AND performs set algebra
(complement, intersection, union) in its head, emitting a flat list of
refs directly.

Each ref is a TimeRange — a set of allowed moments expressed as one or
more half-open intervals. Scoring is mean-of-per-ref-bests over the
flat list (see scoring.py). The planner's only structural decision is
ONE multi-interval ref ("set membership") vs MULTIPLE refs ("graded
coverage").

Compared to V1's `QueryPlanner + plan_to_query_refs`:
- 1 LLM call per query, vs 1 + N (one per leaf phrase) in V1.
- No translation layer between planner output and V7 internals — the
  LLM speaks V7's TimeRange vocabulary directly.
- Anaphoric event references ("since the launch") are correctly
  returned as empty plans; downstream semantic search handles them.

Qualitative viability validated on 42 cases (12 specified + 15 held-out
+ 15 adversarial); see project_temporal_v7_status.md. Verb-polarity
rule in the prompt prevents "did not happen in X" misparsing.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

from .time_range import NEG_INF, POS_INF, Interval, TimeRange

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


MODEL = "gpt-5-mini"
PER_CALL_TIMEOUT_S = 45.0
CONCURRENCY = 8
PROMPT_VERSION = "direct_v3"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_direct"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_direct_plan_cache.json"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT = """You translate a natural-language query into a TIME-RANGE PLAN.

A query describes WHAT MOMENTS IN TIME a matching document's date anchor
should fall inside. You describe that set as a list of REFS. Each ref is
a SET of allowed moments expressed as one or more half-open intervals
[lo, hi). Null endpoints mean unbounded (lo=null is -infinity; hi=null
is +infinity).

OUTPUT SHAPE
============
{{
  "refs": [
    {{"intervals": [{{"lo": "YYYY-MM-DD"|null, "hi": "YYYY-MM-DD"|null}}, ...]}}
  ],
  "extremum": "latest" | "earliest" | null
}}

A doc anchor scores higher when it satisfies MORE of the refs
(mean of per-ref overlap).

KEY CONCEPTS
============
- An INTERVAL is half-open [lo, hi). hi is EXCLUSIVE: "March 2024" =
  lo "2024-03-01", hi "2024-04-01".
- A REF is a SET of allowed moments. The doc anchor satisfies a ref if
  it falls in ANY of the ref's intervals (intra-ref OR via multi-interval).
- Multiple REFS = each ref is a SEPARATE constraint. The doc is scored
  by how many it satisfies (graded coverage).

WHEN TO EMIT ONE REF (MULTI-INTERVAL) vs MULTIPLE REFS
======================================================
This is the planner's only structural decision.

ONE ref with multiple intervals — use when the query describes a SINGLE
allowed REGION that happens to have holes or discontinuities. The
intervals are interchangeable: ANY ONE of them satisfies the user.
  - "not in 2023" → one ref = (-inf, 2023) and (2024, +inf)
  - "in 2024 not in summer" → one ref = [Jan-Jun 2024] and [Sep-Dec 2024]
  - "between A and B" → one ref = [A.lo, B.hi)
  - any single contiguous period ("in 2024", "after March 2020") → one ref

MULTIPLE refs — use when the query lists SEPARATE periods the doc should
match independently. Coverage matters: matching both > matching one.
  - "in 2020 and 2024" (colloquial, disjoint) → two refs (one per year)
  - "in 2020 or 2024" (explicit OR, disjoint) → two refs (graded coverage)
  - "in Q1 or Q4 of 2023" → two refs (each quarter is its own period)

The litmus: if a doc that mentions BOTH periods should rank higher than
one that mentions ONE → emit multiple refs. If they're interchangeable
(any one is fine) → emit one multi-interval ref.

REF_TIME is provided for resolving relative phrases ("recently", "two
weeks ago", "last quarter"). For absolute dates you don't need it.

VERB-POLARITY RULE — CRITICAL
=============================
"not" / "didn't" / "did not" / "wasn't" attached to a VERB is EVENT
POLARITY, not temporal scoping. IGNORE it. Emit the same plan as if the
verb were affirmative. ONLY treat "not" as temporal scoping when it
attaches DIRECTLY to a temporal preposition ("not in X", "not during Y",
"not before Z").

  "what did not happen in 2024" — "not" attaches to the verb "happen",
    NOT to "in 2024". Treat as: "what happened in 2024" → one ref [2024].
  "what wasn't completed by March" — "wasn't" is verb polarity. Treat as
    "what was completed by March" → one ref (-inf, March).

  Contrast with temporal-scoping negation:
  "what happened NOT in 2024" → complement of 2024 (rare phrasing).
  "what happened outside 2024" → complement of 2024.
  "what happened excluding 2024" → complement of 2024.

COMPOSITION RULES (do these AT THE LLM LEVEL — emit the composed result)
=======================================================================
- "in X" → ONE ref = [X.lo, X.hi).
- "after X" → ONE ref = [X.hi, null). (X.hi because "after X" excludes X.)
- "before X" → ONE ref = [null, X.lo).
- "not in X" / "outside X" / "excluding X" → ONE ref = complement of [X]
  = two intervals [null, X.lo) and [X.hi, null).
- "in A not in B" (with B inside A) → ONE ref = A minus B = two intervals
  [A.lo, B.lo) and [B.hi, A.hi).
- "not in A or B" (A and B disjoint) → ONE ref = three intervals
  [null, A.lo), [A.hi, B.lo), [B.hi, null).
- "in A and B" (colloquial; A and B disjoint dates) → TWO refs (one for
  A, one for B). DO NOT intersect them — the intersection is empty.
- "in A or B" / "either A or B" → TWO refs.
- "between A and B" → ONE ref = [A.lo, B.hi) (inclusive of both endpoints).
- "since X" / "starting X" / "from X onwards" → ONE ref = [X.lo, null).
- "until X" → ONE ref = [null, X.hi).

EMPTY OUTPUT
============
If the query has NO temporal scope (e.g., "what happened recently",
"how do I plan my morning?", "lessons from the launch", "how did the
migration go?") emit {{"refs": []}}.

ANAPHORIC EVENT REFERENCES
==========================
If the query references an event by name and you don't know its date
("since the v3 launch", "after the redesign", "before the merger"), emit
{{"refs": []}}. The downstream pipeline resolves anaphoric references
separately via semantic search.

EXTREMUM
========
Set extremum ONLY when the query asks the system to PICK the most-recent /
oldest from MULTIPLE candidates the user knows exist.
  "Most recent meeting in March 2024" → extremum="latest"
  "earliest job" → extremum="earliest"
  "When did Marcus host his first dinner party?" → extremum=null (specific event)

DO NOT set extremum for:
  - "I just had X" / "we just shipped Y" / "I just ran into Z" — "just"
    here means RECENTLY-DEICTIC, NOT a request for the latest-of-many.
    These are statements about a single recent event, often followed by
    a non-retrieval question ("…what severity is this?"). Use extremum=null.
  - "first" / "last" describing a SPECIFIC occurrence the user has in
    mind (e.g., "my first novel", "the first Utah road trip", "his last
    surgery"). The user is naming ONE event, not asking to PICK from many.
    Use extremum=null and let semantic search find that single event.

DEICTIC RESOLUTION
==================
Resolve deictic phrases against REF_TIME. DEICTIC PHRASES ALWAYS GIVE A
TEMPORAL SCOPE, even when they appear alongside an anaphoric event
reference. If a query mixes both, emit the DEICTIC range and drop the
anaphoric.

  "What issue did we hit during the migration last quarter?" → "last
    quarter" is deictic; "the migration" is anaphoric. Emit one ref for
    last quarter (relative to ref_time).
  "How was the launch this morning?" → "this morning" is deictic; "the
    launch" is anaphoric. Emit the morning interval.

Resolutions:
- "this year" → [Jan 1 of ref_time year, Jan 1 of next year)
- "last year" → year before
- "this quarter" / "last quarter" / "next quarter" → corresponding calendar quarter
- "this month" / "last month" / "next month" → corresponding calendar month
- "yesterday" → 1-day interval before ref_time's date
- "today" → ref_time's date (1-day)
- "this week" / "last week" / "next week" → Mon-Sun week-of
- "two weeks ago" / "three months ago" → resolve arithmetically

EXAMPLES
========

Query: "in March 2024"
{{"refs":[{{"intervals":[{{"lo":"2024-03-01","hi":"2024-04-01"}}]}}],"extremum":null}}

Query: "after 2020"
{{"refs":[{{"intervals":[{{"lo":"2021-01-01","hi":null}}]}}],"extremum":null}}

Query: "before 1999"
{{"refs":[{{"intervals":[{{"lo":null,"hi":"1999-01-01"}}]}}],"extremum":null}}

Query: "not in 2023"
{{"refs":[{{"intervals":[{{"lo":null,"hi":"2023-01-01"}},{{"lo":"2024-01-01","hi":null}}]}}],"extremum":null}}

Query: "in 2024 not in summer"
{{"refs":[{{"intervals":[{{"lo":"2024-01-01","hi":"2024-06-01"}},{{"lo":"2024-09-01","hi":"2025-01-01"}}]}}],"extremum":null}}

Query: "in Q1 or Q4 of 2023"
{{"refs":[
  {{"intervals":[{{"lo":"2023-01-01","hi":"2023-04-01"}}]}},
  {{"intervals":[{{"lo":"2023-10-01","hi":"2024-01-01"}}]}}
],"extremum":null}}

Query: "in 2020 and 2024"
{{"refs":[
  {{"intervals":[{{"lo":"2020-01-01","hi":"2021-01-01"}}]}},
  {{"intervals":[{{"lo":"2024-01-01","hi":"2025-01-01"}}]}}
],"extremum":null}}

Query: "between 2020 and 2024"
{{"refs":[{{"intervals":[{{"lo":"2020-01-01","hi":"2025-01-01"}}]}}],"extremum":null}}

Query: "what did NOT happen on May 3 2024" (verb-polarity rule applies)
{{"refs":[{{"intervals":[{{"lo":"2024-05-03","hi":"2024-05-04"}}]}}],"extremum":null}}

Query: "latest budget review in Q2 2024"
{{"refs":[{{"intervals":[{{"lo":"2024-04-01","hi":"2024-07-01"}}]}}],"extremum":"latest"}}

Query: "what did I do recently"
{{"refs":[],"extremum":"latest"}}

Query: "How do I plan my morning?"
{{"refs":[],"extremum":null}}

Query: "since the v3 launch"
{{"refs":[],"extremum":null}}

NOW PRODUCE THE PLAN FOR:

Query: {query}
Reference time: {ref_time}
"""


_PLAN_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["refs", "extremum"],
    "properties": {
        "refs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["intervals"],
                "properties": {
                    "intervals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["lo", "hi"],
                            "properties": {
                                "lo": {
                                    "type": ["string", "null"],
                                },
                                "hi": {
                                    "type": ["string", "null"],
                                },
                            },
                        },
                    },
                },
            },
        },
        "extremum": {
            "type": ["string", "null"],
            "enum": ["latest", "earliest", None],
        },
    },
}


# ---------------------------------------------------------------------------
# Data class for the direct planner's resolved output
# ---------------------------------------------------------------------------


@dataclass
class DirectPlan:
    """Resolved plan: flat list of TimeRange refs + extremum + raw text."""

    refs: list[TimeRange] = field(default_factory=list)
    extremum: str | None = None
    raw: str | None = field(default=None, repr=False)
    parse_error: str | None = field(default=None, repr=False)

    @property
    def latest_intent(self) -> bool:
        return self.extremum == "latest"

    @property
    def earliest_intent(self) -> bool:
        return self.extremum == "earliest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "refs": [
                {
                    "intervals": [
                        {
                            "lo": _us_to_iso(iv.earliest_us),
                            "hi": _us_to_iso(iv.latest_us),
                        }
                        for iv in r.intervals
                    ]
                }
                for r in self.refs
            ],
            "extremum": self.extremum,
        }


# ---------------------------------------------------------------------------
# JSON ↔ TimeRange conversion
# ---------------------------------------------------------------------------


def _iso_to_us(s: str | None) -> int:
    """Parse YYYY-MM-DD (or full ISO) to µs timestamp.

    `None` means unbounded; the caller maps None to NEG_INF or POS_INF.
    """
    if s is None:
        return None  # type: ignore[return-value]
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot parse date {s!r}: {e}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def _us_to_iso(us: int) -> str | None:
    """Convert µs timestamp back to YYYY-MM-DD, with ±∞ → None."""
    if us <= NEG_INF + 1 or us >= POS_INF - 1:
        return None
    dt = datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _json_intervals_to_timerange(json_intervals: list[dict]) -> TimeRange:
    """Convert the LLM's interval JSON to a TimeRange.

    `lo=None` → NEG_INF; `hi=None` → POS_INF. Invalid (lo>=hi) intervals
    are dropped silently; canonicalize handles the rest.
    """
    ivs: list[Interval] = []
    for j in json_intervals:
        lo_s = j.get("lo")
        hi_s = j.get("hi")
        try:
            lo = NEG_INF if lo_s is None else _iso_to_us(lo_s)
            hi = POS_INF if hi_s is None else _iso_to_us(hi_s)
        except ValueError:
            continue
        if lo < hi:
            ivs.append(Interval(lo, hi))
    return TimeRange.from_intervals(ivs)


def _json_to_refs(json_refs: list[dict]) -> list[TimeRange]:
    """Convert the LLM's ref list to a flat list of TimeRanges."""
    out: list[TimeRange] = []
    for jr in json_refs:
        r = _json_intervals_to_timerange(jr.get("intervals", []))
        if r.intervals:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class DirectQueryPlanner:
    """V7 direct planner — emits a flat TimeRange ref list in one LLM call.

    Returns `DirectPlan(refs=list[TimeRange], extremum=str|None)`. No
    intermediate relation enum, no per-leaf extractor calls. The model
    resolves dates and does set algebra in its head.
    """

    def __init__(
        self,
        prompt_template: str | None = None,
        cache_subdir: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._calls = 0
        self._cache_hits = 0
        self._parse_failures = 0
        self._total = 0
        self._prompt_template = prompt_template or PROMPT
        if cache_subdir is None:
            self._cache_file = CACHE_FILE
        else:
            cache_dir = ROOT / "cache" / cache_subdir
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = cache_dir / "llm_direct_plan_cache.json"
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

    async def plan(self, query: str, ref_time: str) -> DirectPlan:
        self._total += 1
        key = _cache_key(query, ref_time)
        if key in self._cache:
            self._cache_hits += 1
            try:
                obj = self._cache[key]
                return DirectPlan(
                    refs=_json_to_refs(obj.get("refs", [])),
                    extremum=obj.get("extremum"),
                    raw=json.dumps(obj),
                )
            except Exception:
                pass

        prompt = self._prompt_template.format(query=query, ref_time=ref_time)
        format_config: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema",
            "name": "direct_plan",
            "strict": True,
            "schema": _PLAN_JSON_SCHEMA,
        }
        text_config: ResponseTextConfigParam = {"format": format_config}
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL,
                    input=prompt,
                    text=text_config,
                )
                self._calls += 1
                raw = resp.output_text
                obj = json.loads(raw)
                refs = _json_to_refs(obj.get("refs", []))
                extremum = obj.get("extremum")
                if extremum not in ("latest", "earliest"):
                    extremum = None
                self._cache[key] = obj
                self._save_cache()
                return DirectPlan(refs=refs, extremum=extremum, raw=raw)
            except Exception as e:
                self._parse_failures += 1
                return DirectPlan(parse_error=str(e), raw="")

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
