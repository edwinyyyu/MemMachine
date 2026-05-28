"""The temporal retrieval planner — emits a flat list of IntervalSet targets.

A single LLM call resolves dates AND performs set algebra (complement,
intersection, union), emitting a flat list of targets directly.

Each target is an `IntervalSet` — a set of allowed moments expressed as
one or more half-open intervals. Scoring is mean-of-per-target-bests
over the flat list (see scoring.py). The planner's only structural
decision is ONE multi-interval target ("set membership") vs MULTIPLE
targets ("graded coverage").
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

from .time_range import NEG_INF, POS_INF, Interval, IntervalSet, Endpoint, is_inf

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


MODEL = "gpt-5-mini"
PER_CALL_TIMEOUT_S = 45.0
CONCURRENCY = 8
PROMPT_VERSION = "v2-anaphora"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT = """You translate a natural-language query into a TIME-RANGE PLAN.

A query describes WHAT MOMENTS IN TIME a matching document's date anchor
should fall inside. You describe that set as a list of TARGETS. Each target
is a SET of allowed moments expressed as one or more half-open intervals
[lo, hi). Null endpoints mean unbounded (lo=null is -infinity; hi=null
is +infinity).

OUTPUT SHAPE
============
{{
  "targets": [
    {{"intervals": [{{"lo": "YYYY-MM-DD"|null, "hi": "YYYY-MM-DD"|null}}, ...]}}
  ],
  "anaphora": [
    {{"phrase": "the migration", "relation": "after"|"since"|"before"|"during"}}
  ],
  "extremum": "latest" | "earliest" | null
}}

`targets` are concrete time sets you've already resolved. `anaphora`
are references to named events whose date you DON'T know — the
downstream resolver finds them in the corpus and converts each to a
target. A doc anchor scores higher when it satisfies MORE of the
(resolved) targets — mean of per-target overlap.

KEY CONCEPTS
============
- An INTERVAL is half-open [lo, hi). hi is EXCLUSIVE: "March 2024" =
  lo "2024-03-01", hi "2024-04-01".
- A TARGET is a SET of allowed moments. The doc anchor satisfies a target
  if it falls in ANY of the target's intervals (intra-target OR via
  multi-interval).
- Multiple TARGETS = each target is a SEPARATE constraint. The doc is
  scored by how many it satisfies (graded coverage).

WHEN TO EMIT ONE TARGET (MULTI-INTERVAL) vs MULTIPLE TARGETS
============================================================
This is the planner's only structural decision.

ONE target with multiple intervals — use when the query describes a SINGLE
allowed REGION that happens to have holes or discontinuities. The
intervals are interchangeable: ANY ONE of them satisfies the user.
  - "not in 2023" → one target = (-inf, 2023) and (2024, +inf)
  - "in 2024 not in summer" → one target = [Jan-Jun 2024] and [Sep-Dec 2024]
  - "between A and B" → one target = [A.lo, B.hi)
  - any single contiguous period ("in 2024", "after March 2020") → one target

MULTIPLE targets — use when the query lists SEPARATE periods the doc should
match independently. Coverage matters: matching both > matching one.
  - "in 2020 and 2024" (colloquial, disjoint) → two targets (one per year)
  - "in 2020 or 2024" (explicit OR, disjoint) → two targets (graded coverage)
  - "in Q1 or Q4 of 2023" → two targets (each quarter is its own period)

The litmus: if a doc that mentions BOTH periods should rank higher than
one that mentions ONE → emit multiple targets. If they're interchangeable
(any one is fine) → emit one multi-interval target.

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
    NOT to "in 2024". Treat as: "what happened in 2024" → one target [2024].
  "what wasn't completed by March" — "wasn't" is verb polarity. Treat as
    "what was completed by March" → one target (-inf, March).

  Contrast with temporal-scoping negation:
  "what happened NOT in 2024" → complement of 2024 (rare phrasing).
  "what happened outside 2024" → complement of 2024.
  "what happened excluding 2024" → complement of 2024.

COMPOSITION RULES (do these AT THE LLM LEVEL — emit the composed result)
=======================================================================
- "in X" → ONE target = [X.lo, X.hi).
- "after X" → ONE target = [X.hi, null). (X.hi because "after X" excludes X.)
- "before X" → ONE target = [null, X.lo).
- "not in X" / "outside X" / "excluding X" → ONE target = complement of [X]
  = two intervals [null, X.lo) and [X.hi, null).
- "in A not in B" (with B inside A) → ONE target = A minus B = two intervals
  [A.lo, B.lo) and [B.hi, A.hi).
- "not in A or B" (A and B disjoint) → ONE target = three intervals
  [null, A.lo), [A.hi, B.lo), [B.hi, null).
- "in A and B" (colloquial; A and B disjoint dates) → TWO targets (one for
  A, one for B). DO NOT intersect them — the intersection is empty.
- "in A or B" / "either A or B" → TWO targets.
- "between A and B" → ONE target = [A.lo, B.hi) (inclusive of both endpoints).
- "since X" / "starting X" / "from X onwards" → ONE target = [X.lo, null).
- "until X" → ONE target = [null, X.hi).

EMPTY OUTPUT
============
If the query has NO temporal scope at all (e.g., "how do I plan my
morning?", "lessons from the launch", "how did the migration go?") emit
{{"targets": [], "anaphora": []}}.

"What did I do recently" / "show me what happened lately" → deictic;
resolve to a recent window (e.g. last 60-90 days from REF_TIME) as a
target with extremum="latest".

ANAPHORIC EVENT REFERENCES
==========================
If the query references an event by NAME without giving its date —
"since the v3 launch", "after the redesign", "before the merger",
"in Q3 2023 after the launch" — do NOT drop it. Emit the noun phrase
and the relation in the `anaphora` field:

  {{"phrase": "the migration", "relation": "after"}}

Relations:
  "after X" / "post X" → relation: "after"   (strictly after the event)
  "since X" / "from X onward" → relation: "since"   (from X inclusive)
  "before X" / "pre X" → relation: "before"
  "during X" / "while X" / "at X" → relation: "during"

CRUCIAL: any ABSOLUTE or DEICTIC temporal scope the query ALSO contains
must STILL be emitted as a target. Only the anaphoric noun phrase goes
in `anaphora`.

  "since the v3 launch"
    → targets: [], anaphora: [{{"phrase":"the v3 launch","relation":"since"}}]
    (no absolute scope — only the anaphor)

  "in Q3 2023 after the launch"
    → targets: [Q3 2023], anaphora: [{{"phrase":"the launch","relation":"after"}}]
    (BOTH the Q3 2023 target AND the anaphoric "after the launch")

  "what did I do in 2024 after the migration"
    → targets: [2024], anaphora: [{{"phrase":"the migration","relation":"after"}}]

The phrase you emit should be the bare noun phrase WITHOUT the relation
word — `"the migration"`, not `"after the migration"`. The relation
goes in its own field.

Multiple anaphoric references → emit each as its own entry in the array.

  "before the launch and after the freeze"
    → anaphora: [
        {{"phrase":"the launch","relation":"before"}},
        {{"phrase":"the freeze","relation":"after"}}
      ]

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
Resolve deictic phrases against REF_TIME and emit them as targets. If a
query mixes deictic AND anaphoric, emit BOTH — deictic to `targets`,
anaphoric to `anaphora`.

  "What issue did we hit during the migration last quarter?" → "last
    quarter" is deictic (a target); "the migration" is anaphoric.
    targets: [last quarter], anaphora: [{{"phrase":"the migration","relation":"during"}}]
  "How was the launch this morning?" → "this morning" is deictic;
    "the launch" is anaphoric.
    targets: [this morning], anaphora: [{{"phrase":"the launch","relation":"during"}}]

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
{{"targets":[{{"intervals":[{{"lo":"2024-03-01","hi":"2024-04-01"}}]}}],"anaphora":[],"extremum":null}}

Query: "after 2020"
{{"targets":[{{"intervals":[{{"lo":"2021-01-01","hi":null}}]}}],"anaphora":[],"extremum":null}}

Query: "before 1999"
{{"targets":[{{"intervals":[{{"lo":null,"hi":"1999-01-01"}}]}}],"anaphora":[],"extremum":null}}

Query: "not in 2023"
{{"targets":[{{"intervals":[{{"lo":null,"hi":"2023-01-01"}},{{"lo":"2024-01-01","hi":null}}]}}],"anaphora":[],"extremum":null}}

Query: "in 2024 not in summer"
{{"targets":[{{"intervals":[{{"lo":"2024-01-01","hi":"2024-06-01"}},{{"lo":"2024-09-01","hi":"2025-01-01"}}]}}],"anaphora":[],"extremum":null}}

Query: "in Q1 or Q4 of 2023"
{{"targets":[
  {{"intervals":[{{"lo":"2023-01-01","hi":"2023-04-01"}}]}},
  {{"intervals":[{{"lo":"2023-10-01","hi":"2024-01-01"}}]}}
],"anaphora":[],"extremum":null}}

Query: "in 2020 and 2024"
{{"targets":[
  {{"intervals":[{{"lo":"2020-01-01","hi":"2021-01-01"}}]}},
  {{"intervals":[{{"lo":"2024-01-01","hi":"2025-01-01"}}]}}
],"anaphora":[],"extremum":null}}

Query: "between 2020 and 2024"
{{"targets":[{{"intervals":[{{"lo":"2020-01-01","hi":"2025-01-01"}}]}}],"anaphora":[],"extremum":null}}

Query: "what did NOT happen on May 3 2024" (verb-polarity rule applies)
{{"targets":[{{"intervals":[{{"lo":"2024-05-03","hi":"2024-05-04"}}]}}],"anaphora":[],"extremum":null}}

Query: "latest budget review in Q2 2024"
{{"targets":[{{"intervals":[{{"lo":"2024-04-01","hi":"2024-07-01"}}]}}],"anaphora":[],"extremum":"latest"}}

Query: "How do I plan my morning?"
{{"targets":[],"anaphora":[],"extremum":null}}

Anaphora examples — events the query names without a date:

Query: "since the v3 launch"
{{"targets":[],"anaphora":[{{"phrase":"the v3 launch","relation":"since"}}],"extremum":null}}

Query: "what happened in 2024 after the migration"
{{"targets":[{{"intervals":[{{"lo":"2024-01-01","hi":"2025-01-01"}}]}}],"anaphora":[{{"phrase":"the migration","relation":"after"}}],"extremum":null}}

Query: "in Q3 2023 after the launch"
{{"targets":[{{"intervals":[{{"lo":"2023-07-01","hi":"2023-10-01"}}]}}],"anaphora":[{{"phrase":"the launch","relation":"after"}}],"extremum":null}}

Query: "things I did in Q4 2024 before year-end review"
{{"targets":[{{"intervals":[{{"lo":"2024-10-01","hi":"2025-01-01"}}]}}],"anaphora":[{{"phrase":"year-end review","relation":"before"}}],"extremum":null}}

Query: "my most recent update after the migration"
{{"targets":[],"anaphora":[{{"phrase":"the migration","relation":"after"}}],"extremum":"latest"}}

NOW PRODUCE THE PLAN FOR:

Query: {query}
Reference time: {ref_time}
"""


_PLAN_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["targets", "anaphora", "extremum"],
    "properties": {
        "targets": {
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
        "anaphora": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["phrase", "relation"],
                "properties": {
                    "phrase": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["after", "since", "before", "during"],
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
# Data class for the resolved planner output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnaphoricTarget:
    """An unresolved temporal reference — the planner identifies the phrase
    and its relation, the resolver finds the date in the corpus."""

    phrase: str
    relation: str  # "after" | "since" | "before" | "during"


_VALID_RELATIONS = frozenset({"after", "since", "before", "during"})


@dataclass
class Plan:
    """Resolved plan: targets (concrete IntervalSets) + anaphora (deferred
    references resolved against the corpus) + extremum."""

    targets: list[IntervalSet] = field(default_factory=list)
    anaphora: list[AnaphoricTarget] = field(default_factory=list)
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
            "targets": [
                {
                    "intervals": [
                        {
                            "lo": _us_to_iso(iv.earliest_us),
                            "hi": _us_to_iso(iv.latest_us),
                        }
                        for iv in t.intervals
                    ]
                }
                for t in self.targets
            ],
            "anaphora": [
                {"phrase": a.phrase, "relation": a.relation}
                for a in self.anaphora
            ],
            "extremum": self.extremum,
        }


# ---------------------------------------------------------------------------
# JSON ↔ IntervalSet conversion
# ---------------------------------------------------------------------------


def _iso_to_us(s: str) -> int:
    """Parse YYYY-MM-DD (or full ISO) to µs timestamp.

    Callers map `None` to NEG_INF/POS_INF before calling this; do not pass
    `None` here.
    """
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


def _us_to_iso(us: Endpoint) -> str | None:
    """Convert µs timestamp back to YYYY-MM-DD, with ±∞ → None."""
    if is_inf(us):
        return None
    dt = datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _json_intervals_to_interval_set(json_intervals: list[dict]) -> IntervalSet:
    """Convert the LLM's interval JSON to an IntervalSet.

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
    return IntervalSet.from_intervals(ivs)


def _json_to_targets(json_targets: list[dict]) -> list[IntervalSet]:
    """Convert the LLM's target list to a flat list of IntervalSets."""
    out: list[IntervalSet] = []
    for jt in json_targets:
        t = _json_intervals_to_interval_set(jt.get("intervals", []))
        if t.intervals:
            out.append(t)
    return out


def _json_to_anaphora(json_anaphora: list[dict]) -> list[AnaphoricTarget]:
    """Convert the LLM's anaphora list to AnaphoricTarget dataclasses.

    Drops entries with empty phrase or unknown relation."""
    out: list[AnaphoricTarget] = []
    for ja in json_anaphora:
        phrase = (ja.get("phrase") or "").strip()
        relation = ja.get("relation")
        if not phrase or relation not in _VALID_RELATIONS:
            continue
        out.append(AnaphoricTarget(phrase=phrase, relation=relation))
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


class QueryPlanner:
    """The temporal retrieval planner — emits a flat IntervalSet target list in one LLM call.

    Returns `Plan(targets=list[IntervalSet], extremum=str|None)`. No
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
            self._cache_file = cache_dir / "llm_plan_cache.json"
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
            try:
                obj = self._cache[key]
                return Plan(
                    targets=_json_to_targets(obj.get("targets", [])),
                    anaphora=_json_to_anaphora(obj.get("anaphora", [])),
                    extremum=obj.get("extremum"),
                    raw=json.dumps(obj),
                )
            except Exception:
                pass

        prompt = self._prompt_template.format(query=query, ref_time=ref_time)
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
                    model=MODEL,
                    input=prompt,
                    text=text_config,
                )
                self._calls += 1
                raw = resp.output_text
                obj = json.loads(raw)
                targets = _json_to_targets(obj.get("targets", []))
                anaphora = _json_to_anaphora(obj.get("anaphora", []))
                extremum = obj.get("extremum")
                if extremum not in ("latest", "earliest"):
                    extremum = None
                self._cache[key] = obj
                self._save_cache()
                return Plan(targets=targets, anaphora=anaphora,
                            extremum=extremum, raw=raw)
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
