"""The temporal retrieval planner — emits a flat list of IntervalSet targets.

A single LLM call resolves dates AND performs set algebra (complement,
intersection, union), emitting a flat list of targets directly.

Each target is an `IntervalSet` — a set of allowed moments expressed as
one or more half-open intervals. Scoring is mean-of-per-target-bests
over the flat list (see scoring.py). The planner's only structural
decision is ONE multi-interval target ("set membership") vs MULTIPLE
targets ("graded coverage").

Recurring patterns are modeled as a SINGLE multi-interval target with
4 intervals: the 3 most-recent past occurrences + 1 next upcoming
occurrence (relative to REF_TIME). The matching doc anchor satisfies
the target if it falls in ANY interval — so the LLM avoids
overcommitting to specific dates that haven't happened ("every
Thursday" → 4 Thursdays, not 13). Intra-day time-of-day qualifiers
("morning", "evening") are encoded with standardized overlapping bands
so morning vs evening docs stay distinguishable while boundary cases
(5pm = afternoon or evening?) still match either qualifier.

Anaphoric references ("since the v3 launch") and "most recently X"
queries emit empty targets — date is unknown, let semantic search
carry. See `research/_variant_b_*` for the A/Bs that validated this
design and `research/_extractor_v3_5_recurring.py` for the matching
doc-side extractor still under iteration.
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
PROMPT_VERSION = "v7.1-point-day-and-deictic-now"

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
  "proximity_anchor": "YYYY-MM-DD" | "latest" | "earliest" | null
}}

`targets` are concrete time sets you have resolved. A doc anchor scores
higher when it satisfies MORE targets — mean of per-target overlap.

`proximity_anchor` is a SEPARATE channel: a single time-point used to
break ties / re-rank by closeness. ORTHOGONAL to targets — set
membership (overlap) and closeness ranking are independent decisions.
See the PROXIMITY ANCHOR section below.

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
   "When did I most recently restring my guitar?"
   "my most recent dental cleaning"
   The user is asking for the LATEST occurrence of a specific
   activity, not "things in the last 60-90 days". Emit empty
   targets with proximity_anchor="latest" so any-date docs about
   that activity remain eligible and the closeness tournament
   picks the latest.

"What did I do recently" / "show me what happened lately" (without a
specific subject) → deictic; resolve to a recent window (e.g. last
60-90 days from REF_TIME) as a target with proximity_anchor="latest".


RECURRING / HABITUAL PATTERNS — past 3 + future 1
=================================================
Queries about a recurring or habitual day-of-week, weekend, month,
quarter, etc. (pure recurring OR singular ambiguous between "this
one" and "the pattern") emit ONE target with FOUR intervals:

- The 3 most-recent past occurrences (relative to REF_TIME)
- The 1 next upcoming occurrence

This expresses the standing pattern without overcommitting to a
year of specific dates that haven't happened. A doc anchored to any
representative occurrence satisfies the target if its anchor falls
in ANY of the four intervals.

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
- Plural day-name: a weekday name in plural form, or after "on", or
  in a possessive routine phrase ("my <weekday> meetings").
- Bare singular day-name without explicit deictic marker: a weekday
  used as a topic word with no "this"/"next"/"last" qualifier.
- Bare weekend / month / quarter without year: "weekend X?",
  "in <month-name>?", "Q<n> X?" with no year specified.
- Possessive routine: "my <time-band> routine", "my <activity>
  schedule".
- "Every X" pattern: "every <weekday>", "every <month>", "every
  <quarter>".

EXCEPTIONS (still emit bounded targets):
- Explicit deictic markers: "THIS Saturday", "NEXT Friday",
  "tomorrow", "yesterday" → bounded to the specific occurrence
  (still respecting the time-of-day qualifier if present).
- Specific calendar period with year: "March 2024", "Q4 2023" →
  bounded to that period.

PROXIMITY ANCHOR
================
`proximity_anchor` is a separate channel from `targets`. It picks a
single time-point to break ties / rerank by closeness. Set membership
(overlap with targets) and closeness ranking are independent decisions —
DO NOT use proximity to express set membership. Always emit `targets`
for the bounded region; use proximity only when the user is pointing at
a SPECIFIC TIME-POINT they want answers to cluster around.

Four allowed values:

  "latest"  — the user wants the MOST RECENT relevant doc.
              Use when picking the most-recent FROM MULTIPLE candidates
              the user knows exist, OR when asking about recency
              without a specific subject ("what happened recently?",
              "anything lately?").

  "earliest" — the user wants the OLDEST relevant doc, again from
              multiple candidates ("his earliest job", "the first
              of these meetings"). Rare.

  ISO date (e.g. "2024-06-15") — the user is pointing AROUND that
              specific date. The doc's anchor closest in either
              direction to that date wins. Use when the query says
              "around / on / closest to / near / at the time of" a
              specific date.

  null      — no closeness scoring. Use for set-membership questions
              ("in March 2024", "during 2023", "not in summer"),
              for queries with no temporal cue at all, and for
              comparative queries that name two alternatives (you want
              BOTH events surfaced; firing a proximity anchor would
              push one out of top-K).

DO NOT set proximity_anchor for:
  - "I just had X" / "we just shipped Y" — "just" here means
    RECENTLY-DEICTIC about a SINGLE recent event, not a request for
    the latest-of-many. Use null.
  - "first" / "last" describing a SPECIFIC occurrence the user has in
    mind ("my first novel", "the first Utah road trip", "his last
    surgery"). The user is naming ONE event. Use null.
  - COMPARATIVE queries that name TWO OR MORE alternatives ("which
    of A or B came first / later", "before or after Y", "X-er than"):
    use null so both events stay surfaced.
      "Which trip happened later, the Iceland trip or the Greece trip?" → null
      "Did the lease signing come before or after the move?" → null
      "Which sample arrived first, the green one or the blue one?" → null
      "Was the conference earlier or later than the launch?" → null
  - DEICTIC-CONTEXT queries where "today" / "this morning" / "right now" /
    "just now" frames the user's CURRENT SITUATION as background and the
    actual question asks about the PAST ("any prior cases?" / "have we
    seen this before?" / "past similar issues?"). The deictic word
    refers to NOW as context, NOT as a proximity target. The retrieval
    intent is historical, so use null.
      "Bakery stock count is off this morning — any past discrepancies?" → null
      "Hit a memory leak just now — what have we seen before?" → null
      "Audit notice arrived today — any prior audit gaps?" → null
      "Server is misbehaving right now — historical similar bugs?" → null
    Distinguishing marker: the query has TWO temporal frames — a
    deictic NOW (the trigger) and an implicit PAST (the search target).
    Fire proximity only if the user is actually asking for the latest
    relevant doc; here they want historical precedents.

Combining with targets:
  "Most recent meeting in March 2024"  → targets=[2024-03-01..2024-04-01], proximity_anchor="latest"
  "Anything around June 2022"          → targets=[],                       proximity_anchor="2022-06-15"
  "Meetings closest to my birthday on 2024-09-12"
                                       → targets=[],                       proximity_anchor="2024-09-12"
  "What didn't happen in March 2024?"  → targets=[complement],             proximity_anchor=null
  "Meetings in 2024"                   → targets=[2024],                   proximity_anchor=null

POINT-DAY TARGETS ALSO GET PROXIMITY
====================================
When `targets` contains a SINGLE INTERVAL of ONE DAY OR LESS (a specific
calendar day, or a within-day time-of-day window on a specific day),
ALSO emit `proximity_anchor` set to that day. Set membership alone is
not enough: docs whose ONLY temporal evidence is the metadata
timestamp ("this conversation took place on YYYY-MM-DD" but the
content doesn't restate the date) need closeness to surface.

  "Find the inventory check from 2019-11-13"
    → targets=[2019-11-13..2019-11-14], proximity_anchor="2019-11-13"
  "Pull up the standup notes from 2022-05-08"
    → targets=[2022-05-08..2022-05-09], proximity_anchor="2022-05-08"
  "What's on the schedule for 2025-04-08?"
    → targets=[2025-04-08..2025-04-09], proximity_anchor="2025-04-08"

Multi-day or wider targets (months, quarters, years) do NOT auto-emit
proximity — those are set-membership intent. The rule fires only for
true point-day single-interval targets.

DEICTIC RESOLUTION
==================
Resolve deictic phrases against REF_TIME and emit them as targets.

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
{{"targets":[{{"intervals":[{{"lo":"2024-03-01","hi":"2024-04-01"}}]}}],"proximity_anchor":null}}

Query: "after 2020"
{{"targets":[{{"intervals":[{{"lo":"2021-01-01","hi":null}}]}}],"proximity_anchor":null}}

Query: "before 1999"
{{"targets":[{{"intervals":[{{"lo":null,"hi":"1999-01-01"}}]}}],"proximity_anchor":null}}

Query: "not in 2023"
{{"targets":[{{"intervals":[{{"lo":null,"hi":"2023-01-01"}},{{"lo":"2024-01-01","hi":null}}]}}],"proximity_anchor":null}}

Query: "in 2024 not in summer"
{{"targets":[{{"intervals":[{{"lo":"2024-01-01","hi":"2024-06-01"}},{{"lo":"2024-09-01","hi":"2025-01-01"}}]}}],"proximity_anchor":null}}

Query: "in Q1 or Q4 of 2023"
{{"targets":[
  {{"intervals":[{{"lo":"2023-01-01","hi":"2023-04-01"}}]}},
  {{"intervals":[{{"lo":"2023-10-01","hi":"2024-01-01"}}]}}
],"proximity_anchor":null}}

Query: "in 2020 and 2024"
{{"targets":[
  {{"intervals":[{{"lo":"2020-01-01","hi":"2021-01-01"}}]}},
  {{"intervals":[{{"lo":"2024-01-01","hi":"2025-01-01"}}]}}
],"proximity_anchor":null}}

Query: "between 2020 and 2024"
{{"targets":[{{"intervals":[{{"lo":"2020-01-01","hi":"2025-01-01"}}]}}],"proximity_anchor":null}}

Query: "what wasn't done on May 3 2024" (verb-polarity rule)
{{"targets":[{{"intervals":[{{"lo":"2024-05-03","hi":"2024-05-04"}}]}}],"proximity_anchor":null}}

Query: "latest X in Q2 2024" where X is a recurring activity type
{{"targets":[{{"intervals":[{{"lo":"2024-04-01","hi":"2024-07-01"}}]}}],"proximity_anchor":"latest"}}

Query: "anything around June 15 2022"
{{"targets":[],"proximity_anchor":"2022-06-15"}}

Query: "meetings closest to my birthday on 2024-09-12"
{{"targets":[],"proximity_anchor":"2024-09-12"}}

Query: "no temporal scope at all"
{{"targets":[],"proximity_anchor":null}}

Recurring / habitual examples (REF_TIME = 2026-04-23 = Thursday):

Query: "on Mondays"
{{"targets":[{{"intervals":[
  {{"lo":"2026-04-20","hi":"2026-04-21"}},
  {{"lo":"2026-04-13","hi":"2026-04-14"}},
  {{"lo":"2026-04-06","hi":"2026-04-07"}},
  {{"lo":"2026-04-27","hi":"2026-04-28"}}
]}}],"proximity_anchor":null}}

Query: "Monday mornings"
{{"targets":[{{"intervals":[
  {{"lo":"2026-04-20T03:00:00Z","hi":"2026-04-20T13:00:00Z"}},
  {{"lo":"2026-04-13T03:00:00Z","hi":"2026-04-13T13:00:00Z"}},
  {{"lo":"2026-04-06T03:00:00Z","hi":"2026-04-06T13:00:00Z"}},
  {{"lo":"2026-04-27T03:00:00Z","hi":"2026-04-27T13:00:00Z"}}
]}}],"proximity_anchor":null}}

Query: "Friday at 3pm"
{{"targets":[{{"intervals":[
  {{"lo":"2026-04-17T15:00:00Z","hi":"2026-04-17T16:00:00Z"}},
  {{"lo":"2026-04-10T15:00:00Z","hi":"2026-04-10T16:00:00Z"}},
  {{"lo":"2026-04-03T15:00:00Z","hi":"2026-04-03T16:00:00Z"}},
  {{"lo":"2026-04-24T15:00:00Z","hi":"2026-04-24T16:00:00Z"}}
]}}],"proximity_anchor":null}}

Query: "in Q3"
{{"targets":[{{"intervals":[
  {{"lo":"2025-07-01","hi":"2025-10-01"}},
  {{"lo":"2024-07-01","hi":"2024-10-01"}},
  {{"lo":"2023-07-01","hi":"2023-10-01"}},
  {{"lo":"2026-07-01","hi":"2026-10-01"}}
]}}],"proximity_anchor":null}}

Query: "in March"
{{"targets":[{{"intervals":[
  {{"lo":"2026-03-01","hi":"2026-04-01"}},
  {{"lo":"2025-03-01","hi":"2025-04-01"}},
  {{"lo":"2024-03-01","hi":"2024-04-01"}},
  {{"lo":"2027-03-01","hi":"2027-04-01"}}
]}}],"proximity_anchor":null}}

Query: "this Friday" (explicit deictic — bounded to that day)
{{"targets":[{{"intervals":[{{"lo":"2026-04-24","hi":"2026-04-25"}}]}}],"proximity_anchor":null}}

Query: "last night" (deictic + night band)
{{"targets":[{{"intervals":[{{"lo":"2026-04-22T19:00:00Z","hi":"2026-04-23T07:00:00Z"}}]}}],"proximity_anchor":null}}

Query: "yesterday morning" (deictic + morning band)
{{"targets":[{{"intervals":[{{"lo":"2026-04-22T03:00:00Z","hi":"2026-04-22T13:00:00Z"}}]}}],"proximity_anchor":null}}

Query: "since X" where X is a named event with no date in the query
{{"targets":[],"proximity_anchor":null}}

Query: "most recently X" where X is a specific activity
{{"targets":[],"proximity_anchor":"latest"}}

NOW PRODUCE THE PLAN FOR:

Query: {query}
Reference time: {ref_time}
"""


_PLAN_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["targets", "proximity_anchor"],
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
        # `proximity_anchor` is a closeness target for ranking (orthogonal
        # to `targets` which is set-membership scope). Accepts:
        # - null               : no closeness scoring
        # - "latest"           : prefer later in time (POS_INF sentinel)
        # - "earliest"         : prefer earlier in time (NEG_INF sentinel)
        # - "YYYY-MM-DD"       : prefer docs whose anchor is closest to
        #                        that date (bidirectional |D − T|)
        "proximity_anchor": {
            "type": ["string", "null"],
        },
    },
}


# ---------------------------------------------------------------------------
# Data class for the resolved planner output
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """Resolved plan: a flat list of IntervalSet targets (set-membership scope)
    + an optional proximity_anchor_us (closeness center for ranking).

    The two fields are orthogonal:
    - `targets` says WHERE the answer's time must fall (overlap-scored).
    - `proximity_anchor_us` says WHICH TIME-POINT to score closeness against
      (None = no closeness scoring; POS_INF = "later is better";
      NEG_INF = "earlier is better"; finite int = "around this time").
    """

    targets: list[IntervalSet] = field(default_factory=list)
    proximity_anchor_us: Endpoint | None = None
    raw: str | None = field(default=None, repr=False)
    parse_error: str | None = field(default=None, repr=False)

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
            "proximity_anchor": _proximity_to_str(self.proximity_anchor_us),
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


def _proximity_from_str(s: str | None) -> Endpoint | None:
    """Parse the LLM's proximity_anchor JSON value to an Endpoint.

    Accepts:
    - null → None (no proximity scoring)
    - "latest" → POS_INF (later in time is closer)
    - "earliest" → NEG_INF (earlier in time is closer)
    - ISO date string → µs timestamp (closeness to that specific date)
    """
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    if s == "latest":
        return POS_INF
    if s == "earliest":
        return NEG_INF
    try:
        return _iso_to_us(s)
    except ValueError:
        return None


def _proximity_to_str(anchor: Endpoint | None) -> str | None:
    """Serialize a proximity anchor for JSON / dict round-trip."""
    if anchor is None:
        return None
    if is_inf(anchor):
        # Distinguish +∞ vs −∞ by comparison against any finite int
        return "latest" if anchor > 0 else "earliest"
    return _us_to_iso(anchor)


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
                    proximity_anchor_us=_proximity_from_str(
                        obj.get("proximity_anchor")
                    ),
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
                anchor = _proximity_from_str(obj.get("proximity_anchor"))
                self._cache[key] = obj
                self._save_cache()
                return Plan(
                    targets=targets, proximity_anchor_us=anchor, raw=raw,
                )
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
