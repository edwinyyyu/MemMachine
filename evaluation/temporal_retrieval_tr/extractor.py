"""The temporal retrieval extractor — emits a list of IntervalSet anchors.

A single LLM call resolves dates AND structures each temporal claim
as an IntervalSet — symmetric with the planner's target shape.

Each reference is an `IntervalSet` — one OR more half-open intervals
on the calendar. Multiple intervals belong to the same reference when
the doc treats them as ONE logical claim with internal structure
(gaps, complements, disjunctions).

When the extractor emits multi-interval refs:
- A bounded period with a gap ("in 2024 except summer", "every
  weekday except holidays").
- A complement claim ("not in March 2024").
- An intrinsic disjunction that's one logical claim.

When the extractor emits singleton refs (most cases):
- A single dated event.
- A specific calendar period without internal gaps.
- A pattern where each occurrence is a DISTINCT event the doc
  describes — one ref per occurrence; multiple refs per doc.

The retriever consumes `list[IntervalSet]` directly. No wrapping.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

from temporal_retrieval_min.extractor_common import _LLMCache, full_ref_context
from temporal_retrieval_min.schema import parse_iso, to_us
from .time_range import Interval, IntervalSet

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v2-worked-examples"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"extractor_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """You are a temporal-reference extractor.

Your job: identify every temporal claim in a passage and resolve each
into a temporal reference. Each reference contains one OR more
half-open intervals on the calendar; multiple intervals belong to the
same reference when the doc treats them as ONE logical claim with
internal structure (gaps, disjunctions, complements).

# Emit / skip test

A reference emits if and only if it locates one or more specific
occurrences on the calendar a reader could later recall, search for,
or reference by date — given the reference time and any in-passage
anchor.

A span SKIPS if its surrounding sentence frames it as a rule, policy,
convention, requirement, format placeholder, or generic constraint
that applies across many possible occurrences. Cues for SKIP:
- Explicit headers: "policy:", "convention:", "rule:", "guideline:".
- Prescriptive modals as main predicate: "must X", "should X",
  "requires X", "never X without Y", "always X before Y".
- Recurrence over an event-CLASS without naming a specific instance.
- Template placeholders for dates.

If the surrounding sentence has SKIP cues, do not emit any temporal-
shaped phrases inside it.

Also skip: vague descriptors with no concrete reference; bare names
of recurring annual events without a year-anchor (unless the phrase
IS the standing arrangement itself).

# One reference (multi-interval) vs multiple references

ONE reference with multiple intervals when the doc describes a
SINGLE logical claim that internally spans multiple calendar
intervals — gaps, complements, or disjunctions inside one claim.

MULTIPLE references (each singleton) when the doc describes
DISTINCT occurrences — separate events the doc reports as separate
things, even if they share a recurring schedule.

Heuristic: ask whether the doc would say "all of these together fit
one claim" (one multi-interval ref) or "each of these is a separate
occurrence" (multiple singleton refs).

# Envelope semantics

- Pinpoint date → single-day interval [day 00:00:00Z, next day 00:00:00Z).
- Calendar span → interval covering the span endpoints, half-open.
- Fuzzy phrase ("around", "about", "roughly", "a few") → widen by
  one granularity.
- Relative phrase resolves against ref time.
- Duration → emit only when attached to a specific calendar anchor;
  emit [anchor, anchor+duration). Do NOT emit IMPACT-MAGNITUDE
  durations.
- earliest is inclusive, latest is exclusive. Use UTC ISO 8601 with
  "Z" suffix.

# Recurring schedules — three emit modes by context

(1) PURE RECURRING — no specific date or era anchor for the
occurrence. Emit ONE reference per distinct occurrence the doc's
pattern projects onto: the 3 most-recent past + 1 next upcoming
occurrences, relative to ref time. Each occurrence is its own ref
(singleton interval). If ref time itself falls on an occurrence,
count it as the most-recent past.

(2) ERA-ANCHORED RECURRING — the pattern is bracketed by a past
era. Emit 4 refs (each a singleton) spread within the era.

(3) SPECIFIC OCCURRENCE WITH RECURRING DESCRIPTOR — the doc gives
a specific date for one occurrence even though the activity is
described as recurring. The date pins the occurrence; emit ONE
singleton ref at that date.

For monthly/quarterly/yearly recurring units, use full unit spans
(past 3 + 1) with no within-day window.

# Per-occurrence within-day window (standardized bands)

| Qualifier                     | Window (UTC)                     |
| ----------------------------- | -------------------------------- |
| "at HH:MM" / "at HHam/pm"     | [HH:00, HH+1:00)  — 1 hour       |
| "morning"                     | [03:00, 13:00)                   |
| "noon"                        | [10:00, 14:00)                   |
| "afternoon"                   | [11:00, 19:00)                   |
| "evening"                     | [16:00, 00:00 next day)          |
| "night"                       | [19:00, 07:00 next day)          |
| no qualifier                  | full day [00:00, 00:00 next day) |

If a clock time and a band qualifier both appear, use the clock-
time window. If multiple band qualifiers appear, use their union.

# Output

A JSON object {"refs": [...]} per the response schema. Each ref has
`intervals` — a list of one or more half-open intervals. Use multiple
intervals within one ref ONLY when the doc treats them as one
logical claim with internal gaps / complement / disjunction. Emit []
when no span meets the bar above.

# WORKED EXAMPLES

All examples assume REF_TIME = 2026-04-23T12:00:00Z (Thursday).

## Pinpoint date

Passage: "shipped on March 15, 2024"
{{"refs":[{{"intervals":[{{"earliest":"2024-03-15T00:00:00Z","latest":"2024-03-16T00:00:00Z"}}]}}]}}

## Calendar span — quarter

Passage: "Q1 2024 was rough"
{{"refs":[{{"intervals":[{{"earliest":"2024-01-01T00:00:00Z","latest":"2024-04-01T00:00:00Z"}}]}}]}}

## Calendar span — decade

Passage: "the 90s were different"
{{"refs":[{{"intervals":[{{"earliest":"1990-01-01T00:00:00Z","latest":"2000-01-01T00:00:00Z"}}]}}]}}

## Relative deictic

Passage: "I went yesterday"
{{"refs":[{{"intervals":[{{"earliest":"2026-04-22T00:00:00Z","latest":"2026-04-23T00:00:00Z"}}]}}]}}

## Fuzzy widening

Passage: "around 2008"
{{"refs":[{{"intervals":[{{"earliest":"2006-01-01T00:00:00Z","latest":"2011-01-01T00:00:00Z"}}]}}]}}

## Pure recurring — weekday, no TOD (4 occurrences, REF_TIME falls on the day)

Passage: "every Thursday"
{{"refs":[
  {{"intervals":[{{"earliest":"2026-04-23T00:00:00Z","latest":"2026-04-24T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-16T00:00:00Z","latest":"2026-04-17T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-09T00:00:00Z","latest":"2026-04-10T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-30T00:00:00Z","latest":"2026-05-01T00:00:00Z"}}]}}
]}}

## Pure recurring — weekday with morning band

Passage: "every Saturday morning"
{{"refs":[
  {{"intervals":[{{"earliest":"2026-04-18T03:00:00Z","latest":"2026-04-18T13:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-11T03:00:00Z","latest":"2026-04-11T13:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-04T03:00:00Z","latest":"2026-04-04T13:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-25T03:00:00Z","latest":"2026-04-25T13:00:00Z"}}]}}
]}}

## Pure recurring — weekday with clock time

Passage: "every Friday at 3pm"
{{"refs":[
  {{"intervals":[{{"earliest":"2026-04-17T15:00:00Z","latest":"2026-04-17T16:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-10T15:00:00Z","latest":"2026-04-10T16:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-03T15:00:00Z","latest":"2026-04-03T16:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2026-04-24T15:00:00Z","latest":"2026-04-24T16:00:00Z"}}]}}
]}}

## Pure recurring — monthly

Passage: "every March"
{{"refs":[
  {{"intervals":[{{"earliest":"2026-03-01T00:00:00Z","latest":"2026-04-01T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2025-03-01T00:00:00Z","latest":"2025-04-01T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2024-03-01T00:00:00Z","latest":"2024-04-01T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2027-03-01T00:00:00Z","latest":"2027-04-01T00:00:00Z"}}]}}
]}}

## Era-anchored recurring (Obama years ≈ 2009-2017)

Passage: "during the Obama years we went every Saturday"
{{"refs":[
  {{"intervals":[{{"earliest":"2010-04-03T00:00:00Z","latest":"2010-04-04T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2012-08-04T00:00:00Z","latest":"2012-08-05T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2014-11-15T00:00:00Z","latest":"2014-11-16T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2016-06-04T00:00:00Z","latest":"2016-06-05T00:00:00Z"}}]}}
]}}

## Specific occurrence with recurring descriptor (date pins it)

Passage: "the weekly check-in on March 11, 2025"
{{"refs":[{{"intervals":[{{"earliest":"2025-03-11T00:00:00Z","latest":"2025-03-12T00:00:00Z"}}]}}]}}

## Multi-interval ref — bounded period with internal gap

Passage: "all of 2024 except summer"
{{"refs":[{{"intervals":[
  {{"earliest":"2024-01-01T00:00:00Z","latest":"2024-06-01T00:00:00Z"}},
  {{"earliest":"2024-09-01T00:00:00Z","latest":"2025-01-01T00:00:00Z"}}
]}}]}}

## Multi-interval ref — complement

Passage: "not in March 2024"
{{"refs":[{{"intervals":[
  {{"earliest":"2024-04-01T00:00:00Z","latest":"2999-12-31T23:59:59Z"}}
]}}]}}

(Note: bounded complement; use a sentinel large date for the unbounded
side since the doc-side schema requires concrete endpoints. The
matching planner emits the symmetric complement as its target.)

## Duration anchored to a specific date

Passage: "for 3 weeks starting June 1, 2024"
{{"refs":[{{"intervals":[{{"earliest":"2024-06-01T00:00:00Z","latest":"2024-06-22T00:00:00Z"}}]}}]}}

## Skip — policy / rule context

Passage: "Policy: backups every Friday at 5pm."
{{"refs":[]}}

## Skip — impact-magnitude duration

Passage: "the outage lasted 45 minutes"
{{"refs":[]}}

## Multiple distinct references in one passage

Passage: "I shipped v3 on March 15, 2024 and again on November 1, 2024"
{{"refs":[
  {{"intervals":[{{"earliest":"2024-03-15T00:00:00Z","latest":"2024-03-16T00:00:00Z"}}]}},
  {{"intervals":[{{"earliest":"2024-11-01T00:00:00Z","latest":"2024-11-02T00:00:00Z"}}]}}
]}}
"""


JSON_SCHEMA: dict[str, Any] = {
    "name": "temporal_references",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "intervals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "earliest": {"type": "string"},
                                    "latest": {"type": "string"},
                                },
                                "required": ["earliest", "latest"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["intervals"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["refs"],
        "additionalProperties": False,
    },
}


class TemporalExtractor:
    """Canonical extractor: emits `list[IntervalSet]`, one IntervalSet per
    temporal reference. Each IntervalSet has one or more intervals.
    """

    def __init__(
        self,
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else CACHE_ROOT
        self.cache = _LLMCache(cd / f"extractor_{PROMPT_VERSION}.json")

    async def _call(self, text: str, ref_time: datetime) -> list[dict]:
        ctx = full_ref_context(ref_time)
        user = f"{ctx}\n\nPassage:\n{text}"
        key = f"{PROMPT_VERSION}|{ctx}|||{text}"
        cached = self.cache.get(self.model, key)
        if cached is None:
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **JSON_SCHEMA}},
            )
            cached = resp.output_text
            self.cache.put(self.model, key, cached)
        try:
            data = json.loads(cached)
            refs = data.get("refs", [])
            if not isinstance(refs, list):
                return []
            return refs
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _to_interval(env: dict) -> Interval | None:
        try:
            earliest = parse_iso(env["earliest"])
            latest = parse_iso(env["latest"])
        except (KeyError, ValueError, TypeError):
            return None
        if latest <= earliest:
            return None
        return Interval(to_us(earliest), to_us(latest))

    async def extract_anchors(
        self, text: str, ref_time: datetime
    ) -> list[IntervalSet]:
        """Extract temporal references as IntervalSets — one IS per ref.

        Each IntervalSet may contain one or more intervals; multiple
        intervals represent one logical claim with internal structure.
        """
        envs = await self._call(text, ref_time)
        out: list[IntervalSet] = []
        for ref in envs:
            ivs: list[Interval] = []
            for env in ref.get("intervals", []):
                iv = self._to_interval(env)
                if iv is not None:
                    ivs.append(iv)
            if ivs:
                out.append(IntervalSet.from_intervals(ivs))
        return out

    def save_caches(self) -> None:
        self.cache.save()
