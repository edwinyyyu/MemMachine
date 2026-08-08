"""Extractor v3.5: recurring patterns → past 3 + future 1 envelopes.

Design 2 co-design: the planner emits recurring queries as past 3 +
future 1 intervals; this extractor mirrors that on the doc side so a
recurring doc's anchors overlap with the planner's enumeration.

Single change vs v3.3: the "recurring phrase" rule emits FOUR envelopes
(past 3 + future 1) instead of ONE (nearest known occurrence).

Principle: model recurrence without overcommitting falsehoods. A few
recent past instances + the next upcoming instance describes a real
standing pattern without claiming a year of fictional dates.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

from temporal_retrieval_min.extractor_common import _LLMCache, full_ref_context
from temporal_retrieval_min.core import Interval
from temporal_retrieval_min.schema import parse_iso, to_us

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v3_5_tod"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SINGLE_PASS_SYSTEM_V3_5 = """You are a temporal-reference extractor.

Your job: identify EVERY span in a passage that names a specific point,
span, or recurring schedule of time, AND directly resolve each one
into one or more temporal envelopes (half-open intervals on the calendar).

# Critical test before emitting anything

Does the passage USE this phrase to locate a specific occurrence on
the calendar — one that the reader could later recall, search for,
or reference by date? Or does the phrase describe a constraint, a
requirement, a format placeholder, or a rule that applies generally
across many possible occurrences?

- Specific occurrence ("yesterday we deployed", "shipped on March 15",
  "Q1 was rough", "during the pandemic", "every Thursday at 3pm I
  have therapy") -> EMIT.
- Constraint / rule / placeholder ("Policy: backups within the last
  hour", "every release requires a 30-minute window", "Subject
  format: [Date]") -> SKIP, even if temporal-shaped.

This is the deciding test for borderline cases. The retriever's job
is to surface timeless rule docs on non-temporal queries; anchoring
a policy at the reference time defeats that.

# What counts as a temporal reference

A span is a temporal reference if and only if, given the reference
time and any explicit anchoring in the passage, you could state WHEN
it is on a calendar — AND the critical test above puts it on the
"specific occurrence" side:

- Absolute dates: "March 5, 2026", "1986", "Q3 2025".
- Relative deictics: "yesterday", "2 weeks ago", "next Thursday".
- Approximations: "around 2010", "a few weeks ago", "recently".
- Eras with a calendar anchor: "the 90s", "back in college", "during
  the pandemic".
- Recurring schedules tied to a real standing pattern: "every
  Thursday at 3pm", "Sundays we do brunch", "Friday afternoons".
  See the recurrence section below for the envelope shape.
- Durations: emit ONLY if attached to a specific calendar anchor.
  In particular, IMPACT-MAGNITUDE durations describe how long an
  effect lasted, not when it was on the calendar — skip them.
  Examples: "over-reported for 6 weeks", "froze for 12 minutes",
  "delayed 3 hours", "outage lasted 45 minutes".

# What does NOT count (skip)

- Bare names of recurring annual events without a year-anchor:
  "summer", "Christmas", "Easter", "graduation day". (EXCEPTION:
  when the phrase IS the recurring schedule itself in a
  standing-arrangement context — "every summer we visit the lake".)
- Vague descriptors: "recent", "modern", "old", "new", "ancient".
- Bare frequency words: "often", "always", "sometimes", "rarely".
- Bare approximators without concrete reference: "about", "around",
  "roughly" used alone.

# Policy / rule / template contexts — skip everything inside

When the surrounding sentence describes a generic policy, rule,
convention, requirement, or format, even temporal-shaped phrases
inside it are CONSTRAINTS or PLACEHOLDERS, not events. Cue patterns:

- Explicit policy header: "policy:", "convention:", "rule:".
- Prescriptive modals as main predicate: "must X", "should X",
  "requires X", "never X without Y", "always X before Y".
- Recurrence over an event-CLASS without naming a specific instance:
  "every release", "every deploy", "every PR", "every sprint".
- Template placeholders: "[Date]", "{date}", "<date>".

# How to think about earliest / latest

- A pinpoint reference (e.g. "March 15, 2024") -> single-day
  envelope: earliest = 2024-03-15T00:00:00Z, latest = 2024-03-16T00:00:00Z.
- A span ("Q1 2024") -> earliest = 2024-01-01T00:00:00Z, latest =
  2024-04-01T00:00:00Z.
- A fuzzy reference ("around 2008") -> widen by one unit: earliest
  = 2006-01-01T00:00:00Z, latest = 2011-01-01T00:00:00Z.
- A relative reference resolves against ref time. "yesterday" ->
  day before ref. "last month" -> calendar month before ref. "the
  90s" -> [1990-01-01, 2000-01-01).
- A duration only counts if attached to an anchor ("for 3 weeks
  starting June 1") -> [anchor, anchor+duration].

# Recurring schedules — past 3 + future 1, preserve time-of-day

A recurring phrase tied to a real standing pattern ("every Thursday
at 3pm", "Sundays we do brunch", "Friday afternoons", "every March")
expresses a STANDING ARRANGEMENT, not a single occurrence. The
passage is a fact about the pattern, retrievable by any of its
recent or upcoming occurrences.

Emit FOUR envelopes per recurring phrase:
- The 3 most-recent past occurrences (relative to ref time).
- The 1 next upcoming occurrence.

If ref time itself falls ON an occurrence (e.g., ref time is a
Thursday and the phrase is "every Thursday"), count it as the most
recent past occurrence.

# Per-occurrence within-day window

Each occurrence's earliest/latest is determined by the time-of-day
qualifier in the phrase, using STANDARDIZED overlapping bands.
Adjacent bands overlap a few hours so phrases that real users would
call by either name still match.

| Qualifier in phrase                | Window (UTC, on that day)        |
| ---------------------------------- | -------------------------------- |
| "at HH:MM" / "at HHam/pm"          | [HH:00, HH+1:00)  — 1 hour       |
| "morning"                          | [03:00, 13:00)                   |
| "noon"                             | [10:00, 14:00)                   |
| "afternoon"                        | [11:00, 19:00)                   |
| "evening"                          | [16:00, 00:00 next day)                   |
| "night"                            | [19:00, 07:00 next day)          |
| no time qualifier                  | full day [00:00, 00:00 next day) |

If BOTH a clock time and a band qualifier appear ("Thursday morning
at 6am"), use the clock-time window. If multiple band qualifiers
appear ("afternoon or evening"), use the UNION (intersect-with-day
of the two windows).

If the passage anchors the schedule earlier ("every Thursday since
March 2024"), the start of the schedule REPLACES the third-most-recent
past occurrence with the start date. Don't go further back than the
schedule's stated start.

If the recurring unit is months/quarters/years (e.g., "every March",
"every Q4", "every December"), apply the same past-3 + future-1
shape using the appropriate calendar unit (no within-day window
since the unit IS the calendar block). For "every March", emit the
last 3 Marches (as full-month spans) + the next March.

EXAMPLES (assume ref time is Thursday 2026-04-23):

"every Thursday at 3pm I have therapy"
-> 4 envelopes, each 1 hour wide at 15:00:
   [2026-04-23T15:00, 2026-04-23T16:00)
   [2026-04-16T15:00, 2026-04-16T16:00)
   [2026-04-09T15:00, 2026-04-09T16:00)
   [2026-04-30T15:00, 2026-04-30T16:00)

"every Saturday morning we hike"
-> 4 morning envelopes (Sat 03:00-13:00):
   [2026-04-18T03:00, 2026-04-18T13:00)
   [2026-04-11T03:00, 2026-04-11T13:00)
   [2026-04-04T03:00, 2026-04-04T13:00)
   [2026-04-25T03:00, 2026-04-25T13:00)

"Sundays we do brunch" (no time qualifier)
-> 4 full-day envelopes:
   [2026-04-19T00:00, 2026-04-20T00:00)
   [2026-04-12T00:00, 2026-04-13T00:00)
   [2026-04-05T00:00, 2026-04-06T00:00)
   [2026-04-26T00:00, 2026-04-27T00:00)

"December is always wall-to-wall holiday parties" (annual, no time)
-> 4 full-month envelopes:
   [2025-12-01, 2026-01-01)
   [2024-12-01, 2025-01-01)
   [2023-12-01, 2024-01-01)
   [2026-12-01, 2027-01-01)

# Rules

- Use UTC ISO 8601 with "Z" suffix.
- earliest is inclusive, latest is exclusive (half-open).
- For "about" / "around" / "roughly", widen non-recurring references
  by one granularity level. Recurring envelopes stay at the unit
  granularity (day/month) per the schedule above.

# Skip — do not emit

If you cannot place a surface on the calendar without falling back
to ref time as a fabricated anchor, DO NOT emit it.

This applies to:
- Policy / rule constraints with no specific occurrence.
- Generic recurrences over an event class with no named instance.
- Template placeholders.
- Phrases that look temporal but lack a calendar anchor (bare
  "the launch" without other context).

# Output

A single JSON object: {"refs": [...]}. Each ref is:
{
  "earliest": ISO-8601 UTC datetime with "Z",
  "latest":   ISO-8601 UTC datetime with "Z"
}

If the passage has no temporal references that meet the bar, output
{"refs": []}.
"""


V3_5_JSON_SCHEMA: dict[str, Any] = {
    "name": "time_envelopes",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "refs": {
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
            }
        },
        "required": ["refs"],
        "additionalProperties": False,
    },
}


class TemporalExtractorV3_5:
    """V3.5: recurring patterns emit past 3 + future 1 envelopes."""

    def __init__(
        self,
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else CACHE_ROOT
        self.cache = _LLMCache(cd / "single_v3_5.json")
        self.shared_pass2_cache = self.cache

    async def _call(self, text: str, ref_time: datetime) -> list[dict]:
        ctx = full_ref_context(ref_time)
        user = f"{ctx}\n\nPassage:\n{text}"
        key = f"{PROMPT_VERSION}|single|{ctx}|||{text}"
        cached = self.cache.get(self.model, key)
        if cached is None:
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_5},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_5_JSON_SCHEMA}},
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
    def _to_interval(env: dict, ref_time: datetime) -> Interval | None:
        try:
            earliest = parse_iso(env["earliest"])
            latest = parse_iso(env["latest"])
        except (KeyError, ValueError, TypeError):
            return None
        if latest <= earliest:
            return None
        return Interval(earliest_us=to_us(earliest), latest_us=to_us(latest))

    async def extract(self, text: str, ref_time: datetime) -> list[Interval]:
        envs = await self._call(text, ref_time)
        out: list[Interval] = []
        for env in envs:
            iv = self._to_interval(env, ref_time)
            if iv is not None:
                out.append(iv)
        return out

    def save_caches(self) -> None:
        self.cache.save()
