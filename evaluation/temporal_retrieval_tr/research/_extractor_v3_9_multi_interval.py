"""Extractor v3.9: multi-interval anchors per reference.

Symmetry with the query-side IntervalSet structure. Each emitted
temporal reference is now an IntervalSet with one OR MORE intervals,
matching the planner's target shape.

When the LLM naturally emits multi-interval per ref:
- A bounded period with a gap ("in 2024 except summer", "every
  weekday except holidays").
- A complement claim ("not in March 2024").
- A disjunction that's intrinsically one claim ("either Q3 or Q4
  of 2024", not two separate events).

When the LLM emits singleton ref (most cases):
- A single dated event.
- A specific calendar period without internal gaps.
- A pattern where each occurrence is a DISTINCT event the doc
  describes (one ref per occurrence; multiple refs per doc).

Distinct from v3.7/v3.8: schema upgrade lets the model produce
multi-interval refs when the doc semantics warrant.
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
from temporal_retrieval_tr.time_range import Interval as TRInterval, IntervalSet
from temporal_retrieval_min.schema import parse_iso, to_us

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v3_9_multi_interval"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SINGLE_PASS_SYSTEM_V3_9 = """You are a temporal-reference extractor.

Your job: identify every temporal claim in a passage and resolve each
into a temporal-reference object. Each reference contains one OR more
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

MULTIPLE references (each a singleton interval) when the doc
describes DISTINCT occurrences — separate events the doc reports
as separate things, even if they share a recurring schedule.

Heuristic: ask whether the doc would say "all of these together
fit one claim" (one multi-interval ref) or "each of these is a
separate occurrence" (multiple singleton refs).

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

A JSON object {"refs": [...]} per the response schema. Each ref
has `intervals` — a list of one or more half-open intervals. Use
multiple intervals within one ref ONLY when the doc treats them as
one logical claim with internal gaps / complement / disjunction.
Emit [] when no span meets the bar above.
"""


# Schema: each ref is now an IntervalSet (multiple intervals)
V3_9_JSON_SCHEMA: dict[str, Any] = {
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


class TemporalExtractorV3_9:
    """v3.9: multi-interval per reference — symmetric with query side.

    Returns list[IntervalSet] directly. The retriever does not need
    extractor_to_anchors wrapping — each ref IS already an IntervalSet.
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
        self.cache = _LLMCache(cd / "single_v3_9.json")
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
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_9},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_9_JSON_SCHEMA}},
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
        return Interval(earliest_us=to_us(earliest), latest_us=to_us(latest))

    async def extract(self, text: str, ref_time: datetime) -> list[Interval]:
        """Legacy adapter: flatten multi-interval refs to a flat Interval list.

        Preserves compatibility with the current retriever (which uses
        extractor_to_anchors to wrap each as a singleton IntervalSet).
        When the retriever is updated to accept list[IntervalSet]
        directly, use `extract_anchors` instead.
        """
        envs = await self._call(text, ref_time)
        out: list[Interval] = []
        for ref in envs:
            for env in ref.get("intervals", []):
                iv = self._to_interval(env)
                if iv is not None:
                    out.append(iv)
        return out

    async def extract_anchors(
        self, text: str, ref_time: datetime
    ) -> list[IntervalSet]:
        """Native shape: each ref → one IntervalSet (potentially multi-interval)."""
        envs = await self._call(text, ref_time)
        out: list[IntervalSet] = []
        for ref in envs:
            ivs: list[TRInterval] = []
            for env in ref.get("intervals", []):
                iv = self._to_interval(env)
                if iv is not None:
                    ivs.append(TRInterval(iv.earliest_us, iv.latest_us))
            if ivs:
                out.append(IntervalSet.from_intervals(ivs))
        return out

    def save_caches(self) -> None:
        self.cache.save()
