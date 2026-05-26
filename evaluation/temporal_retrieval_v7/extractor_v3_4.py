"""V3.4 doc-side temporal extractor — extends V3.3 with explicit directional
surface-form handling (since/after/until/before/from/onwards).

This is purely a DOC-side enhancement. The query side is handled by
`planner_direct.DirectQueryPlanner` which emits TimeRange JSON directly
(including ±∞ via null endpoints).

The doc side bounds open-ended directional claims at the doc's `ref_time`
(SPEC §4.4 convention): a doc can only attest to what was true through
its own publication date, so "Since 2022, the platform scaled" extracts
as `[2022-01-01, doc.ref_time)`. This prevents over-extension into
years beyond the doc's evidence horizon.

Schema unchanged from V3.3 (ISO-string earliest/latest), so the existing
`Interval` adapter is compatible. The new rules just produce different
intervals for directional surface forms.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from temporal_retrieval_min.core import Interval
from temporal_retrieval_min.extractor_common import _LLMCache, full_ref_context
from temporal_retrieval_min.schema import parse_iso, to_us

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v3_4"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"extractor_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SINGLE_PASS_SYSTEM_V3_4 = """You are a temporal-reference extractor for the DOCUMENT side of a
temporal retrieval system.

Your job: identify EVERY span in a passage that names a specific point,
span, or recurring schedule of time, AND directly resolve each one
into a temporal envelope (a half-open interval on the calendar).

# Critical test before emitting anything

Does the passage USE this phrase to locate a specific occurrence on
the calendar — one that the reader could later recall, search for,
or reference by date? Or does the phrase describe a constraint, a
requirement, a format placeholder, or a rule that applies generally
across many possible occurrences?

- Specific occurrence ("yesterday we deployed", "shipped on March 15",
  "Q1 was rough", "during the pandemic") -> EMIT.
- Constraint / rule / placeholder ("Policy: backups within the last
  hour", "every release requires a 30-minute window", "Subject
  format: [Date]") -> SKIP, even if temporal-shaped.

# What counts as a temporal reference

A span is a temporal reference if, given the reference time and any
explicit anchoring, you could state WHEN it is on a calendar:

- Absolute dates: "March 5, 2026", "1986", "Q3 2025".
- Relative deictics: "yesterday", "2 weeks ago", "next Thursday".
- Approximations: "around 2010", "a few weeks ago", "recently".
- Eras with a calendar anchor: "the 90s", "back in college", "during
  the pandemic".
- Recurring schedules tied to a real standing pattern: "every Thursday".
  Emit the first/nearest known occurrence with a sensible window.
- Durations: emit ONLY if attached to a specific calendar anchor.

# DIRECTIONAL SURFACE FORMS (V3.4)

When the passage uses an open-ended directional cue, BOUND THE INTERVAL
AT THE DOCUMENT'S REFERENCE TIME. A document can only attest to what
was true through its own publication date; do NOT extend the interval
into the unbounded future or past.

- "since X" / "from X onwards" / "starting X" / "ever since X" →
  earliest = X.start, latest = REF_TIME (the reference time provided
  to you below).
  Example: "Since 2022 the platform has scaled to 25 microservices."
  ref_time = 2024-08-15. → earliest = 2022-01-01, latest = 2024-08-16.

- "after X" (used as a CONTINUOUS-STATE claim or open-ended span) →
  earliest = X.end (strictly after X), latest = REF_TIME.
  Example: "After the March 2024 launch, we shipped two minor releases."
  ref_time = 2024-08-20. → earliest = 2024-04-01, latest = 2024-08-21.
  Note: "after X" applied to a SINGLE event ("Two days after Election
  Day") is a deictic, not a directional cue — extract the resolved
  date normally.

- "until X" / "by X" / "through X" / "up to X" →
  If a clear lower bound is mentioned in context, emit
  earliest = lower_bound, latest = X.end.
  Otherwise SKIP — do not fabricate a deep-past lower bound.

- "before X" / "prior to X" →
  As a doc-side past-anchor, emit X.start as the date anchor (treat
  the doc as relevantly about the time right before X). If "before"
  is part of a deictic offset ("two weeks before X"), resolve the
  arithmetic and emit the resolved date.

- "between A and B" / "from A to B" → earliest = A.start,
  latest = B.end. Bounded both sides.

These directional rules apply ONLY when the surface form is used to
DESCRIBE A TIME SPAN, not when it's part of a topical phrase
("notes from the team retreat" — "from" is provenance, not span).

# What does NOT count (skip)

- Bare names of recurring annual events without a year-anchor:
  "summer", "Christmas". (Unless used as the recurring schedule:
  "every summer we visit the lake".)
- Vague descriptors: "recent", "modern", "old", "new", "ancient".
- Bare frequency words: "often", "always", "sometimes", "rarely".
- Bare approximators without concrete reference: "about", "around"
  used alone.

# Policy / rule / template contexts — skip everything inside

When the surrounding sentence describes a generic policy, rule,
convention, requirement, or format, even temporal-shaped phrases
inside it are CONSTRAINTS or PLACEHOLDERS, not events.

# How to think about earliest / latest

- A pinpoint reference (e.g. "March 15, 2024") → single-day envelope.
- A span ("Q1 2024") → quarter envelope.
- A fuzzy reference ("around 2008") → widen by one unit.
- A relative reference resolves against ref time.
- A duration only counts if attached to an anchor.

# Rules

- Use UTC ISO 8601 with "Z" suffix.
- earliest is inclusive, latest is exclusive (half-open).
- For "about" / "around" / "roughly" / "a few" — widen by one
  granularity level.

# Skip — do not emit

If you cannot place a surface on the calendar without falling back
to ref time as a fabricated anchor, DO NOT emit it. Omit entirely.

# Output

A single JSON object: {"refs": [...]}. Each ref is:
{
  "earliest": ISO-8601 UTC datetime with "Z",
  "latest":   ISO-8601 UTC datetime with "Z"
}

If the passage has no temporal references that meet the bar, output
{"refs": []}.
"""


V3_4_JSON_SCHEMA: dict[str, Any] = {
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


class TemporalExtractorV3_4:
    """Doc-side V3.4 extractor — extends V3.3 prompt with explicit
    directional surface-form bounding at `ref_time`.

    Drop-in replacement for `TemporalExtractorV3_3`. Same return type
    (`list[Interval]`); the behavioral difference is in HOW it extracts
    "since/after/until/before" cues from doc text.
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
        self.cache = _LLMCache(cd / "single_v3_4.json")

    async def _call(self, text: str, ref_time: datetime) -> list[dict]:
        ctx = full_ref_context(ref_time)
        user = f"{ctx}\n\nPassage:\n{text}"
        key = f"{PROMPT_VERSION}|single|{ctx}|||{text}"
        cached = self.cache.get(self.model, key)
        if cached is None:
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_4},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_4_JSON_SCHEMA}},
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
