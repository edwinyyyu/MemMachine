"""Extractor v3.8: principles-only prompt, no worked examples.

Rewrites v3.3's prompt from scratch with only abstract principles,
explicit rule mappings, and the standardized bands table. No
"yesterday we deployed" / "every Thursday at 3pm" parenthetical
illustrations. No worked input→output examples. The LLM has the JSON
schema (via response_format) for output shape and these principles
for the semantic rules.

Compared to v3.7's surgical patch to v3.3, v3.8 is a clean rewrite of
the entire system prompt. The recurring-pattern logic (three emit
modes + bands) is kept; everything else is principle-form.
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
PROMPT_VERSION = "v3_8_principles"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SINGLE_PASS_SYSTEM_V3_8 = """You are a temporal-reference extractor.

Your job: identify every span in a passage that names a specific
point, span, or recurring schedule of time, AND resolve each into
one or more temporal envelopes (half-open intervals on the calendar).

# Emit / skip test

A span emits if and only if it locates a specific occurrence on the
calendar — one a reader could later recall, search for, or reference
by date — given the reference time and any in-passage anchor.

A span SKIPS if the surrounding sentence frames it as a rule, policy,
convention, requirement, format placeholder, or generic constraint
that applies across many possible occurrences. Cues for SKIP:
- Explicit headers: "policy:", "convention:", "rule:", "guideline:".
- Prescriptive modals as main predicate: "must X", "should X",
  "requires X", "never X without Y", "always X before Y".
- Recurrence over an event-CLASS without naming a specific instance.
- Template placeholders for dates (e.g. "[Date]", "{date}", "<date>").

If the surrounding sentence has SKIP cues, do not emit any temporal-
shaped phrases inside it, even if they look anchorable.

Also skip:
- Vague descriptors with no concrete reference (vague proximity
  adjectives, bare frequency words).
- Bare names of recurring annual events without a year-anchor,
  UNLESS the phrase IS the recurring schedule itself in a standing-
  arrangement context.

# Envelope semantics

- Pinpoint date → single-day envelope: earliest = day 00:00:00Z,
  latest = next day 00:00:00Z.
- Calendar span (quarter, year, decade, era) → envelope covering
  the span's endpoints, half-open.
- Fuzzy phrase ("around", "about", "roughly", "a few", "a couple")
  → widen by one granularity level.
- Relative phrase resolves against ref time.
- Duration → emit only when attached to a specific calendar anchor;
  emit [anchor, anchor+duration). Do NOT emit IMPACT-MAGNITUDE
  durations (how long an effect lasted, distinct from when on the
  calendar).
- earliest is inclusive, latest is exclusive (half-open).
- Use UTC ISO 8601 with "Z" suffix.

# Recurring schedules — three emit modes

A recurring phrase tied to a real standing pattern emits depending
on what the doc anchors it to:

(1) PURE RECURRING — no specific date or era anchor for the
occurrence. Emit 4 envelopes: the 3 most-recent past occurrences +
1 next upcoming, relative to ref time. If ref time itself falls on
an occurrence, count it as the most-recent past.

(2) ERA-ANCHORED RECURRING — the pattern is bracketed by a past
era. Emit 4 envelopes spread within the era, not from ref time.

(3) SPECIFIC OCCURRENCE WITH RECURRING DESCRIPTOR — the doc gives a
specific date for one occurrence even though the activity is
described as recurring. The date pins the occurrence; emit ONE
envelope at that date.

For monthly/quarterly/yearly recurring units, use full unit spans
(past 3 + 1) with no within-day window.

# Per-occurrence within-day window (standardized bands)

Each within-day envelope uses one of these bands based on the time-
of-day qualifier in the phrase. Adjacent bands overlap a few hours
so a phrase real users would call by either name still matches.

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

A JSON object {"refs": [...]} per the response schema. Each ref is
{"earliest": ISO datetime with "Z", "latest": ISO datetime with "Z"}.
Emit [] when no span meets the bar above.
"""


V3_8_JSON_SCHEMA: dict[str, Any] = {
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


class TemporalExtractorV3_8:
    """v3.8: principles-only prompt rewrite."""

    def __init__(
        self,
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else CACHE_ROOT
        self.cache = _LLMCache(cd / "single_v3_8.json")
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
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_8},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_8_JSON_SCHEMA}},
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
