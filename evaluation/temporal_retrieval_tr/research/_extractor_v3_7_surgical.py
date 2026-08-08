"""Extractor v3.7 (surgical): refines v3.6 to fix two over-expansion bugs.

v3.6 expanded any "weekly" / "every" / "Sundays" phrase to 4 envelopes
regardless of context. This caused two regressions:

  Bug 1 (negation_temporal, disc): docs with a SPECIFIC DATE that
  mention a recurring descriptor ("I did the weekly grocery run on
  March 11, 2025") were expanded to 4 envelopes — including ones
  outside the date — falsely matching complement queries.

  Bug 2 (era): recurring patterns anchored to a past ERA ("every
  Saturday during the Obama years") were expanded around REF_TIME
  instead of within the era — missing era-specific queries.

v3.7 adds two emit-mode distinctions to v3.6's recurring bullets:

  - SPECIFIC OCCURRENCE: doc gives a specific date for the occurrence
    even if the activity is described as recurring → emit ONE envelope
    at that date.

  - ERA-ANCHORED RECURRING: recurring pattern with an era anchor
    ("during X", "back in Y") → emit envelopes WITHIN that era.

  - PURE RECURRING: no specific date or era anchor → past 3 + future 1
    from ref_time (same as v3.6).
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
from temporal_retrieval_min.extractor_v3_3 import SINGLE_PASS_SYSTEM_V3_3
from temporal_retrieval_min.schema import parse_iso, to_us

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v3_7_surgical"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


_BANDS = (
    'Time-of-day bands (UTC): "morning"=[03:00,13:00) | "noon"=[10:00,14:00) | '
    '"afternoon"=[11:00,19:00) | "evening"=[16:00,00:00 next day) | '
    '"night"=[19:00,07:00 next day) | "at HH:MM" or "at HHam/pm"=[HH:00,HH+1:00) | '
    'no qualifier=full day.'
)


# v3.3 "What counts" bullet:
_V33_WHATCOUNTS_BULLET = (
    '- Recurring schedules tied to a real standing pattern: "every\n'
    '  Thursday at 3pm". Emit the first/nearest known occurrence.'
)
_V37_WHATCOUNTS_BULLET = (
    '- Recurring schedules tied to a real standing pattern. Three\n'
    '  emit modes by context:\n'
    '\n'
    '  (1) PURE RECURRING — no specific date or era anchor for the\n'
    '      occurrence. Emit 4 envelopes — the 3 most-recent past\n'
    '      occurrences + the 1 next upcoming occurrence relative to\n'
    '      ref time. If ref time falls ON an occurrence, count it as\n'
    '      the most-recent past.\n'
    '      Pattern: present-tense or habitual phrasing with no\n'
    '      pinning date or era ("every <day-of-week>", "<weekday>s",\n'
    '      "<weekday> <TOD>", "I always do X on <day>").\n'
    '\n'
    '  (2) ERA-ANCHORED RECURRING — the pattern is anchored to a\n'
    '      specific past era. Emit 4 envelopes WITHIN that era, not\n'
    '      from ref time. Spread them roughly evenly across the era;\n'
    '      if past-3+future-1 won\'t fit, use up to 4 representative\n'
    '      occurrences in the era.\n'
    '      Pattern: an era-bracketing phrase precedes or attaches to\n'
    '      the schedule ("back in <period> I X every <day>", "during\n'
    '      <event/era>", "in the <decade>s"). The era is the anchor,\n'
    '      not ref time.\n'
    '\n'
    '  (3) SPECIFIC OCCURRENCE WITH RECURRING DESCRIPTOR — the doc\n'
    '      gives a specific date for the occurrence even though the\n'
    '      activity is described as recurring. Emit ONE envelope at\n'
    '      that date; the date pins the occurrence.\n'
    '      Pattern: past tense with a definite date, with the\n'
    '      recurring word as a TYPE descriptor of the activity\n'
    '      ("I did <my recurring activity> on <date>", "the <weekly\n'
    '      type> meeting on <date>"). The date overrides the pattern.\n'
    '\n'
    '  ' + _BANDS + '\n'
    '  For monthly/quarterly/yearly recurring units, use full unit\n'
    '  spans (past 3 + 1) with no within-day window.'
)

# v3.3 "How to think about earliest/latest" bullet:
_V33_HOWTOTHINK_BULLET = (
    '- A recurring phrase ("every Thursday at 3pm"): emit FIRST known\n'
    '  occurrence. If the passage anchors the schedule earlier ("every\n'
    '  Thursday since March"), use that start. Otherwise pick the\n'
    '  nearest past/upcoming occurrence from ref time.'
)
_V37_HOWTOTHINK_BULLET = (
    '- A recurring phrase: see the three emit modes in the recurring\n'
    '  bullet above (pure recurring → 4 envelopes around ref time;\n'
    '  era-anchored → 4 within the era; specific date → 1 at the date).\n'
    '  If the passage anchors a pure-recurring schedule earlier ("every\n'
    '  Thursday since March 2024"), the start REPLACES the third-most-\n'
    '  recent past envelope with the start date.'
)


# Apply surgical patches
SINGLE_PASS_SYSTEM_V3_7 = SINGLE_PASS_SYSTEM_V3_3
assert _V33_WHATCOUNTS_BULLET in SINGLE_PASS_SYSTEM_V3_7
SINGLE_PASS_SYSTEM_V3_7 = SINGLE_PASS_SYSTEM_V3_7.replace(
    _V33_WHATCOUNTS_BULLET, _V37_WHATCOUNTS_BULLET
)
assert _V33_HOWTOTHINK_BULLET in SINGLE_PASS_SYSTEM_V3_7
SINGLE_PASS_SYSTEM_V3_7 = SINGLE_PASS_SYSTEM_V3_7.replace(
    _V33_HOWTOTHINK_BULLET, _V37_HOWTOTHINK_BULLET
)


V3_7_JSON_SCHEMA: dict[str, Any] = {
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


class TemporalExtractorV3_7:
    """v3.7: three emit modes — pure recurring, era-anchored, specific occurrence."""

    def __init__(
        self,
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else CACHE_ROOT
        self.cache = _LLMCache(cd / "single_v3_7.json")
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
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_7},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_7_JSON_SCHEMA}},
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
