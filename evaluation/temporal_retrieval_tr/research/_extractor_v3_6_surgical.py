"""Extractor v3.6 (surgical): tiniest possible patch to v3.3 for recurring.

v3.5_tod added a ~70-line section about recurring patterns; that ~doubled
the prompt length and caused LLM drift on non-recurring docs (b74xotto5:
dense_cluster -0.033, disc -0.033, era -0.050, edge_era_refs -0.083,
plus loss of planner-only's +0.067/+0.083 gains on neg_temp/precedents).

v3.6 takes the OPPOSITE approach: keep the v3.3 prompt byte-identical
EXCEPT for the two existing recurring-bullet edits. No new section.

Two edits vs v3.3:
1. The "Recurring schedules" bullet in "What counts as a temporal
   reference" — note past-3+future-1 + TOD bands inline.
2. The "recurring phrase" bullet in "How to think about earliest /
   latest" — same.

Both are surgical replacements; everything else identical to v3.3.
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
PROMPT_VERSION = "v3_6_surgical"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# Bands string — compact one-liner, kept terse to minimize drift.
_BANDS = (
    'Time-of-day bands (UTC): "morning"=[03:00,13:00) | "noon"=[10:00,14:00) | '
    '"afternoon"=[11:00,19:00) | "evening"=[16:00,00:00 next day) | '
    '"night"=[19:00,07:00 next day) | "at HH:MM" or "at HHam/pm"=[HH:00,HH+1:00) | '
    'no qualifier=full day.'
)


# v3.3 "What counts" bullet on recurring:
_V33_WHATCOUNTS_BULLET = (
    '- Recurring schedules tied to a real standing pattern: "every\n'
    '  Thursday at 3pm". Emit the first/nearest known occurrence.'
)
_V36_WHATCOUNTS_BULLET = (
    '- Recurring schedules tied to a real standing pattern: "every\n'
    '  Thursday at 3pm", "Sundays we do brunch", "Friday afternoons".\n'
    '  Emit 4 envelopes — the 3 most-recent past occurrences + the\n'
    '  1 next upcoming occurrence (relative to ref time). If ref time\n'
    '  falls ON an occurrence, count it as the most-recent past.\n'
    '  ' + _BANDS + '\n'
    '  For monthly/quarterly/yearly units, use full unit spans (past 3\n'
    '  + 1) with no within-day window.'
)

# v3.3 "How to think about earliest/latest" bullet on recurring:
_V33_HOWTOTHINK_BULLET = (
    '- A recurring phrase ("every Thursday at 3pm"): emit FIRST known\n'
    '  occurrence. If the passage anchors the schedule earlier ("every\n'
    '  Thursday since March"), use that start. Otherwise pick the\n'
    '  nearest past/upcoming occurrence from ref time.'
)
_V36_HOWTOTHINK_BULLET = (
    '- A recurring phrase: emit 4 envelopes (past 3 + future 1; see\n'
    '  the "What counts" recurring bullet for the TOD bands). If the\n'
    '  passage anchors the schedule earlier ("every Thursday since\n'
    '  March 2024"), the start of the schedule REPLACES the\n'
    '  third-most-recent past envelope with the start date.'
)


# Apply surgical patches:
SINGLE_PASS_SYSTEM_V3_6 = SINGLE_PASS_SYSTEM_V3_3
assert _V33_WHATCOUNTS_BULLET in SINGLE_PASS_SYSTEM_V3_6, (
    "v3.3 WHATCOUNTS recurring bullet drifted; refresh anchor string"
)
SINGLE_PASS_SYSTEM_V3_6 = SINGLE_PASS_SYSTEM_V3_6.replace(
    _V33_WHATCOUNTS_BULLET, _V36_WHATCOUNTS_BULLET
)
assert _V33_HOWTOTHINK_BULLET in SINGLE_PASS_SYSTEM_V3_6, (
    "v3.3 HOWTOTHINK recurring bullet drifted; refresh anchor string"
)
SINGLE_PASS_SYSTEM_V3_6 = SINGLE_PASS_SYSTEM_V3_6.replace(
    _V33_HOWTOTHINK_BULLET, _V36_HOWTOTHINK_BULLET
)


V3_6_JSON_SCHEMA: dict[str, Any] = {
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


class TemporalExtractorV3_6:
    """v3.6 (surgical): minimal prompt patch to v3.3 for past-3+future-1 recurring."""

    def __init__(
        self,
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else CACHE_ROOT
        self.cache = _LLMCache(cd / "single_v3_6.json")
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
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_6},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_6_JSON_SCHEMA}},
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
