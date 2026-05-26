"""Extractor v3.4: same temporal extraction as v3.3, plus identifies the
PRIMARY event time of the doc (the one the doc is most clearly about).

Two output-schema variants are supported, both producing the same
downstream representation `(list[Interval], primary_index: int | None)`:

- mode="idx" — Option 1: a single `primary_index: int | null` field
  alongside the interval list. The LLM picks an index into its own list.
  Structurally enforces exactly-one-or-none via schema.

- mode="sep" — Option 2: separate top-level `primary` field (or null)
  and `others` list. The primary is structurally distinct from the
  rest, requiring the LLM to commit to a single primary upfront.

Both modes use the same recognition prompt as v3.3, with a primary-
identification instruction appended. Returns same datatype, so the
retriever just stores `(intervals, primary_index)` per doc regardless
of mode.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .core import Interval
from .extractor_common import _LLMCache, full_ref_context
from .extractor_v3_3 import (
    CACHE_ROOT as V33_CACHE_ROOT,
    MODEL,
    SINGLE_PASS_SYSTEM_V3_3,
)
from .schema import parse_iso, to_us


PRIMARY_INSTRUCTION = """

# Primary event time

In addition to listing every temporal reference, identify the ONE that
is the doc's PRIMARY event time — the time the doc is most clearly
about. If the doc is centered on a single event, that event's time is
primary. If the doc mentions other dates only as backdrop / reminiscence
/ comparison, those are NOT primary.

If the doc has no clearly-primary event (e.g., a summary that touches
many events equally, or a list of unrelated items), set primary to
null / primary_index to null.

Examples:
- "Did my morning run today. Reminded me of the marathon I ran in 2010."
  PRIMARY: today's morning run (NOT 2010 — that's reminiscence).
- "I started learning piano in 2019 and gave my first recital last
  month." PRIMARY: last month's recital (the doc is about the recital
  milestone, with 2019 as backdrop).
- "Q1: shipped feature A. Q2: shipped feature B. Q3: shipped feature C."
  PRIMARY: null (the doc is a list, no single event is primary).
"""


# --- Option 1: primary_index in the same flat structure ---
V3_4_IDX_SCHEMA: dict[str, Any] = {
    "name": "time_envelopes_with_primary_idx",
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
            },
            "primary_index": {
                "type": ["integer", "null"],
                "description": (
                    "0-based index into `refs` pointing at the doc's "
                    "primary event time. null if no single ref is "
                    "primary."
                ),
            },
        },
        "required": ["refs", "primary_index"],
        "additionalProperties": False,
    },
}


# --- Option 2: separate primary and others fields ---
_INTERVAL_PROPS = {
    "earliest": {"type": "string"},
    "latest": {"type": "string"},
}
V3_4_SEP_SCHEMA: dict[str, Any] = {
    "name": "time_envelopes_split_primary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "primary": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": _INTERVAL_PROPS,
                        "required": ["earliest", "latest"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ],
                "description": (
                    "The single temporal reference this doc is "
                    "primarily about (its main event time), or null "
                    "if no single ref is primary."
                ),
            },
            "others": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": _INTERVAL_PROPS,
                    "required": ["earliest", "latest"],
                    "additionalProperties": False,
                },
                "description": (
                    "All OTHER temporal references mentioned in the "
                    "doc (excluding the primary)."
                ),
            },
        },
        "required": ["primary", "others"],
        "additionalProperties": False,
    },
}


class TemporalExtractorV3_4Primary:
    """v3.3 extraction + primary-event-time identification.

    extract() returns (intervals, primary_index) — primary_index is the
    0-based index into `intervals` of the primary event, or None.
    """

    def __init__(
        self,
        mode: str = "idx",  # "idx" or "sep"
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if mode not in ("idx", "sep"):
            raise ValueError(f"mode must be 'idx' or 'sep'; got {mode!r}")
        self.mode = mode
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else (V33_CACHE_ROOT.parent / f"v3_4_{mode}")
        cd.mkdir(parents=True, exist_ok=True)
        self.cache = _LLMCache(cd / f"single_v3_4_{mode}.json")
        self.shared_pass2_cache = self.cache

    async def _call(self, text: str, ref_time: datetime) -> dict:
        ctx = full_ref_context(ref_time)
        user = f"{ctx}\n\nPassage:\n{text}"
        key = f"v3_4_{self.mode}|{ctx}|||{text}"
        cached = self.cache.get(self.model, key)
        if cached is None:
            schema = V3_4_IDX_SCHEMA if self.mode == "idx" else V3_4_SEP_SCHEMA
            system = SINGLE_PASS_SYSTEM_V3_3 + PRIMARY_INSTRUCTION
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **schema}},
            )
            cached = resp.output_text
            self.cache.put(self.model, key, cached)
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, AttributeError):
            return {}

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

    async def extract(
        self, text: str, ref_time: datetime
    ) -> tuple[list[Interval], int | None]:
        data = await self._call(text, ref_time)
        if self.mode == "idx":
            refs = data.get("refs", []) or []
            intervals = [iv for iv in (self._to_interval(r) for r in refs) if iv is not None]
            primary_index = data.get("primary_index")
            # Validate: must be in-range int
            if isinstance(primary_index, int) and 0 <= primary_index < len(intervals):
                return intervals, primary_index
            return intervals, None
        else:  # "sep"
            primary_raw = data.get("primary")
            others_raw = data.get("others", []) or []
            primary_iv = self._to_interval(primary_raw) if primary_raw else None
            others_ivs = [iv for iv in (self._to_interval(r) for r in others_raw) if iv is not None]
            if primary_iv is not None:
                # Primary is at index 0 in the combined list
                return [primary_iv] + others_ivs, 0
            return others_ivs, None

    def save_caches(self) -> None:
        self.cache.save()
