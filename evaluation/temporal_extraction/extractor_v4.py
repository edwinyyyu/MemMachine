"""v4 extractor = v3 + recovery pass.

After the chain-of-thought Pass-1, a second call shows the passage and the
references already found, and asks the model to scan specifically for
missed implicit/relative/named-relative/embedded time references. The
recovered spans are unioned with the original extraction.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from extractor_common import (
    FEW_SHOT_EXAMPLES,
    TRIGGER_GAZETTEER,
    BaseImprovedExtractor,
    full_ref_context,
)
from extractor_v3 import PASS1_SYSTEM_V3

RECOVERY_SYSTEM = f"""You are a temporal-reference RECOVERY auditor.

An earlier pass already found some temporal references in a passage. Your
job: carefully audit and find any it MISSED.

{TRIGGER_GAZETTEER}

Focus especially on the patterns that first-pass extractors most often
miss:
- Named-relative calendar units tucked into short queries: "last month",
  "this month", "earlier this month", "later this month", "last year",
  "next week", "this week", "this year", "next year".
- Day-of-week references ("last Tuesday", "next Thursday", "on Friday").
- Decades and eras ("the 90s", "the 2010s", "the mid-80s").
- Partial-period phrases ("the first week of April 2026", "the second
  half of last year", "early March 2025").
- Embedded references in natural speech that are anchored ("when I was in
  college", ONLY if the context anchors it).
- Approximate anchors ("about a month ago", "around 2010", "a few weeks
  ago") — these must be extracted.
- Time-of-day expressions ("at 3pm", "tonight", "this morning").

{FEW_SHOT_EXAMPLES}

Output schema:
{{
  "missed": [
    {{"surface": "...", "kind_guess": "instant|interval|duration|recurrence",
      "context_hint": "<=12 word note",
      "why_missed": "one-line reason this was easy to overlook"}},
    ...
  ]
}}

If nothing was missed, output {{"missed": []}}. Do NOT re-list phrases
that were already found (by exact surface, case-insensitive). Return JSON
ONLY.
"""


class ExtractorV4(BaseImprovedExtractor):
    VERSION = 4

    async def _pass1_primary(
        self, text: str, ref_time: datetime
    ) -> list[dict[str, Any]]:
        ctx = full_ref_context(ref_time)
        user = (
            f"{ctx}\n\n"
            f"Passage:\n{text}\n\n"
            "Return JSON with candidates, review, and refs."
        )
        data = await self._call_json(PASS1_SYSTEM_V3, user, max_completion_tokens=5000)
        if not isinstance(data, dict):
            return []
        return list(data.get("refs", []))

    async def _pass1_recovery(
        self,
        text: str,
        ref_time: datetime,
        already_found: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ctx = full_ref_context(ref_time)
        found_surfaces = [r.get("surface", "") for r in already_found]
        user = (
            f"{ctx}\n\n"
            f"Passage:\n{text}\n\n"
            f"Already found (surfaces): {json.dumps(found_surfaces)}\n\n"
            "Review the passage for MISSED temporal references. "
            'Return {"missed": [...]} as JSON.'
        )
        data = await self._call_json(RECOVERY_SYSTEM, user, max_completion_tokens=4000)
        if not isinstance(data, dict):
            return []
        return list(data.get("missed", []))

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        primary = await self._pass1_primary(text, ref_time)
        missed = await self._pass1_recovery(text, ref_time, primary)
        # Union — dedupe by case-insensitive surface.
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for r in primary + missed:
            surf = (r.get("surface") or "").strip()
            if not surf:
                continue
            k = surf.lower()
            if k in seen:
                continue
            seen.add(k)
            merged.append(r)
        return merged
