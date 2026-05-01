"""v5 extractor = v4 + regex pre-pass.

Run regex over text to find candidate spans for well-known patterns
(numbered-relative, named-relative, year literals, month-day literals,
day-of-week, ISO dates, time-of-day, recurrences, partial-period phrases,
"during X"). Each candidate is then sent to the LLM in a triage call that
asks "is this a real temporal reference in context? if yes, return it
with kind_guess+context_hint; if no, drop it." The kept regex candidates
are unioned with v4's LLM-extracted references.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from extractor_common import (
    full_ref_context,
    regex_candidates,
)
from extractor_v4 import ExtractorV4

REGEX_TRIAGE_SYSTEM = """You are a temporal-reference TRIAGE classifier.

You will be shown a passage and a list of candidate spans found by regex.
Each candidate might be:
- A real temporal reference that should be extracted (KEEP).
- A false positive (numeric literal that is not a date; word that is part
  of a non-temporal phrase; time expression embedded in a quotation or
  title that does not refer to a real event time) — DROP.

For each kept candidate, output:
- surface: the exact substring (use the candidate text, or a trimmed /
  slightly-adjusted substring if the regex over- or under-captured).
- kind_guess: instant, interval, duration, or recurrence.
- context_hint: <=12 word note explaining what it refers to.

Rules:
- "last month"/"this month"/"next month" and similar named-relative: KEEP.
- "the first week of <Month> <Year>": KEEP as kind=interval.
- A year literal like "2019" standing alone: KEEP as instant with
  granularity=year, IF it refers to a time (not e.g. a model number).
- "the 90s"/"the 2010s": KEEP as interval or instant with granularity=decade.
- "morning"/"afternoon"/"evening" without a date anchor: DROP (too vague).
- "noon"/"midnight"/"dawn"/"dusk" alone: DROP unless tied to a date.

Output JSON: {"kept": [{"surface": "...", "kind_guess": "...",
"context_hint": "..."}, ...]}. If none kept, {"kept": []}.
Return JSON ONLY.
"""


class ExtractorV5(ExtractorV4):
    VERSION = 5

    async def _regex_triage(
        self,
        text: str,
        ref_time: datetime,
    ) -> list[dict[str, Any]]:
        cands = regex_candidates(text)
        if not cands:
            return []
        ctx = full_ref_context(ref_time)
        cand_list = [{"surface": s, "start": a, "end": b} for s, a, b in cands]
        user = (
            f"{ctx}\n\n"
            f"Passage:\n{text}\n\n"
            f"Regex candidates:\n{json.dumps(cand_list, indent=2)}\n\n"
            'Review each. Return {"kept": [...]} JSON.'
        )
        data = await self._call_json(
            REGEX_TRIAGE_SYSTEM, user, max_completion_tokens=4000
        )
        if not isinstance(data, dict):
            return []
        return list(data.get("kept", []))

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        primary = await self._pass1_primary(text, ref_time)
        missed = await self._pass1_recovery(text, ref_time, primary)
        triage = await self._regex_triage(text, ref_time)
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for r in primary + missed + triage:
            surf = (r.get("surface") or "").strip()
            if not surf:
                continue
            k = surf.lower()
            if k in seen:
                continue
            seen.add(k)
            merged.append(r)
        return merged
