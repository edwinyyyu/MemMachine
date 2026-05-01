"""v3 extractor = v2 + chain-of-thought Pass 1.

Pass 1 produces a scratchpad listing every potentially-temporal phrase
(even borderline), then reviews each, then outputs the final filtered list.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from extractor_common import (
    FEW_SHOT_EXAMPLES,
    TRIGGER_GAZETTEER,
    BaseImprovedExtractor,
    full_ref_context,
)

PASS1_SYSTEM_V3 = f"""You are a meticulous temporal-reference extractor.

Your job: identify EVERY temporal reference in a passage. A temporal
reference is any span that refers to a moment, span, duration, or recurring
pattern in time (absolute, relative, vague, or recurring).

{TRIGGER_GAZETTEER}

Use the following WORKING PROCESS for every passage:

1. ``candidates``: list every phrase in the passage that might be temporal,
   including borderline ones. Scan the passage end-to-end and check each
   against the gazetteer above. Include: explicit dates, relative phrases,
   day-of-week names, decades, partial-period phrases ("first week of X"),
   time-of-day expressions, recurrences, durations, and embedded phrases
   like "when I was in college".
2. ``review``: for each candidate, decide keep=true/false with a one-line
   reason. Keep if the phrase refers to a concrete or fuzzy point/span in
   time. Drop bare seasons without a year, bare "recent"/"old"/"new", and
   pure frequency words ("often", "always").
3. ``refs``: the final list of kept references, each with:
   - surface: the exact substring (longest natural phrase carrying the
     temporal meaning; include determiners and qualifiers like "the",
     "every", "earlier", "around", "the first week of" when part of the
     phrase).
   - kind_guess: [instant, interval, duration, recurrence].
   - context_hint: <=12 word note.

{FEW_SHOT_EXAMPLES}

Output schema:
{{
  "candidates": [{{"phrase": "...", "reason": "..."}}, ...],
  "review":     [{{"phrase": "...", "keep": true/false, "why": "..."}}, ...],
  "refs":       [{{"surface": "...", "kind_guess": "...", "context_hint": "..."}}, ...]
}}

Return JSON ONLY.
"""


class ExtractorV3(BaseImprovedExtractor):
    VERSION = 3

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
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
