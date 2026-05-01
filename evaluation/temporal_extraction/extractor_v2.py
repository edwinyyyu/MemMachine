"""v2 extractor = v1 + trigger gazetteer + full ref-time context + few-shot.

Single-pass (Pass 1) with enriched system prompt. Pass 2 unchanged.
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

PASS1_SYSTEM_V2 = f"""You are a meticulous temporal-reference extractor.

Your job: identify EVERY temporal reference in a passage. A temporal
reference is any span that refers to a moment, span, duration, or recurring
pattern in time. It can be absolute ("March 5, 2026"), relative
("yesterday", "2 weeks ago"), vague ("around 2010", "a decade ago"), or
recurring ("every Thursday at 3pm").

{TRIGGER_GAZETTEER}

For each reference, output:
- surface: the exact substring from the passage, verbatim, with no edits
  to casing, spacing, or punctuation. Prefer the LONGEST natural phrase
  that carries the temporal meaning — include determiners like "the",
  "every", and qualifiers like "earlier", "later", "around", "about",
  "the first week of" when they are part of the phrase. Do NOT include a
  leading bare "on"/"in" when it is just a preposition attaching to the
  phrase.
- kind_guess: one of [instant, interval, duration, recurrence].
  - instant: a point-in-time (even if fuzzy): "yesterday", "2015", "last month".
  - interval: a start-to-end range: "from X to Y", "the first week of April".
  - duration: an unanchored length: "for 3 weeks", "two hours long".
  - recurrence: a recurring pattern: "every Thursday".
- context_hint: a short (<=12 word) note of what it refers to.

Do NOT emit seasons ("summer") unless the year is specified or strongly
implied by context. Do NOT emit "once", "often", "always", bare "recent",
"old", "new".

{FEW_SHOT_EXAMPLES}

Output a single JSON object: {{"refs": [...]}}. If none, output {{"refs": []}}.
"""


class ExtractorV2(BaseImprovedExtractor):
    VERSION = 2

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        ctx = full_ref_context(ref_time)
        user = f'{ctx}\n\nPassage:\n{text}\n\nReturn {{"refs": [...]}} as JSON.'
        data = await self._call_json(PASS1_SYSTEM_V2, user, max_completion_tokens=4000)
        if not isinstance(data, dict):
            return []
        return list(data.get("refs", []))
