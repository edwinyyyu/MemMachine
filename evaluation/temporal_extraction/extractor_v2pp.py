"""v2'' extractor (v2-prime-prime).

Builds on v2' with three extraction-layer fixes:

1. **Modality detection** in Pass-1: each ref gets a modality tag in
   {actual, fictional, hypothetical, quoted_embedded}. Closes A7 (fictional
   "In the novel I'm reading, the story is set in 1850") and helps A8
   tense-shifted ("if I had been born in 1980").

2. **Wider brackets for colloquial fuzzy modifiers**: "a couple", "a few",
   "several", "not long ago", "back in the day", "a while back" — the
   Pass-2 prompt now explicitly widens the bracket. Closes A3.

3. **Holiday / season / cultural-calendar post-processor**: after Pass-2,
   if a surface matches a holiday/season name (Ramadan, Easter, Christmas,
   Thanksgiving, CNY, Halloween, Valentine's, summer 2024, spring term...)
   we *override* the instant with concrete dates from ``holiday_table``.
   Closes A9.

Schema note: modality is stashed via ``modality_schema.attach_modality``
(per-instance attribute), not as a new dataclass field. Back-compat:
downstream code that does not read ``_modality`` treats all expressions as
actual.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from extractor_common import (
    PASS2_JSON_SCHEMA,
    TRIGGER_GAZETTEER,
    full_ref_context,
)
from extractor_v2p import (
    PASS2_SYSTEM_V2P,
    ExtractorV2P,
)
from holiday_table import resolve_holiday, surface_matches_holiday
from modality_schema import attach_modality, normalize_modality
from resolver import ResolverError, post_process
from schema import FuzzyInstant, TimeExpression, time_expression_from_dict

# ---------------------------------------------------------------------------
# v2'' few-shot examples — adds modality + fuzzy-brackets cases.
# We keep v2' domain-neutral axis-only examples and APPEND the new ones.
# ---------------------------------------------------------------------------
FEW_SHOT_V2PP = """Examples of hard-to-catch temporal references (domain-neutral):

Passage: "My cousin moved out last month and I haven't heard from her since."
Extractions: [{"surface": "last month", "kind_guess": "instant", "modality": "actual", "context_hint": "calendar month prior"}]

Passage: "Earlier this month we repainted the kitchen, and the week before that we ordered the cabinets."
Extractions: [
  {"surface": "Earlier this month", "kind_guess": "instant", "modality": "actual", "context_hint": "within the current calendar month, earlier half"},
  {"surface": "the week before that", "kind_guess": "instant", "modality": "actual", "context_hint": "one week before 'earlier this month'"}
]

Passage: "Back in the 90s, dial-up was the norm; by around 2008 most homes had broadband."
Extractions: [
  {"surface": "the 90s", "kind_guess": "interval", "modality": "actual", "context_hint": "decade 1990-1999"},
  {"surface": "around 2008", "kind_guess": "instant", "modality": "actual", "context_hint": "approximately the year 2008"}
]

Passage: "Every Thursday at 3pm I have a standup, but next Thursday it's cancelled."
Extractions: [
  {"surface": "Every Thursday at 3pm", "kind_guess": "recurrence", "modality": "actual", "context_hint": "weekly Thursday at 15:00"},
  {"surface": "next Thursday", "kind_guess": "instant", "modality": "actual", "context_hint": "upcoming Thursday (cancelled instance)"}
]

--- Axis-only (bare month / season / quarter / part-of-day / weekend-weekday) examples ---

Passage: "March is always chaotic for me."
Extractions: [{"surface": "March", "kind_guess": "recurrence", "modality": "actual", "context_hint": "month of March, any year (axis-only)"}]

Passage: "Q2 is our busiest quarter."
Extractions: [{"surface": "Q2", "kind_guess": "recurrence", "modality": "actual", "context_hint": "second quarter (Apr-Jun), any year (axis-only)"}]

Passage: "Every summer the whole family goes camping."
Extractions: [{"surface": "Every summer", "kind_guess": "recurrence", "modality": "actual", "context_hint": "summer months recurring yearly"}]

Passage: "Tuesday evenings are reserved for book club."
Extractions: [{"surface": "Tuesday evenings", "kind_guess": "recurrence", "modality": "actual", "context_hint": "weekly Tuesday in the evening"}]

Passage: "I usually go running in the morning."
Extractions: [{"surface": "in the morning", "kind_guess": "recurrence", "modality": "actual", "context_hint": "mornings (axis-only)"}]

Passage: "We brunch on weekends."
Extractions: [{"surface": "weekends", "kind_guess": "recurrence", "modality": "actual", "context_hint": "Sat-Sun (axis-only)"}]

--- Modality examples (CRITICAL — determines whether the time is REAL) ---

Passage: "In the novel I'm reading, the story is set in 1850."
Extractions: [{"surface": "1850", "kind_guess": "instant", "modality": "fictional", "context_hint": "fictional year inside a novel reference"}]

Passage: "The movie takes place in 2089, when cars can fly."
Extractions: [{"surface": "2089", "kind_guess": "instant", "modality": "fictional", "context_hint": "fictional future year inside a movie"}]

Passage: "Imagine a world where the year is 2089 and cars can fly."
Extractions: [{"surface": "2089", "kind_guess": "instant", "modality": "hypothetical", "context_hint": "hypothetical year inside 'imagine'"}]

Passage: "What if I had been born in 1980? How different things would be."
Extractions: [{"surface": "1980", "kind_guess": "instant", "modality": "hypothetical", "context_hint": "counterfactual birth year, not real"}]

Passage: "If she had left in 2015, she'd be gone for a decade by now."
Extractions: [{"surface": "2015", "kind_guess": "instant", "modality": "hypothetical", "context_hint": "counterfactual past year"}]

Passage: "Alice said yesterday 'this month has been rough'."
Extractions: [
  {"surface": "yesterday", "kind_guess": "instant", "modality": "actual", "context_hint": "day before ref_time (the saying event)"},
  {"surface": "this month", "kind_guess": "instant", "modality": "quoted_embedded", "context_hint": "this month relative to 'yesterday' — inside a quote"}
]

Passage: "In 2020, my sister said 'this month has been rough'."
Extractions: [
  {"surface": "2020", "kind_guess": "instant", "modality": "actual", "context_hint": "year of the saying event"},
  {"surface": "this month", "kind_guess": "instant", "modality": "quoted_embedded", "context_hint": "some month of 2020, inside the quote"}
]

--- Fuzzy-modifier examples (CRITICAL — these need WIDE brackets) ---

Passage: "A couple of years ago we adopted a cat from the shelter."
Extractions: [{"surface": "A couple of years ago", "kind_guess": "instant", "modality": "actual", "context_hint": "roughly 2-3 years before ref_time; WIDEN bracket ~+/-50%"}]

Passage: "A few weeks back I bumped into my old professor."
Extractions: [{"surface": "A few weeks back", "kind_guess": "instant", "modality": "actual", "context_hint": "roughly 3-5 weeks before ref_time; WIDEN bracket"}]

Passage: "Several months ago I started a new job."
Extractions: [{"surface": "Several months ago", "kind_guess": "instant", "modality": "actual", "context_hint": "roughly 4-8 months before ref_time; WIDE bracket"}]

Passage: "Not long ago, Dana called to say she was moving abroad."
Extractions: [{"surface": "Not long ago", "kind_guess": "instant", "modality": "actual", "context_hint": "vague recent past; WIDE bracket (days to months)"}]

Passage: "Back in the day, we used to swim in the quarry."
Extractions: [{"surface": "Back in the day", "kind_guess": "interval", "modality": "actual", "context_hint": "personal era, decades ago; VERY WIDE bracket"}]

Passage: "A while back we used to hike every Sunday."
Extractions: [{"surface": "A while back", "kind_guess": "instant", "modality": "actual", "context_hint": "vague past; WIDE bracket"}]

--- Holiday / season examples ---

Passage: "During Ramadan last year I reorganized my sleep schedule."
Extractions: [
  {"surface": "Ramadan last year", "kind_guess": "interval", "modality": "actual", "context_hint": "Ramadan in the previous year (resolve to concrete dates)"}
]

Passage: "Easter 2015 was the last time we all gathered."
Extractions: [{"surface": "Easter 2015", "kind_guess": "instant", "modality": "actual", "context_hint": "Easter Sunday 2015"}]

Passage: "We made dumplings during Chinese New Year."
Extractions: [{"surface": "Chinese New Year", "kind_guess": "interval", "modality": "actual", "context_hint": "Lunar New Year period (most recent)"}]

Passage: "Summer 2024 was unusually cool."
Extractions: [{"surface": "Summer 2024", "kind_guess": "interval", "modality": "actual", "context_hint": "meteorological summer of 2024"}]
"""


# ---------------------------------------------------------------------------
# Pass-1 system prompt — v2'' adds modality + fuzzy-modifier rules.
# ---------------------------------------------------------------------------
PASS1_SYSTEM_V2PP = f"""You are a meticulous temporal-reference extractor.

Your job: identify EVERY temporal reference in a passage. A temporal
reference is any span that refers to a moment, span, duration, or recurring
pattern in time. It can be absolute ("March 5, 2026"), relative
("yesterday"), vague ("around 2010"), or recurring ("every Thursday at
3pm"). It can also be AXIS-ONLY — a bare month, season, quarter,
part-of-day, or weekday/weekend category word that identifies a periodic
dimension even without an anchoring year.

{TRIGGER_GAZETTEER}

AXIS-ONLY SURFACES (ALLOWED — always extract these when they appear,
even without a year):
- Bare month names ("January"..."December", "Jan", "Feb", ...)
- Quarters: "Q1"..."Q4", "first quarter"
- Seasons: "spring", "summer", "autumn", "fall", "winter"
- Parts of day: "morning", "afternoon", "evening", "night", "dawn", "dusk"
- Weekend/weekday category words
- Compound axes: "Tuesday evenings", "June weekends", "weekday mornings"

FUZZY COLLOQUIAL MODIFIERS (ALLOWED — extract and flag context_hint
with "WIDEN bracket"):
- "a couple of <unit>", "a few <units>", "several <units>"
- "not long ago", "a while back", "a while ago"
- "back in the day", "in the old days", "once upon a time (personal)"
- "recently", "lately" (only if anchored to a concrete event)
These need WIDER brackets than their literal count — include a
"WIDEN bracket" note in context_hint.

MODALITY (CRITICAL — new field in v2''):

Each reference MUST be tagged with a modality:
- "actual":   the default. The time really happened / will happen.
- "fictional": the time is inside a novel / story / movie / book / game /
               TV show reference. E.g. "In the novel, it's 1850",
               "The movie is set in 2089".
- "hypothetical": the time is inside a conditional / counterfactual /
                  aspirational clause. Triggers: "if I had", "what if",
                  "imagine", "would have", "could have", "suppose", "in a
                  world where". E.g. "If I had been born in 1980".
- "quoted_embedded": the time is inside a direct quote where the
                     speaker's ref_time differs from the passage ref_time.
                     Triggers: "<person> said '...<time>...'",
                     "<person> wrote '...<time>...'". E.g. "Alice said
                     yesterday 'this month has been rough'" — the inner
                     "this month" is quoted_embedded.

If you are UNSURE, emit "actual".

For each reference, output:
- surface: the exact substring from the passage, verbatim.
- kind_guess: one of [instant, interval, duration, recurrence].
- modality: one of [actual, fictional, hypothetical, quoted_embedded].
- context_hint: a short (<=15 word) note; if axis-only, say so; if fuzzy,
  include "WIDEN bracket"; if modality is not actual, explain why briefly.

Do NOT emit: "once", "often", "always", bare "recent"/"old"/"new" with no
concrete referent.

{FEW_SHOT_V2PP}

Output a single JSON object: {{"refs": [...]}}. If none, output {{"refs": []}}.
"""


# ---------------------------------------------------------------------------
# Pass-2 system prompt — v2' base + explicit fuzzy-modifier bracket rules.
# Keep axis-only rules from v2p.
# ---------------------------------------------------------------------------
PASS2_EXTRA_FUZZY = """

FUZZY-MODIFIER BRACKETS (v2'' additions — CRITICAL):

When the surface uses a colloquial fuzzy count, widen the bracket beyond
the naive literal. Concrete policy (ref_time R):

- "a couple of X ago" (X = unit): centered ~2X back, bracket = [R - 3X, R - 1X].
  Example: "a couple of years ago" at 2026-04-23 -> earliest=2023-04-23,
  latest=2025-04-23, best=2024-04-23, granularity=year.
- "a few X ago" (X = unit): centered ~4X back, bracket = [R - 6X, R - 2X].
  Example: "a few weeks back" at 2026-04-23 -> earliest=2026-03-12,
  latest=2026-04-09, best=2026-03-26, granularity=week.
- "several X ago": centered ~6X back, bracket = [R - 10X, R - 3X].
- "not long ago": wide RECENT past, bracket = [R - 6 months, R - 1 day],
  best ~ 3 weeks ago, granularity=month.
- "a while back" / "a while ago": wide past, bracket = [R - 2 years, R - 1 month],
  best ~ 6 months ago, granularity=year.
- "back in the day" / "in the old days": VERY wide, bracket = [R - 40 years,
  R - 10 years], best ~ 25 years ago, granularity=decade. kind="interval".
- "recently" (standalone, rarely extract — only if concrete): bracket =
  [R - 30 days, R], best = R - 7 days.

Always fill earliest, latest, and best.
"""


PASS2_SYSTEM_V2PP = PASS2_SYSTEM_V2P + PASS2_EXTRA_FUZZY


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------
class ExtractorV2PP(ExtractorV2P):
    """v2-prime-prime: modality + fuzzy brackets + holiday post-processor."""

    VERSION = 2  # share pass-2 cache; own cache subdir for pass-1.

    def __init__(self, concurrency: int = 5, **kwargs: Any) -> None:
        # Skip ExtractorV2P.__init__ which hardcodes its cache_subdir.
        from extractor_common import BaseImprovedExtractor

        BaseImprovedExtractor.__init__(
            self, concurrency=concurrency, cache_subdir="extractor_v2pp", **kwargs
        )
        # Per-ref modality tags (keyed by surface.lower()) — populated in pass1,
        # consumed in extract().
        self._modality_by_surface: dict[str, str] = {}

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        ctx = full_ref_context(ref_time)
        user = f'{ctx}\n\nPassage:\n{text}\n\nReturn {{"refs": [...]}} as JSON.'
        data = await self._call_json(
            PASS1_SYSTEM_V2PP, user, max_completion_tokens=4000
        )
        if not isinstance(data, dict):
            return []
        refs = list(data.get("refs", []))
        # Record modality per surface for later attach_modality().
        for r in refs:
            surf = (r.get("surface") or "").strip()
            if surf:
                self._modality_by_surface[surf.lower()] = normalize_modality(
                    r.get("modality")
                )
        return refs

    async def pass2(
        self,
        surface: str,
        kind_guess: str,
        context_hint: str,
        surrounding: str,
        ref_time: datetime,
        *,
        error_hint: str = "",
    ) -> dict[str, Any] | None:
        """Override Pass-2 to use v2'' prompt (v2' + fuzzy bracket rules)."""
        wk = ref_time.strftime("%A")
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Surrounding context: {surrounding}\n"
            f'Reference: "{surface}"\n'
            f"Kind hint: {kind_guess}\n"
            f"Context hint: {context_hint}\n"
        )
        if error_hint:
            user += f"\nCorrection hint: {error_hint}\n"
        user += "\nReturn JSON matching the schema."
        raw = await self._call(
            PASS2_SYSTEM_V2PP,
            user,
            json_schema=PASS2_JSON_SCHEMA,
            max_completion_tokens=3000,
            share_cache=False,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            user2 = (
                user
                + "\n\n[Your previous response was invalid JSON. Return ONLY the JSON object.]"
            )
            raw2 = await self._call(
                PASS2_SYSTEM_V2PP,
                user2,
                json_schema=PASS2_JSON_SCHEMA,
                max_completion_tokens=3000,
                share_cache=False,
            )
            try:
                return json.loads(raw2)
            except json.JSONDecodeError:
                return None

    # ------------------------------------------------------------------
    # Override extract() to attach modality and run holiday post-processor.
    # ------------------------------------------------------------------
    async def extract(self, text: str, ref_time: datetime) -> list[TimeExpression]:
        # Reset per-call modality map.
        self._modality_by_surface = {}

        refs = await self.pass1(text, ref_time)

        # Dedupe by lowercased surface.
        seen: set[str] = set()
        unique_refs: list[dict[str, Any]] = []
        for ref in refs:
            surf = (ref.get("surface") or "").strip()
            if not surf:
                continue
            key = surf.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_refs.append(ref)

        coros = []
        surfaces_ordered: list[str] = []
        for ref in unique_refs:
            surface = ref["surface"].strip()
            kind_guess = ref.get("kind_guess", "instant")
            context_hint = ref.get("context_hint", "")
            surfaces_ordered.append(surface)
            coros.append(
                self.pass2_with_retry(surface, kind_guess, context_hint, text, ref_time)
            )
        results = await asyncio.gather(*coros)

        out: list[TimeExpression] = []
        for surface, pred in zip(surfaces_ordered, results):
            if pred is None:
                continue
            pred["reference_time"] = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                te = time_expression_from_dict(pred)
            except Exception:
                continue
            idx = text.find(surface)
            if idx >= 0:
                te.span_start = idx
                te.span_end = idx + len(surface)

            # --- v2'' addition: holiday / season post-processor --------
            try:
                _apply_holiday_override(te, ref_time)
            except Exception:
                pass

            # Post-process (normalizes / validates)
            try:
                te, _warn = post_process(te, auto_correct=True)
            except ResolverError:
                continue

            # --- v2'' addition: attach modality tag --------
            modality = self._modality_by_surface.get(surface.lower(), "actual")
            attach_modality(te, modality)

            out.append(te)
        return out


# ---------------------------------------------------------------------------
# Holiday override helper
# ---------------------------------------------------------------------------
def _apply_holiday_override(te: TimeExpression, ref_time: datetime) -> None:
    """If the TE's surface matches a holiday/season gazetteer entry,
    OVERRIDE the instant/interval with the concrete date range.

    Rules:
    - Only fire if surface_matches_holiday (saves work).
    - If year not explicit in surface, use ref_time year-resolution.
    - Respect the original kind when reasonable; for day-level holidays
      with granularity day, keep kind=instant; for ranges (Ramadan, CNY,
      season), convert to kind=interval.
    """
    surface = te.surface or ""
    if not surface_matches_holiday(surface):
        return
    res = resolve_holiday(surface, ref_time)
    if res is None:
        return
    earliest = res["earliest"]
    latest = res["latest"]
    best = res["best"]
    gran = res["granularity"]
    span_days = (latest - earliest).days
    # Day-level holiday -> instant
    if span_days <= 1:
        te.kind = "instant"
        te.instant = FuzzyInstant(
            earliest=earliest,
            latest=latest,
            best=best,
            granularity=gran,
        )
        te.interval = None
        te.recurrence = None
    else:
        # Range -> interval
        from schema import FuzzyInterval

        te.kind = "interval"
        te.interval = FuzzyInterval(
            start=FuzzyInstant(
                earliest=earliest,
                latest=earliest + (latest - earliest) * 0,  # same timepoint
                best=None,
                granularity=gran,
            ),
            end=FuzzyInstant(
                earliest=latest,
                latest=latest,
                best=None,
                granularity=gran,
            ),
        )
        # Fix: interval.start.latest should be >= earliest
        te.interval.start.latest = earliest
        te.instant = None
        te.recurrence = None
