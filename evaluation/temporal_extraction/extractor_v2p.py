"""v2' extractor (v2-prime).

Identical to v2 (gazetteer + full ref-time context + few-shot) except:

1. Pass 1 prompt: REMOVES the "Do NOT emit seasons without a year" rule.
   Explicitly ALLOWS and prompts for bare temporal axis surfaces:
       - bare months: "March", "December"
       - quarters: "Q1", "Q2", "Q3", "Q4"
       - seasons: "spring", "summer", "autumn", "fall", "winter"
       - part-of-day: "morning", "afternoon", "evening", "night", "dawn",
         "dusk"
       - weekend/weekday category words: "weekend", "weekday", "weekends",
         "weekdays"
   Each has a domain-neutral few-shot extraction example.

2. Pass 2: when resolving a bare axis surface with no year/context, emit a
   recurrence that encodes the axis constraint, reusing existing
   axes_for_recurrence infrastructure (BYMONTH for month/season/quarter,
   BYHOUR for part-of-day, BYDAY for weekend/weekday).

Design choice: no new schema field ``axis_only`` is needed — an axis-only
reference is naturally expressible as a yearly/daily/weekly recurrence
whose axis-specific dimension is constrained via the RRULE and whose
other axes are uniform. The multi_axis_scorer already consumes this
shape correctly via ``axes_for_recurrence``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from extractor_common import (
    PASS2_JSON_SCHEMA,
    TRIGGER_GAZETTEER,
    BaseImprovedExtractor,
    full_ref_context,
)

# ---------------------------------------------------------------------------
# Pass-1 few-shots — include v2's domain-neutral examples plus NEW axis-only
# examples for bare months, quarters, seasons, parts-of-day, weekend/weekday.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES_V2P = """Examples of hard-to-catch temporal references (domain-neutral):

Passage: "My cousin moved out last month and I haven't heard from her since."
Extractions: [{"surface": "last month", "kind_guess": "instant", "context_hint": "calendar month prior"}]

Passage: "Earlier this month we repainted the kitchen, and the week before that we ordered the cabinets."
Extractions: [
  {"surface": "Earlier this month", "kind_guess": "instant", "context_hint": "within the current calendar month, earlier half"},
  {"surface": "the week before that", "kind_guess": "instant", "context_hint": "one week before 'earlier this month'"}
]

Passage: "I saw her last Tuesday at the coffee shop."
Extractions: [{"surface": "last Tuesday", "kind_guess": "instant", "context_hint": "most recent past Tuesday"}]

Passage: "Back in the 90s, dial-up was the norm; by around 2008 most homes had broadband."
Extractions: [
  {"surface": "the 90s", "kind_guess": "interval", "context_hint": "decade 1990-1999"},
  {"surface": "around 2008", "kind_guess": "instant", "context_hint": "approximately the year 2008"}
]

Passage: "Every Thursday at 3pm I have a standup, but next Thursday it's cancelled."
Extractions: [
  {"surface": "Every Thursday at 3pm", "kind_guess": "recurrence", "context_hint": "weekly Thursday at 15:00"},
  {"surface": "next Thursday", "kind_guess": "instant", "context_hint": "upcoming Thursday (cancelled instance)"}
]

Passage: "Sometime in the first week of July 2020, we moved to the city."
Extractions: [{"surface": "the first week of July 2020", "kind_guess": "interval", "context_hint": "first calendar week of July 2020"}]

--- Axis-only (bare month / season / quarter / part-of-day / weekend-weekday) examples ---

Passage: "March is always chaotic for me."
Extractions: [{"surface": "March", "kind_guess": "recurrence", "context_hint": "month of March, any year (axis-only: bare month without year)"}]

Passage: "I tend to travel a lot in December."
Extractions: [{"surface": "December", "kind_guess": "recurrence", "context_hint": "month of December, any year (axis-only)"}]

Passage: "Q2 is our busiest quarter."
Extractions: [{"surface": "Q2", "kind_guess": "recurrence", "context_hint": "second quarter (Apr-Jun), any year (axis-only)"}]

Passage: "Every summer the whole family goes camping."
Extractions: [{"surface": "Every summer", "kind_guess": "recurrence", "context_hint": "summer months recurring yearly"}]

Passage: "I hate winter — it drags on forever."
Extractions: [{"surface": "winter", "kind_guess": "recurrence", "context_hint": "winter season, any year (axis-only)"}]

Passage: "Autumn is my favorite — cool air, good light."
Extractions: [{"surface": "Autumn", "kind_guess": "recurrence", "context_hint": "autumn/fall season, any year (axis-only)"}]

Passage: "Tuesday evenings are reserved for book club."
Extractions: [
  {"surface": "Tuesday evenings", "kind_guess": "recurrence", "context_hint": "weekly Tuesday in the evening"}
]

Passage: "I usually go running in the morning."
Extractions: [{"surface": "in the morning", "kind_guess": "recurrence", "context_hint": "mornings (axis-only: part-of-day without date)"}]

Passage: "Afternoons are when I'm most productive."
Extractions: [{"surface": "Afternoons", "kind_guess": "recurrence", "context_hint": "afternoons (axis-only: part-of-day)"}]

Passage: "We brunch on weekends."
Extractions: [{"surface": "weekends", "kind_guess": "recurrence", "context_hint": "Saturdays and Sundays (axis-only: weekend)"}]

Passage: "Weekday mornings I take the train."
Extractions: [{"surface": "Weekday mornings", "kind_guess": "recurrence", "context_hint": "Mon-Fri in the morning (axis-only: weekday + part-of-day)"}]

Passage: "June weekends are for the beach."
Extractions: [{"surface": "June weekends", "kind_guess": "recurrence", "context_hint": "Sat-Sun in the month of June (axis-only: month + weekend)"}]

Passage: "October is apple-harvest season up north."
Extractions: [{"surface": "October", "kind_guess": "recurrence", "context_hint": "month of October, any year (axis-only)"}]"""


# ---------------------------------------------------------------------------
# Pass-1 system prompt — v2 base with axis-allow section replacing the
# season-block rule.
# ---------------------------------------------------------------------------
PASS1_SYSTEM_V2P = f"""You are a meticulous temporal-reference extractor.

Your job: identify EVERY temporal reference in a passage. A temporal
reference is any span that refers to a moment, span, duration, or recurring
pattern in time. It can be absolute ("March 5, 2026"), relative
("yesterday", "2 weeks ago"), vague ("around 2010", "a decade ago"), or
recurring ("every Thursday at 3pm"). It can also be AXIS-ONLY — a bare
month, season, quarter, part-of-day, or weekday/weekend category word that
identifies a periodic dimension even without an anchoring year.

{TRIGGER_GAZETTEER}

AXIS-ONLY SURFACES (ALLOWED — always extract these when they appear,
even without a year):
- Bare month names: "January"..."December", "Jan", "Feb", "Mar", ...
  (Treat as kind=recurrence describing the month axis, any year.)
- Quarters: "Q1", "Q2", "Q3", "Q4", "first quarter", "last quarter".
- Seasons: "spring", "summer", "autumn", "fall", "winter" (any casing).
- Parts of day: "morning", "afternoon", "evening", "night", "dawn",
  "dusk", and their plurals ("mornings", "evenings"), and "in the
  morning/afternoon/evening/night".
- Weekend/weekday category: "weekend", "weekends", "weekday", "weekdays".
- Compound axes: "Tuesday evenings", "June weekends", "weekday mornings",
  "summer Saturdays" — extract as a single compound axis recurrence.

For each reference, output:
- surface: the exact substring from the passage, verbatim, with no edits
  to casing, spacing, or punctuation. Prefer the LONGEST natural phrase
  that carries the temporal meaning — include determiners like "the",
  "every", qualifiers like "earlier", "later", "around", "about",
  "the first week of" when they are part of the phrase. Do NOT include a
  leading bare "on"/"in" when it is just a preposition attaching to the
  phrase (except for "in the morning/afternoon/evening/night" idioms
  where the whole phrase IS the temporal reference).
- kind_guess: one of [instant, interval, duration, recurrence].
  - instant: a point-in-time: "yesterday", "2015", "last month".
  - interval: a start-to-end range: "from X to Y".
  - duration: an unanchored length: "for 3 weeks".
  - recurrence: a recurring pattern OR an axis-only reference
    ("every Thursday", "March", "summer", "weekends", "afternoons").
- context_hint: a short (<=12 word) note of what it refers to; if
  axis-only, say so explicitly (e.g. "axis-only: month of March").

Do NOT emit: "once", "often", "always", bare "recent"/"old"/"new" with
no concrete referent.

{FEW_SHOT_EXAMPLES_V2P}

Output a single JSON object: {{"refs": [...]}}. If none, output {{"refs": []}}.
"""


# ---------------------------------------------------------------------------
# Pass-2 system prompt — v2 base + explicit rules for axis-only recurrences.
# We keep the schema unchanged and instruct the LLM to emit YEARLY/DAILY/
# WEEKLY recurrences with BYMONTH/BYHOUR/BYDAY when the surface is
# axis-only.
# ---------------------------------------------------------------------------
PASS2_SYSTEM_V2P = """You resolve ONE temporal reference into absolute wall-clock form.

Reference time is given; use it to resolve all relative expressions.
Compute carefully - check weekday alignment, month length, year rollovers.

Output JSON matching this schema exactly. Omit fields not relevant to the
kind by setting them to null.

{
  "kind": "instant" | "interval" | "duration" | "recurrence",
  "surface": string,
  "confidence": float in [0,1],
  "instant": { "earliest": ISO, "latest": ISO, "best": ISO|null, "granularity": string } | null,
  "interval": { "start": {...instant...}, "end": {...instant...} } | null,
  "duration": { "seconds": int } | null,
  "recurrence": {
    "rrule": string,
    "dtstart": {...instant...},
    "until": {...instant...} | null,
    "exdates": [{...instant...}]
  } | null
}

Granularity is one of: second, minute, hour, day, week, month, quarter, year, decade, century.

Rules:
- Use UTC ISO 8601 with "Z" suffix for all datetimes.
- earliest is inclusive, latest is exclusive.
- For "about"/"around"/"roughly"/"a few"/"a couple", widen the interval by one granularity level; set best to the centered estimate.
- "yesterday" at ref R -> day bracket of R-1d, granularity=day.
- "N weeks ago" at ref R -> day bracket of R-7N days, granularity=day.
- "last week" -> week bracket of the ISO week before R, granularity=week.
- "last month" -> MONTH bracket of R.month-1 (wrapping year if needed), granularity=month.
- "this month" -> MONTH bracket of R.month, granularity=month.
- "last year" -> year bracket of R.year-1, granularity=year.
- "this year" -> year bracket of R.year, granularity=year.
- "in YYYY" -> year bracket of YYYY, granularity=year.
- "the 90s"/"the 1990s" -> decade bracket [1990, 2000), granularity=decade.
- "around YYYY" -> year bracket of YYYY plus +/- 2 years, granularity=year.
- "the first week of <month> <year>" -> interval starting the first day of that month.
- For recurrences "every Thursday": rrule "FREQ=WEEKLY;BYDAY=TH". No DTSTART inside the rrule string.

AXIS-ONLY RESOLUTION (CRITICAL):

When the surface is a bare month, season, quarter, part-of-day, weekend/
weekday, or a compound of these WITHOUT a year anchor, emit a recurrence
(kind="recurrence") that encodes the axis constraint via the RRULE. Set
dtstart.earliest = dtstart.latest = the reference time (we only need
dtstart as an anchor; the RRULE does the work). Set dtstart.best = null.
Set granularity appropriately (month / quarter / season / hour / week).
Set until=null. The BY-clauses below make the scorer concentrate
probability on the correct axis dimension.

- Bare month ("March", "Dec"): rrule "FREQ=YEARLY;BYMONTH=<n>" where n is
  the 1-12 month number. granularity=month.
- Multiple bare months or a quarter ("Q2"): rrule "FREQ=YEARLY;BYMONTH=<a>,<b>,<c>".
  - Q1 -> BYMONTH=1,2,3;  Q2 -> BYMONTH=4,5,6
  - Q3 -> BYMONTH=7,8,9;  Q4 -> BYMONTH=10,11,12
  granularity=quarter.
- Season: rrule "FREQ=YEARLY;BYMONTH=<months>".
  - spring -> BYMONTH=3,4,5
  - summer -> BYMONTH=6,7,8
  - autumn / fall -> BYMONTH=9,10,11
  - winter -> BYMONTH=12,1,2
  granularity=month.
- Part-of-day: rrule "FREQ=DAILY;BYHOUR=<hours>".
  - morning -> BYHOUR=6,7,8,9,10,11
  - afternoon -> BYHOUR=12,13,14,15,16,17
  - evening -> BYHOUR=18,19,20,21
  - night -> BYHOUR=22,23,0,1,2,3,4,5
  - dawn -> BYHOUR=5,6
  - dusk -> BYHOUR=18,19
  granularity=hour.
- Weekend: rrule "FREQ=WEEKLY;BYDAY=SA,SU". granularity=week.
- Weekday: rrule "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR". granularity=week.
- Compound axes combine BY-clauses:
  - "Tuesday evenings" -> rrule "FREQ=WEEKLY;BYDAY=TU;BYHOUR=18,19,20,21".
  - "June weekends" -> rrule "FREQ=YEARLY;BYMONTH=6;BYDAY=SA,SU".
  - "weekday mornings" -> rrule "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=6,7,8,9,10,11".
  - "summer Saturdays" -> rrule "FREQ=YEARLY;BYMONTH=6,7,8;BYDAY=SA".

For ALL recurrence dtstart, fill instant.earliest and instant.latest with
valid ISO datetimes: use the reference time as both earliest and latest
(or earliest = ref, latest = ref + 1 second). dtstart.best = null.

For non-axis-only recurrences ("every Thursday at 3pm"), set dtstart to
the first matching instance on/after the reference time, granularity=day.

For recurrences with no explicit end, set until to null.
Always fill the instant.earliest/latest/best triple for non-recurrence
kinds. Best is the typical point estimate.
"""


class ExtractorV2P(BaseImprovedExtractor):
    """v2-prime: axis-aware single-pass extractor."""

    VERSION = 2  # share extractor_shared_pass2 cache; use our own subdir for pass1.

    def __init__(self, concurrency: int = 5, **kwargs: Any) -> None:
        super().__init__(
            concurrency=concurrency, cache_subdir="extractor_v2p", **kwargs
        )

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        ctx = full_ref_context(ref_time)
        user = f'{ctx}\n\nPassage:\n{text}\n\nReturn {{"refs": [...]}} as JSON.'
        data = await self._call_json(PASS1_SYSTEM_V2P, user, max_completion_tokens=4000)
        if not isinstance(data, dict):
            return []
        return list(data.get("refs", []))

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
        """Override to use v2p Pass-2 prompt (axis-aware). Keeps cache
        share disabled for Pass-2 since prompt differs from v2."""
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
            PASS2_SYSTEM_V2P,
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
                PASS2_SYSTEM_V2P,
                user2,
                json_schema=PASS2_JSON_SCHEMA,
                max_completion_tokens=3000,
                share_cache=False,
            )
            try:
                return json.loads(raw2)
            except json.JSONDecodeError:
                return None
