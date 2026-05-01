"""Adversarial stress-test corpus generator.

Generates hand-authored docs + queries covering the categories in
ADVERSARIAL.md (A1-A9 extraction, R1-R7 representation, S1-S8 retrieval).

Each entry carries a `category` field. Docs have `gold_extraction`
descriptions (textual / semi-structured). Queries have `gold_retrieval`
lists of doc_ids they should retrieve in top-5, and `expected_behavior`
describing what correct behavior looks like (especially for A7/A8/S3
where "correct" = refrain / return empty).

Writes:
    data/adversarial_docs.jsonl
    data/adversarial_queries.jsonl
    data/adversarial_gold.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# A Thursday — A4 examples use this ref_time heavily.
REF_TIME_DEFAULT = "2026-04-23T12:00:00Z"  # Thursday 2026-04-23


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------
DOCS: list[dict] = [
    # ----- A1 — Self-anchored / embedded reference time -----
    {
        "doc_id": "adv_a1_0",
        "category": "A1",
        "text": "Alice told me yesterday that she would be gone next week.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Two refs: 'yesterday' -> 2026-04-22; 'next week' is relative to yesterday's speaker perspective (2026-04-26..05-03), NOT relative to utterance-time next week.",
        "expected_behavior": "extractor should emit 'yesterday' and 'next week'; ideally 'next week' anchored to yesterday — but current system resolves to utterance ref_time's next week (2026-04-27..05-04). Accept either resolution; flag if 'next week' is omitted.",
    },
    {
        "doc_id": "adv_a1_1",
        "category": "A1",
        "text": "In 2020, my sister said 'this month has been rough'.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Two refs: '2020' (year), 'this month' which inside a 2020-embedded quote resolves to SOME month in 2020, not April 2026.",
        "expected_behavior": "extractor SHOULD emit '2020'; 'this month' inside the quote is ambiguous — if emitted, should resolve to some 2020 month.",
    },
    {
        "doc_id": "adv_a1_2",
        "category": "A1",
        "text": "When I saw her at the conference, it was pouring rain. A week later, I got sick.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'A week later' is relative to 'when I saw her at the conference', not utterance time. The conference is an unresolved event anchor.",
        "expected_behavior": "extractor SHOULD emit 'a week later' as relational; Allen extractor should bind it to 'the conference' as anchor.",
    },
    # ----- A2 — Compositional relative expressions -----
    {
        "doc_id": "adv_a2_0",
        "category": "A2",
        "text": "Three weeks after my birthday last year we finally closed on the house.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Compositional: 'my birthday last year' + '+3 weeks'. Birthday is an event anchor; 'last year' scopes it.",
        "expected_behavior": "extractor SHOULD emit 'three weeks after my birthday last year'; proper resolution requires binding 'my birthday' + resolving to 2025 + adding 3w.",
    },
    {
        "doc_id": "adv_a2_1",
        "category": "A2",
        "text": "The Thursday of the week after next, we have the kickoff.",
        "ref_time": REF_TIME_DEFAULT,  # ref_time is Thursday 2026-04-23
        "gold_extraction": "'the week after next' = week of 2026-05-04..05-10; its Thursday = 2026-05-07.",
        "expected_behavior": "extractor SHOULD resolve to Thursday 2026-05-07 at day granularity.",
    },
    {
        "doc_id": "adv_a2_2",
        "category": "A2",
        "text": "Two days before three weeks from now, we'll ship the beta.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "3 weeks from 2026-04-23 = 2026-05-14; minus 2 days = 2026-05-12.",
        "expected_behavior": "extractor SHOULD resolve to 2026-05-12; composition risks arithmetic error.",
    },
    {
        "doc_id": "adv_a2_3",
        "category": "A2",
        "text": "The month after my first anniversary we moved into the new place.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Compositional on event anchor 'my first anniversary'. Cannot resolve without event-resolver.",
        "expected_behavior": "extractor SHOULD emit the phrase as relational; resolution likely fails.",
    },
    # ----- A3 — Fuzzy modifier edge cases -----
    {
        "doc_id": "adv_a3_0",
        "category": "A3",
        "text": "A couple of years ago we adopted a cat from the shelter.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'a couple of years ago' -> roughly 2023-2024, wide bracket. Expected granularity=year.",
        "expected_behavior": "extractor should emit with widened bracket covering 2023-2024.",
    },
    {
        "doc_id": "adv_a3_1",
        "category": "A3",
        "text": "A few weeks back I bumped into my old professor at the bookstore.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'a few weeks back' ~ 3-5 weeks. Expected bracket roughly 2026-03-19..2026-04-02.",
        "expected_behavior": "extractor should emit widened bracket ~3-5 weeks back.",
    },
    {
        "doc_id": "adv_a3_2",
        "category": "A3",
        "text": "Not long ago, Dana called to say she was moving abroad.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'Not long ago' is nearly undefined. Any bracket from 'days' to 'months' is defensible.",
        "expected_behavior": "extractor may emit vague past bracket or may skip. Either is defensible.",
    },
    {
        "doc_id": "adv_a3_3",
        "category": "A3",
        "text": "Back in the day, we used to swim in the quarry until the sheriff came.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'Back in the day' is a personal era; decades-ago bracket defensible.",
        "expected_behavior": "extractor should emit (likely via era_extractor) with decade-level granularity.",
    },
    # ----- A4 — Same-day / weekday ambiguity (ref_time = Thursday 2026-04-23) -----
    {
        "doc_id": "adv_a4_0",
        "category": "A4",
        "text": "I'll see you Thursday for drinks.",
        "ref_time": REF_TIME_DEFAULT,  # Thursday
        "gold_extraction": "'Thursday' on a Thursday is ambiguous: today or next Thursday (2026-04-30)? Convention: usually next Thursday if the day hasn't happened / if future tense.",
        "expected_behavior": "extractor should emit Thursday; either 2026-04-23 or 2026-04-30 is defensible.",
    },
    {
        "doc_id": "adv_a4_1",
        "category": "A4",
        "text": "Last Thursday we finally got the report approved.",
        "ref_time": REF_TIME_DEFAULT,  # Thursday
        "gold_extraction": "'Last Thursday' on a Thursday could mean today or 7 days ago (2026-04-16). Convention: a week ago.",
        "expected_behavior": "extractor should emit 'last Thursday'; 2026-04-16 preferred.",
    },
    {
        "doc_id": "adv_a4_2",
        "category": "A4",
        "text": "Next Tuesday I'll finally finish the tax paperwork.",
        "ref_time": REF_TIME_DEFAULT,  # Thursday 2026-04-23
        "gold_extraction": "'Next Tuesday' from a Thursday: 5 days away (2026-04-28) or 12 days away (2026-05-05)?",
        "expected_behavior": "extractor should emit Tuesday; 2026-04-28 preferred (nearest upcoming).",
    },
    {
        "doc_id": "adv_a4_3",
        "category": "A4",
        "text": "This weekend we're driving up to the cabin.",
        "ref_time": REF_TIME_DEFAULT,  # Thursday 2026-04-23
        "gold_extraction": "'This weekend' on a Thursday = the upcoming Sat-Sun, 2026-04-25..2026-04-26.",
        "expected_behavior": "extractor should emit 2026-04-25..2026-04-26 weekend.",
    },
    # ----- A5 — Temporal references to unknown entities -----
    {
        "doc_id": "adv_a5_0",
        "category": "A5",
        "text": "Since the divorce, life has been surprisingly calm.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Event anchor 'the divorce' with no date. Allen-style: 'after the divorce'.",
        "expected_behavior": "Allen extractor should emit relation=after, anchor='the divorce'.",
    },
    {
        "doc_id": "adv_a5_1",
        "category": "A5",
        "text": "Post-lockdown I picked up running in the mornings again.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'Post-lockdown' is an era (COVID 2020-2021 lockdown most likely).",
        "expected_behavior": "era_extractor should resolve post-2020 interval; or Allen extractor should pick relation=after.",
    },
    {
        "doc_id": "adv_a5_2",
        "category": "A5",
        "text": "Before the move, we had to pack twenty boxes of books.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Event anchor 'the move' with no date.",
        "expected_behavior": "Allen extractor should emit relation=before, anchor='the move'.",
    },
    {
        "doc_id": "adv_a5_3",
        "category": "A5",
        "text": "When I was in college I took a class on classical Latin.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Personal era 'when I was in college'. Era_extractor should rough-bracket.",
        "expected_behavior": "era_extractor should emit a fuzzy 4-year interval with low confidence.",
    },
    # ----- A6 — Non-standard recurrence cycles -----
    {
        "doc_id": "adv_a6_0",
        "category": "A6",
        "text": "I get the allergy shot every 13 days.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Recurrence with interval=13 days. rrule: FREQ=DAILY;INTERVAL=13.",
        "expected_behavior": "extractor should emit recurrence with non-standard interval; rrule likely malformed.",
    },
    {
        "doc_id": "adv_a6_1",
        "category": "A6",
        "text": "Every other Thursday we have the senior team sync.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Biweekly Thursday. rrule: FREQ=WEEKLY;INTERVAL=2;BYDAY=TH.",
        "expected_behavior": "extractor should emit biweekly; risk of treating as plain weekly.",
    },
    {
        "doc_id": "adv_a6_2",
        "category": "A6",
        "text": "The last Monday of each month is the retrospective.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "rrule: FREQ=MONTHLY;BYDAY=-1MO.",
        "expected_behavior": "extractor should emit last-Monday-of-month rrule.",
    },
    {
        "doc_id": "adv_a6_3",
        "category": "A6",
        "text": "I have PT every Tuesday and Thursday.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "rrule: FREQ=WEEKLY;BYDAY=TU,TH.",
        "expected_behavior": "extractor should emit multi-day weekly rrule.",
    },
    # ----- A7 — Fictional / hypothetical contexts -----
    {
        "doc_id": "adv_a7_0",
        "category": "A7",
        "text": "In the novel I'm reading, the story is set in 1850.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Fictional '1850' — should NOT be emitted as a real temporal reference about the narrator.",
        "expected_behavior": "extractor SHOULD refrain (or emit with very low confidence). Emitting 1850 as a real timestamp is a failure.",
    },
    {
        "doc_id": "adv_a7_1",
        "category": "A7",
        "text": "What if I had been born in 1980? How different things would be.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Hypothetical '1980' — not a real birth date.",
        "expected_behavior": "extractor should refrain; emitting 1980 as a fact is wrong.",
    },
    {
        "doc_id": "adv_a7_2",
        "category": "A7",
        "text": "Imagine a world where the year is 2089 and cars can fly.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Hypothetical '2089' — no real event.",
        "expected_behavior": "extractor should refrain; emitting 2089 confidently is wrong.",
    },
    # ----- A8 — Tense + aspect shifts -----
    {
        "doc_id": "adv_a8_0",
        "category": "A8",
        "text": "I had been living in Boston since 2015 before I finally moved west.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Past-perfect continuous. Interval [2015, <move date>). Move date unknown.",
        "expected_behavior": "extractor should emit '2015' as interval-start; granularity=year.",
    },
    {
        "doc_id": "adv_a8_1",
        "category": "A8",
        "text": "I will have finished the manuscript by next Tuesday.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Future-perfect. 'by next Tuesday' = deadline 2026-04-28. Should be emitted as point-in-time deadline.",
        "expected_behavior": "extractor should emit 'next Tuesday' (2026-04-28). The 'will have' aspect is a deadline, not a past fact.",
    },
    {
        "doc_id": "adv_a8_2",
        "category": "A8",
        "text": "I was going to go to the gala last week, but the flight was cancelled.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Past unrealized future. 'last week' refers to a planned but unattended event.",
        "expected_behavior": "extractor should emit 'last week' as interval but ideally flag polarity=negated.",
    },
    # ----- A9 — Era / holiday references -----
    {
        "doc_id": "adv_a9_0",
        "category": "A9",
        "text": "During Ramadan last year I reorganized my sleep schedule.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Ramadan 2025: approximately 2025-02-28..2025-03-29. Varies annually.",
        "expected_behavior": "era_extractor may not resolve; gpt-5-mini may know Ramadan 2025 dates.",
    },
    {
        "doc_id": "adv_a9_1",
        "category": "A9",
        "text": "Easter 2015 was the last time we all gathered at grandma's.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Easter 2015 = 2015-04-05. Variable date.",
        "expected_behavior": "extractor should resolve to 2015-04-05; risk is generic '2015'.",
    },
    {
        "doc_id": "adv_a9_2",
        "category": "A9",
        "text": "Before the last World Cup, I barely watched any soccer.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Last FIFA World Cup = Qatar 2022 (Nov-Dec 2022).",
        "expected_behavior": "era_extractor likely fails; emission is 'the last World Cup' as event anchor only.",
    },
    {
        "doc_id": "adv_a9_3",
        "category": "A9",
        "text": "During Chinese New Year we made dumplings for the whole family.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Most recent CNY = 2026-02-17. Varies annually. Could be any year.",
        "expected_behavior": "era_extractor may fail; likely emits vague bracket.",
    },
    # ----- R1 — Massive span vs point-interval -----
    {
        "doc_id": "adv_r1_0",
        "category": "R1",
        "text": "I was married from 2015 to 2023.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Interval 2015-01-01..2023-01-01. 8-year span.",
        "expected_behavior": "extractor should emit interval; representation risk is over-retrieving day-specific queries.",
    },
    {
        "doc_id": "adv_r1_1",
        "category": "R1",
        "text": "On April 5, 2019 we celebrated Mira's third birthday at the park.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Day-level instant 2019-04-05.",
        "expected_behavior": "extractor should emit day-granularity.",
    },
    # ----- R2 — Recurrence density skew -----
    {
        "doc_id": "adv_r2_0",
        "category": "R2",
        "text": "I've been going to book club every Thursday for the past 5 years.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Recurrence weekly Thursday; ~260 instances over 5 years.",
        "expected_behavior": "extractor should emit recurrence; representation risk is ~260 intervals.",
    },
    {
        "doc_id": "adv_r2_1",
        "category": "R2",
        "text": "I had book club on April 4, 2024.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Single instant 2024-04-04.",
        "expected_behavior": "extractor should emit single day.",
    },
    # ----- R3 — Degenerate / zero-width intervals -----
    {
        "doc_id": "adv_r3_0",
        "category": "R3",
        "text": "Right now I'm drafting the email.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'Right now' = ref_time as zero-width instant. Granularity=minute at most.",
        "expected_behavior": "extractor should emit 'right now' or skip. If emitted, bracket may be zero-width.",
    },
    {
        "doc_id": "adv_r3_1",
        "category": "R3",
        "text": "Just then the alarm went off.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Refers to an unspecified recent moment. 'just then' may be skipped per gazetteer rules.",
        "expected_behavior": "extractor may skip 'just then'. Defensible either way.",
    },
    # ----- R4 — Infinite / open-ended references -----
    {
        "doc_id": "adv_r4_0",
        "category": "R4",
        "text": "Since 1990 I've kept a journal almost every day.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Open-ended interval [1990, now). latest should be ref_time (or +∞).",
        "expected_behavior": "extractor should emit interval starting 1990; latest may be set to ref_time.",
    },
    {
        "doc_id": "adv_r4_1",
        "category": "R4",
        "text": "I plan to keep this house for the rest of my life.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Open-ended forward interval [now, +∞).",
        "expected_behavior": "extractor may skip; emission risks a finite fake latest.",
    },
    {
        "doc_id": "adv_r4_2",
        "category": "R4",
        "text": "For as long as I can remember, Dad has drunk his coffee black.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Open-ended backward interval [birth, now).",
        "expected_behavior": "extractor may skip; emission risks arbitrary earliest.",
    },
    # ----- R5 — Paraphrastic / non-standard synonyms -----
    {
        "doc_id": "adv_r5_0",
        "category": "R5",
        "text": "Last spring we finally planted the apple trees.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "'Last spring' = March-May 2025 (season convention).",
        "expected_behavior": "extractor should emit spring 2025 interval; gazetteer currently suppresses seasons without a year.",
    },
    {
        "doc_id": "adv_r5_1",
        "category": "R5",
        "text": "During the Obama years I finished grad school.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Obama years = 2009-2017. Era.",
        "expected_behavior": "era_extractor should emit 2009-01-20..2017-01-20 interval.",
    },
    {
        "doc_id": "adv_r5_2",
        "category": "R5",
        "text": "Halloween 2022 was the night everything went sideways.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Halloween 2022 = 2022-10-31. Fixed date.",
        "expected_behavior": "extractor should resolve to 2022-10-31 at day granularity.",
    },
    # ----- R6 — Multi-cycle recurrence -----
    {
        "doc_id": "adv_r6_0",
        "category": "R6",
        "text": "The third Saturday of every odd-numbered month in 2024 we have the open house.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Multi-cycle: Jan, Mar, May, Jul, Sep, Nov 2024; 3rd Saturday each. Cannot express as single rrule.",
        "expected_behavior": "extractor's rrule almost certainly incomplete or malformed.",
    },
    # ----- R7 — Temporal duration without anchor -----
    {
        "doc_id": "adv_r7_0",
        "category": "R7",
        "text": "The meeting was a 3-hour slog with no breaks.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Duration only (3 hours). No anchor.",
        "expected_behavior": "extractor should emit kind=duration. Retrieval skips duration-only by design.",
    },
    {
        "doc_id": "adv_r7_1",
        "category": "R7",
        "text": "The project lasted 2 weeks before it was scrapped.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Duration only (2 weeks). No anchor given.",
        "expected_behavior": "extractor should emit kind=duration.",
    },
    # ----- S-category supporting docs (retrieval bed for S1, S4, S5, S6, S8) -----
    {
        "doc_id": "adv_s1_year",
        "category": "S1-bed",
        "text": "In 2024 we relocated to a smaller apartment.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Year-granularity 2024.",
        "expected_behavior": "baseline extraction.",
    },
    {
        "doc_id": "adv_s1_month",
        "category": "S1-bed",
        "text": "In March 2024 we signed the lease for the new apartment.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Month-granularity March 2024.",
        "expected_behavior": "baseline extraction.",
    },
    {
        "doc_id": "adv_s1_day",
        "category": "S1-bed",
        "text": "On March 15, 2024 we handed the old landlord the keys.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Day-granularity 2024-03-15.",
        "expected_behavior": "baseline extraction.",
    },
    {
        "doc_id": "adv_s4_source",
        "category": "S4",
        "text": "I texted Priya last Thursday and then we met up 3 days ago.",
        "ref_time": "2026-04-21T12:00:00Z",  # Tuesday
        "gold_extraction": "'last Thursday' from a Tuesday = 2026-04-16; '3 days ago' = 2026-04-18. Different dates; cross-check required.",
        "expected_behavior": "extractor emits two references; temporal coherence risk.",
    },
    {
        "doc_id": "adv_s5_hour",
        "category": "S5",
        "text": "Five years ago at 3pm we cut the ribbon on the store.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "2021-04-23 at 15:00 UTC (hour preserved).",
        "expected_behavior": "extractor should preserve hour=15.",
    },
    {
        "doc_id": "adv_s5_date",
        "category": "S5",
        "text": "Five years ago today we closed the acquisition.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "2021-04-23 (date-of-year preserved).",
        "expected_behavior": "extractor should preserve month+day, shift year.",
    },
    {
        "doc_id": "adv_s6_tiny",
        "category": "S6",
        "text": "Thirty seconds ago I clicked the send button.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "second-granularity point 30 seconds before ref_time.",
        "expected_behavior": "extractor should emit with second granularity.",
    },
    {
        "doc_id": "adv_s6_huge",
        "category": "S6",
        "text": "Back in the 15th century, printing changed everything.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Century-granularity 1400-01-01..1500-01-01.",
        "expected_behavior": "extractor should emit century bracket.",
    },
    {
        "doc_id": "adv_s8_meet",
        "category": "S8",
        "text": "I first met my wife at a software conference in Austin.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "No time expression. This is the event a 'year I met my wife' query should surface via non-temporal evidence.",
        "expected_behavior": "extractor emits nothing (correct). Retrieval must cross-reference.",
    },
    {
        "doc_id": "adv_s8_2018",
        "category": "S8",
        "text": "In 2018 I flew to Austin for the first time, for a conference.",
        "ref_time": REF_TIME_DEFAULT,
        "gold_extraction": "Year 2018; non-temporal link via 'Austin' + 'conference' to adv_s8_meet.",
        "expected_behavior": "extractor emits '2018'. Retrieval needs cross-doc reasoning to tie to meet doc.",
    },
]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
QUERIES: list[dict] = [
    # ----- A1 -----
    {
        "query_id": "q_a1_0",
        "category": "A1",
        "text": "What did Alice tell me about being gone?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a1_0"],
        "expected_behavior": "semantic should find it; temporal scoring is noisy here.",
    },
    {
        "query_id": "q_a1_1",
        "category": "A1",
        "text": "What happened a week after I saw her at the conference?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a1_2"],
        "expected_behavior": "Allen-relation query; event anchor 'the conference'.",
    },
    # ----- A2 -----
    {
        "query_id": "q_a2_0",
        "category": "A2",
        "text": "What happened on May 7, 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a2_1"],
        "expected_behavior": "Requires compositional resolution of 'Thursday of the week after next' to 2026-05-07.",
    },
    {
        "query_id": "q_a2_1",
        "category": "A2",
        "text": "What happened on May 12, 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a2_2"],
        "expected_behavior": "Requires compositional arithmetic to 2026-05-12.",
    },
    # ----- A3 -----
    {
        "query_id": "q_a3_0",
        "category": "A3",
        "text": "When did we adopt the cat?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a3_0"],
        "expected_behavior": "semantic match likely dominates.",
    },
    {
        "query_id": "q_a3_1",
        "category": "A3",
        "text": "What happened in early April 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a3_1"],
        "expected_behavior": "'A few weeks back' from 2026-04-23 ~ late March/early April — should overlap.",
    },
    # ----- A4 -----
    {
        "query_id": "q_a4_0",
        "category": "A4",
        "text": "What happened on April 30, 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a4_0"],
        "expected_behavior": "If 'Thursday' resolves to next Thursday (2026-04-30), retrieval should hit.",
    },
    {
        "query_id": "q_a4_1",
        "category": "A4",
        "text": "What happened on April 16, 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a4_1"],
        "expected_behavior": "'last Thursday' should resolve to 2026-04-16.",
    },
    {
        "query_id": "q_a4_2",
        "category": "A4",
        "text": "What happened on April 28, 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a4_2"],
        "expected_behavior": "'next Tuesday' should resolve to 2026-04-28.",
    },
    {
        "query_id": "q_a4_3",
        "category": "A4",
        "text": "What did we do the weekend of April 25, 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a4_3"],
        "expected_behavior": "'This weekend' should resolve to 2026-04-25..26.",
    },
    # ----- A5 -----
    {
        "query_id": "q_a5_0",
        "category": "A5",
        "text": "What happened after the divorce?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a5_0"],
        "expected_behavior": "Allen relation=after with unresolved event anchor.",
    },
    {
        "query_id": "q_a5_1",
        "category": "A5",
        "text": "What did I do post-lockdown?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a5_1"],
        "expected_behavior": "Semantic + era match.",
    },
    {
        "query_id": "q_a5_2",
        "category": "A5",
        "text": "What did I study in college?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a5_3"],
        "expected_behavior": "Semantic; era query.",
    },
    # ----- A6 -----
    {
        "query_id": "q_a6_0",
        "category": "A6",
        "text": "What do I do every other Thursday?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a6_1"],
        "expected_behavior": "Recurrence query.",
    },
    {
        "query_id": "q_a6_1",
        "category": "A6",
        "text": "What is the retrospective schedule?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a6_2"],
        "expected_behavior": "semantic dominates.",
    },
    # ----- A7 (expected empty / low score) -----
    {
        "query_id": "q_a7_0",
        "category": "A7",
        "text": "What happened in 1850?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": [],  # nothing really happened; if extractor emitted adv_a7_0 as 1850, that's a false positive
        "expected_behavior": "Retrieval SHOULD return empty (or low). If adv_a7_0 ranks high, the extractor emitted fictional 1850 as real — this is a failure.",
    },
    {
        "query_id": "q_a7_1",
        "category": "A7",
        "text": "Was I born in 1980?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": [],
        "expected_behavior": "Retrieval SHOULD not confidently match adv_a7_1.",
    },
    # ----- A8 -----
    {
        "query_id": "q_a8_0",
        "category": "A8",
        "text": "What will I finish by next Tuesday?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a8_1"],
        "expected_behavior": "'next Tuesday' resolution + semantic.",
    },
    {
        "query_id": "q_a8_1",
        "category": "A8",
        "text": "What did I miss last week?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a8_2"],
        "expected_behavior": "'last week' overlap. Polarity-aware retrieval would score this differently.",
    },
    # ----- A9 -----
    {
        "query_id": "q_a9_0",
        "category": "A9",
        "text": "What did I do during Ramadan 2025?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a9_0"],
        "expected_behavior": "Era resolution + semantic.",
    },
    {
        "query_id": "q_a9_1",
        "category": "A9",
        "text": "What happened on April 5, 2015?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_a9_1"],
        "expected_behavior": "Requires Easter 2015 resolution to 2015-04-05.",
    },
    # ----- R1 -----
    {
        "query_id": "q_r1_0",
        "category": "R1",
        "text": "What happened on April 5, 2019?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r1_1", "adv_r1_0"],
        "expected_behavior": "Day-specific doc adv_r1_1 should rank ABOVE wide-interval adv_r1_0. Tests granularity ranking.",
    },
    # ----- R2 -----
    {
        "query_id": "q_r2_0",
        "category": "R2",
        "text": "What book club event happened on April 4, 2024?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r2_1", "adv_r2_0"],
        "expected_behavior": "Single-instance doc should NOT be buried under recurrence.",
    },
    # ----- R3 -----
    {
        "query_id": "q_r3_0",
        "category": "R3",
        "text": "What are you doing right now?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r3_0"],
        "expected_behavior": "Zero-width interval matching risk.",
    },
    # ----- R4 -----
    {
        "query_id": "q_r4_0",
        "category": "R4",
        "text": "What have I been doing since 1995?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r4_0"],
        "expected_behavior": "Open-ended interval matching.",
    },
    # ----- R5 -----
    {
        "query_id": "q_r5_0",
        "category": "R5",
        "text": "What did I do in Q2 2025?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r5_0"],
        "expected_behavior": "'last spring' (Mar-May 2025) overlaps with Q2 (Apr-Jun). Paraphrastic match needed.",
    },
    {
        "query_id": "q_r5_1",
        "category": "R5",
        "text": "What did I do in the 2010s?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r5_1"],
        "expected_behavior": "Obama years (2009-2017) overlaps 2010s.",
    },
    {
        "query_id": "q_r5_2",
        "category": "R5",
        "text": "What happened on October 31, 2022?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r5_2"],
        "expected_behavior": "Halloween 2022 resolution.",
    },
    # ----- R6 -----
    {
        "query_id": "q_r6_0",
        "category": "R6",
        "text": "When is the open house in March 2024?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_r6_0"],
        "expected_behavior": "Multi-cycle recurrence; March's 3rd Saturday = 2024-03-16.",
    },
    # ----- R7 -----
    {
        "query_id": "q_r7_0",
        "category": "R7",
        "text": "What 2-hour meetings did I attend?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": [],  # no 2-hr doc in corpus
        "expected_behavior": "Retrieval should return empty / low-score; only 3-hour and 2-week docs exist.",
    },
    # ----- S1 -----
    {
        "query_id": "q_s1_0",
        "category": "S1",
        "text": "What happened on March 15, 2024?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_s1_day", "adv_s1_month", "adv_s1_year"],
        "expected_behavior": "Day > month > year ranking.",
    },
    # ----- S2 -----
    {
        "query_id": "q_s2_0",
        "category": "S2",
        "text": "What happened after my move?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": [
            "adv_a5_2"
        ],  # the 'before the move' doc; after-of-move is sort of undefined
        "expected_behavior": "No re-resolution at query time; Allen after-query with unresolved anchor.",
    },
    # ----- S3 — Negative queries (expected empty / low) -----
    {
        "query_id": "q_s3_0",
        "category": "S3",
        "text": "What did I NOT do last week?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": [],
        "expected_behavior": "Retrieval should not confidently return anything. Negation is not modeled.",
    },
    # ----- S4 — Coherence violation query -----
    {
        "query_id": "q_s4_0",
        "category": "S4",
        "text": "What did I do with Priya on April 16, 2026?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_s4_source"],
        "expected_behavior": "Expected date 2026-04-16 should match 'last Thursday' from Tuesday ref_time.",
    },
    # ----- S5 -----
    {
        "query_id": "q_s5_0",
        "category": "S5",
        "text": "What happened on April 23, 2021 at 3pm?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_s5_hour"],
        "expected_behavior": "Hour-preservation test.",
    },
    {
        "query_id": "q_s5_1",
        "category": "S5",
        "text": "What happened on April 23, 2021?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_s5_date", "adv_s5_hour"],
        "expected_behavior": "Date-preservation test.",
    },
    # ----- S6 -----
    {
        "query_id": "q_s6_0",
        "category": "S6",
        "text": "What happened in the last 60 seconds?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_s6_tiny"],
        "expected_behavior": "Tiny-window retrieval.",
    },
    {
        "query_id": "q_s6_1",
        "category": "S6",
        "text": "What happened in the 1400s?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_s6_huge"],
        "expected_behavior": "Huge-window retrieval.",
    },
    # ----- S7 — Multi-anchor -----
    {
        "query_id": "q_s7_0",
        "category": "S7",
        "text": "What happened between my first surgery and my second surgery?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": [],  # no corpus doc; tests the multi-anchor handling gap
        "expected_behavior": "Two event anchors; current Allen handles one. Should gracefully return nothing / semantic fallback.",
    },
    # ----- S8 — Conflated temporal + non-temporal -----
    {
        "query_id": "q_s8_0",
        "category": "S8",
        "text": "What year did I meet my wife?",
        "ref_time": REF_TIME_DEFAULT,
        "gold_retrieval": ["adv_s8_meet", "adv_s8_2018"],
        "expected_behavior": "Answer requires linking meet doc (no date) to 2018 doc (date) via shared context 'Austin conference'.",
    },
]


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------
def _build_gold() -> list[dict]:
    out: list[dict] = []
    for q in QUERIES:
        out.append(
            {
                "query_id": q["query_id"],
                "category": q["category"],
                "relevant_doc_ids": list(q.get("gold_retrieval", [])),
                "expected_behavior": q.get("expected_behavior", ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    docs_path = DATA_DIR / "adversarial_docs.jsonl"
    queries_path = DATA_DIR / "adversarial_queries.jsonl"
    gold_path = DATA_DIR / "adversarial_gold.jsonl"

    with docs_path.open("w") as f:
        for d in DOCS:
            f.write(json.dumps(d) + "\n")
    with queries_path.open("w") as f:
        for q in QUERIES:
            f.write(json.dumps(q) + "\n")
    with gold_path.open("w") as f:
        for g in _build_gold():
            f.write(json.dumps(g) + "\n")

    # Category coverage summary
    from collections import Counter

    doc_cats = Counter(d["category"] for d in DOCS)
    q_cats = Counter(q["category"] for q in QUERIES)
    print(f"Wrote {len(DOCS)} docs to {docs_path}")
    print(f"Wrote {len(QUERIES)} queries to {queries_path}")
    print(f"Wrote {len(_build_gold())} gold entries to {gold_path}")
    print("\nDoc categories:")
    for c, n in sorted(doc_cats.items()):
        print(f"  {c}: {n}")
    print("\nQuery categories:")
    for c, n in sorted(q_cats.items()):
        print(f"  {c}: {n}")
    print(f"\nTotal examples: {len(DOCS) + len(QUERIES)}")


if __name__ == "__main__":
    main()
