# Adversarial Stress Test — Where This System Should Break

The current ship-best configuration (v2 extractor + multi-axis scorer +
utterance anchor + era extraction + Allen channel for relational queries)
achieves high numbers on synthetic data. But the data is our own design.
This doc enumerates cases we expect the system to struggle with — testing
these will reveal systematic blind spots.

## Extraction-layer adversaries

### A1 — Self-anchored / embedded reference time

Natural language resolves relative time against the speaker's ref_time, but
reported speech and narrative nest inner ref_times inside outer ones.

Examples:
- "Alice told me yesterday that she would be gone next week" — "next week"
  is relative to yesterday's perspective, not today's.
- "In 2020, he said 'this month has been rough'" — "this month" is a 2020
  month, not utterance-time month.
- "When I saw her, it was pouring rain. A week later, I got sick." — "a
  week later" is relative to "when I saw her", not to utterance time.

Predicted failure: extractor resolves against utterance ref_time, not the
embedded anchor.

### A2 — Compositional relative expressions

Expressions that compose multiple relative offsets.

Examples:
- "Three weeks after my birthday last year"
- "The Thursday of the week after next"
- "Two days before three weeks from now"
- "The month after my first anniversary"

Predicted failure: partial resolution, arithmetic errors on week/month
boundaries.

### A3 — Fuzzy modifier edge cases

How much does the system widen brackets for colloquial vagueness?

Examples:
- "A couple of years ago" — 2 or 3? User's idiolect varies.
- "A few weeks back" — 3? 5?
- "Not long ago" — an hour? a day? a year?
- "Recently" — nearly undefined
- "A while back"
- "Back in the day"

Predicted failure: extractor over-tightens brackets or omits entirely.

### A4 — Same-day/weekday ambiguity

"Last Thursday" on Thursday — today or 7 days ago?

Examples (all said on Thursday 2026-04-23):
- "I'll see you Thursday" — today or next Thursday?
- "Last Thursday" — today or a week ago?
- "Next Tuesday" — 5 days away or 12 days away?
- "This weekend" — coming weekend or the one that just ended?

Predicted failure: inconsistent defaults across these; system likely picks
one interpretation silently.

### A5 — Temporal references to unknown entities

Expressions whose resolution depends on events not defined anywhere yet.

Examples:
- "Since the divorce" — when was the divorce?
- "Post-lockdown" — which lockdown? 2020? 2021?
- "Before the move" — which move?
- "When I was in college" — which years?
- "During grandma's illness"

Predicted failure: left unresolved or resolved confidently-wrongly to a
recent default.

### A6 — Non-standard recurrence cycles

Beyond FREQ=DAILY/WEEKLY/MONTHLY/YEARLY.

Examples:
- "Every 13 days"
- "Every other Thursday" (biweekly)
- "The last Monday of each month"
- "Every Tuesday and Thursday"
- "Every weekend except holidays"
- "Every quarter except Q1"

Predicted failure: rrule malformed, or the exception clauses dropped.

### A7 — Time within fictional / hypothetical contexts

- "In the story, it was set in 1850"
- "What if I had been born in 1980?"
- "Imagine a world where the year is 2089"

Predicted failure: extractor treats these as real times or omits them;
unclear which is correct.

### A8 — Tense + aspect shifts

- "I had been living there since 2015" (past perfect continuous)
- "I will have finished by next Tuesday" (future perfect)
- "I was going to go last week" (past unrealized future)

Predicted failure: tense signal lost; system treats "I will have finished
by next Tuesday" as a simple "next Tuesday" reference.

### A9 — Era/holiday references

Culture-specific or variable dates.

Examples:
- "Ramadan last year" — varies annually
- "Easter 2015" — varies
- "Before the World Cup" — which World Cup?
- "During Chinese New Year" — which year's?
- "The week of Thanksgiving" — US vs. Canada vs. Liberia?

Predicted failure: era extractor handles common English holidays; fails on
culture-specific.

## Representation-layer adversaries

### R1 — Massive span vs point-interval

Doc says "I was married from 2015 to 2023" — an 8-year interval. Query
asks "what happened on April 5, 2019?" — a single day. The interval
contains the day. Does overlap correctly rank this doc, or does it
dominate more specific day-granularity docs?

Predicted failure: 8-year interval gets matched against every day-specific
query; over-retrieves.

### R2 — Recurrence density skew

Doc A: "book club every Thursday for the past 5 years" — ~260 instances.
Doc B: "I had book club on April 4, 2024" — 1 instance.
Query: "my book club events".

Predicted failure: Doc A's 260 overlaps dominate doc B under sum-aggregation;
any single-event correctness gets buried.

### R3 — Degenerate / zero-width intervals

"Right now", "this moment", "just then". Brackets collapse to 0 duration;
overlap with anything non-zero has measure zero.

Predicted failure: scoring returns 0 for instant-vs-instant match; system
requires nonzero fuzzy bracket to match anything.

### R4 — Infinite / open-ended references

"Since 1990" — interval [1990, now) or [1990, ∞)?
"Until I die" — [now, future-unknown)?
"For as long as I can remember" — [birth, now)?
"The rest of my life" — [now, ∞)?

Predicted failure: extractor forces a finite `latest` that doesn't match
semantics.

### R5 — Paraphrastic / non-standard temporal synonyms

Doc: "last spring" (March-May 2025).
Query: "what did I do in Q2 2025?" (April-June).

Doc: "during the Obama years" (2009-2017).
Query: "things from the 2010s".

Doc: "Halloween 2022".
Query: "events on October 31, 2022".

Predicted failure: representation axes don't align; unless extraction
normalizes, doc and query don't match even though semantically identical.

### R6 — Multi-cycle recurrence

"The third Saturday of every odd month in 2024."

Predicted failure: current schema supports simple RRULE; multi-cycle logic
likely produces a malformed rrule.

### R7 — Temporal duration without anchor

"A 3-hour meeting" (no time specified), "lasted 2 weeks" (when?).

Query: "2-hour meetings?" — pure duration, no anchor.

Predicted failure: system skips duration-only for retrieval (by design);
no way to match duration-vs-duration.

## Retrieval-layer adversaries

### S1 — Granularity mismatch on day-specific queries

Query: "on March 15, 2024" (exact day).
Docs: one says "in 2024" (year), one says "March 2024" (month), one says
"March 15" (day). Correct ranking: day > month > year. Current scorer may
rank them near-equally.

### S2 — Context-dependent query

Query: "after my move" — the query alone has no anchor; requires resolution
from co-present corpus knowledge about "my move". Today we don't re-resolve
at query time.

### S3 — Negative temporal queries

"What did I NOT do last week?" — requires retrieving the absence of
something, which doesn't map to a standard retrieval predicate.

### S4 — Temporal coherence violations between extracted expressions

Doc has both "last Thursday" and "3 days ago" at the same ref_time but
these must resolve to different dates (if ref_time is a Sunday, last
Thursday=Thu 3 days ago, fine; but if ref_time is Tuesday, last Thursday
= 5 days ago, not 3 days). Today we don't cross-check.

### S5 — Systematic time-of-day-preserving vs -losing

"5 years ago at 3pm" should preserve hour=15; "5 years ago today" should
preserve date-of-year. But "5 years ago" alone — time-of-day uniform?

### S6 — Scale-extreme queries

"What happened in the last 5 minutes?" — tiny window, likely no
extractions in most docs fall there.

"What happened in the last 500 years?" — enormous window, matches almost
everything.

### S7 — Multi-anchor queries

"What was between my first and second surgeries?" — needs both anchors
resolved AND Allen retrieval parameterized by BOTH; current Allen handles
one anchor.

### S8 — Queries that conflate temporal and non-temporal evidence

"The year I met my wife" — the answer requires identifying the meeting
event, then returning its year. Retrieval needs to surface the meeting
doc, not just docs with years.

## Implementation plan

Build a curated adversarial test set with ~30-40 examples across the
categories above. For each:

1. `text` — the adversarial doc/query
2. `ref_time` — utterance time
3. `category` — which adversary class (A1-A9, R1-R7, S1-S8)
4. `gold_extraction` — what a correct extractor would output
5. `gold_retrieval` — for query examples, which docs should rank top-5
6. `expected_failure_mode` — what we think will go wrong

Run the current ship-best pipeline (v2 extractor + multi-axis scorer +
utterance anchor + era + Allen) and measure per-category failure rates.
The goal isn't to fix all of them — it's to know which ones break, so
we can prioritize the next wave and avoid overconfidence.

## Output

- `results/adversarial.md` — per-category failure analysis
- `results/adversarial.json` — raw per-example outcomes
- `data/adversarial_docs.jsonl`, `data/adversarial_queries.jsonl`, `data/adversarial_gold.jsonl`
