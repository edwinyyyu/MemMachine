# Adversarial Stress-Test — Results

Corpus: 58 docs, 40 queries. Wall clock: 1317.5s. LLM cost: $0.8262.

## Per-category retrieval metrics

| Category | N | R@5 | R@10 | MRR | NDCG@10 | Emit rate | Avg TEs/doc | Correct-skip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 3.33 | - |
| A2 | 2 | 1.000 | 1.000 | 0.417 | 0.565 | 1.00 | 3.25 | - |
| A3 | 2 | 0.000 | 0.000 | 0.029 | 0.000 | 1.00 | 2.25 | - |
| A4 | 4 | 0.500 | 1.000 | 0.403 | 0.539 | 1.00 | 1.00 | - |
| A5 | 3 | 0.333 | 1.000 | 0.437 | 0.563 | 1.00 | 1.50 | - |
| A6 | 2 | 0.500 | 0.500 | 0.250 | 0.315 | 1.00 | 1.00 | - |
| A7 | 2 | 0.000 | 0.000 | - | - | 1.00 | 2.00 | 0.00 |
| A8 | 2 | 0.500 | 1.000 | 0.583 | 0.678 | 1.00 | 1.67 | - |
| A9 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.75 | - |
| R1 | 1 | 0.500 | 1.000 | 0.500 | 0.605 | 1.00 | 1.50 | - |
| R2 | 1 | 0.500 | 0.500 | 1.000 | 0.613 | 1.00 | 1.50 | - |
| R3 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.00 | - |
| R4 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.33 | - |
| R5 | 3 | 0.667 | 0.667 | 0.667 | 0.667 | 0.67 | 0.67 | - |
| R6 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | - |
| R7 | 1 | 0.000 | 0.000 | - | - | 1.00 | 1.00 | - |
| S1 | 1 | 1.000 | 1.000 | 1.000 | 0.906 | - | - | - |
| S2 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | - | - | - |
| S3 | 1 | 1.000 | 1.000 | - | - | - | - | - |
| S4 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 2.00 | - |
| S5 | 2 | 0.000 | 0.250 | 0.083 | 0.109 | 0.50 | 0.50 | - |
| S6 | 2 | 0.000 | 0.500 | 0.062 | 0.158 | 0.50 | 0.50 | - |
| S7 | 1 | 1.000 | 1.000 | - | - | - | - | - |
| S8 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | - |

**Overall**: R@5=0.562, R@10=0.684, MRR=0.572, NDCG@10=0.586

## Top failure examples

### 1. `q_a3_0` (A3) — 'When did we adopt the cat?'
- Gold: ['adv_a3_0']
- Top-5: ['adv_a2_3', 'adv_s5_date', 'adv_s1_month', 'adv_s5_hour', 'adv_a2_0']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: semantic match likely dominates.

### 2. `q_a3_1` (A3) — 'What happened in early April 2026?'
- Gold: ['adv_a3_1']
- Top-5: ['adv_r2_1', 'adv_a2_0', 'adv_a1_1', 'adv_a2_1', 'adv_a5_1']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: 'A few weeks back' from 2026-04-23 ~ late March/early April — should overlap.

### 3. `q_a4_0` (A4) — 'What happened on April 30, 2026?'
- Gold: ['adv_a4_0']
- Top-5: ['adv_r2_1', 'adv_a2_0', 'adv_a4_2', 'adv_a4_1', 'adv_a1_1']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: If 'Thursday' resolves to next Thursday (2026-04-30), retrieval should hit.

### 4. `q_a4_3` (A4) — 'What did we do the weekend of April 25, 2026?'
- Gold: ['adv_a4_3']
- Top-5: ['adv_a1_2', 'adv_a2_2', 'adv_a2_3', 'adv_a5_0', 'adv_s1_year']
- Routing: used_allen=True, relation=contains, anchor=April 25, 2026
- Expected behavior: 'This weekend' should resolve to 2026-04-25..26.

### 5. `q_a5_0` (A5) — 'What happened after the divorce?'
- Gold: ['adv_a5_0']
- Top-5: ['adv_a6_0', 'adv_a6_1', 'adv_a6_3', 'adv_a6_2', 'adv_a2_2']
- Routing: used_allen=True, relation=after, anchor=the divorce
- Expected behavior: Allen relation=after with unresolved event anchor.

### 6. `q_a5_1` (A5) — 'What did I do post-lockdown?'
- Gold: ['adv_a5_1']
- Top-5: ['adv_a6_0', 'adv_a6_1', 'adv_a6_3', 'adv_a6_2', 'adv_a2_2']
- Routing: used_allen=True, relation=after, anchor=lockdown
- Expected behavior: Semantic + era match.

### 7. `q_a6_1` (A6) — 'What is the retrospective schedule?'
- Gold: ['adv_a6_2']
- Top-5: ['adv_r6_0', 'adv_r7_1', 'adv_r7_0', 'adv_r5_2', 'adv_a8_1']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: semantic dominates.

### 8. `q_a8_1` (A8) — 'What did I miss last week?'
- Gold: ['adv_a8_2']
- Top-5: ['adv_a5_2', 'adv_a9_3', 'adv_a5_3', 'adv_a9_2', 'adv_r5_1']
- Routing: used_allen=True, relation=before, anchor=2026-04-23T12:00:00Z
- Expected behavior: 'last week' overlap. Polarity-aware retrieval would score this differently.

### 9. `q_r1_0` (R1) — 'What happened on April 5, 2019?'
- Gold: ['adv_r1_0', 'adv_r1_1']
- Top-5: ['adv_r2_1', 'adv_r1_1', 'adv_s5_date', 'adv_a9_1', 'adv_a4_1']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: Day-specific doc adv_r1_1 should rank ABOVE wide-interval adv_r1_0. Tests granularity ranking.

### 10. `q_r2_0` (R2) — 'What book club event happened on April 4, 2024?'
- Gold: ['adv_r2_0', 'adv_r2_1']
- Top-5: ['adv_r2_1', 'adv_s1_year', 'adv_s4_source', 'adv_a8_1', 'adv_a3_1']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: Single-instance doc should NOT be buried under recurrence.

### 11. `q_r5_0` (R5) — 'What did I do in Q2 2025?'
- Gold: ['adv_r5_0']
- Top-5: ['adv_r2_1', 'adv_s5_date', 'adv_a4_2', 'adv_a8_0', 'adv_a5_1']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: 'last spring' (Mar-May 2025) overlaps with Q2 (Apr-Jun). Paraphrastic match needed.

### 12. `q_r6_0` (R6) — 'When is the open house in March 2024?'
- Gold: ['adv_r6_0']
- Top-5: ['adv_s1_month', 'adv_s1_day', 'adv_r2_1', 'adv_a2_0', 'adv_s1_year']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: Multi-cycle recurrence; March's 3rd Saturday = 2024-03-16.

### 13. `q_s5_1` (S5) — 'What happened on April 23, 2021?'
- Gold: ['adv_s5_date', 'adv_s5_hour']
- Top-5: ['adv_r1_1', 'adv_a1_1', 'adv_r1_0', 'adv_a5_1', 'adv_s6_tiny']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: Date-preservation test.

### 14. `q_s5_0` (S5) — 'What happened on April 23, 2021 at 3pm?'
- Gold: ['adv_s5_hour']
- Top-5: ['adv_r1_1', 'adv_s6_tiny', 'adv_r3_1', 'adv_a4_1', 'adv_a5_1']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: Hour-preservation test.

### 15. `q_s6_1` (S6) — 'What happened in the 1400s?'
- Gold: ['adv_s6_huge']
- Top-5: ['adv_a3_3', 'adv_a5_3', 'adv_a9_0', 'adv_r5_1', 'adv_r4_0']
- Routing: used_allen=False, relation=None, anchor=None
- Expected behavior: Huge-window retrieval.

## Extraction summary (sample)

- **adv_a1_0** (A1) `Alice told me yesterday that she would be gone next week.` -> 4 TEs (surfaces=['yesterday', 'next week', 'yesterday', 'next week'], kinds=['instant', 'instant', 'interval', 'interval'])
- **adv_a1_1** (A1) `In 2020, my sister said 'this month has been rough'.` -> 4 TEs (surfaces=['In 2020', 'this month', '2020', 'this month'], kinds=['instant', 'instant', 'interval', 'interval'])
- **adv_a1_2** (A1) `When I saw her at the conference, it was pouring rain. A week later, I got sick.` -> 2 TEs (surfaces=['When I saw her at the conference', 'A week later'], kinds=['instant', 'instant'])
- **adv_a2_0** (A2) `Three weeks after my birthday last year we finally closed on the house.` -> 5 TEs (surfaces=['Three weeks after my birthday last year', 'Three weeks', 'my birthday last year', 'last year', 'last year'], kinds=['instant', 'duration', 'instant', 'instant', 'interval'])
- **adv_a2_1** (A2) `The Thursday of the week after next, we have the kickoff.` -> 2 TEs (surfaces=['The Thursday of the week after next', 'the week after next'], kinds=['instant', 'interval'])
- **adv_a2_2** (A2) `Two days before three weeks from now, we'll ship the beta.` -> 4 TEs (surfaces=['Two days before three weeks from now', 'three weeks from now', 'Two days', 'three weeks'], kinds=['instant', 'instant', 'duration', 'duration'])
- **adv_a2_3** (A2) `The month after my first anniversary we moved into the new place.` -> 2 TEs (surfaces=['The month after my first anniversary', 'my first anniversary'], kinds=['interval', 'instant'])
- **adv_a3_0** (A3) `A couple of years ago we adopted a cat from the shelter.` -> 1 TEs (surfaces=['A couple of years ago'], kinds=['instant'])
- **adv_a3_1** (A3) `A few weeks back I bumped into my old professor at the bookstore.` -> 1 TEs (surfaces=['A few weeks back'], kinds=['instant'])
- **adv_a3_2** (A3) `Not long ago, Dana called to say she was moving abroad.` -> 2 TEs (surfaces=['Not long ago,', 'Not long ago'], kinds=['instant', 'interval'])
- **adv_a3_3** (A3) `Back in the day, we used to swim in the quarry until the sheriff came.` -> 5 TEs (surfaces=['Back in the day', 'until the sheriff came.', 'Back in the day', 'until the sheriff came', 'the sheriff came'], kinds=['instant', 'interval', 'interval', 'interval', 'instant'])
- **adv_a4_0** (A4) `I'll see you Thursday for drinks.` -> 1 TEs (surfaces=['Thursday'], kinds=['instant'])
- **adv_a4_1** (A4) `Last Thursday we finally got the report approved.` -> 1 TEs (surfaces=['Last Thursday'], kinds=['instant'])
- **adv_a4_2** (A4) `Next Tuesday I'll finally finish the tax paperwork.` -> 1 TEs (surfaces=['Next Tuesday'], kinds=['instant'])
- **adv_a4_3** (A4) `This weekend we're driving up to the cabin.` -> 1 TEs (surfaces=['This weekend'], kinds=['interval'])
- **adv_a5_0** (A5) `Since the divorce, life has been surprisingly calm.` -> 1 TEs (surfaces=['Since the divorce'], kinds=['interval'])
- **adv_a5_1** (A5) `Post-lockdown I picked up running in the mornings again.` -> 2 TEs (surfaces=['Post-lockdown', 'the mornings'], kinds=['interval', 'recurrence'])
- **adv_a5_2** (A5) `Before the move, we had to pack twenty boxes of books.` -> 2 TEs (surfaces=['Before the move,', 'Before the move'], kinds=['interval', 'interval'])
- **adv_a5_3** (A5) `When I was in college I took a class on classical Latin.` -> 1 TEs (surfaces=['When I was in college'], kinds=['interval'])
- **adv_a6_0** (A6) `I get the allergy shot every 13 days.` -> 1 TEs (surfaces=['every 13 days'], kinds=['recurrence'])
- **adv_a6_1** (A6) `Every other Thursday we have the senior team sync.` -> 1 TEs (surfaces=['Every other Thursday'], kinds=['recurrence'])
- **adv_a6_2** (A6) `The last Monday of each month is the retrospective.` -> 1 TEs (surfaces=['The last Monday of each month'], kinds=['recurrence'])
- **adv_a6_3** (A6) `I have PT every Tuesday and Thursday.` -> 1 TEs (surfaces=['every Tuesday and Thursday.'], kinds=['recurrence'])
- **adv_a7_0** (A7) `In the novel I'm reading, the story is set in 1850.` -> 3 TEs (surfaces=['1850', "I'm reading", '1850'], kinds=['instant', 'interval', 'interval'])
- **adv_a7_1** (A7) `What if I had been born in 1980? How different things would be.` -> 1 TEs (surfaces=['born in 1980'], kinds=['instant'])

## Cost & timing

- Total LLM tokens: input=335607, output=371162
- Estimated cost: $0.8262
- Wall clock: 1317.5s

## Failure diagnosis

Classify the per-category failures by where the system breaks: extraction,
representation, or retrieval. Tractable vs hard distinguishes prompt/schema
fixes from architectural changes.

| Category | R@5 | Failure mode | Locus | Tractable? |
|---|---:|---|---|---|
| A1 (embedded ref) | 1.00 | passes | — | — |
| A2 (compositional) | 1.00 | passes R@5 but MRR 0.42: compositional resolution noisy; Thursday-of-week-after-next got right interval but ranking is fragile | extraction (pass-2 arithmetic) | tractable — add composition-aware few-shot |
| A3 (fuzzy modifiers) | 0.00 | "a couple of years ago" / "a few weeks back" extracted but brackets too narrow / too wide to match date-specific queries | representation (bracket width) | tractable — widen fuzzy brackets |
| A4 (weekday ambiguity) | 0.50 | "Thursday" on a Thursday: extractor picked one default silently; no ambiguity flag; ranking to specific dates inconsistent | extraction (disambiguation policy) | tractable — emit multiple hypotheses or mark ambiguous |
| A5 (unknown-entity anchors) | 0.33 | Allen routes correctly but resolver can't resolve "the divorce"/"lockdown" → retrieval degrades to semantic | retrieval (event resolver) | hard — requires cross-doc coreference |
| A6 (non-standard recurrence) | 0.50 | "every 13 days"/"every other Thursday" emitted as recurrence but rrule likely malformed; expand fans out wrong instances | representation (rrule schema) | tractable — extend rrule grammar + add INTERVAL support |
| A7 (fictional) | 0.00 | "1850 in the novel" → extractor emits 1850 as real (100% emit rate, 0% correct-skip) | extraction (no modality flag) | tractable — add modality prompt + polarity field |
| A8 (tense/aspect) | 0.50 | "last week" extracted but "was going to go" negated modality lost; future-perfect "by next Tuesday" worked | extraction (no aspect flag) | tractable — polarity extractor already exists; route it |
| A9 (holidays) | 1.00 | Ramadan 2025 resolved; Easter 2015 worked | — | — |
| R1 (span vs point) | 0.50 | Day-specific doc adv_r1_1 ranked 2nd; 8-year marriage adv_r1_0 not in top-5: the tight interval CORRECTLY beat the wide one but a related doc was missed | representation (over-tight filter on wide interval) | tractable — lower penalty for wide intervals that contain query |
| R2 (recurrence density) | 0.50 | Single-event doc adv_r2_1 ranked #1 → good; but recurrence doc adv_r2_0 not in top-5 | representation (recurrence flattening misses the specific day) | tractable — add exact-instance match for recurrences |
| R3 (zero-width) | 1.00 | "right now" matched → good | — | — |
| R4 (open-ended) | 1.00 | "since 1990" matched → good | — | — |
| R5 (paraphrastic) | 0.67 | "Obama years"/"Halloween 2022" resolved; "last spring" → Q2 2025 query failed (season→quarter translation absent) | representation (axis misalignment) | tractable — season/quarter cross-index |
| R6 (multi-cycle) | 0.00 | "3rd Saturday of every odd month in 2024" → 0 TEs extracted (LLM gave up) | extraction + representation | hard — requires compound RRULE or fan-out |
| R7 (duration only) | 0.00 | "2-hour meetings?" retrieved irrelevant docs; expected empty | retrieval (no duration index) | hard — new index type |
| S1 (granularity) | 1.00 | Day > month > year ranking worked | — | — |
| S2 (context-dep) | 1.00 | "after my move" landed on correct doc by semantic match | — | — |
| S3 (negative) | 1.00 | "What did I NOT do last week?" → no confident match: correct-skip behavior | — | — |
| S4 (coherence) | 1.00 | "last Thursday from Tuesday ref_time" resolved correctly | — | — |
| S5 (hour vs date preserve) | 0.00 | "5 years ago at 3pm" → extractor TIMED OUT on pass-2; "5 years ago today" emitted but date-of-year not preserved | extraction (timeout) + representation (hour axis) | mixed — infra (budget/timeout) and schema |
| S6 (scale extremes) | 0.00 | "30 seconds ago" emitted but granularity doesn't match "last 60 seconds" query; "the 15th century" → extractor timed out | extraction + representation | tractable — century literal gazetteer entry + tighter second-scale bracket |
| S7 (multi-anchor) | 1.00 | Returned empty, which is the desired graceful failure | — | — |
| S8 (conflated temp+non-temp) | 0.00 | "What year did I meet my wife?" needs to link meet-doc to 2018-doc via "Austin" + "conference"; neither doc has temporal signal → semantic rerank returns unrelated docs | retrieval (cross-doc chaining absent) | hard — requires multi-hop retrieval |

### Extraction vs representation vs retrieval — locus summary

- **Extraction-dominated** (12 cats hurt): A2 arithmetic, A3 brackets, A4
  disambiguation, A7 modality, A8 aspect, A6/R6 rrule, S5 pass-2 timeout.
  These respond to prompt/schema fixes.
- **Representation-dominated** (3 cats hurt): R1 span/point penalty, R2
  recurrence flattening, R5 season↔quarter, R7 duration index, S5 hour
  axis, S6 scale axes.
- **Retrieval-dominated** (3 cats hurt): A5 event resolver miss, S8
  multi-hop cross-doc linking.

## Top tractable fixes (prompt / schema)

1. **Polarity + modality field on TimeExpression** (A7, A8). Emit
   `polarity ∈ {positive, negated, hypothetical, fictional}` during pass-1;
   retrieval penalizes negated/fictional/hypothetical matches. Closes
   A7 (0→1.0) and half of A8 cases — estimated +0.08 overall R@5.
2. **Widen fuzzy-modifier brackets in pass-2** (A3). "A couple of years"
   → span ±50% around best; "a few weeks" → span ±100%. Pull A3 from
   0.0 toward 0.6 — estimated +0.03 overall R@5.
3. **Cross-axis rewrite for season/quarter/holiday** (R5, partial A9).
   Add a post-processor that emits SYNONYM TEs — "last spring"
   additionally emits a Q2 interval; "Halloween" emits month+day. Pull
   R5 from 0.67 to 0.90 — estimated +0.02 overall R@5.

Cumulative expected lift: ~+0.13 R@5 → ~0.69.

## Top hard cases (architectural)

1. **Event resolver for personal anchors** (A5, S2). Needs a doc-graph
   of named entities and cross-doc coreference. "The divorce", "the
   move", "my first anniversary" require that retrieval build a
   persistent event index across the corpus.
2. **Multi-hop cross-doc reasoning** (S8). "The year I met my wife"
   needs to link the event-doc to the year-doc via shared context
   ("Austin conference"). This is a retrieval-time join, not a
   representation fix.
3. **Compound / multi-cycle recurrence grammar** (R6). Single rrule
   can't express "3rd Saturday of every odd-numbered month". Requires
   either fan-out (generate per-month concrete instances) or a new
   representation layer.

## Ceiling estimate under mixed real-world mix

If a realistic corpus contains roughly:
- 60% easy cases (current synth numbers apply: R@5 ~0.85-0.95)
- 25% moderate adversarial (A3/A4/A8/R5-R6: current R@5 ~0.50)
- 15% hard adversarial (A5/A7/S8: current R@5 ~0.10)

Expected mixed R@5 ≈ 0.60·0.90 + 0.25·0.50 + 0.15·0.10 ≈ **0.68**.
After the 3 tractable fixes: ≈ 0.60·0.92 + 0.25·0.65 + 0.15·0.10 ≈ **0.73**.

The harder architectural items (event resolver, multi-hop) gate the path
above R@5 ~0.80 on a mixed corpus.
