# Adversarial Re-Eval with v2' (v2-prime) Extractor

Corpus: 58 docs, 40 queries. Wall: 1056.4s. LLM cost: $0.6455.

## Per-category — v2' vs v2 baseline

| Cat | N | v2 R@5 | **v2' R@5** | ΔR@5 | v2 Emit | **v2' Emit** | v2' Avg TEs | Correct-skip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 2 | 1.000 | **1.000** | 0.000 | 1.00 | **1.00** | 3.33 | - |
| A2 | 2 | 1.000 | **1.000** | 0.000 | 1.00 | **1.00** | 3.25 | - |
| A3 | 2 | 0.000 | **0.000** | 0.000 | 1.00 | **1.00** | 1.75 | - |
| A4 | 4 | 0.500 | **0.750** | 0.250 | 1.00 | **1.00** | 1.50 | - |
| A5 | 3 | 0.333 | **0.667** | 0.333 | 1.00 | **1.00** | 1.25 | - |
| A6 | 2 | 0.500 | **0.500** | 0.000 | 1.00 | **1.00** | 1.00 | - |
| A7 | 2 | 0.000 | **0.000** | 0.000 | 1.00 | **1.00** | 2.33 | 0.00 |
| A8 | 2 | 0.500 | **0.500** | 0.000 | 1.00 | **0.67** | 1.33 | - |
| A9 | 2 | 1.000 | **0.000** | -1.000 | 1.00 | **0.50** | 1.00 | - |
| R1 | 1 | 0.500 | **0.500** | 0.000 | 1.00 | **1.00** | 2.50 | - |
| R2 | 1 | 0.500 | **1.000** | 0.500 | 1.00 | **1.00** | 2.00 | - |
| R3 | 1 | 1.000 | **1.000** | 0.000 | 1.00 | **1.00** | 1.00 | - |
| R4 | 1 | 1.000 | **0.000** | -1.000 | 1.00 | **0.67** | 0.67 | - |
| R5 | 3 | 0.667 | **0.667** | 0.000 | 0.67 | **1.00** | 1.33 | - |
| R6 | 1 | 0.000 | **0.000** | 0.000 | 0.00 | **0.00** | 0.00 | - |
| R7 | 1 | 0.000 | **0.000** | 0.000 | 1.00 | **1.00** | 1.00 | - |
| S1 | 1 | 1.000 | **1.000** | 0.000 | - | **-** | - | - |
| S2 | 1 | 1.000 | **1.000** | 0.000 | - | **-** | - | - |
| S3 | 1 | 1.000 | **1.000** | 0.000 | - | **-** | - | - |
| S4 | 1 | 1.000 | **1.000** | 0.000 | 1.00 | **1.00** | 2.00 | - |
| S5 | 2 | 0.000 | **0.750** | 0.750 | 0.50 | **1.00** | 1.50 | - |
| S6 | 2 | 0.000 | **0.500** | 0.500 | 0.50 | **1.00** | 1.00 | - |
| S7 | 1 | 1.000 | **1.000** | 0.000 | - | **-** | - | - |
| S8 | 1 | 0.000 | **0.500** | 0.500 | 0.00 | **1.00** | 1.50 | - |

**Overall v2' R@5**: 0.597 (v2 baseline 0.562, Δ +0.035)
**Overall v2' R@10**: 0.660, MRR: 0.574, NDCG@10: 0.579

## Extraction-level diagnosis

Per-category emit rate and TEs/doc: how often does v2' produce SOMETHING vs v2?

**Categories with 0% emit rate even under v2'**: ['R6']

## Top failure examples (after v2')

### 1. `q_a3_0` (A3) — 'When did we adopt the cat?'
- Gold: ['adv_a3_0']
- Top-5: ['adv_s5_date', 'adv_s1_month', 'adv_r2_1', 'adv_a8_0', 'adv_a9_3']
- Expected: semantic match likely dominates.

### 2. `q_a3_1` (A3) — 'What happened in early April 2026?'
- Gold: ['adv_a3_1']
- Top-5: ['adv_r2_1', 'adv_a2_0', 'adv_a1_1', 'adv_a2_1', 'adv_a5_1']
- Expected: 'A few weeks back' from 2026-04-23 ~ late March/early April — should overlap.

### 3. `q_a4_0` (A4) — 'What happened on April 30, 2026?'
- Gold: ['adv_a4_0']
- Top-5: ['adv_r2_1', 'adv_a2_0', 'adv_a4_2', 'adv_a4_1', 'adv_a1_1']
- Expected: If 'Thursday' resolves to next Thursday (2026-04-30), retrieval should hit.

### 4. `q_a5_0` (A5) — 'What happened after the divorce?'
- Gold: ['adv_a5_0']
- Top-5: ['adv_a4_0', 'adv_a6_0', 'adv_a6_1', 'adv_a6_3', 'adv_a8_1']
- Expected: Allen relation=after with unresolved event anchor.

### 5. `q_a6_1` (A6) — 'What is the retrospective schedule?'
- Gold: ['adv_a6_2']
- Top-5: ['adv_a9_0', 'adv_r6_0', 'adv_r7_1', 'adv_r7_0', 'adv_r4_0']
- Expected: semantic dominates.

### 6. `q_a8_1` (A8) — 'What did I miss last week?'
- Gold: ['adv_a8_2']
- Top-5: ['adv_a2_0', 'adv_r2_1', 'adv_s1_day', 'adv_s5_hour', 'adv_a1_1']
- Expected: 'last week' overlap. Polarity-aware retrieval would score this differently.

### 7. `q_a9_0` (A9) — 'What did I do during Ramadan 2025?'
- Gold: ['adv_a9_0']
- Top-5: ['adv_a2_0', 'adv_r4_2', 'adv_a2_2', 'adv_a5_2', 'adv_a8_1']
- Expected: Era resolution + semantic.

### 8. `q_a9_1` (A9) — 'What happened on April 5, 2015?'
- Gold: ['adv_a9_1']
- Top-5: ['adv_r5_1', 'adv_r4_2', 'adv_a2_2', 'adv_a5_2', 'adv_a8_1']
- Expected: Requires Easter 2015 resolution to 2015-04-05.

### 9. `q_r1_0` (R1) — 'What happened on April 5, 2019?'
- Gold: ['adv_r1_0', 'adv_r1_1']
- Top-5: ['adv_r2_1', 'adv_r1_1', 'adv_s5_date', 'adv_a9_1', 'adv_s5_hour']
- Expected: Day-specific doc adv_r1_1 should rank ABOVE wide-interval adv_r1_0. Tests granularity ranking.

### 10. `q_r4_0` (R4) — 'What have I been doing since 1995?'
- Gold: ['adv_r4_0']
- Top-5: ['adv_r4_2', 'adv_r1_1', 'adv_a1_1', 'adv_s5_hour', 'adv_r5_1']
- Expected: Open-ended interval matching.

### 11. `q_r5_0` (R5) — 'What did I do in Q2 2025?'
- Gold: ['adv_r5_0']
- Top-5: ['adv_r2_1', 'adv_s1_year', 'adv_a4_2', 'adv_a5_1', 'adv_a2_3']
- Expected: 'last spring' (Mar-May 2025) overlaps with Q2 (Apr-Jun). Paraphrastic match needed.

### 12. `q_r6_0` (R6) — 'When is the open house in March 2024?'
- Gold: ['adv_r6_0']
- Top-5: ['adv_s1_day', 'adv_r4_2', 'adv_a2_2', 'adv_a5_2', 'adv_a8_1']
- Expected: Multi-cycle recurrence; March's 3rd Saturday = 2024-03-16.

### 13. `q_s5_1` (S5) — 'What happened on April 23, 2021?'
- Gold: ['adv_s5_date', 'adv_s5_hour']
- Top-5: ['adv_r2_1', 'adv_s1_year', 'adv_a1_1', 'adv_r1_0', 'adv_s5_hour']
- Expected: Date-preservation test.

### 14. `q_s6_0` (S6) — 'What happened in the last 60 seconds?'
- Gold: ['adv_s6_tiny']
- Top-5: ['adv_a5_2', 'adv_s4_source', 'adv_a4_1', 'adv_a1_2', 'adv_a2_0']
- Expected: Tiny-window retrieval.

### 15. `q_s8_0` (S8) — 'What year did I meet my wife?'
- Gold: ['adv_s8_2018', 'adv_s8_meet']
- Top-5: ['adv_s8_meet', 'adv_r1_0', 'adv_s4_source', 'adv_a3_1', 'adv_a1_2']
- Expected: Answer requires linking meet doc (no date) to 2018 doc (date) via shared context 'Austin conference'.

## Extraction sample (first 25 docs)

- **adv_a1_0** (A1) `Alice told me yesterday that she would be gone next week.` -> 4 TEs (surfaces=['yesterday', 'next week', 'yesterday', 'next week'], kinds=['instant', 'instant', 'interval', 'interval'])
- **adv_a1_1** (A1) `In 2020, my sister said 'this month has been rough'.` -> 4 TEs (surfaces=['In 2020', 'this month', '2020', 'this month'], kinds=['instant', 'instant', 'interval', 'interval'])
- **adv_a1_2** (A1) `When I saw her at the conference, it was pouring rain. A week later, I got sick.` -> 2 TEs (surfaces=['A week later', 'When I saw her at the conference'], kinds=['instant', 'instant'])
- **adv_a2_0** (A2) `Three weeks after my birthday last year we finally closed on the house.` -> 5 TEs (surfaces=['Three weeks', 'my birthday last year', 'last year', 'Three weeks after my birthday last year', 'last year'], kinds=['duration', 'instant', 'instant', 'instant', 'interval'])
- **adv_a2_1** (A2) `The Thursday of the week after next, we have the kickoff.` -> 2 TEs (surfaces=['The Thursday of the week after next', 'the week after next'], kinds=['instant', 'interval'])
- **adv_a2_2** (A2) `Two days before three weeks from now, we'll ship the beta.` -> 4 TEs (surfaces=['Two days before three weeks from now', 'three weeks from now', 'Two days', 'three weeks'], kinds=['instant', 'instant', 'duration', 'duration'])
- **adv_a2_3** (A2) `The month after my first anniversary we moved into the new place.` -> 2 TEs (surfaces=['The month after my first anniversary', 'my first anniversary'], kinds=['interval', 'instant'])
- **adv_a3_0** (A3) `A couple of years ago we adopted a cat from the shelter.` -> 1 TEs (surfaces=['A couple of years ago'], kinds=['instant'])
- **adv_a3_1** (A3) `A few weeks back I bumped into my old professor at the bookstore.` -> 1 TEs (surfaces=['A few weeks back'], kinds=['instant'])
- **adv_a3_2** (A3) `Not long ago, Dana called to say she was moving abroad.` -> 2 TEs (surfaces=['Not long ago', 'Not long ago'], kinds=['instant', 'interval'])
- **adv_a3_3** (A3) `Back in the day, we used to swim in the quarry until the sheriff came.` -> 3 TEs (surfaces=['Back in the day', 'until the sheriff came', 'the sheriff came'], kinds=['interval', 'interval', 'instant'])
- **adv_a4_0** (A4) `I'll see you Thursday for drinks.` -> 2 TEs (surfaces=['Thursday', 'Thursday'], kinds=['recurrence', 'instant'])
- **adv_a4_1** (A4) `Last Thursday we finally got the report approved.` -> 1 TEs (surfaces=['Last Thursday'], kinds=['instant'])
- **adv_a4_2** (A4) `Next Tuesday I'll finally finish the tax paperwork.` -> 1 TEs (surfaces=['Next Tuesday'], kinds=['instant'])
- **adv_a4_3** (A4) `This weekend we're driving up to the cabin.` -> 2 TEs (surfaces=['This weekend', 'This weekend'], kinds=['instant', 'interval'])
- **adv_a5_0** (A5) `Since the divorce, life has been surprisingly calm.` -> 1 TEs (surfaces=['Since the divorce'], kinds=['interval'])
- **adv_a5_1** (A5) `Post-lockdown I picked up running in the mornings again.` -> 1 TEs (surfaces=['Post-lockdown'], kinds=['interval'])
- **adv_a5_2** (A5) `Before the move, we had to pack twenty boxes of books.` -> 2 TEs (surfaces=['Before the move', 'Before the move'], kinds=['instant', 'interval'])
- **adv_a5_3** (A5) `When I was in college I took a class on classical Latin.` -> 1 TEs (surfaces=['When I was in college'], kinds=['interval'])
- **adv_a6_0** (A6) `I get the allergy shot every 13 days.` -> 1 TEs (surfaces=['every 13 days'], kinds=['recurrence'])
- **adv_a6_1** (A6) `Every other Thursday we have the senior team sync.` -> 1 TEs (surfaces=['Every other Thursday'], kinds=['recurrence'])
- **adv_a6_2** (A6) `The last Monday of each month is the retrospective.` -> 1 TEs (surfaces=['The last Monday of each month'], kinds=['recurrence'])
- **adv_a6_3** (A6) `I have PT every Tuesday and Thursday.` -> 1 TEs (surfaces=['every Tuesday and Thursday'], kinds=['recurrence'])
- **adv_a7_0** (A7) `In the novel I'm reading, the story is set in 1850.` -> 3 TEs (surfaces=['1850', "I'm reading", '1850'], kinds=['instant', 'interval', 'interval'])
- **adv_a7_1** (A7) `What if I had been born in 1980? How different things would be.` -> 2 TEs (surfaces=['1980', 'born in 1980'], kinds=['instant', 'instant'])

## Cost & timing

- Total LLM tokens: input=474017, output=263479
- Estimated cost: $0.6455
- Wall clock: 1056.4s
