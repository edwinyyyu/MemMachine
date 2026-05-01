# Adversarial Re-Eval with v2'' Extractor (modality + fuzzy + holidays)

Corpus: 58 docs, 40 queries. Wall: 638.0s. LLM cost: $0.9032.
filter_modality = True

## Per-category — v2 vs v2' vs v2''

| Cat | N | v2 R@5 | v2' R@5 | **v2'' R@5** | ΔvsV2' | v2'' Emit | v2'' Avg TEs | CorrSkip(A7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 2 | 1.000 | 1.000 | **1.000** | 0.000 | 1.00 | 2.00 | - |
| A2 | 2 | 1.000 | 1.000 | **1.000** | 0.000 | 1.00 | 3.50 | - |
| A3 | 2 | 0.000 | 0.000 | **0.000** | 0.000 | 1.00 | 1.50 | - |
| A4 | 4 | 0.500 | 0.750 | **0.750** | 0.000 | 1.00 | 1.00 | - |
| A5 | 3 | 0.333 | 0.667 | **0.333** | -0.333 | 1.00 | 1.25 | - |
| A6 | 2 | 0.500 | 0.500 | **1.000** | 0.500 | 1.00 | 1.50 | - |
| A7 | 2 | 0.000 | 0.000 | **1.000** | 1.000 | 1.00 | 1.00 | 1.00 |
| A8 | 2 | 0.500 | 0.500 | **0.500** | 0.000 | 1.00 | 1.67 | - |
| A9 | 2 | 1.000 | 0.000 | **1.000** | 1.000 | 1.00 | 1.50 | - |
| R1 | 1 | 0.500 | 0.500 | **0.500** | 0.000 | 1.00 | 2.50 | - |
| R2 | 1 | 0.500 | 1.000 | **1.000** | 0.000 | 1.00 | 1.50 | - |
| R3 | 1 | 1.000 | 1.000 | **1.000** | 0.000 | 1.00 | 1.00 | - |
| R4 | 1 | 1.000 | 0.000 | **0.000** | 0.000 | 1.00 | 1.33 | - |
| R5 | 3 | 0.667 | 0.667 | **0.667** | 0.000 | 1.00 | 1.33 | - |
| R6 | 1 | 0.000 | 0.000 | **1.000** | 1.000 | 0.00 | 0.00 | - |
| R7 | 1 | 0.000 | 0.000 | **0.000** | 0.000 | 1.00 | 1.00 | - |
| S1 | 1 | 1.000 | 1.000 | **1.000** | 0.000 | - | - | - |
| S2 | 1 | 1.000 | 1.000 | **1.000** | 0.000 | - | - | - |
| S3 | 1 | 1.000 | 1.000 | **1.000** | 0.000 | - | - | - |
| S4 | 1 | 1.000 | 1.000 | **1.000** | 0.000 | 1.00 | 2.00 | - |
| S5 | 2 | 0.000 | 0.750 | **0.500** | -0.250 | 1.00 | 2.00 | - |
| S6 | 2 | 0.000 | 0.500 | **1.000** | 0.500 | 1.00 | 1.00 | - |
| S7 | 1 | 1.000 | 1.000 | **1.000** | 0.000 | - | - | - |
| S8 | 1 | 0.000 | 0.500 | **0.500** | 0.000 | 0.50 | 0.50 | - |

**Overall v2'' R@5**: 0.740 (v2 0.562, v2' 0.597, Δ vs v2' +0.142)
**Overall v2'' R@10**: 0.806, MRR: 0.635, NDCG@10: 0.656

## Modality partition

Docs skipped (ALL extracted TEs non-actual): 3
- `adv_a7_0` (A7): `In the novel I'm reading, the story is set in 1850.` -> modalities=['fictional']
- `adv_a7_1` (A7): `What if I had been born in 1980? How different things would be.` -> modalities=['hypothetical']
- `adv_a7_2` (A7): `Imagine a world where the year is 2089 and cars can fly.` -> modalities=['hypothetical']

## Top failures (after v2'')

### 1. `q_a3_0` (A3) — 'When did we adopt the cat?'
- Gold: ['adv_a3_0']
- Top-5: ['adv_a2_0', 'adv_s6_tiny', 'adv_a5_2', 'adv_r4_0', 'adv_a6_2']
- Expected: semantic match likely dominates.

### 2. `q_a3_1` (A3) — 'What happened in early April 2026?'
- Gold: ['adv_a3_1']
- Top-5: ['adv_r2_1', 'adv_a2_0', 'adv_a1_1', 'adv_a5_1', 'adv_a4_1']
- Expected: 'A few weeks back' from 2026-04-23 ~ late March/early April — should overlap.

### 3. `q_a4_0` (A4) — 'What happened on April 30, 2026?'
- Gold: ['adv_a4_0']
- Top-5: ['adv_a2_0', 'adv_a2_1', 'adv_a4_1', 'adv_a1_1', 'adv_a9_0']
- Expected: If 'Thursday' resolves to next Thursday (2026-04-30), retrieval should hit.

### 4. `q_a5_0` (A5) — 'What happened after the divorce?'
- Gold: ['adv_a5_0']
- Top-5: ['adv_a4_0', 'adv_a6_0', 'adv_a6_1', 'adv_a6_3', 'adv_a8_1']
- Expected: Allen relation=after with unresolved event anchor.

### 5. `q_a5_1` (A5) — 'What did I do post-lockdown?'
- Gold: ['adv_a5_1']
- Top-5: ['adv_a1_0', 'adv_a4_0', 'adv_a6_0', 'adv_a6_1', 'adv_a6_3']
- Expected: Semantic + era match.

### 6. `q_a8_1` (A8) — 'What did I miss last week?'
- Gold: ['adv_a8_2']
- Top-5: ['adv_a9_3', 'adv_a2_0', 'adv_a9_0', 'adv_r5_0', 'adv_r6_0']
- Expected: 'last week' overlap. Polarity-aware retrieval would score this differently.

### 7. `q_r1_0` (R1) — 'What happened on April 5, 2019?'
- Gold: ['adv_r1_0', 'adv_r1_1']
- Top-5: ['adv_r2_1', 'adv_r1_1', 'adv_s5_date', 'adv_a9_1', 'adv_s5_hour']
- Expected: Day-specific doc adv_r1_1 should rank ABOVE wide-interval adv_r1_0. Tests granularity ranking.

### 8. `q_r4_0` (R4) — 'What have I been doing since 1995?'
- Gold: ['adv_r4_0']
- Top-5: ['adv_r4_2', 'adv_a3_3', 'adv_a8_0', 'adv_r1_0', 'adv_a9_1']
- Expected: Open-ended interval matching.

### 9. `q_r5_0` (R5) — 'What did I do in Q2 2025?'
- Gold: ['adv_r5_0']
- Top-5: ['adv_r2_1', 'adv_s1_year', 'adv_s5_date', 'adv_a9_0', 'adv_a8_0']
- Expected: 'last spring' (Mar-May 2025) overlaps with Q2 (Apr-Jun). Paraphrastic match needed.

### 10. `q_s5_1` (S5) — 'What happened on April 23, 2021?'
- Gold: ['adv_s5_date', 'adv_s5_hour']
- Top-5: ['adv_r6_0', 'adv_a1_1', 'adv_r1_0', 'adv_a2_0', 'adv_a5_1']
- Expected: Date-preservation test.

### 11. `q_s8_0` (S8) — 'What year did I meet my wife?'
- Gold: ['adv_s8_2018', 'adv_s8_meet']
- Top-5: ['adv_s8_meet', 'adv_r1_0', 'adv_a8_0', 'adv_a9_1', 'adv_r2_0']
- Expected: Answer requires linking meet doc (no date) to 2018 doc (date) via shared context 'Austin conference'.

## Extraction sample (first 30 docs)

- **adv_a1_0** (A1) `Alice told me yesterday that she would be gone next week.` -> 2 TEs (surfaces=['yesterday', 'next week'], modalities=['actual', 'quoted_embedded'])
- **adv_a1_1** (A1) `In 2020, my sister said 'this month has been rough'.` -> 2 TEs (surfaces=['In 2020', 'this month'], modalities=['actual', 'quoted_embedded'])
- **adv_a1_2** (A1) `When I saw her at the conference, it was pouring rain. A week later, I got sick.` -> 2 TEs (surfaces=['When I saw her at the conference', 'A week later'], modalities=['actual', 'actual'])
- **adv_a2_0** (A2) `Three weeks after my birthday last year we finally closed on the house.` -> 5 TEs (surfaces=['Three weeks', 'my birthday', 'last year', 'my birthday last year', 'Three weeks after my birthday last year'], modalities=['actual', 'actual', 'actual', 'actual', 'actual'])
- **adv_a2_1** (A2) `The Thursday of the week after next, we have the kickoff.` -> 3 TEs (surfaces=['The Thursday of the week after next', 'the week after next', 'Thursday'], modalities=['actual', 'actual', 'actual'])
- **adv_a2_2** (A2) `Two days before three weeks from now, we'll ship the beta.` -> 4 TEs (surfaces=['Two days', 'three weeks from now', 'Two days before three weeks from now', 'Two days before three weeks from now'], modalities=['actual', 'actual', 'actual', 'actual'])
- **adv_a2_3** (A2) `The month after my first anniversary we moved into the new place.` -> 2 TEs (surfaces=['The month after my first anniversary', 'my first anniversary'], modalities=['actual', 'actual'])
- **adv_a3_0** (A3) `A couple of years ago we adopted a cat from the shelter.` -> 1 TEs (surfaces=['A couple of years ago'], modalities=['actual'])
- **adv_a3_1** (A3) `A few weeks back I bumped into my old professor at the bookstore.` -> 1 TEs (surfaces=['A few weeks back'], modalities=['actual'])
- **adv_a3_2** (A3) `Not long ago, Dana called to say she was moving abroad.` -> 1 TEs (surfaces=['Not long ago'], modalities=['actual'])
- **adv_a3_3** (A3) `Back in the day, we used to swim in the quarry until the sheriff came.` -> 3 TEs (surfaces=['Back in the day', 'until the sheriff came', 'the sheriff came'], modalities=['actual', 'actual', 'actual'])
- **adv_a4_0** (A4) `I'll see you Thursday for drinks.` -> 1 TEs (surfaces=['Thursday'], modalities=['actual'])
- **adv_a4_1** (A4) `Last Thursday we finally got the report approved.` -> 1 TEs (surfaces=['Last Thursday'], modalities=['actual'])
- **adv_a4_2** (A4) `Next Tuesday I'll finally finish the tax paperwork.` -> 1 TEs (surfaces=['Next Tuesday'], modalities=['actual'])
- **adv_a4_3** (A4) `This weekend we're driving up to the cabin.` -> 1 TEs (surfaces=['This weekend'], modalities=['actual'])
- **adv_a5_0** (A5) `Since the divorce, life has been surprisingly calm.` -> 1 TEs (surfaces=['Since the divorce'], modalities=['actual'])
- **adv_a5_1** (A5) `Post-lockdown I picked up running in the mornings again.` -> 2 TEs (surfaces=['Post-lockdown', 'in the mornings'], modalities=['actual', 'actual'])
- **adv_a5_2** (A5) `Before the move, we had to pack twenty boxes of books.` -> 1 TEs (surfaces=['Before the move,'], modalities=['actual'])
- **adv_a5_3** (A5) `When I was in college I took a class on classical Latin.` -> 1 TEs (surfaces=['When I was in college'], modalities=['actual'])
- **adv_a6_0** (A6) `I get the allergy shot every 13 days.` -> 1 TEs (surfaces=['every 13 days'], modalities=['actual'])
- **adv_a6_1** (A6) `Every other Thursday we have the senior team sync.` -> 1 TEs (surfaces=['Every other Thursday'], modalities=['actual'])
- **adv_a6_2** (A6) `The last Monday of each month is the retrospective.` -> 3 TEs (surfaces=['The last Monday of each month', 'last Monday', 'each month'], modalities=['actual', 'actual', 'actual'])
- **adv_a6_3** (A6) `I have PT every Tuesday and Thursday.` -> 1 TEs (surfaces=['every Tuesday and Thursday'], modalities=['actual'])
- **adv_a7_0** (A7) [SKIP] `In the novel I'm reading, the story is set in 1850.` -> 1 TEs (surfaces=['1850'], modalities=['fictional'])
- **adv_a7_1** (A7) [SKIP] `What if I had been born in 1980? How different things would be.` -> 1 TEs (surfaces=['1980'], modalities=['hypothetical'])
- **adv_a7_2** (A7) [SKIP] `Imagine a world where the year is 2089 and cars can fly.` -> 1 TEs (surfaces=['the year is 2089'], modalities=['hypothetical'])
- **adv_a8_0** (A8) `I had been living in Boston since 2015 before I finally moved west.` -> 2 TEs (surfaces=['since 2015', '2015'], modalities=['actual', 'actual'])
- **adv_a8_1** (A8) `I will have finished the manuscript by next Tuesday.` -> 2 TEs (surfaces=['by next Tuesday', 'next Tuesday'], modalities=['actual', 'actual'])
- **adv_a8_2** (A8) `I was going to go to the gala last week, but the flight was cancelled.` -> 1 TEs (surfaces=['last week'], modalities=['actual'])
- **adv_a9_0** (A9) `During Ramadan last year I reorganized my sleep schedule.` -> 1 TEs (surfaces=['During Ramadan last year'], modalities=['actual'])

## Cost & timing

- Total LLM tokens: input=673938, output=367335
- Estimated cost: $0.9032
- Wall clock: 638.0s
