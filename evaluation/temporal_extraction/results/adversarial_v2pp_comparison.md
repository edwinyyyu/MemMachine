# v2'' Ship Recommendation — Adversarial Comparison

## Headline

| Metric | v2 | v2' | **v2''** |
|---|---:|---:|---:|
| Overall R@5 | 0.562 | 0.597 | **0.740** |
| Overall R@10 | — | 0.660 | **0.806** |
| MRR | — | 0.574 | **0.635** |
| NDCG@10 | — | 0.579 | **0.656** |
| LLM cost | $0.83 | $0.65 | $0.90 |
| Wall (s) | 1318 | 1056 | 638 |

v2'' beats the projected 0.69 target by +5pp (actual 0.740).

## Per-category lift (v2'' − v2')

| Cat | N | v2 R@5 | v2' R@5 | v2'' R@5 | Δv2' |
|---|---:|---:|---:|---:|---:|
| A1 | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| A2 | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| A3 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| A4 | 4 | 0.500 | 0.750 | 0.750 | 0.000 |
| A5 | 3 | 0.333 | 0.667 | 0.333 | -0.333 |
| A6 | 2 | 0.500 | 0.500 | 1.000 | +0.500 |
| **A7** | 2 | 0.000 | 0.000 | **1.000** | **+1.000** |
| A8 | 2 | 0.500 | 0.500 | 0.500 | 0.000 |
| **A9** | 2 | 1.000 | 0.000 | **1.000** | **+1.000** |
| R1 | 1 | 0.500 | 0.500 | 0.500 | 0.000 |
| R2 | 1 | 0.500 | 1.000 | 1.000 | 0.000 |
| R3 | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| R4 | 1 | 1.000 | 0.000 | 0.000 | 0.000 |
| R5 | 3 | 0.667 | 0.667 | 0.667 | 0.000 |
| R6 | 1 | 0.000 | 0.000 | 1.000 | +1.000 |
| R7 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| S1 | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| S2 | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| S3 | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| S4 | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| S5 | 2 | 0.000 | 0.750 | 0.500 | -0.250 |
| S6 | 2 | 0.000 | 0.500 | 1.000 | +0.500 |
| S7 | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| S8 | 1 | 0.000 | 0.500 | 0.500 | 0.000 |

## Target tests

### A7 correct-skip (critical test for modality fix)

- v2 correct-skip: 0% (0/3)
- v2' correct-skip: 0% (0/3)
- **v2'' correct-skip: 100% (3/3)** — all three fictional/hypothetical docs skipped
- Modality partition: `adv_a7_0` (fictional 1850), `adv_a7_1` (hypothetical 1980), `adv_a7_2` (hypothetical 2089) — all modality-filtered out.

### A3 fuzzy modifiers

- v2'' R@5 = 0.000 — widened brackets DID fire at extraction (e.g. "A couple of years ago" -> [2023-04-23, 2025-04-23], "A few weeks back" -> [2026-03-12, 2026-04-09]) but retrieval pipeline ranked them below stronger semantic matches. The fix is extraction-layer; retrieval-layer ranking overrides it.
- Example failure: q_a3_1 "What happened in early April 2026?" — adv_a3_1 has overlapping bracket but ranks 9th in ma_top20 (narrow-interval docs dominate Jaccard).
- **Verdict**: target ~0.60 not met; extraction fix worked but retrieval still needs wider-bracket weighting.

### A9 named-era+year

- v2 R@5: 1.000 (gpt-5-mini knew some dates)
- v2' R@5: 0.000 (v2' emitted generic year only, losing the date)
- **v2'' R@5: 1.000** — holiday post-processor resolved:
  - "Easter 2015" -> 2015-04-05 (concrete day)
  - "During Ramadan last year" (ref 2026) -> Ramadan 2026 dates (Feb 17 - Mar 19)
- **Verdict**: target ~0.80 exceeded.

## Base regression probe

Quick parallel sample on 20 base items (`data/queries.jsonl` + `data/docs.jsonl`):
- 2 items with surface-span drift: v2' said "1987", v2'' said "in 1987"; v2' said "1776", v2'' said "in 1776". Both encode the same year — not semantic regression.
- **0 non-actual modality tags** on the base corpus — the modality field defaults to "actual" and only fires on adversarial cases.

## Cost

- Adversarial v2'' eval: $0.9032 (within $1 budget).
- Total session spend: ~$0.90.

## Ship recommendation

**Ship v2'' as new default.**

Rationale:
1. +14pp overall R@5 vs v2' (0.597 -> 0.740), beats projected 0.69 target.
2. A7 modality filter works perfectly (0% -> 100% correct-skip).
3. A9 holiday resolver recovers the R@5=1.0 that v2' lost (regression reversed).
4. Bonus lifts on A6 (+0.5), R6 (+1.0), S6 (+0.5).
5. A3 is not a regression — extraction-layer fix is in place but retrieval scoring dominates.
6. A5 and S5 regressed (-0.33, -0.25); worth watching but small-N (3, 2 queries).
7. No base regressions detected beyond cosmetic surface-span drift.

### Risks / follow-ups

- A3 retrieval-layer: wider fuzzy brackets alone don't win R@5; may need bracket-width-aware Jaccard weighting.
- A5 (-0.33), S5 (-0.25): investigate whether modality-filter or holiday-override accidentally displaced real temporal matches on these.
- LLM cost up ~$0.25 vs v2' (longer Pass-1 prompt). Still well under $1 budget.
