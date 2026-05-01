# Scorer Containment Sweep

Wall: 753.0s. Zero LLM calls (cache-only, modulo a few uncached Allen extractions).

Four configs compared against the ship-best v2'' + multi-axis + score-blend pipeline:

- baseline_jaccard: existing jaccard_composite.
- containment_{log2,sqrt,dice}: new max(jaccard_composite, q_in_s_score, s_in_q_score) with decay formula.

## Key finding — no lift, no regression, recommend NOT shipping

All three decay formulas produce identical rankings to the baseline on the
adversarial 40-query set and near-identical (+0.004) on the base 55-query
set. **A3 R@5 stays at 0.00.**

Root cause: `max(jaccard_composite, q_in_s, s_in_q)` is dominated by
`jaccard_composite` in every case where containment fires on this corpus.
`jaccard_composite = 0.5·Jaccard + 0.3·proximity + 0.2·gran_compat` — the
proximity and granularity terms already reward best-point closeness even
when raw set-Jaccard is small. Empirical spot-checks:

| pair                        | jaccard | log2 | sqrt | dice |
|-----------------------------|--------:|-----:|-----:|-----:|
| q_a3_1 (early Apr) vs adv_a3_1 ("few weeks") | 0.588 | 0.34 | 0.26 | 0.36 |
| 1-second Q inside 10-year S | 0.250 | 0.03 | 0.02 | ≈0 |
| 1-day Q inside 2-year S     | 0.285 | 0.10 | 0.07 | 0.001 |

The decayed containment score is consistently smaller than the jaccard
composite, so `max()` picks the jaccard. The containment branch never
contributes.

### Why A3 failures are unfixable by this intervention

- **q_a3_0** ("When did we adopt the cat?"): query extractor emits *reference
  time* (2026-04-23), which is **outside** the gold doc's [2023-04-23,
  2025-04-23] bracket. Q.best ∉ S → no containment term fires. Needs
  query-side fix (don't emit "when" as a TE, or emit as open past).
- **q_a3_1** ("What happened in early April 2026?"): Q.best **is** contained
  in adv_a3_1's bracket, but jaccard_composite already scores 0.588 for
  that pair (adv_a3_1 ranks 4th in interval-only rank). The gold gets
  buried further by multi-axis + anchor + semantic rerank, not by the
  interval scorer. The fix for A3 lies in blend weights or reranker, not
  in the per-pair interval scorer.

## Overall comparison

| config | Adv R@5 | Adv R@10 | A3 R@5 | Base R@5 | Base R@10 |
|---|---:|---:|---:|---:|---:|
| baseline_jaccard | 0.688 | 0.743 | 0.000 | 0.425 | 0.549 |
| containment_log2 | 0.688 | 0.743 | 0.000 | 0.429 | 0.566 |
| containment_sqrt | 0.688 | 0.743 | 0.000 | 0.429 | 0.566 |
| containment_dice | 0.688 | 0.743 | 0.000 | 0.429 | 0.566 |

## Per-category (Adversarial) — delta vs baseline_jaccard

| cat | n | baseline_jaccard R@5 | containment_log2 R@5 | containment_sqrt R@5 | containment_dice R@5 |
|---|---:|---:|---:|---:|---:|
| A1 | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| A2 | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| A3 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| A4 | 4 | 0.750 | 0.750 | 0.750 | 0.750 |
| A5 | 3 | 0.333 | 0.333 | 0.333 | 0.333 |
| A6 | 2 | 0.500 | 0.500 | 0.500 | 0.500 |
| A7 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| A8 | 2 | 0.500 | 0.500 | 0.500 | 0.500 |
| A9 | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| R1 | 1 | 0.500 | 0.500 | 0.500 | 0.500 |
| R2 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| R3 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| R4 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| R5 | 3 | 0.667 | 0.667 | 0.667 | 0.667 |
| R6 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| R7 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| S1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| S2 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| S3 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| S4 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| S5 | 2 | 0.750 | 0.750 | 0.750 | 0.750 |
| S6 | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| S7 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| S8 | 1 | 0.500 | 0.500 | 0.500 | 0.500 |

## A3 per-query R@5

| qid | baseline_jaccard | containment_log2 | containment_sqrt | containment_dice |
|---|---|---|---|---|
| q_a3_0 | 0.00 | 0.00 | 0.00 | 0.00 |
| q_a3_1 | 0.00 | 0.00 | 0.00 | 0.00 |

## Regressions vs baseline

| config | Δ Adv R@5 | Δ Base R@5 | max cat regression | worst cat |
|---|---:|---:|---:|---|
| containment_log2 | +0.000 | +0.004 | +0.000 | - |
| containment_sqrt | +0.000 | +0.004 | +0.000 | - |
| containment_dice | +0.000 | +0.004 | +0.000 | - |
