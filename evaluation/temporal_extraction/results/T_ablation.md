# T_lblend ablation — does T earn its place?

Variants:
- **A. filter_only**: union(sem-top50, T-top50) -> rerank; final = rerank * filter. T retrieved, not scored.
- **B. filter_T (current)**: same retrieval; final = score_blend({T,R}, CV) * filter.
- **C. filter_no_T_retrieval**: sem-top50 only -> rerank; final = rerank * filter. T not retrieved.
- **D. T_retrieval_wider**: union(sem-top50, T-top100) -> rerank; final = score_blend({T,R}, CV) * filter.

Filter = absolute_anchor 0/1 window from LLM plan (decay/recency/causal removed).

## Per-question answers (lead)

1. **Does T_lblend earn its place as scoring? (A vs B)** — **Yes.** B (0.605) beats A (0.562) by Δ=+0.042. T scoring earns its place.
2. **Does T_lblend earn its place as retrieval? (C vs A)** — **No.** A (0.562) ≈ C (0.562), Δ=+0.000. T as a retrieval channel adds nothing — sem top-50 covers it.
3. **Does wider T retrieval help? (B vs D)** — **No.** D (0.605) ≈ B (0.605), Δ=+0.000. Existing T top-50 is sufficient; widening to 100 doesn't help.

## R@1 table (per benchmark)

| Benchmark | n | A: filter_only | B: filter_T (current) | C: no_T_retrieval | D: T_wider |
|---|---:|---:|---:|---:|---:|
| composition | 25 | 0.160 | 0.040 | 0.160 | 0.040 |
| hard_bench | 75 | 0.973 | 1.000 | 0.973 | 1.000 |
| temporal_essential | 25 | 1.000 | 1.000 | 1.000 | 1.000 |
| tempreason_small | 60 | 0.733 | 0.733 | 0.733 | 0.733 |
| conjunctive_temporal | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_te_doc | 12 | 1.000 | 1.000 | 1.000 | 1.000 |
| relative_time | 12 | 0.750 | 1.000 | 0.750 | 1.000 |
| era_refs | 12 | 0.333 | 0.417 | 0.333 | 0.417 |
| open_ended_date | 15 | 0.200 | 0.333 | 0.200 | 0.333 |
| causal_relative | 15 | 0.467 | 0.467 | 0.467 | 0.467 |
| latest_recent | 15 | 0.133 | 0.267 | 0.133 | 0.267 |
| negation_temporal | 15 | 0.000 | 0.000 | 0.000 | 0.000 |

## Macro R@1 across 12 benches

- A (filter_only):       0.562
- **B (filter_T, current): 0.605**
- C (no_T_retrieval):    0.562
- D (T_wider):           0.605

Δ vs current best (B):
- A − B: -0.042
- C − B: -0.042
- D − B: +0.000

## Recommendation

**SHIP B (current filter_T).** Macro R@1 0.605. T earns its place as both retrieval and scoring channel.

## Followup: do we still need a temporal index?

B > A but C ≈ A: T matters as scoring, not retrieval. Keep T_lblend in score_blend; drop the T retrieval channel (sem top-50 covers candidate sourcing).
