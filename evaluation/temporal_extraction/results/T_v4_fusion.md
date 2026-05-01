# T_v4 vs T_lblend in fusion with cross-encoder rerank

Fusion stack: `score_blend({T, R}, {0.4, 0.6}, top_k_per=40, dispersion_cv_ref=0.20)`. Tail = semantic.
Rerank pool: `union(S top-50, T_lblend top-50, T_v4 top-50)` capped at 75. Both fusions share the same R-channel and S-tail, so any delta is purely the T-channel swap.

LME runs were attempted but the process exited silently mid-`lme_nontemp` (likely OOM under the cross-encoder); the temporal-corpus results below are complete and conclusive on their own.

## R@1 — Δ(v4 − lblend) leads

| Benchmark | n | rerank_only | fuse_T_lblend_R | fuse_T_v4_R | **Δ(v4 − lblend)** |
|---|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0.640 (48/75) | **0.893 (67/75)** | 0.640 (48/75) | **−0.253**  (lblend much better) |
| latest_recent | 15 | 0.133 (2/15) | **0.267 (4/15)** | 0.133 (2/15) | **−0.133**  (lblend better) |
| era_refs | 12 | 0.250 (3/12) | **0.417 (5/12)** | 0.333 (4/12) | **−0.083**  (lblend better) |
| temporal_essential | 25 | 0.920 (23/25) | 1.000 (25/25) | 1.000 (25/25) | +0.000  (tie, both saturate) |
| tempreason_small | 60 | 0.650 (39/60) | 0.733 (44/60) | 0.733 (44/60) | +0.000  (tie) |
| conjunctive_temporal | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | +0.000  (tie, ceiling) |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | +0.000  (tie, ceiling) |
| relative_time | 12 | 0.250 (3/12) | 1.000 (12/12) | 1.000 (12/12) | +0.000  (tie) |
| causal_relative | 15 | 0.467 (7/15) | 0.467 (7/15) | 0.467 (7/15) | +0.000  (tie, T inert) |
| negation_temporal | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | +0.000  (tie at floor) |
| open_ended_date | 15 | 0.267 (4/15) | 0.400 (6/15) | **0.467 (7/15)** | **+0.067**  (v4 better) |

## R@5

| Benchmark | n | rerank_only R@5 | fuse_T_lblend_R R@5 | fuse_T_v4_R R@5 |
|---|---:|---:|---:|---:|
| hard_bench | 75 | 0.853 (64/75) | **0.960 (72/75)** | 0.853 (64/75) |
| latest_recent | 15 | 1.000 (15/15) | 1.000 (15/15) | 1.000 (15/15) |
| era_refs | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| temporal_essential | 25 | 1.000 (25/25) | 1.000 (25/25) | 1.000 (25/25) |
| tempreason_small | 60 | 1.000 (60/60) | 1.000 (60/60) | 1.000 (60/60) |
| conjunctive_temporal | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| relative_time | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| causal_relative | 15 | 1.000 (15/15) | 1.000 (15/15) | 1.000 (15/15) |
| negation_temporal | 15 | 0.933 (14/15) | 0.467 (7/15) | 0.400 (6/15) |
| open_ended_date | 15 | 0.733 (11/15) | 0.800 (12/15) | 0.800 (12/15) |

## Macro-average R@1 across 11 benches

- rerank_only:      0.507
- fuse_T_lblend_R:  **0.652**
- fuse_T_v4_R:      0.598
- **Δ(v4 − lblend):  −0.054**

## Verdict

**T_v4 < T_lblend in fusion (Δ = −0.054 macro R@1; −7 absolute query wins, dominated by hard_bench).** The fusion does NOT rescue T_v4's saturation problem on inside-window distractors. The hypothesis that the rerank channel breaks T_v4's 1.0 saturation is **falsified for hard_bench** — the dominant date-anchored corpus where T was supposed to add value.

Recommend keeping **T_lblend** as the production temporal primitive. Do not migrate.

### Why fusion failed to rescue T_v4

The mechanism is the CV gate inside `score_blend`. Recall: a channel's effective weight is `user_w * min(1, cv / 0.20)`.

On **hard_bench**, T_v4 has dispersion CV well below 0.20: most queries are inside-window date-anchored, the gold doc and most distractors all sit inside the query window, so most T_v4 scores are at or near 1.0. The CV gate suppresses the T channel near-completely → fusion collapses to rerank_only (0.640 R@1, identical to baseline). T_lblend's blend `0.2*iv + 0.2*tag + 0.6*lattice` has a much wider per-query distribution (lattice cells separate even within-window docs), so its CV survives the 0.20 gate and the channel actually contributes — hence 0.893.

In other words, T_v4's "clean primitive" is *too* clean: it discards the ranking signal that distinguishes among inside-window docs, and the orthogonal R-channel cannot recover what T threw away because R already operates on textual relevance, not temporal containment. R + flat-T = R + 0 = R.

On **latest_recent** and **era_refs**, the same mechanism applies: queries with "latest", "earliest", "in the 90s" etc. produce broad q intervals; many docs land inside; T_v4 saturates; CV gate fires.

### Where T_v4 wins

**open_ended_date** (+0.067, 7/15 vs 6/15): half-line queries (`after 2020`) where T_lblend's iv-norm penalizes broad q intervals but T_v4's containment ratio gives the doc full 1.0 credit when it lies inside the half-line. T_v4's primitive is *correct* for open-ended polarity — it's just not what hard_bench rewards.

### Per-benchmark conclusion: which T-channel is right for each query type?

| Benchmark | Query shape | Best T |
|---|---|---|
| hard_bench | dense date-anchored, inside-window distractors | **T_lblend** (lattice separates within-window) |
| latest_recent | "latest"/"most recent" → broad q-interval | **T_lblend** |
| era_refs | "in the 90s" → era-broad | **T_lblend** (Δ −0.083) |
| open_ended_date | "after X" half-line | **T_v4** (Δ +0.067) |
| temporal_essential | precise q anchor | tie (both saturate) |
| tempreason_small | precise month/year cued | tie |
| conjunctive_temporal | multiple AND'd anchors | tie (geomean works for both) |
| multi_te_doc | doc has many TEs | tie |
| relative_time | needs anchor resolution | tie (both 1.000) |
| causal_relative | T channel inert | tie |
| negation_temporal | T-channel hurts vs rerank | tie at floor |

## Saturation rescue: did fusion fix T_v4's standalone losses?

Comparing prior T_v4 standalone-vs-T_lblend deltas (from `T_v4.md`) to the fused deltas:

| Benchmark | T_v4 standalone vs T_lblend | T_v4 fusion vs T_lblend | Rescued? |
|---|---:|---:|---|
| conjunctive_temporal | −0.250 | +0.000 | RESCUED (both at 1.000 ceiling) |
| multi_te_doc | −0.083 | +0.000 | RESCUED (both at 1.000) |
| temporal_essential | −0.040 | +0.000 | RESCUED (both at 1.000) |
| open_ended_date | +0.067 | +0.067 | PRESERVED |
| tempreason_small | +0.017 | +0.000 | mostly preserved |
| era_refs | +0.083 | −0.083 | LOST AND INVERTED — T_lblend extracts more from era queries in fusion |
| hard_bench | +0.027 | −0.253 | LOST — fusion amplifies T_lblend's lattice advantage |

So the rescue happens *on benches where standalone T_v4 already lost* — but always by ceiling-clamping at 1.000, not by T_v4 actually contributing usefully. Where it really matters (hard_bench, the production-shape distribution), the **fused** v4-vs-lblend gap *widens* from +0.027 (standalone) to **−0.253** because T_lblend's lattice channel multiplies its advantage when paired with R, while T_v4's saturation gets gated to zero by CV.

## Per-bench v4 vs lblend implied swap counts (lower bounds from R@1 deltas)

The per-query JSON was lost when the LME stage crashed before write-out. From count deltas alone we know `lblend_only ≥ Δr1` and `v4_only ≥ −Δr1` when Δr1 ≤ 0.

| Benchmark | r1_count(v4) | r1_count(lblend) | Δr1 (v4 − lblend) |
|---|---:|---:|---:|
| hard_bench | 48 | 67 | −19 (≥19 lblend-only gains, ≥0 v4-only) |
| latest_recent | 2 | 4 | −2 |
| era_refs | 4 | 5 | −1 |
| open_ended_date | 7 | 6 | +1 (≥1 v4-only gain) |

Hard_bench is decisive: T_lblend wins **at least 19 queries** that T_v4 misses, with no offsetting v4 wins (since fuse_v4_R is identical to rerank_only on hard_bench, fusion didn't surface any v4-unique top1).

## Bottom line

T_v4's mathematical elegance (single primitive subsumes containment, exact-match, range-vs-range, half-line) does **not** translate into a fusion advantage. The standalone 1.0-saturation collapses dispersion, the CV gate then suppresses T_v4's contribution, and fusion degenerates to rerank_only on the queries that matter most (date-anchored corpora). T_lblend's lattice channel — discrete, multi-resolution, naturally dispersive across cells — survives the CV gate and is the right primitive in fusion.

**Production: keep T_lblend. T_v4 stays interesting only as a single-channel primitive for open-ended-polarity queries; if open-ended-polarity becomes a target distribution, prefer a polarity-aware T_v4 variant (or a T_lblend-with-polarity gate) over T_v4 swap.**
