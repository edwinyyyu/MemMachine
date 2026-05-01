# Hierarchical Granularity Tags (F2) — Results

Tag-based retrieval: every extracted TimeExpression emits a set of discrete granularity tags (day:..., week:..., month:..., quarter:..., year:..., decade:..., century:...). Matching is set intersection; scoring is Jaccard (unweighted or rarity-weighted) over tag sets. Per-doc aggregation over expression pairs is ``sum`` or ``max``.

## Index statistics

- Docs indexed: **79**
- Distinct tags: **297**
- (tag, expr) rows: **824**
- Tags by granularity: century=3, day=82, decade=8, month=14, pattern=3, quarter=103, week=26, weekday=7, year=51

## Variant comparison

|   Variant   | all R@5 | all NDCG | base R@5 | base NDCG | disc R@5 | disc NDCG | utt R@5 | utt NDCG |
|-------------|--------:|---------:|---------:|----------:|---------:|----------:|--------:|---------:|
| T1_jaccard_sum | 0.318 | 0.285 | 0.273 | 0.309 | 0.433 | 0.287 | 0.200 | 0.163 |
| T2_jaccard_max | 0.428 | 0.410 | 0.370 | 0.393 | 0.600 | 0.521 | 0.200 | 0.163 |
| T3_weighted_sum | 0.385 | 0.400 | 0.313 | 0.397 | 0.567 | 0.485 | 0.200 | 0.163 |
| T4_weighted_max | 0.383 | 0.405 | 0.309 | 0.387 | 0.567 | 0.515 | 0.200 | 0.163 |
| T5_jaccard_sum_hybrid | 0.418 | 0.435 | 0.320 | 0.390 | 0.567 | 0.503 | 0.467 | 0.454 |
| BracketOnly_jaccard_sum | 0.463 | 0.378 | 0.453 | 0.434 | 0.567 | 0.339 | 0.200 | 0.210 |
| Baseline_bracket_jaccard_sum_hybrid | 0.528 | 0.551 | 0.513 | 0.557 | 0.567 | 0.567 | 0.487 | 0.475 |

## Per-subset N sizes

- T1_jaccard_sum: all=90, base=50, disc=30, utt=10
- T2_jaccard_max: all=90, base=50, disc=30, utt=10
- T3_weighted_sum: all=90, base=50, disc=30, utt=10
- T4_weighted_max: all=90, base=50, disc=30, utt=10
- T5_jaccard_sum_hybrid: all=90, base=50, disc=30, utt=10
- BracketOnly_jaccard_sum: all=90, base=50, disc=30, utt=10
- Baseline_bracket_jaccard_sum_hybrid: all=90, base=50, disc=30, utt=10

## Catastrophic diffs (|ΔR@5| >= 0.2, T1 tag vs bracket)

- Tag-better queries: **3**
- Bracket-better queries: **20**

### Sample tag wins
- `q_crit_1` (Δ=+0.20): 'What did I have scheduled 14 days ago?'; gold=['doc_crit_1', 'doc_interval_2', 'doc_rec_cancel_0', 'doc_rec_simple_1', 'doc_rec_simple_2']; bracket_top5=['doc_crit_1', 'doc_multi_3']; tag_top5=['doc_interval_2', 'doc_interval_0', 'doc_crit_1', 'doc_cm_9_near', 'doc_rd_2_recur']
- `q_spec_day_7` (Δ=+0.50): 'What happened on February 14, 2026?'; gold=['doc_interval_2', 'doc_rec_simple_2']; bracket_top5=['doc_multi_3']; tag_top5=['doc_interval_2', 'doc_interval_0', 'doc_cm_9_near', 'doc_rd_2_recur', 'doc_multi_1']
- `q_spec_day_9` (Δ=+0.50): 'What happened on February 15, 2026?'; gold=['doc_interval_2', 'doc_rec_simple_2']; bracket_top5=['doc_multi_3']; tag_top5=['doc_interval_2', 'doc_interval_0', 'doc_cm_9_near', 'doc_rd_2_recur', 'doc_multi_1']

### Sample bracket wins
- `q_cm_0` (Δ=-1.00): 'What happened on April 2, 2026?'; gold=['doc_cm_0_near']; bracket_top5=['doc_multi_3', 'doc_cm_0_near', 'doc_cm_3_near', 'doc_multi_2', 'doc_cm_1_near']; tag_top5=['doc_cm_9_near', 'doc_rd_2_recur', 'doc_cm_8_near', 'doc_cm_4_near', 'doc_multi_1']
- `q_cm_1` (Δ=-1.00): 'What happened on April 3, 2026?'; gold=['doc_cm_1_near']; bracket_top5=['doc_wvn_4', 'doc_cm_1_near', 'doc_multi_2', 'doc_cm_0_near', 'doc_cm_3_near']; tag_top5=['doc_cm_9_near', 'doc_rd_2_recur', 'doc_cm_8_near', 'doc_cm_4_near', 'doc_multi_1']
- `q_cm_3` (Δ=-1.00): 'What happened on April 5, 2026?'; gold=['doc_cm_3_near']; bracket_top5=['doc_cm_4_near', 'doc_multi_2', 'doc_cm_1_near', 'doc_cm_0_near', 'doc_cm_3_near']; tag_top5=['doc_cm_9_near', 'doc_rd_2_recur', 'doc_cm_8_near', 'doc_cm_4_near', 'doc_multi_1']
- `q_cm_7` (Δ=-1.00): 'What happened on April 9, 2026?'; gold=['doc_cm_7_near']; bracket_top5=['doc_multi_3', 'doc_cm_4_near', 'doc_cm_7_near', 'doc_cm_9_near', 'doc_cm_8_near']; tag_top5=['doc_cm_9_near', 'doc_cm_8_near', 'doc_cm_4_near', 'doc_rd_2_recur', 'doc_multi_1']
- `q_fuzzy_2` (Δ=-1.00): 'What happened around 1998?'; gold=['doc_decade_0']; bracket_top5=['doc_utt_7', 'doc_decade_0']; tag_top5=['doc_utt_7', 'doc_utt_1', 'doc_abs_distant_0', 'doc_abs_distant_1', 'doc_utt_2']

## Summary (≤400 words)

**1. Tag vs bracket (temporal only).** Tag-Jaccard R@5: all=0.318, base=0.273, disc=0.433, utt=0.200. Bracket R@5: all=0.463, base=0.453, disc=0.567, utt=0.200. Delta all=-0.145.

**2. Rarity-weighting.** T3 (weighted/sum) R@5=0.385 vs T1 (jaccard/sum) R@5=0.318; delta=+0.067.

**3. Max vs sum aggregation.** Jaccard max=0.428 vs sum=0.318 (Δ=+0.109); weighted max=0.383 vs sum=0.385 (Δ=-0.002).

**4. Hybrid (tags + semantic) vs ship-best (bracket + semantic).** T5 (hybrid) R@5=0.418 vs baseline R@5=0.528 (Δ=-0.110 all; base Δ=-0.193).

**5. Catastrophic failure modes.** 3 queries where tags beat brackets by ≥0.2 R@5; 20 queries where brackets beat tags by ≥0.2. See JSON `failures` section for details.

**6. Ship recommendation.** DEPRIORITIZE — tags lose to brackets even with semantic rerank.

**7. Cost.** $0 LLM (all extractions cached), ~$0 embeddings (all cached). Tag index build is purely deterministic from cached extractions.
