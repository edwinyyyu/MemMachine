# Iterative Query Refinement (Hopfield-style attractor)

Mechanism: after hop0 retrieval, centroid of retrieved embeddings is used to pull the query embedding toward the topic cluster. Pure IQR variants use zero LLM calls; iqr_plus_v2f adds one v2f LLM call anchored on the refined probe's retrieved context.

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | d@20 | base@50 | arch@50 | d@50 | llm | embed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine_baseline | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.508 | +0.000 | 0.0 | 2.0 |
| cosine_baseline | synthetic_19q | 0.569 | 0.569 | +0.000 | 0.824 | 0.824 | +0.000 | 0.0 | 2.0 |
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 | 4.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 | 4.0 |
| iqr_beta_0.2_t1 | locomo_30q | 0.383 | 0.250 | -0.133 | 0.508 | 0.400 | -0.108 | 0.0 | 2.0 |
| iqr_beta_0.2_t1 | synthetic_19q | 0.569 | 0.585 | +0.016 | 0.824 | 0.826 | +0.003 | 0.0 | 2.0 |
| iqr_beta_0.4_t1 | locomo_30q | 0.383 | 0.217 | -0.167 | 0.508 | 0.300 | -0.208 | 0.0 | 2.0 |
| iqr_beta_0.4_t1 | synthetic_19q | 0.569 | 0.575 | +0.005 | 0.824 | 0.848 | +0.024 | 0.0 | 2.0 |
| iqr_beta_0.6_t1 | locomo_30q | 0.383 | 0.150 | -0.233 | 0.508 | 0.267 | -0.242 | 0.0 | 2.0 |
| iqr_beta_0.6_t1 | synthetic_19q | 0.569 | 0.594 | +0.025 | 0.824 | 0.842 | +0.018 | 0.0 | 2.0 |
| iqr_beta_0.4_t2 | locomo_30q | 0.383 | 0.150 | -0.233 | 0.508 | 0.250 | -0.258 | 0.0 | 2.0 |
| iqr_beta_0.4_t2 | synthetic_19q | 0.569 | 0.572 | +0.002 | 0.824 | 0.842 | +0.018 | 0.0 | 2.0 |
| iqr_beta_0.4_filtered | locomo_30q | 0.383 | 0.250 | -0.133 | 0.508 | 0.300 | -0.208 | 0.0 | 2.0 |
| iqr_beta_0.4_filtered | synthetic_19q | 0.569 | 0.554 | -0.016 | 0.824 | 0.830 | +0.006 | 0.0 | 2.0 |
| iqr_plus_v2f | locomo_30q | 0.383 | 0.672 | +0.289 | 0.508 | 0.739 | +0.231 | 1.0 | 4.0 |
| iqr_plus_v2f | synthetic_19q | 0.569 | 0.602 | +0.033 | 0.824 | 0.831 | +0.007 | 1.0 | 4.0 |

## Geometry check: cos(q_refined, gold) - cos(q_0, gold)

Mean cosine-to-gold delta across questions (positive = refinement moves the query closer to gold embeddings).

| Arch | Dataset | n | mean Δcos | frac questions closer |
|---|---|---:|---:|---:|
| iqr_beta_0.2_t1 | locomo_30q | 30 | +0.0557 | 1.000 |
| iqr_beta_0.2_t1 | synthetic_19q | 19 | +0.0640 | 1.000 |
| iqr_beta_0.4_t1 | locomo_30q | 30 | +0.1042 | 1.000 |
| iqr_beta_0.4_t1 | synthetic_19q | 19 | +0.1215 | 1.000 |
| iqr_beta_0.6_t1 | locomo_30q | 30 | +0.1382 | 1.000 |
| iqr_beta_0.6_t1 | synthetic_19q | 19 | +0.1660 | 1.000 |
| iqr_beta_0.4_t2 | locomo_30q | 30 | +0.1381 | 1.000 |
| iqr_beta_0.4_t2 | synthetic_19q | 19 | +0.1711 | 1.000 |
| iqr_beta_0.4_filtered | locomo_30q | 30 | +0.0982 | 1.000 |
| iqr_beta_0.4_filtered | synthetic_19q | 19 | +0.1062 | 1.000 |
| iqr_plus_v2f | locomo_30q | 30 | +0.1042 | 1.000 |
| iqr_plus_v2f | synthetic_19q | 19 | +0.1215 | 1.000 |

## Top categories by Δr@50 (iqr_beta_0.4_t1, LoCoMo-30)

Gaining:
Losing:
  - locomo_temporal (n=16): Δ=-0.250 W/T/L=0/12/4
  - locomo_multi_hop (n=4): Δ=-0.375 W/T/L=0/2/2

## Verdict

**ABANDON**: all IQR variants at or below cosine on LoCoMo-30 @K=50 (cosine=0.508, best_pure=0.400, plus_v2f=0.739, v2f=0.858).
