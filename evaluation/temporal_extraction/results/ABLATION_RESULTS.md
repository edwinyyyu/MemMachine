# Temporal Ablation — Results

64 cells = {narrow, quarter, half, full_unit} × {jaccard, gaussian, gaussian_integrated, hard_overlap} × {sum, max, top_k=3, log_sum}. All extractions reused from cache; realized LLM cost this run: $0.00 (baseline cost $ 0.3288).

## Top 10 by overall NDCG@10

| bracket | score | agg | all R@5 | all R@10 | all MRR | all NDCG@10 | disc_wvn NDCG@10 | disc_cm NDCG@10 | disc_rd NDCG@10 | crit top1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quarter | jaccard_composite | sum | 0.487 | 0.537 | 0.445 | 0.402 | 0.117 | 0.435 | 0.559 | 5/5 |
| quarter | jaccard_composite | top_k | 0.487 | 0.537 | 0.445 | 0.402 | 0.117 | 0.435 | 0.559 | 5/5 |
| quarter | jaccard_composite | log_sum | 0.487 | 0.537 | 0.445 | 0.402 | 0.117 | 0.435 | 0.559 | 5/5 |
| quarter | jaccard_composite | max | 0.487 | 0.537 | 0.445 | 0.402 | 0.117 | 0.435 | 0.559 | 5/5 |
| half | jaccard_composite | sum | 0.502 | 0.527 | 0.447 | 0.400 | 0.145 | 0.402 | 0.559 | 5/5 |
| half | jaccard_composite | max | 0.502 | 0.527 | 0.447 | 0.400 | 0.145 | 0.402 | 0.559 | 5/5 |
| half | jaccard_composite | top_k | 0.502 | 0.527 | 0.447 | 0.400 | 0.145 | 0.402 | 0.559 | 5/5 |
| half | jaccard_composite | log_sum | 0.502 | 0.527 | 0.447 | 0.400 | 0.145 | 0.402 | 0.559 | 5/5 |
| quarter | gaussian | sum | 0.479 | 0.509 | 0.444 | 0.397 | 0.036 | 0.476 | 0.450 | 5/5 |
| quarter | gaussian | top_k | 0.479 | 0.509 | 0.444 | 0.397 | 0.036 | 0.476 | 0.450 | 5/5 |

## Best cell per metric

- **all_ndcg@10**: quarter/jaccard_composite/sum = 0.402
- **all_recall@5**: half/jaccard_composite/sum = 0.502
- **all_mrr**: half/jaccard_composite/sum = 0.447
- **disc_wvn_ndcg@10**: half/jaccard_composite/sum = 0.145
- **disc_cm_ndcg@10**: quarter/gaussian/sum = 0.476
- **disc_rd_ndcg@10**: narrow/jaccard_composite/sum = 0.559

## Jaccard vs Gaussian — mean across all 16 subcells

| score | all_ndcg@10 | all_recall@5 | all_mrr | disc_cm_ndcg@10 | disc_wvn_ndcg@10 | disc_rd_ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|
| jaccard_composite | 0.396 | 0.475 | 0.441 | 0.412 | 0.080 | 0.559 |
| gaussian | 0.392 | 0.470 | 0.439 | 0.458 | 0.050 | 0.450 |
| gaussian_integrated | 0.392 | 0.470 | 0.439 | 0.458 | 0.050 | 0.450 |
| hard_overlap | 0.363 | 0.448 | 0.403 | 0.308 | 0.033 | 0.394 |

## Bracket-width comparison — best per bracket (any score/agg)

| bracket | best config (score/agg) | all NDCG@10 | disc_wvn NDCG@10 | disc_cm NDCG@10 | disc_rd NDCG@10 |
|---|---|---:|---:|---:|---:|
| narrow | gaussian/sum | 0.392 | 0.065 | 0.439 | 0.450 |
| quarter | jaccard_composite/sum | 0.402 | 0.117 | 0.435 | 0.559 |
| half | jaccard_composite/sum | 0.400 | 0.145 | 0.402 | 0.559 |
| full_unit | jaccard_composite/sum | 0.394 | 0.029 | 0.421 | 0.559 |

## Aggregation comparison — mean across all 16 subcells

| agg | all_ndcg@10 | disc_rd_ndcg@10 | disc_cm_ndcg@10 |
|---|---:|---:|---:|
| sum | 0.386 | 0.463 | 0.409 |
| max | 0.386 | 0.463 | 0.409 |
| top_k | 0.386 | 0.463 | 0.409 |
| log_sum | 0.386 | 0.463 | 0.409 |

**Aggregation sensitivity**: Only 2/16 (bracket, score) combinations produce different NDCG@10 across agg modes (>0.001 spread). Most queries have a single temporal expression, so per-doc scores collapse to one pairwise score regardless of agg function. When aggregation *does* matter (multi-expression queries / docs), `sum` = `top_k` ≥ `max` ≥ `log_sum` in this dataset.

## Gaussian vs gaussian_integrated — identical?

| bracket | gaussian NDCG@10 | gaussian_integrated NDCG@10 | diff |
|---|---:|---:|---:|
| full_unit | 0.3920 | 0.3920 | 0.000000 |
| half | 0.3868 | 0.3868 | 0.000000 |
| narrow | 0.3920 | 0.3920 | 0.000000 |
| quarter | 0.3975 | 0.3975 | 0.000000 |

Expected: identical. Both compute exp(−d²/(2(σq²+σs²))) after normalization. Difference verifies they're numerically equivalent.

## Convolution-of-spikes check (H3)

The current pipeline expands each recurrence into instance intervals and indexes them independently. Under `score=gaussian` + `agg=sum`, the total doc score for a recurrence against a query Gaussian is:

  score(doc) = Σ_i exp(-(μq - μi)² / (2(σq² + σi²)))

which is *exactly* the product-integral of the query Gaussian against the spike-train-convolved-with-σi recurrence density (H3). Under `agg=max` this collapses to nearest-instance. Numerical evidence from disc_rd cells below.

### disc_rd subset under {gaussian, jaccard} × {sum, max, top_k, log_sum} (full_unit bracket):

| score | agg | disc_rd R@5 | disc_rd NDCG@10 |
|---|---|---:|---:|
| jaccard_composite | sum | 1.000 | 0.559 |
| jaccard_composite | max | 1.000 | 0.559 |
| jaccard_composite | top_k | 1.000 | 0.559 |
| jaccard_composite | log_sum | 1.000 | 0.559 |
| gaussian | sum | 1.000 | 0.450 |
| gaussian | max | 1.000 | 0.450 |
| gaussian | top_k | 1.000 | 0.450 |
| gaussian | log_sum | 1.000 | 0.450 |

## Broken configurations (all_recall@5 < 0.05)

None.

## Base-query-only comparison (vs base REPORT.md)

Base REPORT.md T-only retrieval: R@5 0.460, MRR 0.625, NDCG@10 0.476 (indexed over 39 base docs only). The ablation scores are computed against an index of ALL 89 docs (39 base + 50 discriminator), so disc docs compete with and sometimes outrank base gold — which is why absolute base_* numbers here are lower than REPORT.md. Relative comparisons across cells are still valid. Top 5 configs by base NDCG@10:

| bracket | score | agg | base R@5 | base MRR | base NDCG@10 |
|---|---|---|---:|---:|---:|
| full_unit | hard_overlap | sum | 0.430 | 0.589 | 0.448 |
| full_unit | hard_overlap | max | 0.430 | 0.589 | 0.448 |
| full_unit | hard_overlap | top_k | 0.430 | 0.589 | 0.448 |
| full_unit | hard_overlap | log_sum | 0.430 | 0.589 | 0.448 |
| narrow | hard_overlap | sum | 0.430 | 0.589 | 0.448 |

## Ship-best recommendation

**quarter / jaccard_composite / sum** — NDCG@10 0.402, R@5 0.487, MRR 0.445, critical 5/5.

Tie-break preference if multiple cells are within 0.005 NDCG@10: prefer (a) the simpler agg (sum > log_sum > top_k > max), then (b) narrower bracket (narrow > quarter > half > full_unit) to minimize fanout.
