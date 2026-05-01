# Dense Time-Cluster Stress Test

- Extractor: **v2** (reasoning_effort=minimal)
- Corpus: **100 docs**, all dated April 2024 (days 1..30)
- Queries: **30**, each with exactly 1 gold doc
- Embedding model: text-embedding-3-small
- Cost: **$0.0000**, wall: 0.4s

## Per-variant retrieval metrics

| Variant | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| SEMANTIC-ONLY | 1.000 | 1.000 | 1.000 | 1.000 |
| T-ONLY | 0.967 | 0.967 | 0.967 | 0.967 |
| V7 (T=0.5, S=0.5) | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 (T=0.3, S=0.7) [TempReason] | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 AUTO (T=0.0, S=1.0) | 1.000 | 1.000 | 1.000 | 1.000 |

## Auto-tune weight sweep (V7 T+S)

| T | S | R@1 | R@3 | R@5 | MRR |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.1 | 0.9 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.2 | 0.8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.3 | 0.7 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | 0.6 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5 | 0.5 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.6 | 0.4 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.7 | 0.3 | 0.967 | 1.000 | 1.000 | 0.983 |

**Best**: T=0.0, S=1.0 → R@1=1.000, MRR=1.000

## T-score variance diagnostic

Within each query's top-10 candidate pool (union of S/T-only/V7 top-10):

- Mean T std-dev in pool: **0.0014**
- Mean T mean (in pool): 0.0135
- Mean T max (in pool): 0.0170
- Mean T min (in pool): 0.0127
- Mean S std-dev in pool (reference): 0.1366
- Queries with ZERO T variance in pool: 29/30

## Conclusion (auto-generated)

- Auto-tune chose T=0.0 (low T weight). V7 with default 0.5/0.5 lost +0.000 R@1 vs SEMANTIC-ONLY.
- Verdict: **In dense time-cluster regime, T should be down-weighted or dropped.** Default V7 weights are not robust to this regime.

## Failure cases (SEMANTIC R@1 → V7 default lost rank)

(none — V7 default did not demote any sem-rank-1 below)
