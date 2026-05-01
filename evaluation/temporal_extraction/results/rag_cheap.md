# Cheap RAG Fusion Re-Eval (v2' extractor; V1-V4, V7 only)

Docs: 165. Queries: 155. Wall: 126.8s.

Extraction coverage: v2p docs=27/165, v2p queries=70/155, era docs=35/165.

Variants skipped: V5 (ROUTED-SINGLE) and V6 (ROUTED-MULTI) — router omitted.
V8 (LLM-RERANK) and V9 (HYBRID) skipped per task spec.

## R@5 by variant × subset

| Variant | base | discriminator | utterance | era | axis | allen | combined |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1_CASCADE | 0.131 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.177 |
| V2_TEMPORAL-ONLY | 0.132 | 0.000 | 0.000 | 0.000 | 0.958 | 0.000 | 0.172 |
| V3_SEMANTIC-ONLY | 0.287 | 0.667 | 0.467 | 0.850 | 0.817 | 0.950 | 0.609 |
| V4_RRF-ALL | 0.334 | 0.633 | 0.100 | 0.925 | 1.000 | 0.875 | 0.618 |
| V7_SCORE-BLEND | 0.269 | 0.667 | 0.467 | 0.900 | 0.933 | 0.950 | 0.625 |

## NDCG@10 by variant × subset

| Variant | base | discriminator | utterance | era | axis | allen | combined |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1_CASCADE | 0.172 | 0.000 | 0.000 | 0.000 | 0.925 | 0.000 | 0.181 |
| V2_TEMPORAL-ONLY | 0.173 | 0.000 | 0.000 | 0.000 | 0.860 | 0.000 | 0.172 |
| V3_SEMANTIC-ONLY | 0.343 | 0.511 | 0.411 | 0.848 | 0.694 | 0.705 | 0.544 |
| V4_RRF-ALL | 0.408 | 0.421 | 0.177 | 0.698 | 0.984 | 0.698 | 0.549 |
| V7_SCORE-BLEND | 0.323 | 0.511 | 0.411 | 0.847 | 0.946 | 0.780 | 0.581 |

## MRR by variant × subset

| Variant | base | discriminator | utterance | era | axis | allen | combined |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1_CASCADE | 0.340 | 0.036 | 0.029 | 0.046 | 0.917 | 0.045 | 0.257 |
| V2_TEMPORAL-ONLY | 0.358 | 0.021 | 0.013 | 0.011 | 0.825 | 0.008 | 0.237 |
| V3_SEMANTIC-ONLY | 0.594 | 0.469 | 0.390 | 0.865 | 0.640 | 0.604 | 0.599 |
| V4_RRF-ALL | 0.608 | 0.346 | 0.078 | 0.637 | 1.000 | 0.620 | 0.578 |
| V7_SCORE-BLEND | 0.556 | 0.469 | 0.388 | 0.856 | 0.975 | 0.714 | 0.644 |

## Per-subset winner (R@5; ties by NDCG@10)

| Subset | Best | R@5 | NDCG@10 | MRR |
|---|---|---:|---:|---:|
| base | V4_RRF-ALL | 0.334 | 0.408 | 0.608 |
| discriminator | V3_SEMANTIC-ONLY | 0.667 | 0.511 | 0.469 |
| utterance | V3_SEMANTIC-ONLY | 0.467 | 0.411 | 0.390 |
| era | V4_RRF-ALL | 0.925 | 0.698 | 0.637 |
| axis | V4_RRF-ALL | 1.000 | 0.984 | 1.000 |
| allen | V7_SCORE-BLEND | 0.950 | 0.780 | 0.714 |
| combined | V7_SCORE-BLEND | 0.625 | 0.581 | 0.644 |

## Combined ranking

| Variant | Combined R@5 | LLM calls/q |
|---|---:|---:|
| V7_SCORE-BLEND | 0.625 | 0 |
| V4_RRF-ALL | 0.618 | 0 |
| V3_SEMANTIC-ONLY | 0.609 | 0 |
| V1_CASCADE | 0.177 | 0 |
| V2_TEMPORAL-ONLY | 0.172 | 0 |
