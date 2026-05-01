# Deep-Narrow Iterative Retrieval Study

Tests a 15-20 hop x 1 cue per hop retrieval architecture (untested shape) against v2f (shallow, 1x3 cues) and chain_with_scratchpad (3-5 hops x 1 cue).

## Headline: fair-backfill recall

| Variant | Dataset | n | base@20 | arch@20 | d@20 | base@50 | arch@50 | d@50 | avg LLM |
|---|---|---|---|---|---|---|---|---|---|
| meta_v2f | locomo_30q | 30 | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| chain_with_scratchpad | locomo_30q | 30 | 0.383 | 0.486 | +0.103 | 0.508 | 0.619 | +0.111 | 4.1 |
| v15_control | locomo_30q | 30 | 0.383 | 0.706 | +0.322 | 0.508 | 0.739 | +0.231 | 1.0 |
| deep_narrow_v1 | locomo_30q | 5 | 0.300 | 0.400 | +0.100 | 0.300 | 0.700 | +0.400 | 14.0 |

## Hop behavior (deep-narrow only)

| Variant | Dataset | avg hops | % hit max_hops | sat rate | stop reasons |
|---|---|---|---|---|---|
| deep_narrow_v1 | locomo_30q | 15.00 | 0.0% | 0.000 | segment_cap=5 |

## Per-category recall (deep_narrow_v1 vs meta_v2f)

### locomo_30q

| Category | n | v2f arch@20 | DN arch@20 | DN-v2f | v2f arch@50 | DN arch@50 | DN-v2f@50 |
|---|---|---|---|---|---|---|---|
| locomo_multi_hop | 1 | 0.625 | 0.000 | -0.625 | 0.875 | 0.500 | -0.375 |
| locomo_single_hop | 2 | 0.617 | 0.500 | -0.117 | 0.825 | 1.000 | +0.175 |
| locomo_temporal | 2 | 0.875 | 0.500 | -0.375 | 0.875 | 0.500 | -0.375 |

## Recall vs hop curve (deep_narrow_v1)

### locomo_30q

| hop | avg r@20 | avg r@50 | n_questions |
|---|---|---|---|
| 0 | 0.3000 | 0.3000 | 5 |
| 1 | 0.4000 | 0.5000 | 5 |
| 2 | 0.4000 | 0.5000 | 5 |
| 3 | 0.4000 | 0.5000 | 5 |
| 4 | 0.4000 | 0.5000 | 5 |
| 5 | 0.4000 | 0.8000 | 5 |
| 6 | 0.4000 | 0.8000 | 5 |
| 7 | 0.4000 | 0.8000 | 5 |
| 8 | 0.4000 | 0.7000 | 5 |
| 9 | 0.4000 | 0.7000 | 5 |
| 10 | 0.4000 | 0.7000 | 5 |
| 11 | 0.4000 | 0.7000 | 5 |
| 12 | 0.4000 | 0.7000 | 5 |
| 13 | 0.4000 | 0.7000 | 5 |
| 14 | 0.4000 | 0.7000 | 5 |
| 15 | 0.4000 | 0.7000 | 5 |

## Verdict

- LoCoMo d(DN-v2f) r@20: -0.3556, r@50: -0.1583. LLM calls: DN=14.0 vs v2f=1.0.