# Final composition v2 — speaker-driven stacking

Composition v2 enters **speaker_user_filter** (+8.3pp @ K=20 on LoCoMo when it fires) into the stack, alongside the other shipped narrow wins. Evaluated on 4 datasets at K=20 and K=50, fair-backfilled.


Elapsed: 1007s.


## K=20 recall matrix

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q | overall |
|---|---|---|---|---|---|
| v2f | 0.7556 | 0.6130 | 0.4804 | 0.5931 | **0.6323** |
| speaker_filter_only | 0.8389 | 0.6130 | 0.4804 | 0.5931 | **0.6607** |
| speaker_plus_router | 0.8389 | 0.6183 | 0.4905 | 0.5873 | **0.6622** |
| speaker_plus_alias | 0.7944 | 0.6028 | 0.4922 | 0.5959 | **0.6462** |
| speaker_plus_critical | 0.8389 | 0.6467 | 0.4903 | 0.5803 | **0.6664** |
| speaker_all_in | 0.8472 | 0.5853 | 0.5194 | 0.5606 | **0.6561** |

## K=50 recall matrix

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q | overall |
|---|---|---|---|---|---|
| v2f | 0.8583 | 0.8513 | 0.9169 | 0.9021 | **0.8789** |
| ens_all_plus_crit | 0.9167 | 0.8968 | 0.9488 | 0.9299 | **0.9217** |
| composition_v1 | 0.8833 | 0.8829 | 0.9327 | 0.9345 | **0.9056** |
| composition_v2_all | 0.9167 | 0.8829 | 0.9327 | 0.9345 | **0.9170** |
| drop_speaker | 0.8833 | 0.8829 | 0.9327 | 0.9345 | **0.9056** |
| drop_ens_2 | 0.9167 | 0.8882 | 0.9223 | 0.9287 | **0.9147** |
| drop_alias | 0.9167 | 0.8829 | 0.9327 | 0.9374 | **0.9177** |
| drop_clause | 0.9167 | 0.8829 | 0.9327 | 0.9345 | **0.9170** |
| drop_context | 0.9167 | 0.8829 | 0.9327 | 0.9345 | **0.9170** |
| drop_critical | 0.9167 | 0.8789 | 0.9351 | 0.9345 | **0.9165** |
| drop_router | 0.9083 | 0.8753 | 0.9180 | 0.9115 | **0.9038** |

## Ablation: drop-one from composition_v2_all (LoCoMo K=50)

| Drop | r@50 | Δ vs v2_all |
|---|---:|---:|
| (none: v2_all) | 0.9167 | — |
| drop_speaker | 0.8833 | -0.0334 |
| drop_ens_2 | 0.9167 | +0.0000 |
| drop_alias | 0.9167 | +0.0000 |
| drop_clause | 0.9167 | +0.0000 |
| drop_context | 0.9167 | +0.0000 |
| drop_critical | 0.9167 | +0.0000 |
| drop_router | 0.9083 | -0.0084 |

## LoCoMo per-subset delta analysis at K=20

Comparing speaker_all_in vs v2f on subsets defined by which narrow win fires.

| Subset | n | v2f@20 | speaker_all_in@20 | Δ |
|---|---:|---:|---:|---:|
| speaker_fires | 18 | 0.8056 | 0.9444 | +0.1388 |
| alias_fires | 30 | 0.7556 | 0.8472 | +0.0916 |
| speaker_and_alias | 18 | 0.8056 | 0.9444 | +0.1388 |
| speaker_not_alias | 0 | - | - | - |
| neither | 0 | - | - | - |

## Per-(dataset, K) production recipes

Per-cell best variant. K=20 draws from K20_VARIANTS; K=50 draws from K50_VARIANTS.

| Dataset | K | Best variant | Recall | Δ vs v2f |
|---|---:|---|---:|---:|
| locomo_30q | 20 | speaker_all_in | 0.8472 | +0.0916 |
| locomo_30q | 50 | ens_all_plus_crit | 0.9167 | +0.0584 |
| synthetic_19q | 20 | speaker_plus_critical | 0.6467 | +0.0337 |
| synthetic_19q | 50 | ens_all_plus_crit | 0.8968 | +0.0455 |
| puzzle_16q | 20 | speaker_all_in | 0.5194 | +0.0390 |
| puzzle_16q | 50 | ens_all_plus_crit | 0.9488 | +0.0319 |
| advanced_23q | 20 | speaker_plus_alias | 0.5959 | +0.0028 |
| advanced_23q | 50 | drop_alias | 0.9374 | +0.0353 |

## Decision rules

- LoCoMo K=20 v2f = 0.7556
- LoCoMo K=20 speaker_all_in = 0.8472
- Δ speaker_all_in vs v2f = +0.0916
  => **NEW K=20 SHIP (LoCoMo): speaker_all_in** (+9.2pp >= 5pp threshold)

- LoCoMo K=50 ens_all_plus_crit (prior reference) = 0.9167
- LoCoMo K=50 composition_v2_all = 0.9167
- Δ composition_v2_all vs ens_all_plus_crit = +0.0000
  => composition_v2_all did NOT clearly beat prior ceiling.
