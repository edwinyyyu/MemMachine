# Composition Study — do the three shipped wins compose?

Three shipped wins being composed: **keyword router**, **ens_2_v2f_typeenum**, **critical-info always_top_M**.

Variants evaluated on LoCoMo-30 + synthetic-19 + puzzle-16 + advanced-23 at K=20 and K=50, fair-backfilled.


Elapsed: 1133s.


## Recall table (r@20)

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q | overall |
|---|---|---|---|---|---|
| v2f | 0.7556 | 0.6130 | 0.4804 | 0.5931 | **0.6323** |
| router_v2fplus_default | 0.7556 | 0.6276 | 0.4905 | 0.5844 | **0.6350** |
| ens_2 | 0.5806 | 0.5864 | 0.5185 | 0.5693 | **0.5676** |
| crit_only | 0.7556 | 0.6427 | 0.4775 | 0.5923 | **0.6380** |
| ens_2_plus_crit | 0.5806 | 0.5887 | 0.5218 | 0.5726 | **0.5696** |
| router_ens | 0.7556 | 0.6183 | 0.4905 | 0.5873 | **0.6338** |
| router_ens_plus_crit | 0.7556 | 0.6427 | 0.4875 | 0.5865 | **0.6383** |
| ens_all_plus_crit | 0.6806 | 0.6318 | 0.5065 | 0.6079 | **0.6194** |

## Recall table (r@50)

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q | overall |
|---|---|---|---|---|---|
| v2f | 0.8583 | 0.8513 | 0.9169 | 0.9021 | **0.8789** |
| router_v2fplus_default | 0.8833 | 0.8789 | 0.9299 | 0.9345 | **0.9042** |
| ens_2 | 0.9083 | 0.8606 | 0.9213 | 0.8949 | **0.8969** |
| crit_only | 0.8583 | 0.8554 | 0.9117 | 0.9021 | **0.8788** |
| ens_2_plus_crit | 0.9083 | 0.8647 | 0.9150 | 0.8949 | **0.8966** |
| router_ens | 0.8833 | 0.8789 | 0.9299 | 0.9316 | **0.9034** |
| router_ens_plus_crit | 0.8833 | 0.8829 | 0.9280 | 0.9316 | **0.9040** |
| ens_all_plus_crit | 0.9167 | 0.8968 | 0.9488 | 0.9299 | **0.9217** |

## Additivity check: ens_2 + crit vs sum of gains

Compute Δ(ens_2 + crit) vs Δ(ens_2) + Δ(crit_only) per dataset × K.

| Dataset | K | v2f | ens_2 | crit_only | ens_2+crit | Δ_ens_2 | Δ_crit | Δ_both | sum_indiv | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| locomo_30q | 20 | 0.7556 | 0.5806 | 0.7556 | 0.5806 | -0.1750 | +0.0000 | -0.1750 | -0.1750 | cannibalize |
| locomo_30q | 50 | 0.8583 | 0.9083 | 0.8583 | 0.9083 | +0.0500 | +0.0000 | +0.0500 | +0.0500 | partial (dominant single sticks) |
| synthetic_19q | 20 | 0.6130 | 0.5864 | 0.6427 | 0.5887 | -0.0266 | +0.0297 | -0.0243 | +0.0031 | cannibalize |
| synthetic_19q | 50 | 0.8513 | 0.8606 | 0.8554 | 0.8647 | +0.0093 | +0.0041 | +0.0134 | +0.0134 | additive |
| puzzle_16q | 20 | 0.4804 | 0.5185 | 0.4775 | 0.5218 | +0.0381 | -0.0029 | +0.0414 | +0.0352 | additive |
| puzzle_16q | 50 | 0.9169 | 0.9213 | 0.9117 | 0.9150 | +0.0044 | -0.0052 | -0.0019 | -0.0008 | cannibalize |
| advanced_23q | 20 | 0.5931 | 0.5693 | 0.5923 | 0.5726 | -0.0238 | -0.0008 | -0.0205 | -0.0246 | cannibalize |
| advanced_23q | 50 | 0.9021 | 0.8949 | 0.9021 | 0.8949 | -0.0072 | +0.0000 | -0.0072 | -0.0072 | cannibalize |

## Keyword-router label distribution

| Dataset | v2f | v2f_plus_types | type_enumerated | chain | v2f_style_explicit |
|---|---|---|---|---|---|
| locomo_30q | 0 | 30 | 0 | 0 | 0 |
| synthetic_19q | 0 | 17 | 0 | 2 | 0 |
| puzzle_16q | 0 | 12 | 0 | 1 | 3 |
| advanced_23q | 0 | 22 | 0 | 1 | 0 |

## Per-category r@50 on locomo_30q

| category | n | v2f | router_v2fplus_default | ens_2 | crit_only | ens_2_plus_crit | router_ens | router_ens_plus_crit | ens_all_plus_crit |
|---|---|---|---|---|---|---|---|---|---|
| locomo_multi_hop | 4 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| locomo_single_hop | 10 | 0.825 | 0.900 | 0.875 | 0.825 | 0.875 | 0.900 | 0.900 | 0.900 |
| locomo_temporal | 16 | 0.875 | 0.875 | 0.938 | 0.875 | 0.938 | 0.875 | 0.875 | 0.938 |

## Per-category r@50 on synthetic_19q

| category | n | v2f | router_v2fplus_default | ens_2 | crit_only | ens_2_plus_crit | router_ens | router_ens_plus_crit | ens_all_plus_crit |
|---|---|---|---|---|---|---|---|---|---|
| completeness | 4 | 0.865 | 0.865 | 0.827 | 0.865 | 0.827 | 0.865 | 0.865 | 0.885 |
| conjunction | 3 | 1.000 | 0.952 | 1.000 | 1.000 | 1.000 | 0.952 | 0.952 | 1.000 |
| control | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| inference | 3 | 0.939 | 0.970 | 0.970 | 0.939 | 0.970 | 0.970 | 0.970 | 0.970 |
| proactive | 4 | 0.643 | 0.791 | 0.753 | 0.663 | 0.773 | 0.791 | 0.811 | 0.805 |
| procedural | 2 | 0.661 | 0.653 | 0.561 | 0.661 | 0.561 | 0.653 | 0.653 | 0.686 |

## LLM retrieval cost per question (relative to 1 v2f call)

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q |
|---|---|---|---|---|
| v2f | 1.00× | 1.00× | 1.00× | 1.00× |
| router_v2fplus_default | 2.00× | 2.32× | 2.00× | 2.13× |
| ens_2 | 2.00× | 2.00× | 2.00× | 2.00× |
| crit_only | 1.00× | 1.00× | 1.00× | 1.00× |
| ens_2_plus_crit | 2.00× | 2.00× | 2.00× | 2.00× |
| router_ens | 2.00× | 2.42× | 2.06× | 2.17× |
| router_ens_plus_crit | 2.00× | 2.42× | 2.06× | 2.17× |
| ens_all_plus_crit | 10.00× | 10.00× | 10.00× | 10.00× |

## Critical-info classifier (ingest-time, one-off cost)

- Prompt version: v3
- New calls this run: 1729, cached: 866
- Input tokens: 727241 output tokens: 414077
- Est USD (gpt-5-mini @ $0.25/M in, $2/M out): $1.0100

  - locomo_30q: flag rate 0.00% (0/419 turns), 0 alt-keys

  - synthetic_19q: flag rate 3.68% (17/462 turns), 51 alt-keys

  - puzzle_16q: flag rate 1.08% (9/834 turns), 27 alt-keys

  - advanced_23q: flag rate 1.14% (10/880 turns), 30 alt-keys


## Verdict

- Best variant overall @ K=50 (weighted): **ens_all_plus_crit** r@50=0.9217

- LoCoMo-30 @ K=50: v2f=0.8583 ens_2=0.9083 crit=0.8583 ens_2+crit=0.9083 router_ens=0.8833 router_ens+crit=0.8833

- synthetic-19 @ K=20: v2f=0.6130 ens_2=0.5864 crit=0.6427 ens_2+crit=0.5887
