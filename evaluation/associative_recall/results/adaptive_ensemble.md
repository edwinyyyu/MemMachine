# Novelty-Gated Adaptive Ensemble

Sequentially run specialists in order `['v2f', 'type_enumerated', 'chain_with_scratchpad', 'v2f_plus_types', 'v2f_style_explicit']`. After each specialist past the first, measure novelty = |R_s \ R_accumulated| / K over its top-K turn_ids. If novelty > tau, merge (sum_cosine) and continue; else stop. Final top-K is fair-backfilled from raw-query cosine.

**Tau values**: [0.1, 0.2, 0.3]

**Budgets**: K=[20, 50]


## Specialist costs (× v2f)

| Specialist | cost |
|---|---|
| v2f | 1.0 |
| type_enumerated | 1.0 |
| chain_with_scratchpad | 5.0 |
| v2f_plus_types | 2.0 |
| v2f_style_explicit | 1.0 |

Total cost if all 5 called: **10.0**×


## Dataset: locomo_30q  (n_with_gold=30)

| Variant | r@20 | r@50 | mean_n_called@20 | mean_n_called@50 | mean_cost@20 | mean_cost@50 |
|---|---|---|---|---|---|---|
| v2f | 0.7556 | 0.8583 | 1.0 | 1.0 | 1.0× | 1.0× |
| ens_2_v2f_typeenum | 0.5806 | 0.9083 | 2.0 | 2.0 | 2.0× | 2.0× |
| ens_5 | 0.6806 | 0.9167 | 5.0 | 5.0 | 10.0× | 10.0× |
| adaptive_tau_0.1 | 0.5472 | 0.9167 | 2.533 | 2.6 | 4.667× | 3.767× |
| adaptive_tau_0.2 | 0.5583 | 0.9083 | 2.067 | 1.867 | 3.133× | 1.867× |
| adaptive_tau_0.3 | 0.5917 | 0.9083 | 1.8 | 1.633 | 2.2× | 1.633× |

## Dataset: synthetic_19q  (n_with_gold=19)

| Variant | r@20 | r@50 | mean_n_called@20 | mean_n_called@50 | mean_cost@20 | mean_cost@50 |
|---|---|---|---|---|---|---|
| v2f | 0.6130 | 0.8513 | 1.0 | 1.0 | 1.0× | 1.0× |
| ens_2_v2f_typeenum | 0.5864 | 0.8606 | 2.0 | 2.0 | 2.0× | 2.0× |
| ens_5 | 0.6224 | 0.8928 | 5.0 | 5.0 | 10.0× | 10.0× |
| adaptive_tau_0.1 | 0.5757 | 0.8752 | 2.316 | 2.158 | 4.0× | 2.684× |
| adaptive_tau_0.2 | 0.5867 | 0.8599 | 1.895 | 1.579 | 2.316× | 1.579× |
| adaptive_tau_0.3 | 0.6356 | 0.8432 | 1.211 | 1.053 | 1.211× | 1.053× |

## Combined (LoCoMo + synthetic)

| Variant | r@20 | r@50 | mean_n_called@20 | mean_n_called@50 | mean_cost@20 | mean_cost@50 |
|---|---|---|---|---|---|---|
| v2f | 0.7003 | 0.8556 | 1.0 | 1.0 | 1.0× | 1.0× |
| ens_2_v2f_typeenum | 0.5828 | 0.8898 | 2.0 | 2.0 | 2.0× | 2.0× |
| ens_5 | 0.6580 | 0.9074 | 5.0 | 5.0 | 10.0× | 10.0× |
| adaptive_tau_0.1 | 0.5583 | 0.9006 | 2.449 | 2.429 | 4.408× | 3.347× |
| adaptive_tau_0.2 | 0.5693 | 0.8895 | 2.0 | 1.755 | 2.816× | 1.755× |
| adaptive_tau_0.3 | 0.6087 | 0.8831 | 1.571 | 1.408 | 1.816× | 1.408× |

## Distribution of specialists-called (adaptive) @K=50


### locomo_30q

| Variant | n=1 | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|---|
| adaptive_tau_0.1 | 0 | 23 | 0 | 3 | 4 |
| adaptive_tau_0.2 | 4 | 26 | 0 | 0 | 0 |
| adaptive_tau_0.3 | 11 | 19 | 0 | 0 | 0 |

### synthetic_19q

| Variant | n=1 | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|---|
| adaptive_tau_0.1 | 1 | 16 | 0 | 2 | 0 |
| adaptive_tau_0.2 | 8 | 11 | 0 | 0 | 0 |
| adaptive_tau_0.3 | 18 | 1 | 0 | 0 | 0 |

## Per-category recall @K=50 (combined)

| Category | n | v2f | ens_2 | ens_5 | a_0.1 | a_0.2 | a_0.3 |
|---|---|---|---|---|---|---|---|
| control | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| conjunction | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| inference | 3 | 0.9394 | 0.9697 | 0.9697 | 0.9697 | 0.9394 | 0.9394 |
| locomo_temporal | 16 | 0.8750 | 0.9375 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| locomo_single_hop | 10 | 0.8250 | 0.8750 | 0.9000 | 0.9000 | 0.8750 | 0.8750 |
| completeness | 4 | 0.8654 | 0.8269 | 0.8846 | 0.8461 | 0.8461 | 0.8269 |
| locomo_multi_hop | 4 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 |
| proactive | 4 | 0.6434 | 0.7534 | 0.7856 | 0.8034 | 0.7534 | 0.6434 |
| procedural | 2 | 0.6607 | 0.5607 | 0.6863 | 0.5607 | 0.5607 | 0.6607 |

## Cost-per-pp-gain vs v2f (combined, K=50)

| Variant | r@50 | Δ vs v2f (pp) | mean_cost@50 | pp / cost |
|---|---|---|---|---|
| v2f | 0.8556 | +0.00 | 1.0× | +0.00 |
| ens_2_v2f_typeenum | 0.8898 | +3.42 | 2.0× | +1.71 |
| ens_5 | 0.9074 | +5.18 | 10.0× | +0.52 |
| adaptive_tau_0.1 | 0.9006 | +4.50 | 3.347× | +1.34 |
| adaptive_tau_0.2 | 0.8895 | +3.39 | 1.755× | +1.93 |
| adaptive_tau_0.3 | 0.8831 | +2.75 | 1.408× | +1.95 |

## Verdict

- **ens_5 @K=50 combined**: r@50=0.9074, cost=10.0×

- **best adaptive @K=50**: `adaptive_tau_0.1` r@50=0.9006, mean cost=3.347×

- **Recommendation**: NOTEWORTHY — `adaptive_tau_0.1` within 0.68pp of ens_5 at 33% cost. Cost saver but slight recall loss.
