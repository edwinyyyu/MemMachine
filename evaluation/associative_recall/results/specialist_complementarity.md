# Specialist Complementarity Analysis

Does ensembling retrieval specialists (by union of gold-turns retrieved) produce a meaningful recall lift over v2f alone? Or do specialists mostly rediscover the same gold turns?

## Data coverage

| Dataset | n_questions | specialists re-run (all cache-hits) |
|---|---|---|
| locomo_30q | 30 | ['v2f', 'v2f_plus_types', 'type_enumerated', 'chain_with_scratchpad', 'v2f_style_explicit'] |
| synthetic_19q | 19 | ['v2f', 'v2f_plus_types', 'type_enumerated', 'chain_with_scratchpad', 'v2f_style_explicit'] |
| puzzle_16q | 16 | ['v2f', 'v2f_plus_types', 'type_enumerated', 'chain_with_scratchpad', 'v2f_style_explicit'] |
| advanced_23q | 23 | ['v2f', 'v2f_plus_types', 'type_enumerated', 'chain_with_scratchpad', 'v2f_style_explicit'] |

## Unique gains vs v2f (per-specialist)

Across all 88 questions, for each specialist s ≠ v2f at each K, how often does s retrieve a gold turn that v2f missed, and by how many turns on average?


### K=20

| Specialist | mean unique vs v2f | frac q with unique gain | mean overlap with v2f | mean recall |
|---|---|---|---|---|
| v2f_plus_types | 0.0227 | 0.0114 | 3.8068 | 0.6323 |
| type_enumerated | 0.9091 | 0.4886 | 2.5795 | 0.4926 |
| chain_with_scratchpad | 0.8295 | 0.4432 | 2.5795 | 0.5144 |
| v2f_style_explicit | 0.5909 | 0.3523 | 3.2841 | 0.6308 |
| v2f (ref) | — | — | — | 0.6323 |

### K=50

| Specialist | mean unique vs v2f | frac q with unique gain | mean overlap with v2f | mean recall |
|---|---|---|---|---|
| v2f_plus_types | 0.5682 | 0.3182 | 6.3182 | 0.9062 |
| type_enumerated | 0.2955 | 0.2159 | 5.9773 | 0.7955 |
| chain_with_scratchpad | 0.1591 | 0.125 | 6.25 | 0.7893 |
| v2f_style_explicit | 0.1932 | 0.1705 | 6.4318 | 0.8683 |
| v2f (ref) | — | — | — | 0.8789 |

## Union-ensemble ceiling (all 5 specialists)

| K | v2f-alone recall | union-5 recall | best-single recall | Δ union over v2f | Δ best-single over v2f |
|---|---|---|---|---|---|
| 20 | 0.6323 | 0.7827 | 0.7008 | +0.1504 | +0.0685 |
| 50 | 0.8789 | 0.9492 | 0.9326 | +0.0703 | +0.0537 |

## Ensemble candidates (v2f + subset of others)


### K=20

| size | ensemble | mean_recall | Δ vs v2f-alone |
|---|---|---|---|
| 1 | v2f | 0.6323 | +0.0000 |
| 2 | v2f+type_enumerated | 0.7131 | +0.0808 |
| 3 | v2f+type_enumerated+chain_with_scratchpad | 0.7585 | +0.1262 |
| 4 | v2f+type_enumerated+chain_with_scratchpad+v2f_style_explicit | 0.7827 | +0.1504 |
| 5 | v2f+v2f_plus_types+type_enumerated+chain_with_scratchpad+v2f_style_explicit | 0.7827 | +0.1504 |

### K=50

| size | ensemble | mean_recall | Δ vs v2f-alone |
|---|---|---|---|
| 1 | v2f | 0.8789 | +0.0000 |
| 2 | v2f+v2f_plus_types | 0.9291 | +0.0502 |
| 3 | v2f+v2f_plus_types+type_enumerated | 0.9443 | +0.0654 |
| 4 | v2f+v2f_plus_types+type_enumerated+chain_with_scratchpad | 0.9477 | +0.0688 |
| 5 | v2f+v2f_plus_types+type_enumerated+chain_with_scratchpad+v2f_style_explicit | 0.9492 | +0.0703 |

## Per-category unique gains (top 2 per specialist at K=20)


### v2f_plus_types

| Category | n | mean unique vs v2f | frac q with unique gain |
|---|---|---|---|
| conjunction | 3 | 0.6667 | 0.3333 |
| locomo_temporal | 16 | 0.0 | 0.0 |

### type_enumerated

| Category | n | mean unique vs v2f | frac q with unique gain |
|---|---|---|---|
| open_exploration | 2 | 4.0 | 1.0 |
| logic_constraint | 3 | 3.6667 | 1.0 |

### chain_with_scratchpad

| Category | n | mean unique vs v2f | frac q with unique gain |
|---|---|---|---|
| open_exploration | 2 | 3.5 | 1.0 |
| frequency_detection | 1 | 3.0 | 1.0 |

### v2f_style_explicit

| Category | n | mean unique vs v2f | frac q with unique gain |
|---|---|---|---|
| quantitative_aggregation | 3 | 3.3333 | 0.6667 |
| sequential_chain | 3 | 1.6667 | 0.6667 |

## Recommended ensemble composition

- **K=20 best pair**: `v2f+type_enumerated` → 0.7131 (Δ +0.0808 vs v2f-alone; cost ~2× v2f).

- **K=20 best trio**: `v2f+type_enumerated+chain_with_scratchpad` → 0.7585 (Δ +0.1262 vs v2f-alone; cost ~3× v2f).

- **K=50 best pair**: `v2f+v2f_plus_types` → 0.9291 (Δ +0.0502 vs v2f-alone; cost ~2× v2f).

- **K=50 best trio**: `v2f+v2f_plus_types+type_enumerated` → 0.9443 (Δ +0.0654 vs v2f-alone; cost ~3× v2f).


## Verdict

- Union-5 lift over v2f: **Δ@20 = +0.1504**, **Δ@50 = +0.0703** (absolute recall pp).

- Per-question best-single specialist lift: Δ@20 = +0.0685, Δ@50 = +0.0537. Union-5 beats the best-single by +0.0819 @20 and +0.0166 @50 — this gap is the *true complementarity* signal: specialists rescue different gold turns even within the same question.

- Overall mean unique-gains per (specialist, question, K): **0.446** gold turns (marginal/promising threshold: 0.15 / 0.50).

- **Ensemble verdict: worth building — the union delivers a real recall lift.**


### Cost vs benefit
- Per-query LLM cost (rough): v2f=1 call, type_enumerated=1, v2f_plus_types=2 (v2f stage + types stage), v2f_style_explicit=1, chain_with_scratchpad ≤5.

- **K=20**: best 2-specialist pair `v2f+type_enumerated` gains +8.08 pp at ~2× v2f cost → 4.04 pp per v2f-equivalent call.

- **K=50**: best 2-specialist pair `v2f+v2f_plus_types` gains +5.02 pp at ~2× v2f cost → 2.51 pp per v2f-equivalent call.

- Full union-5 at K=50: +7.03 pp for ~10× cost → ~0.70 pp per v2f-equivalent call. Diminishing returns past a 2- or 3-specialist ensemble.
