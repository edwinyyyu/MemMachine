# Ensemble Retrieval Study

Does a specialist ensemble beat v2f-alone at **fair K budget** — where the ensemble's merged top-K is truncated to the same number of segments as v2f's top-K? If so, which merging strategy and ensemble composition ships?

**Compositions**: ['ens_2_v2f_v2fplus', 'ens_2_v2f_typeenum', 'ens_3', 'ens_5']

**Merging strategies**: ['max_cosine', 'sum_cosine', 'rrf', 'round_robin']

**Budgets**: K=(20, 50)

Each specialist is re-run cache-only; if a prompt is a cache miss the specialist emits `ACTION: DONE` (no new cues). Ensemble merging is followed by fair-backfill from cosine top-K of the raw query.


## LLM-call cost per question (relative)

| Setup | LLM calls / question |
|---|---|
| v2f-alone | 1.0 |
| ens_2_v2f_v2fplus (v2f+v2f_plus_types) | 3.0 |
| ens_2_v2f_typeenum (v2f+type_enumerated) | 2.0 |
| ens_3 (v2f+v2f_plus_types+type_enumerated) | 4.0 |
| ens_5 (v2f+v2f_plus_types+type_enumerated+chain_with_scratchpad+v2f_style_explicit) | 10.0 |

## Headline: LoCoMo @ K=50 (primary fair-budget test)

- v2f-alone @K=50 on LoCoMo: **0.8583**

| Ensemble | Strategy | r@50 | Δ vs v2f | LLM×v2f |
|---|---|---|---|---|
| ens_3 | max_cosine | 0.9167 | +0.0584 | 4.0× |
| ens_3 | sum_cosine | 0.9167 | +0.0584 | 4.0× |
| ens_5 | sum_cosine | 0.9167 | +0.0584 | 10.0× |
| ens_2_v2f_typeenum | max_cosine | 0.9083 | +0.0500 | 2.0× |
| ens_2_v2f_typeenum | sum_cosine | 0.9083 | +0.0500 | 2.0× |
| ens_2_v2f_typeenum | rrf | 0.9083 | +0.0500 | 2.0× |
| ens_2_v2f_typeenum | round_robin | 0.9083 | +0.0500 | 2.0× |
| ens_3 | rrf | 0.9083 | +0.0500 | 4.0× |
| ens_5 | rrf | 0.9083 | +0.0500 | 10.0× |
| ens_5 | round_robin | 0.9083 | +0.0500 | 10.0× |
| ens_2_v2f_v2fplus | max_cosine | 0.8833 | +0.0250 | 3.0× |
| ens_2_v2f_v2fplus | sum_cosine | 0.8833 | +0.0250 | 3.0× |
| ens_2_v2f_v2fplus | rrf | 0.8833 | +0.0250 | 3.0× |
| ens_2_v2f_v2fplus | round_robin | 0.8833 | +0.0250 | 3.0× |
| ens_5 | max_cosine | 0.8833 | +0.0250 | 10.0× |
| ens_3 | round_robin | 0.8750 | +0.0167 | 4.0× |

## Dataset: locomo_30q

n questions with gold = 30


### K=20

v2f-alone recall = **0.7556**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.6222 (-0.1334) | 0.7083 (-0.0473) | 0.7556 (+0.0000) | 0.7556 (+0.0000) |
| ens_2_v2f_typeenum | 0.5917 (-0.1639) | 0.5806 (-0.1750) | 0.6889 (-0.0667) | 0.6556 (-0.1000) |
| ens_3 | 0.5306 (-0.2250) | 0.6639 (-0.0917) | 0.7722 (+0.0166) | 0.6889 (-0.0667) |
| ens_5 | 0.4417 (-0.3139) | 0.6806 (-0.0750) | 0.7389 (-0.0167) | 0.6444 (-0.1112) |

### K=50

v2f-alone recall = **0.8583**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.8833 (+0.0250) | 0.8833 (+0.0250) | 0.8833 (+0.0250) | 0.8833 (+0.0250) |
| ens_2_v2f_typeenum | 0.9083 (+0.0500) | 0.9083 (+0.0500) | 0.9083 (+0.0500) | 0.9083 (+0.0500) |
| ens_3 | 0.9167 (+0.0584) | 0.9167 (+0.0584) | 0.9083 (+0.0500) | 0.8750 (+0.0167) |
| ens_5 | 0.8833 (+0.0250) | 0.9167 (+0.0584) | 0.9083 (+0.0500) | 0.9083 (+0.0500) |

## Dataset: synthetic_19q

n questions with gold = 19


### K=20

v2f-alone recall = **0.6130**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.5830 (-0.0300) | 0.6006 (-0.0124) | 0.5980 (-0.0150) | 0.6206 (+0.0076) |
| ens_2_v2f_typeenum | 0.6092 (-0.0038) | 0.5864 (-0.0266) | 0.5741 (-0.0389) | 0.5958 (-0.0172) |
| ens_3 | 0.5870 (-0.0260) | 0.5951 (-0.0179) | 0.5942 (-0.0188) | 0.6140 (+0.0010) |
| ens_5 | 0.5870 (-0.0260) | 0.6224 (+0.0094) | 0.6277 (+0.0147) | 0.6240 (+0.0110) |

### K=50

v2f-alone recall = **0.8513**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.8821 (+0.0308) | 0.8821 (+0.0308) | 0.8892 (+0.0379) | 0.8892 (+0.0379) |
| ens_2_v2f_typeenum | 0.8606 (+0.0093) | 0.8606 (+0.0093) | 0.8606 (+0.0093) | 0.8606 (+0.0093) |
| ens_3 | 0.8680 (+0.0167) | 0.8736 (+0.0223) | 0.8843 (+0.0330) | 0.8843 (+0.0330) |
| ens_5 | 0.8796 (+0.0283) | 0.8928 (+0.0415) | 0.9028 (+0.0515) | 0.9068 (+0.0555) |

## Dataset: puzzle_16q

n questions with gold = 16


### K=20

v2f-alone recall = **0.4804**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.4451 (-0.0353) | 0.4376 (-0.0428) | 0.4804 (+0.0000) | 0.4804 (+0.0000) |
| ens_2_v2f_typeenum | 0.4386 (-0.0418) | 0.5185 (+0.0381) | 0.5597 (+0.0793) | 0.4716 (-0.0088) |
| ens_3 | 0.4369 (-0.0435) | 0.5019 (+0.0215) | 0.5646 (+0.0842) | 0.4721 (-0.0083) |
| ens_5 | 0.4362 (-0.0442) | 0.5032 (+0.0228) | 0.5393 (+0.0589) | 0.4642 (-0.0162) |

### K=50

v2f-alone recall = **0.9169**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.9425 (+0.0256) | 0.9425 (+0.0256) | 0.9416 (+0.0247) | 0.9416 (+0.0247) |
| ens_2_v2f_typeenum | 0.9213 (+0.0044) | 0.9213 (+0.0044) | 0.9213 (+0.0044) | 0.9213 (+0.0044) |
| ens_3 | 0.9482 (+0.0313) | 0.9554 (+0.0385) | 0.9354 (+0.0185) | 0.9354 (+0.0185) |
| ens_5 | 0.9380 (+0.0211) | 0.9488 (+0.0319) | 0.9489 (+0.0320) | 0.9463 (+0.0294) |

## Dataset: advanced_23q

n questions with gold = 23


### K=20

v2f-alone recall = **0.5931**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.5130 (-0.0801) | 0.5399 (-0.0532) | 0.5931 (+0.0000) | 0.5931 (+0.0000) |
| ens_2_v2f_typeenum | 0.5095 (-0.0836) | 0.5693 (-0.0238) | 0.5945 (+0.0014) | 0.5735 (-0.0196) |
| ens_3 | 0.5046 (-0.0885) | 0.5685 (-0.0246) | 0.6014 (+0.0083) | 0.5953 (+0.0022) |
| ens_5 | 0.5038 (-0.0893) | 0.6049 (+0.0118) | 0.6121 (+0.0190) | 0.5847 (-0.0084) |

### K=50

v2f-alone recall = **0.9021**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.9357 (+0.0336) | 0.9357 (+0.0336) | 0.9316 (+0.0295) | 0.9316 (+0.0295) |
| ens_2_v2f_typeenum | 0.8949 (-0.0072) | 0.8949 (-0.0072) | 0.8949 (-0.0072) | 0.8949 (-0.0072) |
| ens_3 | 0.9415 (+0.0394) | 0.9415 (+0.0394) | 0.9180 (+0.0159) | 0.9180 (+0.0159) |
| ens_5 | 0.9327 (+0.0306) | 0.9330 (+0.0309) | 0.9268 (+0.0247) | 0.9268 (+0.0247) |

## All datasets aggregated


### K=20

v2f-alone recall = **0.6323**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.5530 (-0.0793) | 0.5918 (-0.0405) | 0.6291 (-0.0032) | 0.6339 (+0.0016) |
| ens_2_v2f_typeenum | 0.5462 (-0.0861) | 0.5676 (-0.0647) | 0.6159 (-0.0164) | 0.5878 (-0.0445) |
| ens_3 | 0.5189 (-0.1134) | 0.5947 (-0.0376) | 0.6514 (+0.0191) | 0.6088 (-0.0235) |
| ens_5 | 0.4883 (-0.1440) | 0.6160 (-0.0163) | 0.6455 (+0.0132) | 0.5917 (-0.0406) |

### K=50

v2f-alone recall = **0.8789**

| Ensemble | max_cosine | sum_cosine | rrf | round_robin |
|---|---|---|---|---|
| ens_2_v2f_v2fplus | 0.9075 (+0.0286) | 0.9075 (+0.0286) | 0.9078 (+0.0289) | 0.9078 (+0.0289) |
| ens_2_v2f_typeenum | 0.8969 (+0.0180) | 0.8969 (+0.0180) | 0.8969 (+0.0180) | 0.8969 (+0.0180) |
| ens_3 | 0.9184 (+0.0395) | 0.9209 (+0.0420) | 0.9106 (+0.0317) | 0.8992 (+0.0203) |
| ens_5 | 0.9054 (+0.0265) | 0.9216 (+0.0427) | 0.9193 (+0.0404) | 0.9197 (+0.0408) |

## Per-category gains at K=50 (best ensemble×strategy)

Winner: `ens_5 × sum_cosine` @ r@50=0.9216 (Δ = +0.0427).

| Category | n | v2f r@50 | ens r@50 | Δ |
|---|---|---|---|---|
| proactive | 4 | 0.6434 | 0.7856 | +0.1422 |
| logic_constraint | 3 | 0.7581 | 0.8918 | +0.1337 |
| locomo_single_hop | 10 | 0.8250 | 0.9000 | +0.0750 |
| consistency_checking | 2 | 0.9285 | 1.0000 | +0.0715 |
| locomo_temporal | 16 | 0.8750 | 0.9375 | +0.0625 |
| evolving_terminology | 5 | 0.8394 | 0.8996 | +0.0602 |
| frequency_detection | 1 | 0.8947 | 0.9474 | +0.0527 |
| unfinished_business | 3 | 0.8718 | 0.9231 | +0.0513 |
| contradiction | 2 | 0.9584 | 1.0000 | +0.0416 |
| inference | 3 | 0.9394 | 0.9697 | +0.0303 |
| procedural | 2 | 0.6607 | 0.6863 | +0.0256 |
| open_exploration | 2 | 0.8810 | 0.9047 | +0.0237 |
| completeness | 4 | 0.8654 | 0.8846 | +0.0192 |
| absence_inference | 3 | 0.9315 | 0.9500 | +0.0185 |
| negation | 3 | 0.8697 | 0.8872 | +0.0175 |
| quantitative_aggregation | 3 | 0.8889 | 0.8910 | +0.0021 |
| locomo_multi_hop | 4 | 0.8750 | 0.8750 | +0.0000 |
| control | 3 | 1.0000 | 1.0000 | +0.0000 |
| conjunction | 3 | 1.0000 | 1.0000 | +0.0000 |
| state_change | 3 | 1.0000 | 1.0000 | +0.0000 |
| perspective_separation | 4 | 0.9773 | 0.9773 | +0.0000 |
| constraint_propagation | 2 | 1.0000 | 1.0000 | +0.0000 |
| sequential_chain | 3 | 0.9744 | 0.9487 | -0.0257 |

## Verdict

- **LoCoMo K=50**: best ensemble×strategy = `ens_3 × max_cosine` → r@50=0.9167, Δ=+0.0584 pp vs v2f-alone (cost 4.0×).

- Union-5 set-theoretic ceiling on LoCoMo K=50 = 0.9492 (from specialist_complementarity). This is what we'd get if merge had no K-budget truncation penalty.

- **Recommendation: SHIP — `ens_2_v2f_typeenum × max_cosine` gives Δ=+0.0500 pp at 2.0× cost (meets the ≥3pp @ <3× ship-rule)**


All LoCoMo K=50 ensemble×strategy settings meeting the ≥3pp ship threshold (sorted by cost):

| cost | ensemble | strategy | r@50 | Δ | pp/call |
|---|---|---|---|---|---|
| 2.0× | ens_2_v2f_typeenum | max_cosine | 0.9083 | +0.0500 | 2.50 |
| 2.0× | ens_2_v2f_typeenum | round_robin | 0.9083 | +0.0500 | 2.50 |
| 2.0× | ens_2_v2f_typeenum | rrf | 0.9083 | +0.0500 | 2.50 |
| 2.0× | ens_2_v2f_typeenum | sum_cosine | 0.9083 | +0.0500 | 2.50 |
| 4.0× | ens_3 | max_cosine | 0.9167 | +0.0584 | 1.46 |
| 4.0× | ens_3 | rrf | 0.9083 | +0.0500 | 1.25 |
| 4.0× | ens_3 | sum_cosine | 0.9167 | +0.0584 | 1.46 |
| 10.0× | ens_5 | round_robin | 0.9083 | +0.0500 | 0.50 |
| 10.0× | ens_5 | rrf | 0.9083 | +0.0500 | 0.50 |
| 10.0× | ens_5 | sum_cosine | 0.9167 | +0.0584 | 0.58 |

### Strategy comparison (all datasets, K=50)

Mean r@50 across all compositions by strategy:

| Strategy | mean r@50 across ensembles |
|---|---|
| sum_cosine | 0.9117 |
| rrf | 0.9086 |
| max_cosine | 0.9071 |
| round_robin | 0.9059 |