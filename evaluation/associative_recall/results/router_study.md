# Router Study

Question: can a cheap router dispatch across specialists and beat v2f-only across many question categories?

## Setup

- 5 specialists: v2f (baseline), v2f_plus_types (K=50 Pareto), type_enumerated (logic_constraint), chain (chain_with_scratchpad), v2f_style_explicit (cross-dataset winner).

- 6 routers: v2f_only (control), oracle (ceiling via per-category best specialist), llm_router_mini (gpt-5-mini), llm_router_nano (gpt-5-nano), keyword_router (regex rules), embedding_router (nearest-exemplar cosine).

- 4 datasets (88 questions total) × {K=20, K=50}. Per-question specialist recalls loaded from existing cached per-question result files; no specialist code was modified.

## Overall recall

| Router | r@20 | Δ@20 | r@50 | Δ@50 | routing accuracy vs oracle @20 / @50 |
|---|---|---|---|---|---|
| v2f_only | 0.6323 | +0.1730 | 0.8789 | +0.1346 | 0.53 / 0.48 |
| oracle | 0.6679 | +0.2086 | 0.9133 | +0.1691 | 1.00 / 1.00 |
| llm_router_mini | 0.6430 | +0.1837 | 0.8747 | +0.1305 | 0.48 / 0.36 |
| llm_router_nano | 0.6432 | +0.1839 | 0.8741 | +0.1298 | 0.39 / 0.40 |
| keyword_router | 0.6350 | +0.1758 | 0.9042 | +0.1599 | 0.00 / 0.40 |
| embedding_router | 0.6363 | +0.1771 | 0.8718 | +0.1275 | 0.40 / 0.31 |

## Per-dataset r@20

| Router | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q |
|---|---|---|---|---|
| v2f_only | 0.7556 | 0.6130 | 0.4804 | 0.5931 |
| oracle | 0.7722 | 0.6624 | 0.5360 | 0.6282 |
| llm_router_mini | 0.7556 | 0.6076 | 0.5096 | 0.6182 |
| llm_router_nano | 0.7722 | 0.6291 | 0.4788 | 0.6009 |
| keyword_router | 0.7556 | 0.6276 | 0.4905 | 0.5844 |
| embedding_router | 0.7556 | 0.5885 | 0.5100 | 0.6082 |

## Per-dataset r@50

| Router | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q |
|---|---|---|---|---|
| v2f_only | 0.8583 | 0.8513 | 0.9169 | 0.9021 |
| oracle | 0.8833 | 0.8941 | 0.9566 | 0.9383 |
| llm_router_mini | 0.8417 | 0.8464 | 0.9317 | 0.9016 |
| llm_router_nano | 0.8500 | 0.8407 | 0.9098 | 0.9082 |
| keyword_router | 0.8833 | 0.8789 | 0.9299 | 0.9345 |
| embedding_router | 0.8583 | 0.8389 | 0.9232 | 0.8809 |

## Per-category r@20

| Category | n | v2f_only | oracle | llm_router_mini | llm_router_nano | keyword_router | embedding_router |
|---|---|---|---|---|---|---|---|
| absence_inference | 3 | 0.4574 | 0.4870 | 0.4797 | 0.4797 | 0.4574 | 0.4519 |
| completeness | 4 | 0.4551 | 0.5577 | 0.4551 | 0.4551 | 0.4551 | 0.4359 |
| conjunction | 3 | 0.8095 | 0.8571 | 0.7143 | 0.8095 | 0.8095 | 0.6667 |
| consistency_checking | 2 | 0.6785 | 0.7857 | 0.7857 | 0.6785 | 0.6785 | 0.7857 |
| constraint_propagation | 2 | 0.6131 | 0.6131 | 0.5605 | 0.5605 | 0.6131 | 0.4369 |
| contradiction | 2 | 0.7333 | 0.7333 | 0.7333 | 0.6500 | 0.7333 | 0.7333 |
| control | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| evolving_terminology | 5 | 0.4809 | 0.5045 | 0.5412 | 0.5045 | 0.4409 | 0.5012 |
| frequency_detection | 1 | 0.5789 | 0.5789 | 0.5789 | 0.5789 | 0.5789 | 0.5789 |
| inference | 3 | 0.7662 | 0.8615 | 0.8182 | 0.9091 | 0.7662 | 0.7705 |
| locomo_multi_hop | 4 | 0.6250 | 0.7500 | 0.6250 | 0.7500 | 0.6250 | 0.6250 |
| locomo_single_hop | 10 | 0.6167 | 0.6167 | 0.6167 | 0.6167 | 0.6167 | 0.6167 |
| locomo_temporal | 16 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 |
| logic_constraint | 3 | 0.1663 | 0.3503 | 0.3328 | 0.2243 | 0.2218 | 0.3503 |
| negation | 3 | 0.5201 | 0.5201 | 0.5201 | 0.5201 | 0.5201 | 0.4674 |
| open_exploration | 2 | 0.5000 | 0.5000 | 0.4762 | 0.4762 | 0.5000 | 0.5000 |
| perspective_separation | 4 | 0.7159 | 0.7424 | 0.7576 | 0.7576 | 0.7159 | 0.7159 |
| proactive | 4 | 0.3514 | 0.3765 | 0.3582 | 0.3707 | 0.4207 | 0.3582 |
| procedural | 2 | 0.3470 | 0.3470 | 0.3470 | 0.2470 | 0.3470 | 0.3470 |
| quantitative_aggregation | 3 | 0.4915 | 0.6149 | 0.4915 | 0.4915 | 0.4915 | 0.5769 |
| sequential_chain | 3 | 0.5093 | 0.5920 | 0.5683 | 0.5683 | 0.5349 | 0.5170 |
| state_change | 3 | 0.6071 | 0.6071 | 0.5307 | 0.5307 | 0.5794 | 0.5784 |
| unfinished_business | 3 | 0.7253 | 0.7253 | 0.7253 | 0.7253 | 0.7253 | 0.8205 |

## Per-category r@50

| Category | n | v2f_only | oracle | llm_router_mini | llm_router_nano | keyword_router | embedding_router |
|---|---|---|---|---|---|---|---|
| absence_inference | 3 | 0.9315 | 0.9537 | 0.9537 | 0.9537 | 0.9315 | 0.9537 |
| completeness | 4 | 0.8654 | 0.8846 | 0.8654 | 0.8654 | 0.8654 | 0.8654 |
| conjunction | 3 | 1.0000 | 1.0000 | 0.9524 | 0.9524 | 0.9524 | 0.9047 |
| consistency_checking | 2 | 0.9285 | 1.0000 | 0.9285 | 1.0000 | 1.0000 | 0.9285 |
| constraint_propagation | 2 | 1.0000 | 1.0000 | 0.9500 | 0.9500 | 0.9237 | 1.0000 |
| contradiction | 2 | 0.9584 | 1.0000 | 1.0000 | 0.9584 | 1.0000 | 0.9584 |
| control | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| evolving_terminology | 5 | 0.8394 | 0.8980 | 0.8159 | 0.8010 | 0.9114 | 0.7740 |
| frequency_detection | 1 | 0.8947 | 1.0000 | 0.8947 | 0.8947 | 1.0000 | 0.8947 |
| inference | 3 | 0.9394 | 0.9697 | 0.9394 | 0.9697 | 0.9697 | 0.9394 |
| locomo_multi_hop | 4 | 0.8750 | 0.8750 | 0.7500 | 0.7500 | 0.8750 | 0.8750 |
| locomo_single_hop | 10 | 0.8250 | 0.9000 | 0.8250 | 0.8500 | 0.9000 | 0.8250 |
| locomo_temporal | 16 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 |
| logic_constraint | 3 | 0.7581 | 0.9196 | 0.8187 | 0.7454 | 0.8363 | 0.8012 |
| negation | 3 | 0.8697 | 0.8872 | 0.8872 | 0.8872 | 0.8872 | 0.8697 |
| open_exploration | 2 | 0.8810 | 0.8810 | 0.8334 | 0.8095 | 0.8572 | 0.8334 |
| perspective_separation | 4 | 0.9773 | 0.9773 | 0.9773 | 0.9773 | 0.9773 | 0.9356 |
| proactive | 4 | 0.6434 | 0.8048 | 0.6559 | 0.6059 | 0.7913 | 0.6559 |
| procedural | 2 | 0.6607 | 0.6607 | 0.6607 | 0.6607 | 0.6529 | 0.6607 |
| quantitative_aggregation | 3 | 0.8889 | 0.9167 | 0.8889 | 0.9167 | 0.9167 | 0.8397 |
| sequential_chain | 3 | 0.9744 | 0.9744 | 0.9744 | 0.9744 | 0.9744 | 0.9744 |
| state_change | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9792 | 1.0000 |
| unfinished_business | 3 | 0.8718 | 0.9231 | 0.9231 | 0.9231 | 0.9231 | 0.9231 |

## Oracle per-category picks


### K=20

| Category | Specialist |
|---|---|
| absence_inference | type_enumerated |
| completeness | v2f_style_explicit |
| conjunction | v2f_style_explicit |
| consistency_checking | type_enumerated |
| constraint_propagation | v2f |
| contradiction | v2f |
| control | v2f |
| evolving_terminology | v2f_style_explicit |
| frequency_detection | v2f |
| inference | v2f_style_explicit |
| locomo_multi_hop | v2f_style_explicit |
| locomo_single_hop | v2f |
| locomo_temporal | v2f |
| logic_constraint | type_enumerated |
| negation | v2f |
| open_exploration | v2f |
| perspective_separation | chain |
| proactive | type_enumerated |
| procedural | v2f |
| quantitative_aggregation | v2f_style_explicit |
| sequential_chain | type_enumerated |
| state_change | v2f |
| unfinished_business | v2f |

### K=50

| Category | Specialist |
|---|---|
| absence_inference | v2f_style_explicit |
| completeness | v2f_style_explicit |
| conjunction | v2f |
| consistency_checking | v2f_plus_types |
| constraint_propagation | v2f |
| contradiction | v2f_plus_types |
| control | v2f |
| evolving_terminology | v2f_plus_types |
| frequency_detection | v2f_plus_types |
| inference | v2f_plus_types |
| locomo_multi_hop | v2f |
| locomo_single_hop | v2f_plus_types |
| locomo_temporal | v2f |
| logic_constraint | v2f_plus_types |
| negation | v2f_plus_types |
| open_exploration | v2f |
| perspective_separation | v2f |
| proactive | v2f_plus_types |
| procedural | v2f |
| quantitative_aggregation | v2f_plus_types |
| sequential_chain | v2f |
| state_change | v2f |
| unfinished_business | v2f_plus_types |

## llm_router_mini: categories where routing helps / hurts vs v2f_only


### K=20

- Top helps:
  - logic_constraint (n=3): 0.166 → 0.333 (+0.166)
  - consistency_checking (n=2): 0.678 → 0.786 (+0.107)
  - evolving_terminology (n=5): 0.481 → 0.541 (+0.060)
- Top hurts:
  - conjunction (n=3): 0.809 → 0.714 (-0.095)
  - state_change (n=3): 0.607 → 0.531 (-0.076)
  - constraint_propagation (n=2): 0.613 → 0.560 (-0.053)

### K=50

- Top helps:
  - logic_constraint (n=3): 0.758 → 0.819 (+0.061)
  - unfinished_business (n=3): 0.872 → 0.923 (+0.051)
  - contradiction (n=2): 0.958 → 1.000 (+0.042)
- Top hurts:
  - locomo_multi_hop (n=4): 0.875 → 0.750 (-0.125)
  - constraint_propagation (n=2): 1.000 → 0.950 (-0.050)
  - conjunction (n=3): 1.000 → 0.952 (-0.048)

## llm_router_nano: categories where routing helps / hurts vs v2f_only


### K=20

- Top helps:
  - inference (n=3): 0.766 → 0.909 (+0.143)
  - locomo_multi_hop (n=4): 0.625 → 0.750 (+0.125)
  - sequential_chain (n=3): 0.509 → 0.568 (+0.059)
- Top hurts:
  - procedural (n=2): 0.347 → 0.247 (-0.100)
  - contradiction (n=2): 0.733 → 0.650 (-0.083)
  - state_change (n=3): 0.607 → 0.531 (-0.076)

### K=50

- Top helps:
  - consistency_checking (n=2): 0.928 → 1.000 (+0.072)
  - unfinished_business (n=3): 0.872 → 0.923 (+0.051)
  - inference (n=3): 0.939 → 0.970 (+0.030)
- Top hurts:
  - locomo_multi_hop (n=4): 0.875 → 0.750 (-0.125)
  - open_exploration (n=2): 0.881 → 0.809 (-0.072)
  - constraint_propagation (n=2): 1.000 → 0.950 (-0.050)

## keyword_router: categories where routing helps / hurts vs v2f_only


### K=20

- Top helps:
  - proactive (n=4): 0.351 → 0.421 (+0.069)
  - logic_constraint (n=3): 0.166 → 0.222 (+0.055)
  - sequential_chain (n=3): 0.509 → 0.535 (+0.026)
- Top hurts:
  - evolving_terminology (n=5): 0.481 → 0.441 (-0.040)
  - state_change (n=3): 0.607 → 0.579 (-0.028)

### K=50

- Top helps:
  - proactive (n=4): 0.643 → 0.791 (+0.148)
  - frequency_detection (n=1): 0.895 → 1.000 (+0.105)
  - logic_constraint (n=3): 0.758 → 0.836 (+0.078)
- Top hurts:
  - constraint_propagation (n=2): 1.000 → 0.924 (-0.076)
  - conjunction (n=3): 1.000 → 0.952 (-0.048)
  - open_exploration (n=2): 0.881 → 0.857 (-0.024)

## embedding_router: categories where routing helps / hurts vs v2f_only


### K=20

- Top helps:
  - logic_constraint (n=3): 0.166 → 0.350 (+0.184)
  - consistency_checking (n=2): 0.678 → 0.786 (+0.107)
  - unfinished_business (n=3): 0.725 → 0.821 (+0.095)
- Top hurts:
  - constraint_propagation (n=2): 0.613 → 0.437 (-0.176)
  - conjunction (n=3): 0.809 → 0.667 (-0.143)
  - negation (n=3): 0.520 → 0.467 (-0.053)

### K=50

- Top helps:
  - unfinished_business (n=3): 0.872 → 0.923 (+0.051)
  - logic_constraint (n=3): 0.758 → 0.801 (+0.043)
  - absence_inference (n=3): 0.931 → 0.954 (+0.022)
- Top hurts:
  - conjunction (n=3): 1.000 → 0.905 (-0.095)
  - evolving_terminology (n=5): 0.839 → 0.774 (-0.065)
  - quantitative_aggregation (n=3): 0.889 → 0.840 (-0.049)

## oracle: categories where routing helps / hurts vs v2f_only


### K=20

- Top helps:
  - logic_constraint (n=3): 0.166 → 0.350 (+0.184)
  - locomo_multi_hop (n=4): 0.625 → 0.750 (+0.125)
  - quantitative_aggregation (n=3): 0.491 → 0.615 (+0.123)
- Top hurts:
  - (none)

### K=50

- Top helps:
  - logic_constraint (n=3): 0.758 → 0.920 (+0.161)
  - proactive (n=4): 0.643 → 0.805 (+0.161)
  - frequency_detection (n=1): 0.895 → 1.000 (+0.105)
- Top hurts:
  - (none)

## Router cost

- Router LLM calls this run (cache misses only): 0

- Cache hits: 352

- Fresh input tokens this run: 0

- Fresh output tokens this run: 0

- Estimated per-question cost (cold cache): ~164 input tokens + ~150 output tokens for gpt-5-mini (including reasoning tokens); ~80 for nano.

- At gpt-5-mini list pricing (~$0.25/M input, $2/M output), one mini router call is ~$0.0004/question; one nano call ~$0.0002. 88 questions ≈ $0.04 for mini, $0.02 for nano.

- Keyword + embedding routers use $0 of new LLM budget per question (embedding router uses one cached cosine lookup).


## Verdict

- **v2f_only overall r@20 = 0.6323, r@50 = 0.8789.**

- **Oracle ceiling r@20 = 0.6679 (Δ vs v2f = +0.0356), r@50 = 0.9133 (Δ = +0.0344).**

- **Best cheap router at K=20: llm_router_nano = 0.6432 (Δ vs v2f = +0.0109).**

- **Best cheap router at K=50: keyword_router = 0.9042 (Δ vs v2f = +0.0253).**
