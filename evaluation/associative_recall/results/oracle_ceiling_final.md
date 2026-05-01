# Final Oracle Ceiling Across All Architectures

Datasets: locomo_30q, synthetic_19q, puzzle_16q, advanced_23q
Architectures collected per dataset: locomo_30q=156, synthetic_19q=151, puzzle_16q=57, advanced_23q=57

## Oracle ceiling vs shipped recipe

| Dataset | N | Archs (≥ K=50) | Shipped @20 | Oracle @20 | Gap @20 | Shipped @50 | Oracle @50 | Gap @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locomo_30q | 30 | 132.5 | 0.8917 (two_speaker_filter@20) | 0.9083 | 0.0167 | 0.9167 (composition_v2_all@50) | 0.9333 | 0.0167 |
| synthetic_19q | 19 | 130.4 | 0.6130 (two_speaker_filter@20) | 0.7636 | 0.1506 | 0.8829 (composition_v2_all@50) | 0.9379 | 0.0549 |
| puzzle_16q | 16 | 43.8 | 0.4804 (v2f@20) | 0.6337 | 0.1532 | 0.9327 (composition_v2_all@50) | 0.9763 | 0.0436 |
| advanced_23q | 23 | 42.0 | 0.5931 (v2f@20) | 0.7351 | 0.1420 | 0.9345 (composition_v2_all@50) | 0.9707 | 0.0362 |

**Overall (question-weighted):**

- Oracle @20 = 0.7819
- Shipped @20 = 0.6787
- Gap @20 = 0.1032
- Oracle @50 = 0.9519
- Shipped @50 = 0.9170
- Gap @50 = 0.0349

## Stubborn failures (best_r@50 < 0.5 across ALL archs)

Count: **1**

| Dataset | Conv | Category | # gold | Best @20 | Best @50 | Best arch @50 | Question |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| locomo_30q | locomo_conv-26 | locomo_temporal | 1 | 0.0 | 0.0 | adaptive_adaptive | When did Melanie paint a sunrise? |

## Per-category headroom @50 (gap = oracle - shipped)

Sorted by largest gap first (bigger gap = more routing headroom).

| Dataset | Category | N | Oracle @50 | Shipped @50 | Gap @50 | Oracle @20 | Shipped @20 | Gap @20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locomo_30q | locomo_multi_hop | 4 | 1.0000 | 0.8750 | 0.1250 | 0.8750 | 0.8750 | 0.0000 |
| synthetic_19q | procedural | 2 | 0.7530 | 0.6529 | 0.1000 | 0.3765 | 0.3470 | 0.0294 |
| puzzle_16q | logic_constraint | 3 | 0.9371 | 0.8513 | 0.0859 | 0.4132 | 0.1663 | 0.2469 |
| synthetic_19q | completeness | 4 | 0.9423 | 0.8654 | 0.0769 | 0.6731 | 0.4551 | 0.2179 |
| advanced_23q | constraint_propagation | 2 | 1.0000 | 0.9237 | 0.0763 | 0.6895 | 0.6131 | 0.0763 |
| synthetic_19q | proactive | 4 | 0.8861 | 0.8106 | 0.0755 | 0.6684 | 0.3514 | 0.3170 |
| puzzle_16q | absence_inference | 3 | 1.0000 | 0.9315 | 0.0685 | 0.6259 | 0.4574 | 0.1685 |
| advanced_23q | negation | 3 | 0.9524 | 0.8872 | 0.0651 | 0.6093 | 0.5201 | 0.0893 |
| advanced_23q | evolving_terminology | 5 | 0.9733 | 0.9114 | 0.0620 | 0.6684 | 0.4809 | 0.1875 |
| synthetic_19q | conjunction | 3 | 1.0000 | 0.9524 | 0.0476 | 0.8571 | 0.8095 | 0.0476 |
| puzzle_16q | open_exploration | 2 | 0.9047 | 0.8572 | 0.0476 | 0.5715 | 0.5000 | 0.0715 |
| synthetic_19q | inference | 3 | 1.0000 | 0.9697 | 0.0303 | 0.9394 | 0.7662 | 0.1732 |
| advanced_23q | quantitative_aggregation | 3 | 0.9444 | 0.9167 | 0.0278 | 0.7073 | 0.4915 | 0.2158 |
| puzzle_16q | sequential_chain | 3 | 1.0000 | 0.9744 | 0.0256 | 0.7058 | 0.5093 | 0.1965 |
| advanced_23q | perspective_separation | 4 | 1.0000 | 0.9773 | 0.0227 | 0.8674 | 0.7159 | 0.1515 |
| puzzle_16q | state_change | 3 | 1.0000 | 0.9792 | 0.0208 | 0.7649 | 0.6071 | 0.1577 |
| locomo_30q | locomo_temporal | 16 | 0.9375 | 0.9375 | 0.0000 | 0.9375 | 0.9375 | 0.0000 |
| locomo_30q | locomo_single_hop | 10 | 0.9000 | 0.9000 | 0.0000 | 0.8750 | 0.8250 | 0.0500 |
| synthetic_19q | control | 3 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| puzzle_16q | contradiction | 2 | 1.0000 | 1.0000 | 0.0000 | 0.7333 | 0.7333 | 0.0000 |
| advanced_23q | unfinished_business | 3 | 0.9231 | 0.9231 | 0.0000 | 0.8718 | 0.7253 | 0.1465 |
| advanced_23q | frequency_detection | 1 | 1.0000 | 1.0000 | 0.0000 | 0.5789 | 0.5789 | 0.0000 |
| advanced_23q | consistency_checking | 2 | 1.0000 | 1.0000 | 0.0000 | 0.7857 | 0.6785 | 0.1072 |

## Verdict

- K=20 overall: gap = 0.1032; meaningful headroom (>=5pp) — routing could still help.
- K=50 overall: gap = 0.0349; modest headroom (3-5pp) via routing/composition.

- Stubborn failures at K=50 (< 0.5 best): **1**

Captured fraction of oracle ceiling: K=20 = 0.868 of oracle (0.6787/0.7819), K=50 = 0.963 of oracle (0.9170/0.9519).

## Pointers

- Raw matrix: `results/oracle_ceiling_final.json`
- Source script: `oracle_ceiling_analysis.py`
