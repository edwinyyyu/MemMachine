# Cross-Architecture Failure Overlap

Failure threshold: recall @K=50 < 0.5. Stubborn threshold: >= 80% of archs fail. Easy threshold: >= 80% of archs succeed.

## Architectures catalogued

| Dataset | Architecture | Questions |
| --- | --- | --- |
| locomo_30q | budget_baseline_50 | 30 |
| locomo_30q | budget_gencheck_50 | 30 |
| locomo_30q | budget_v15_tight_50 | 30 |
| locomo_30q | cot_chain_of_thought | 30 |
| locomo_30q | domain_agnostic_v2f_style_explicit | 30 |
| locomo_30q | fairbackfill_hybrid_v2f_gencheck | 30 |
| locomo_30q | fairbackfill_meta_v2f | 30 |
| locomo_30q | fairbackfill_v15_control | 30 |
| locomo_30q | goal_chain_chain_goal_tracking | 30 |
| locomo_30q | goal_chain_chain_with_scratchpad | 30 |
| locomo_30q | self_cot | 30 |
| locomo_30q | self_v2 | 30 |
| locomo_30q | self_v3 | 30 |
| locomo_30q | two_call | 30 |
| locomo_30q | type_enum_type_enumerated | 30 |
| locomo_30q | type_enum_type_enumerated_selective | 30 |
| locomo_30q | type_enum_v2f_plus_types | 30 |
| locomo_30q | v15_hybrid_hybrid_v15_dual | 30 |
| synthetic_19q | budget_baseline_50 | 19 |
| synthetic_19q | budget_gencheck_50 | 19 |
| synthetic_19q | budget_v15_tight_50 | 19 |
| synthetic_19q | cot_chain_of_thought | 19 |
| synthetic_19q | domain_agnostic_v2f_register_inferred | 19 |
| synthetic_19q | domain_agnostic_v2f_style_explicit | 19 |
| synthetic_19q | fairbackfill_hybrid_v2f_gencheck | 19 |
| synthetic_19q | fairbackfill_meta_v2f | 19 |
| synthetic_19q | fairbackfill_v15_control | 19 |
| synthetic_19q | fulleval_frontier_v2_iterative | 19 |
| synthetic_19q | fulleval_meta_v2f | 19 |
| synthetic_19q | fulleval_v15_control | 19 |
| synthetic_19q | goal_chain_chain_goal_tracking | 19 |
| synthetic_19q | goal_chain_chain_with_scratchpad | 19 |
| synthetic_19q | self_cot | 19 |
| synthetic_19q | self_v2 | 19 |
| synthetic_19q | self_v3 | 19 |
| synthetic_19q | two_call | 19 |
| synthetic_19q | type_enum_type_enumerated | 19 |
| synthetic_19q | type_enum_type_enumerated_selective | 19 |
| synthetic_19q | type_enum_v2f_plus_types | 19 |
| synthetic_19q | v15_hybrid_hybrid_v15_dual | 19 |
| puzzle_16q | budget_baseline_50 | 16 |
| puzzle_16q | budget_gencheck_50 | 16 |
| puzzle_16q | budget_v15_tight_50 | 16 |
| puzzle_16q | cot_chain_of_thought | 16 |
| puzzle_16q | domain_agnostic_v2f_register_inferred | 16 |
| puzzle_16q | domain_agnostic_v2f_style_explicit | 16 |
| puzzle_16q | fairbackfill_hybrid_v2f_gencheck | 16 |
| puzzle_16q | fairbackfill_meta_v2f | 16 |
| puzzle_16q | fairbackfill_v15_control | 16 |
| puzzle_16q | fulleval_frontier_v2_iterative | 16 |
| puzzle_16q | fulleval_hybrid_v2f_gencheck | 16 |
| puzzle_16q | fulleval_meta_v2f | 16 |
| puzzle_16q | fulleval_v15_control | 16 |
| puzzle_16q | goal_chain_chain_goal_tracking | 16 |
| puzzle_16q | goal_chain_chain_with_scratchpad | 16 |
| puzzle_16q | self_cot | 16 |
| puzzle_16q | self_v2 | 16 |
| puzzle_16q | self_v3 | 16 |
| puzzle_16q | two_call | 16 |
| puzzle_16q | type_enum_type_enumerated | 16 |
| puzzle_16q | type_enum_type_enumerated_selective | 16 |
| puzzle_16q | type_enum_v2f_plus_types | 16 |
| puzzle_16q | v15_hybrid_hybrid_v15_dual | 16 |
| advanced_23q | budget_baseline_50 | 23 |
| advanced_23q | budget_gencheck_50 | 23 |
| advanced_23q | budget_v15_tight_50 | 23 |
| advanced_23q | cot_chain_of_thought | 23 |
| advanced_23q | domain_agnostic_v2f_style_explicit | 23 |
| advanced_23q | fairbackfill_hybrid_v2f_gencheck | 23 |
| advanced_23q | fairbackfill_meta_v2f | 23 |
| advanced_23q | fairbackfill_v15_control | 23 |
| advanced_23q | fulleval_frontier_v2_iterative | 23 |
| advanced_23q | fulleval_hybrid_v2f_gencheck | 23 |
| advanced_23q | fulleval_meta_v2f | 23 |
| advanced_23q | fulleval_v15_control | 23 |
| advanced_23q | goal_chain_chain_goal_tracking | 23 |
| advanced_23q | goal_chain_chain_with_scratchpad | 23 |
| advanced_23q | self_cot | 23 |
| advanced_23q | self_v2 | 23 |
| advanced_23q | self_v3 | 23 |
| advanced_23q | two_call | 23 |
| advanced_23q | type_enum_type_enumerated | 23 |
| advanced_23q | type_enum_type_enumerated_selective | 23 |
| advanced_23q | type_enum_v2f_plus_types | 23 |
| advanced_23q | v15_hybrid_hybrid_v15_dual | 23 |

## Per-dataset summary

| Dataset | # Arch | # Questions assessed | Stubborn failures (>=80% fail) | Easy (>=80% pass) |
| --- | ---: | ---: | ---: | ---: |
| locomo_30q | 18 | 30 | 2 | 20 |
| synthetic_19q | 22 | 19 | 0 | 18 |
| puzzle_16q | 23 | 16 | 0 | 16 |
| advanced_23q | 22 | 23 | 0 | 23 |

## Stubborn failures -- locomo_30q

| Conv | Category | # gold | Fail frac | Best recall | Best arch | Question |
| --- | --- | ---: | ---: | ---: | --- | --- |
| locomo_conv-26 | locomo_temporal | 1 | 18/18 (1.00) | 0.00 | budget_baseline_50 | When did Melanie paint a sunrise? |
| locomo_conv-26 | locomo_single_hop | 4 | 15/18 (0.83) | 0.75 | budget_v15_tight_50 | What activities does Melanie partake in? |

## Top-5 hardest questions per dataset (any fail frac)

### locomo_30q

| Fail frac | Best recall | Best arch | Cat | Gold | Question |
| ---: | ---: | --- | --- | ---: | --- |
| 1.00 | 0.00 | budget_baseline_50 | locomo_temporal | 1 | When did Melanie paint a sunrise? |
| 0.83 | 0.75 | budget_v15_tight_50 | locomo_single_hop | 4 | What activities does Melanie partake in? |
| 0.56 | 1.00 | budget_gencheck_50 | locomo_temporal | 1 | How long ago was Caroline's 18th birthday? |
| 0.56 | 1.00 | budget_gencheck_50 | locomo_single_hop | 1 | What did Caroline research? |
| 0.50 | 1.00 | budget_gencheck_50 | locomo_temporal | 1 | When did Caroline have a picnic? |

### synthetic_19q

| Fail frac | Best recall | Best arch | Cat | Gold | Question |
| ---: | ---: | --- | --- | ---: | --- |
| 0.23 | 0.88 | budget_v15_tight_50 | proactive | 8 | What needs to happen to set up the bedroom smart home features? |
| 0.14 | 0.71 | type_enum_v2f_plus_types | procedural | 17 | Create a complete checklist of remaining tasks for the June 15th anniversary party, organized by timeline. |
| 0.09 | 0.87 | budget_v15_tight_50 | procedural | 15 | What are the remaining phases of the smart home setup and what's in each phase? |
| 0.05 | 1.00 | budget_baseline_50 | conjunction | 7 | What are the exact specifications for the living room movie mode automation? |
| 0.05 | 1.00 | budget_v15_tight_50 | proactive | 6 | I want to cook dinner for Bob tonight. What should I keep in mind? |

### puzzle_16q

| Fail frac | Best recall | Best arch | Cat | Gold | Question |
| ---: | ---: | --- | --- | ---: | --- |
| 0.09 | 1.00 | fulleval_hybrid_v2f_gencheck | logic_constraint | 12 | Based on all constraints discussed, what is the final valid desk arrangement for the 6 desks? |
| 0.09 | 0.89 | fulleval_hybrid_v2f_gencheck | logic_constraint | 19 | What is the final conference room schedule for next week, including all the changes that were made? |
| 0.04 | 1.00 | budget_baseline_50 | sequential_chain | 16 | What chain of discoveries led to successfully recreating the grandmother's lamb stew? |
| 0.04 | 1.00 | budget_baseline_50 | contradiction | 10 | Where is the company retreat being held and when does it start? |
| 0.04 | 1.00 | budget_gencheck_50 | absence_inference | 18 | Based on the conversation, does the user follow any specific dietary pattern? What evidence supports your conclusion? |

### advanced_23q

| Fail frac | Best recall | Best arch | Cat | Gold | Question |
| ---: | ---: | --- | --- | ---: | --- |
| 0.14 | 0.86 | budget_baseline_50 | negation | 7 | What was the final technology decision and what mitigation strategies were agreed upon to address known weaknesses? |
| 0.09 | 0.83 | budget_v15_tight_50 | quantitative_aggregation | 12 | How did the project estimate compare to the client's budget, and what was the resolution? |
| 0.05 | 1.00 | budget_v15_tight_50 | perspective_separation | 11 | What was Bob's position on the launch deadline and how did it differ from Carol's? |
| 0.05 | 1.00 | budget_baseline_50 | perspective_separation | 6 | What is Harris the CFO's position on pricing for the CloudDeck Enterprise tier, and how does it contrast with Gina's? |
| 0.05 | 0.87 | fairbackfill_v15_control | evolving_terminology | 15 | What is the current status of Project Phoenix? Include any milestones reached and upcoming work. |

## Category-level stubbornness rates

Percent of questions in a category that are stubborn failures (assessed questions only).

### locomo_30q

| Category | N | # Stubborn | Stubborn rate | # Easy | Easy rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| locomo_single_hop | 10 | 1 | 0.10 | 4 | 0.40 |
| locomo_temporal | 16 | 1 | 0.06 | 13 | 0.81 |
| locomo_multi_hop | 4 | 0 | 0.00 | 3 | 0.75 |

### synthetic_19q

| Category | N | # Stubborn | Stubborn rate | # Easy | Easy rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| inference | 3 | 0 | 0.00 | 3 | 1.00 |
| completeness | 4 | 0 | 0.00 | 4 | 1.00 |
| control | 3 | 0 | 0.00 | 3 | 1.00 |
| procedural | 2 | 0 | 0.00 | 2 | 1.00 |
| conjunction | 3 | 0 | 0.00 | 3 | 1.00 |
| proactive | 4 | 0 | 0.00 | 3 | 0.75 |

### puzzle_16q

| Category | N | # Stubborn | Stubborn rate | # Easy | Easy rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| logic_constraint | 3 | 0 | 0.00 | 3 | 1.00 |
| contradiction | 2 | 0 | 0.00 | 2 | 1.00 |
| absence_inference | 3 | 0 | 0.00 | 3 | 1.00 |
| sequential_chain | 3 | 0 | 0.00 | 3 | 1.00 |
| open_exploration | 2 | 0 | 0.00 | 2 | 1.00 |
| state_change | 3 | 0 | 0.00 | 3 | 1.00 |

### advanced_23q

| Category | N | # Stubborn | Stubborn rate | # Easy | Easy rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| quantitative_aggregation | 3 | 0 | 0.00 | 3 | 1.00 |
| perspective_separation | 4 | 0 | 0.00 | 4 | 1.00 |
| unfinished_business | 3 | 0 | 0.00 | 3 | 1.00 |
| frequency_detection | 1 | 0 | 0.00 | 1 | 1.00 |
| consistency_checking | 2 | 0 | 0.00 | 2 | 1.00 |
| constraint_propagation | 2 | 0 | 0.00 | 2 | 1.00 |
| evolving_terminology | 5 | 0 | 0.00 | 5 | 1.00 |
| negation | 3 | 0 | 0.00 | 3 | 1.00 |

## Surface features: stubborn vs. easy

| Dataset | Group | N | mean Q length (chars) | mean # gold turns | multi-part rate |
| --- | --- | ---: | ---: | ---: | ---: |
| locomo_30q | stubborn | 2 | 36.5 | 2.50 | 0.00 |
| locomo_30q | easy | 20 | 49.2 | 1.35 | 0.05 |
| synthetic_19q | stubborn | 0 | 0.0 | 0.00 | 0.00 |
| synthetic_19q | easy | 18 | 94.4 | 8.11 | 0.61 |
| puzzle_16q | stubborn | 0 | 0.0 | 0.00 | 0.00 |
| puzzle_16q | easy | 16 | 99.5 | 14.06 | 0.88 |
| advanced_23q | stubborn | 0 | 0.0 | 0.00 | 0.00 |
| advanced_23q | easy | 23 | 97.9 | 11.00 | 0.74 |

## Which architecture is best on stubborn failures?

Count of stubborn questions where this arch was the best (and achieved recall >= 0.5).

### locomo_30q

| Arch | # stubborn Qs recovered |
| --- | ---: |
| budget_v15_tight_50 | 1 |

## Verdict

- Total stubborn failures across all datasets: **2**
- Of those, **1** are *structural ceilings* (no architecture reached recall >= 0.5).
- **1** are *architecture-specific solvable but unstable* (at least one architecture achieved recall >= 0.5).

**Verdict: mixed.** A meaningful fraction of stubborn questions are structurally unreachable, but others have at least one arch that solves them — suggesting both cue-generation instability *and* some genuine ceilings.

## Pointers

- Raw failure matrix: `results/failure_overlap_analysis.json`
- Source script: `failure_overlap.py`
