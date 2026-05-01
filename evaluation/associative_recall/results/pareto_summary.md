# Pareto Frontier Summary (Recall vs Cost)

Cost metric used for Pareto analysis: `embed_calls + 5 * llm_calls`.
Dominance: architecture A dominates B iff A has >= r@20, >= r@50, and <= cost, with strict inequality somewhere.
Only architectures with complete recall + cost data are included (n=211 total (arch,dataset) pairs).

## Dataset: `advanced_23q` (21 architectures)

| Rank | Arch | r@20 | r@50 | LLM | Embed | Cost | Status |
|------|------|------|------|-----|-------|------|--------|
| 1 | memindex:v2f_v2_with_index | 0.598 | 0.914 | 1.0 | 4.0 | 9.0 | **PARETO** |
| 2 | fairbackfill:hybrid_v2f_gencheck | 0.593 | 0.923 | 2.0 | 5.7 | 15.7 | **PARETO** |
| 3 | fairbackfill:meta_v2f | 0.593 | 0.902 | 1.0 | 4.0 | 9.0 | dominated by memindex:v2f_v2_with_index |
| 4 | self:self_cot | 0.587 | 0.899 | 2.0 | 9.3 | 19.3 | dominated by fairbackfill:hybrid_v2f_gencheck, fairbackfill:meta_v2f, +1 more |
| 5 | memindex:v2f_without_index | 0.578 | 0.906 | 1.0 | 4.0 | 9.0 | dominated by memindex:v2f_v2_with_index |
| 6 | entity:entity_entity_simple | 0.576 | 0.921 | 1.0 | 4.0 | 9.0 | **PARETO** |
| 7 | self:self_v2 | 0.567 | 0.916 | 1.9 | 6.6 | 16.0 | dominated by entity:entity_entity_simple, fairbackfill:hybrid_v2f_gencheck |
| 8 | memindex:v15_with_index | 0.554 | 0.894 | 1.0 | 4.0 | 9.0 | dominated by entity:entity_entity_simple, fairbackfill:meta_v2f, +2 more |
| 9 | cot:cot_chain_of_thought | 0.553 | 0.894 | 2.0 | 10.7 | 20.7 | dominated by entity:entity_entity_simple, fairbackfill:hybrid_v2f_gencheck, +5 more |
| 10 | memindex:index_only | 0.551 | 0.901 | 1.0 | 4.0 | 9.0 | dominated by entity:entity_entity_simple, fairbackfill:meta_v2f, +2 more |
| 11 | fairbackfill:v15_control | 0.543 | 0.911 | 1.0 | 4.0 | 9.0 | dominated by entity:entity_entity_simple, memindex:v2f_v2_with_index |
| 12 | self:self_v3 | 0.542 | 0.900 | 1.8 | 7.9 | 17.0 | dominated by entity:entity_entity_simple, fairbackfill:hybrid_v2f_gencheck, +6 more |
| 13 | chain:hybrid_v15_term | 0.503 | 0.866 | 2.0 | 8.0 | 18.0 | dominated by entity:entity_entity_simple, fairbackfill:hybrid_v2f_gencheck, +8 more |
| 14 | chain:v15_reference | 0.503 | 0.839 | 1.0 | 4.0 | 9.0 | dominated by entity:entity_entity_simple, fairbackfill:meta_v2f, +5 more |
| 15 | chain:hybrid_full | 0.503 | 0.839 | 3.0 | 9.0 | 24.0 | dominated by chain:hybrid_v15_term, chain:v15_reference, +12 more |
| 16 | entity:entity_entity_weighted_question | 0.489 | 0.885 | 1.0 | 3.0 | 8.0 | **PARETO** |
| 17 | chain:iterative_chain_nostop | 0.481 | 0.762 | 4.0 | 6.0 | 26.0 | dominated by chain:hybrid_full, chain:hybrid_v15_term, +14 more |
| 18 | chain:iterative_chain | 0.443 | 0.764 | 2.0 | 3.0 | 13.0 | dominated by chain:v15_reference, entity:entity_entity_simple, +7 more |
| 19 | chain:terminology_discovery | 0.419 | 0.819 | 1.0 | 8.0 | 13.0 | dominated by chain:v15_reference, entity:entity_entity_simple, +7 more |
| 20 | chain:chain_of_thought | 0.417 | 0.786 | 2.0 | 10.0 | 20.0 | dominated by chain:hybrid_v15_term, chain:terminology_discovery, +13 more |
| 21 | chain:embedding_explore | 0.407 | 0.717 | 0.0 | 2.0 | 2.0 | **PARETO** |

## Dataset: `beam_30q` (26 architectures)

| Rank | Arch | r@20 | r@50 | LLM | Embed | Cost | Status |
|------|------|------|------|-----|-------|------|--------|
| 1 | tree:tree_tree_explicit_d4_s8_k10_nr1_beam | 0.667 | 0.667 | 4.0 | 2.3 | 22.3 | **PARETO** |
| 2 | arch:retrieve_summarize_retrieve | 0.650 | 0.686 | 2.0 | 4.0 | 14.0 | **PARETO** |
| 3 | arch:arch_retrieve_summarize_retrieve | 0.650 | 0.686 | 2.0 | 4.0 | 14.0 | dominated by arch:retrieve_summarize_retrieve |
| 4 | arch:mmr_diversified | 0.636 | 0.743 | 0.0 | 2.0 | 2.0 | **PARETO** |
| 5 | arch:arch_mmr_diversified | 0.636 | 0.743 | 0.0 | 2.0 | 2.0 | **PARETO** |
| 6 | agent:agent_hypothesis_driven | 0.634 | 0.721 | 2.2 | 4.7 | 15.7 | dominated by arch:arch_mmr_diversified, arch:mmr_diversified |
| 7 | agent:agent_v15_conditional_hop2 | 0.629 | 0.764 | 2.0 | 5.0 | 15.0 | **PARETO** |
| 8 | agent:agent_v15_conditional_forced2 | 0.629 | 0.749 | 1.5 | 4.9 | 12.5 | **PARETO** |
| 9 | agent:agent_v15_control | 0.629 | 0.724 | 1.0 | 4.0 | 9.0 | dominated by arch:arch_mmr_diversified, arch:mmr_diversified |
| 10 | agent:v15_conditional_forced2 | 0.629 | 0.749 | 1.5 | 4.9 | 12.4 | **PARETO** |
| 11 | arch:arch_multi_query_fusion | 0.621 | 0.722 | 1.0 | 6.7 | 11.7 | dominated by agent:agent_v15_control, arch:arch_mmr_diversified, +1 more |
| 12 | arch:multi_query_fusion | 0.621 | 0.722 | 1.0 | 6.7 | 11.7 | dominated by agent:agent_v15_control, arch:arch_mmr_diversified, +1 more |
| 13 | arch:arch_negative_space | 0.621 | 0.735 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:mmr_diversified |
| 14 | arch:negative_space | 0.621 | 0.735 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +1 more |
| 15 | agent:agent_v15_variable_cues | 0.617 | 0.679 | 1.0 | 3.8 | 8.8 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +2 more |
| 16 | agent:agent_agentic_loop | 0.596 | 0.662 | 2.0 | 5.7 | 15.6 | dominated by agent:agent_v15_conditional_forced2, agent:agent_v15_conditional_hop2, +11 more |
| 17 | arch:hybrid_gap_fill | 0.563 | 0.681 | 1.0 | 4.0 | 9.0 | dominated by agent:agent_v15_control, arch:arch_mmr_diversified, +3 more |
| 18 | arch:arch_hybrid_gap_fill | 0.563 | 0.681 | 1.0 | 4.0 | 9.0 | dominated by agent:agent_v15_control, arch:arch_mmr_diversified, +3 more |
| 19 | arch:centroid_walk | 0.560 | 0.632 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +2 more |
| 20 | arch:arch_centroid_walk | 0.560 | 0.632 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +3 more |
| 21 | arch:segment_as_query | 0.512 | 0.646 | 0.0 | 14.0 | 14.0 | dominated by agent:agent_v15_conditional_forced2, agent:agent_v15_control, +12 more |
| 22 | arch:arch_segment_as_query | 0.512 | 0.646 | 0.0 | 14.0 | 14.0 | dominated by agent:agent_v15_conditional_forced2, agent:agent_v15_control, +12 more |
| 23 | arch:cluster_diversify | 0.496 | 0.666 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +2 more |
| 24 | arch:arch_cluster_diversify | 0.496 | 0.666 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +3 more |
| 25 | arch:arch_agent_working_set | 0.417 | 0.580 | 2.5 | 3.6 | 16.1 | dominated by agent:agent_agentic_loop, agent:agent_hypothesis_driven, +21 more |
| 26 | arch:agent_working_set | 0.417 | 0.580 | 2.5 | 3.6 | 16.1 | dominated by agent:agent_agentic_loop, agent:agent_hypothesis_driven, +21 more |

## Dataset: `locomo_30q` (121 architectures)

| Rank | Arch | r@20 | r@50 | LLM | Embed | Cost | Status |
|------|------|------|------|-----|-------|------|--------|
| 1 | agent:agent_v15_rerank | 0.772 | 0.772 | 2.0 | 4.0 | 14.0 | **PARETO** |
| 2 | adaptive:v2f | 0.756 | 0.858 | 1.0 | 3.0 | 8.0 | **PARETO** |
| 3 | bestshot:meta_v2f | 0.756 | 0.858 | 1.0 | 4.0 | 9.0 | dominated by adaptive:v2f |
| 4 | fairbackfill:meta_v2f | 0.756 | 0.858 | 1.0 | 4.0 | 9.0 | dominated by adaptive:v2f |
| 5 | fairbackfill:hybrid_v2f_gencheck | 0.756 | 0.858 | 2.0 | 5.8 | 15.8 | dominated by adaptive:v2f, bestshot:meta_v2f, +1 more |
| 6 | optim:optim_meta_v2f_antipattern | 0.756 | 0.858 | 1.0 | 4.0 | 9.0 | **PARETO** |
| 7 | optim:optim_double_v2f | 0.756 | 0.858 | 2.0 | 5.8 | 15.8 | dominated by optim:optim_meta_v2f_antipattern |
| 8 | bestshot:frontier_v2_iterative | 0.739 | 0.783 | 2.2 | 3.4 | 14.4 | dominated by adaptive:v2f, bestshot:meta_v2f, +2 more |
| 9 | optim:optim_meta_v2h_diverse_cues | 0.733 | 0.783 | 1.0 | 4.0 | 9.0 | dominated by adaptive:v2f, bestshot:meta_v2f, +2 more |
| 10 | optim:meta_v2h_diverse_cues | 0.733 | 0.783 | 1.0 | 4.0 | 9.0 | dominated by adaptive:v2f, bestshot:meta_v2f, +3 more |
| 11 | tree:tree_v15_multi_sub_k10_nr1 | 0.722 | 0.822 | 0.0 | 0.0 | 0.0 | **PARETO** |
| 12 | tree:tree_actual_v15_control_k10_nr1 | 0.722 | 0.822 | 1.0 | 0.0 | 5.0 | dominated by tree:tree_v15_multi_sub_k10_nr1 |
| 13 | tree:tree_v15_targeted_second_hop_k10_nr1 | 0.722 | 0.822 | 1.0 | 0.1 | 5.1 | dominated by tree:tree_actual_v15_control_k10_nr1, tree:tree_v15_multi_sub_k10_nr1 |
| 14 | tree:tree_actual_v15_plus_gaps_k10_nr1 | 0.722 | 0.822 | 1.9 | 0.0 | 9.5 | dominated by adaptive:v2f, bestshot:meta_v2f, +5 more |
| 15 | agent:agent_v15_conditional_hop2 | 0.722 | 0.822 | 2.0 | 4.7 | 14.7 | dominated by adaptive:v2f, bestshot:meta_v2f, +6 more |
| 16 | tree:tree_v15_branch_cues_k10_nr1 | 0.722 | 0.822 | 2.6 | 2.2 | 15.4 | dominated by adaptive:v2f, agent:agent_v15_conditional_hop2, +7 more |
| 17 | meta:meta_v5_additive_v15 | 0.722 | 0.822 | 2.0 | 6.0 | 16.0 | dominated by adaptive:v2f, agent:agent_v15_conditional_hop2, +10 more |
| 18 | agent:agent_v15_conditional_forced2 | 0.722 | 0.783 | 1.4 | 4.8 | 11.9 | dominated by adaptive:v2f, bestshot:meta_v2f, +7 more |
| 19 | agent:agent_v15_control | 0.722 | 0.772 | 1.0 | 4.0 | 9.0 | dominated by adaptive:v2f, bestshot:meta_v2f, +7 more |
| 20 | adaptive:adaptive | 0.722 | 0.825 | 1.0 | 3.0 | 8.0 | dominated by adaptive:v2f |
| 21 | meta:v5_additive_v15 | 0.722 | 0.822 | 2.0 | 6.0 | 16.0 | dominated by adaptive:adaptive, adaptive:v2f, +12 more |
| 22 | frontier:double_v15 | 0.706 | 0.800 | 2.0 | 5.9 | 15.9 | dominated by adaptive:adaptive, adaptive:v2f, +11 more |
| 23 | frontier:triple_v15 | 0.706 | 0.800 | 3.0 | 7.9 | 22.9 | dominated by adaptive:adaptive, adaptive:v2f, +14 more |
| 24 | frontier:hybrid_v15_constrained | 0.706 | 0.797 | 2.0 | 5.9 | 15.9 | dominated by adaptive:adaptive, adaptive:v2f, +12 more |
| 25 | frontier:hybrid_v15_frontier | 0.706 | 0.789 | 2.0 | 5.4 | 15.4 | dominated by adaptive:adaptive, adaptive:v2f, +9 more |
| 26 | frontier:hybrid_v15_frontier_3gap | 0.706 | 0.789 | 2.0 | 5.4 | 15.4 | dominated by adaptive:adaptive, adaptive:v2f, +9 more |
| 27 | frontier:hybrid_v15_frontier_deep | 0.706 | 0.789 | 2.7 | 6.6 | 20.1 | dominated by adaptive:adaptive, adaptive:v2f, +17 more |
| 28 | adaptive:v15 | 0.706 | 0.739 | 1.0 | 3.0 | 8.0 | dominated by adaptive:adaptive, adaptive:v2f, +3 more |
| 29 | bestshot:v15_control | 0.706 | 0.739 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +10 more |
| 30 | fairbackfill:v15_control | 0.706 | 0.739 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +10 more |
| 31 | frontier:simple_frontier | 0.706 | 0.739 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +10 more |
| 32 | cot:cot_chain_of_thought | 0.706 | 0.850 | 2.0 | 11.7 | 21.7 | dominated by adaptive:v2f, bestshot:meta_v2f, +4 more |
| 33 | frontier:frontier_double_v15 | 0.706 | 0.800 | 2.0 | 5.9 | 15.9 | dominated by adaptive:adaptive, adaptive:v2f, +12 more |
| 34 | frontier:frontier_triple_v15 | 0.706 | 0.800 | 3.0 | 7.9 | 22.9 | dominated by adaptive:adaptive, adaptive:v2f, +17 more |
| 35 | frontier:frontier_hybrid_v15_constrained | 0.706 | 0.797 | 2.0 | 5.9 | 15.9 | dominated by adaptive:adaptive, adaptive:v2f, +11 more |
| 36 | optim:optim_meta_v2g_no_boolean | 0.706 | 0.789 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +6 more |
| 37 | frontier:frontier_hybrid_v15_frontier_3gap | 0.706 | 0.789 | 2.0 | 5.4 | 15.4 | dominated by adaptive:adaptive, adaptive:v2f, +9 more |
| 38 | frontier:frontier_hybrid_v15_frontier | 0.706 | 0.789 | 2.0 | 5.4 | 15.4 | dominated by adaptive:adaptive, adaptive:v2f, +9 more |
| 39 | frontier:frontier_hybrid_v15_frontier_deep | 0.706 | 0.789 | 2.7 | 6.6 | 20.1 | dominated by adaptive:adaptive, adaptive:v2f, +23 more |
| 40 | optim:optim_meta_v2i_anti_question_only | 0.706 | 0.783 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +8 more |
| 41 | frontier:frontier_simple_frontier | 0.706 | 0.739 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +15 more |
| 42 | optim:optim_v15_control | 0.706 | 0.739 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +15 more |
| 43 | self:self_v2 | 0.700 | 0.933 | 2.0 | 6.4 | 16.4 | **PARETO** |
| 44 | bestshot:retrieve_then_decompose | 0.689 | 0.842 | 2.0 | 5.6 | 15.6 | dominated by adaptive:v2f, bestshot:meta_v2f, +2 more |
| 45 | optim:optim_v2f_general | 0.689 | 0.767 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +11 more |
| 46 | tree:tree_v15_boosted_tail_k10_nr1 | 0.678 | 0.789 | 0.0 | 0.0 | 0.0 | dominated by tree:tree_v15_multi_sub_k10_nr1 |
| 47 | optim:optim_meta_v2b_v2_vocab | 0.650 | 0.733 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +19 more |
| 48 | self:self_v3 | 0.644 | 0.842 | 2.0 | 6.3 | 16.3 | dominated by adaptive:v2f, bestshot:meta_v2f, +5 more |
| 49 | meta:meta_v2b_improved_strategist | 0.644 | 0.803 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +6 more |
| 50 | meta:v2b_improved_strategist | 0.644 | 0.803 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +6 more |
| 51 | meta:v2_strategist_only | 0.639 | 0.775 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +13 more |
| 52 | meta:meta_v2_strategist_only | 0.639 | 0.775 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +14 more |
| 53 | optim:optim_meta_v2c_v15_plus_completeness | 0.611 | 0.736 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +23 more |
| 54 | optim:optim_meta_v2c_completeness | 0.611 | 0.694 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +25 more |
| 55 | agent:agent_hypothesis_driven | 0.600 | 0.842 | 2.6 | 5.7 | 18.5 | dominated by adaptive:v2f, bestshot:meta_v2f, +7 more |
| 56 | meta:v1_strategist_searcher | 0.600 | 0.803 | 2.0 | 3.9 | 13.9 | dominated by adaptive:adaptive, adaptive:v2f, +8 more |
| 57 | meta:v4_iterative | 0.600 | 0.803 | 4.0 | 5.9 | 25.9 | dominated by adaptive:adaptive, adaptive:v2f, +20 more |
| 58 | meta:meta_v1_strategist_searcher | 0.600 | 0.803 | 2.0 | 3.9 | 13.9 | dominated by adaptive:adaptive, adaptive:v2f, +10 more |
| 59 | meta:meta_v4_iterative | 0.600 | 0.803 | 4.0 | 5.9 | 25.9 | dominated by adaptive:adaptive, adaptive:v2f, +23 more |
| 60 | agent:agent_orient_then_v15 | 0.600 | 0.739 | 2.0 | 4.0 | 14.0 | dominated by adaptive:adaptive, adaptive:v15, +28 more |
| 61 | memindex:v2f_without_index | 0.589 | 0.700 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +25 more |
| 62 | optim:optim_meta_v2a_strategist_format | 0.578 | 0.833 | 1.0 | 4.0 | 9.0 | dominated by adaptive:v2f, bestshot:meta_v2f, +2 more |
| 63 | arch:agent_working_set | 0.575 | 0.642 | 3.2 | 4.5 | 20.5 | dominated by adaptive:adaptive, adaptive:v15, +55 more |
| 64 | arch:arch_agent_working_set | 0.575 | 0.642 | 3.2 | 4.5 | 20.5 | dominated by adaptive:adaptive, adaptive:v15, +55 more |
| 65 | optim:optim_meta_v2j_completeness_only | 0.572 | 0.708 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +26 more |
| 66 | meta:meta_v3_domain_knowledge | 0.561 | 0.681 | 2.0 | 4.0 | 14.0 | dominated by adaptive:adaptive, adaptive:v15, +35 more |
| 67 | meta:v3_domain_knowledge | 0.561 | 0.681 | 2.0 | 4.0 | 14.0 | dominated by adaptive:adaptive, adaptive:v15, +35 more |
| 68 | frontier:iterative_frontier | 0.556 | 0.742 | 2.7 | 5.7 | 19.2 | dominated by adaptive:adaptive, adaptive:v2f, +42 more |
| 69 | frontier:frontier_iterative_frontier | 0.556 | 0.742 | 2.7 | 5.7 | 19.0 | dominated by adaptive:adaptive, adaptive:v2f, +42 more |
| 70 | optim:optim_meta_v2d_minimal | 0.550 | 0.783 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v2f, +13 more |
| 71 | agent:agent_v15_variable_cues | 0.544 | 0.594 | 1.0 | 3.5 | 8.5 | dominated by adaptive:adaptive, adaptive:v15, +5 more |
| 72 | agent:agent_adaptive_strategy | 0.533 | 0.756 | 2.0 | 5.4 | 15.4 | dominated by adaptive:adaptive, adaptive:v2f, +31 more |
| 73 | agent:agent_agentic_loop | 0.533 | 0.733 | 2.1 | 6.5 | 17.2 | dominated by adaptive:adaptive, adaptive:v15, +52 more |
| 74 | frontier:frontier_retrieval_grounded_decomp | 0.533 | 0.692 | 3.2 | 4.8 | 20.8 | dominated by adaptive:adaptive, adaptive:v15, +61 more |
| 75 | optim:optim_meta_v2e_length | 0.533 | 0.617 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +30 more |
| 76 | frontier:retrieval_grounded_decomp | 0.533 | 0.692 | 3.2 | 4.8 | 20.8 | dominated by adaptive:adaptive, adaptive:v15, +61 more |
| 77 | agent:agent_working_memory_buffer | 0.528 | 0.717 | 2.0 | 5.9 | 15.7 | dominated by adaptive:adaptive, adaptive:v15, +42 more |
| 78 | tree:tree_v15_rrf_rerank_k10_nr1 | 0.517 | 0.764 | 0.0 | 0.0 | 0.0 | dominated by tree:tree_v15_boosted_tail_k10_nr1, tree:tree_v15_multi_sub_k10_nr1 |
| 79 | bestshot:flat_multi_cue | 0.511 | 0.639 | 1.0 | 4.8 | 9.8 | dominated by adaptive:adaptive, adaptive:v15, +32 more |
| 80 | memindex:v15_with_index | 0.506 | 0.656 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +31 more |
| 81 | agent:agent_focused_agentic | 0.506 | 0.775 | 1.7 | 4.8 | 13.1 | dominated by adaptive:adaptive, adaptive:v2f, +19 more |
| 82 | agent:agent_v15_with_stop | 0.506 | 0.650 | 1.0 | 3.5 | 8.5 | dominated by adaptive:adaptive, adaptive:v15, +6 more |
| 83 | bestshot:decompose_then_retrieve | 0.503 | 0.756 | 3.9 | 7.7 | 27.2 | dominated by adaptive:adaptive, adaptive:v2f, +52 more |
| 84 | frontier:frontier_priority_frontier | 0.494 | 0.722 | 3.4 | 7.3 | 24.5 | dominated by adaptive:adaptive, adaptive:v15, +63 more |
| 85 | agent:agent_dual_perspective | 0.494 | 0.606 | 2.0 | 4.0 | 14.0 | dominated by adaptive:adaptive, adaptive:v15, +44 more |
| 86 | frontier:priority_frontier | 0.494 | 0.722 | 3.4 | 7.3 | 24.3 | dominated by adaptive:adaptive, adaptive:v15, +63 more |
| 87 | tree:tree_v15_maxsim_rerank_k10_nr1 | 0.483 | 0.728 | 0.0 | 0.0 | 0.0 | dominated by tree:tree_v15_boosted_tail_k10_nr1, tree:tree_v15_multi_sub_k10_nr1, +1 more |
| 88 | tree:tree_v15_plus_tree_k10_nr1 | 0.483 | 0.608 | 2.0 | 0.8 | 10.8 | dominated by adaptive:adaptive, adaptive:v15, +37 more |
| 89 | memindex:v2f_v2_with_index | 0.469 | 0.586 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +36 more |
| 90 | tree:tree_actual_v15_plus_gaps_reranked_k10_nr1 | 0.467 | 0.856 | 1.0 | 0.0 | 5.0 | **PARETO** |
| 91 | bestshot:interleaved | 0.461 | 0.717 | 3.9 | 7.7 | 27.2 | dominated by adaptive:adaptive, adaptive:v15, +70 more |
| 92 | memindex:index_only | 0.458 | 0.631 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +35 more |
| 93 | entity:entity_entity_simple | 0.456 | 0.572 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +39 more |
| 94 | arch:arch_mmr_diversified | 0.450 | 0.619 | 0.0 | 2.0 | 2.0 | dominated by tree:tree_v15_boosted_tail_k10_nr1, tree:tree_v15_maxsim_rerank_k10_nr1, +2 more |
| 95 | arch:mmr_diversified | 0.450 | 0.619 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, tree:tree_v15_boosted_tail_k10_nr1, +3 more |
| 96 | arch:arch_negative_space | 0.450 | 0.508 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:mmr_diversified, +4 more |
| 97 | arch:negative_space | 0.450 | 0.508 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +5 more |
| 98 | self:self_cot | 0.444 | 0.594 | 2.0 | 6.6 | 16.6 | dominated by adaptive:adaptive, adaptive:v15, +72 more |
| 99 | tree:tree_decompose_then_retrieve_k10_nr1 | 0.433 | 0.728 | 3.6 | 4.7 | 22.8 | dominated by adaptive:adaptive, adaptive:v15, +63 more |
| 100 | tree:tree_interleaved_prioritized_k10_nr1 | 0.433 | 0.650 | 0.0 | 0.0 | 0.0 | dominated by tree:tree_v15_boosted_tail_k10_nr1, tree:tree_v15_maxsim_rerank_k10_nr1, +2 more |
| 101 | tree:tree_v15_then_gaps_k10_nr1 | 0.433 | 0.650 | 0.0 | 0.0 | 0.0 | dominated by tree:tree_v15_boosted_tail_k10_nr1, tree:tree_v15_maxsim_rerank_k10_nr1, +2 more |
| 102 | tree:tree_interleaved_k10_nr1 | 0.433 | 0.650 | 0.1 | 0.0 | 0.7 | dominated by tree:tree_interleaved_prioritized_k10_nr1, tree:tree_v15_boosted_tail_k10_nr1, +4 more |
| 103 | tree:tree_retrieve_then_decompose_k10_nr1 | 0.433 | 0.650 | 1.1 | 0.1 | 5.6 | dominated by tree:tree_actual_v15_control_k10_nr1, tree:tree_actual_v15_plus_gaps_reranked_k10_nr1, +8 more |
| 104 | entity:entity_entity_v2f | 0.433 | 0.500 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +48 more |
| 105 | tree:tree_rerank_pool_k10_nr1 | 0.417 | 0.619 | 0.0 | 0.0 | 0.0 | dominated by tree:tree_interleaved_prioritized_k10_nr1, tree:tree_v15_boosted_tail_k10_nr1, +4 more |
| 106 | entity:entity_entity_per_segment | 0.411 | 0.594 | 1.0 | 5.0 | 10.0 | dominated by adaptive:adaptive, adaptive:v15, +47 more |
| 107 | arch:arch_cluster_diversify | 0.400 | 0.433 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_mmr_diversified, arch:arch_negative_space, +10 more |
| 108 | arch:cluster_diversify | 0.400 | 0.433 | 0.0 | 2.0 | 2.0 | dominated by arch:arch_cluster_diversify, arch:arch_mmr_diversified, +11 more |
| 109 | arch:arch_hybrid_gap_fill | 0.383 | 0.603 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +44 more |
| 110 | entity:entity_entity_weighted_question | 0.383 | 0.519 | 1.0 | 3.0 | 8.0 | dominated by adaptive:adaptive, adaptive:v15, +15 more |
| 111 | arch:hybrid_gap_fill | 0.383 | 0.603 | 1.0 | 4.0 | 9.0 | dominated by adaptive:adaptive, adaptive:v15, +44 more |
| 112 | adaptive:baseline | 0.383 | 0.508 | 0.0 | 1.0 | 1.0 | dominated by tree:tree_interleaved_k10_nr1, tree:tree_interleaved_prioritized_k10_nr1, +6 more |
| 113 | arch:retrieve_summarize_retrieve | 0.378 | 0.533 | 2.0 | 4.0 | 14.0 | dominated by adaptive:adaptive, adaptive:v15, +62 more |
| 114 | agent:agent_context_bootstrapping | 0.378 | 0.719 | 2.0 | 6.0 | 16.0 | dominated by adaptive:adaptive, adaptive:v15, +54 more |
| 115 | arch:arch_retrieve_summarize_retrieve | 0.378 | 0.533 | 2.0 | 4.0 | 14.0 | dominated by adaptive:adaptive, adaptive:v15, +62 more |
| 116 | arch:arch_multi_query_fusion | 0.333 | 0.461 | 1.0 | 7.0 | 12.0 | dominated by adaptive:adaptive, adaptive:baseline, +59 more |
| 117 | arch:multi_query_fusion | 0.333 | 0.461 | 1.0 | 7.0 | 12.0 | dominated by adaptive:adaptive, adaptive:baseline, +60 more |
| 118 | arch:arch_segment_as_query | 0.222 | 0.339 | 0.0 | 14.0 | 14.0 | dominated by adaptive:adaptive, adaptive:baseline, +73 more |
| 119 | arch:segment_as_query | 0.222 | 0.339 | 0.0 | 14.0 | 14.0 | dominated by adaptive:adaptive, adaptive:baseline, +73 more |
| 120 | arch:centroid_walk | 0.217 | 0.217 | 0.0 | 2.0 | 2.0 | dominated by adaptive:baseline, arch:arch_cluster_diversify, +13 more |
| 121 | arch:arch_centroid_walk | 0.217 | 0.217 | 0.0 | 2.0 | 2.0 | dominated by adaptive:baseline, arch:arch_cluster_diversify, +14 more |

## Dataset: `puzzle_16q` (24 architectures)

| Rank | Arch | r@20 | r@50 | LLM | Embed | Cost | Status |
|------|------|------|------|-----|-------|------|--------|
| 1 | chain:chain_of_thought | 0.693 | 0.974 | 2.0 | 10.3 | 20.3 | **PARETO** |
| 2 | chain:iterative_chain_nostop | 0.592 | 0.974 | 4.0 | 6.0 | 26.0 | dominated by chain:chain_of_thought |
| 3 | chain:terminology_discovery | 0.564 | 0.974 | 1.0 | 8.0 | 13.0 | **PARETO** |
| 4 | chain:hybrid_full | 0.549 | 0.979 | 3.0 | 9.0 | 24.0 | **PARETO** |
| 5 | chain:v15_reference | 0.549 | 0.974 | 1.0 | 4.0 | 9.0 | **PARETO** |
| 6 | chain:hybrid_v15_term | 0.549 | 0.954 | 2.0 | 8.0 | 18.0 | dominated by chain:terminology_discovery, chain:v15_reference |
| 7 | chain:embedding_explore | 0.538 | 0.974 | 0.0 | 2.0 | 2.0 | **PARETO** |
| 8 | entity:entity_entity_simple | 0.509 | 0.898 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference |
| 9 | memindex:v15_with_index | 0.509 | 0.895 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference, +1 more |
| 10 | fairbackfill:v15_control | 0.508 | 0.909 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference |
| 11 | memindex:index_only | 0.501 | 0.885 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference, +3 more |
| 12 | memindex:v2f_v2_with_index | 0.495 | 0.890 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference, +3 more |
| 13 | self:self_cot | 0.487 | 0.907 | 2.0 | 11.1 | 21.1 | dominated by chain:chain_of_thought, chain:embedding_explore, +4 more |
| 14 | self:self_v3 | 0.487 | 0.908 | 1.8 | 8.0 | 17.1 | dominated by chain:embedding_explore, chain:terminology_discovery, +2 more |
| 15 | memindex:v2f_without_index | 0.486 | 0.898 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference, +2 more |
| 16 | chain:iterative_chain | 0.484 | 0.974 | 2.0 | 3.3 | 13.3 | dominated by chain:embedding_explore, chain:terminology_discovery, +1 more |
| 17 | self:self_v2 | 0.481 | 0.922 | 1.9 | 7.4 | 16.8 | dominated by chain:embedding_explore, chain:iterative_chain, +2 more |
| 18 | fairbackfill:hybrid_v2f_gencheck | 0.480 | 0.922 | 2.0 | 6.0 | 16.0 | dominated by chain:embedding_explore, chain:iterative_chain, +2 more |
| 19 | fairbackfill:meta_v2f | 0.480 | 0.917 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference |
| 20 | cot:cot_chain_of_thought | 0.479 | 0.922 | 1.9 | 9.5 | 18.9 | dominated by chain:embedding_explore, chain:hybrid_v15_term, +4 more |
| 21 | entity:entity_entity_v2f | 0.447 | 0.877 | 1.0 | 4.0 | 9.0 | dominated by chain:embedding_explore, chain:v15_reference, +7 more |
| 22 | query_rewrite:cosine_only | 0.432 | 0.883 | 0.0 | 1.0 | 1.0 | **PARETO** |
| 23 | entity:entity_entity_weighted_question | 0.430 | 0.889 | 1.0 | 3.0 | 8.0 | dominated by chain:embedding_explore |
| 24 | entity:entity_entity_per_segment | 0.402 | 0.836 | 1.0 | 5.0 | 10.0 | dominated by chain:embedding_explore, chain:v15_reference, +10 more |

## Dataset: `synthetic_19q` (19 architectures)

| Rank | Arch | r@20 | r@50 | LLM | Embed | Cost | Status |
|------|------|------|------|-----|-------|------|--------|
| 1 | entity:entity_entity_v2f | 0.660 | 0.823 | 1.0 | 4.0 | 9.0 | **PARETO** |
| 2 | memindex:v2f_v2_with_index | 0.650 | 0.848 | 1.0 | 4.0 | 9.0 | **PARETO** |
| 3 | fairbackfill:v15_control | 0.638 | 0.839 | 1.0 | 4.0 | 9.0 | dominated by memindex:v2f_v2_with_index |
| 4 | bestshot:v15_control | 0.638 | 0.739 | 1.0 | 4.0 | 9.0 | dominated by entity:entity_entity_v2f, fairbackfill:v15_control, +1 more |
| 5 | self:self_v2 | 0.631 | 0.869 | 1.8 | 6.4 | 15.6 | **PARETO** |
| 6 | memindex:v15_with_index | 0.617 | 0.837 | 1.0 | 4.0 | 9.0 | dominated by fairbackfill:v15_control, memindex:v2f_v2_with_index |
| 7 | fairbackfill:hybrid_v2f_gencheck | 0.613 | 0.887 | 2.0 | 5.8 | 15.8 | **PARETO** |
| 8 | fairbackfill:meta_v2f | 0.613 | 0.851 | 1.0 | 4.0 | 9.0 | **PARETO** |
| 9 | bestshot:meta_v2f | 0.613 | 0.748 | 1.0 | 4.0 | 9.0 | dominated by entity:entity_entity_v2f, fairbackfill:meta_v2f, +3 more |
| 10 | adaptive:v15 | 0.613 | 0.831 | 1.0 | 3.0 | 8.0 | **PARETO** |
| 11 | self:self_cot | 0.607 | 0.888 | 2.0 | 9.6 | 19.6 | **PARETO** |
| 12 | entity:entity_entity_simple | 0.604 | 0.833 | 1.0 | 4.0 | 9.0 | dominated by fairbackfill:meta_v2f, fairbackfill:v15_control, +2 more |
| 13 | entity:entity_entity_weighted_question | 0.592 | 0.824 | 1.0 | 3.0 | 8.0 | dominated by adaptive:v15 |
| 14 | memindex:index_only | 0.588 | 0.825 | 1.0 | 4.0 | 9.0 | dominated by adaptive:v15, entity:entity_entity_simple, +4 more |
| 15 | self:self_v3 | 0.586 | 0.845 | 1.7 | 6.4 | 14.8 | dominated by fairbackfill:meta_v2f, memindex:v2f_v2_with_index |
| 16 | memindex:v2f_without_index | 0.581 | 0.838 | 1.0 | 4.0 | 9.0 | dominated by fairbackfill:meta_v2f, fairbackfill:v15_control, +1 more |
| 17 | adaptive:baseline | 0.569 | 0.824 | 0.0 | 1.0 | 1.0 | **PARETO** |
| 18 | entity:entity_entity_per_segment | 0.569 | 0.839 | 1.0 | 5.0 | 10.0 | dominated by fairbackfill:meta_v2f, memindex:v2f_v2_with_index |
| 19 | cot:cot_chain_of_thought | 0.524 | 0.885 | 2.0 | 10.2 | 20.2 | dominated by fairbackfill:hybrid_v2f_gencheck, self:self_cot |

## Cross-Dataset Pareto Membership

| Arch | #Datasets Pareto-Optimal | Tested on | Pareto on |
|------|--------------------------|-----------|-----------|
| chain:embedding_explore | 2 | advanced_23q,puzzle_16q | advanced_23q,puzzle_16q |
| fairbackfill:hybrid_v2f_gencheck | 2 | advanced_23q,locomo_30q,puzzle_16q,synthetic_19q | advanced_23q,synthetic_19q |
| memindex:v2f_v2_with_index | 2 | advanced_23q,locomo_30q,puzzle_16q,synthetic_19q | advanced_23q,synthetic_19q |
| self:self_v2 | 2 | advanced_23q,locomo_30q,puzzle_16q,synthetic_19q | locomo_30q,synthetic_19q |
| adaptive:baseline | 1 | locomo_30q,synthetic_19q | synthetic_19q |
| adaptive:v15 | 1 | locomo_30q,synthetic_19q | synthetic_19q |
| adaptive:v2f | 1 | locomo_30q | locomo_30q |
| agent:agent_v15_conditional_forced2 | 1 | beam_30q,locomo_30q | beam_30q |
| agent:agent_v15_conditional_hop2 | 1 | beam_30q,locomo_30q | beam_30q |
| agent:agent_v15_rerank | 1 | locomo_30q | locomo_30q |
| agent:v15_conditional_forced2 | 1 | beam_30q | beam_30q |
| arch:arch_mmr_diversified | 1 | beam_30q,locomo_30q | beam_30q |
| arch:mmr_diversified | 1 | beam_30q,locomo_30q | beam_30q |
| arch:retrieve_summarize_retrieve | 1 | beam_30q,locomo_30q | beam_30q |
| chain:chain_of_thought | 1 | advanced_23q,puzzle_16q | puzzle_16q |
| chain:hybrid_full | 1 | advanced_23q,puzzle_16q | puzzle_16q |
| chain:terminology_discovery | 1 | advanced_23q,puzzle_16q | puzzle_16q |
| chain:v15_reference | 1 | advanced_23q,puzzle_16q | puzzle_16q |
| entity:entity_entity_simple | 1 | advanced_23q,locomo_30q,puzzle_16q,synthetic_19q | advanced_23q |
| entity:entity_entity_v2f | 1 | locomo_30q,puzzle_16q,synthetic_19q | synthetic_19q |
| entity:entity_entity_weighted_question | 1 | advanced_23q,locomo_30q,puzzle_16q,synthetic_19q | advanced_23q |
| fairbackfill:meta_v2f | 1 | advanced_23q,locomo_30q,puzzle_16q,synthetic_19q | synthetic_19q |
| optim:optim_meta_v2f_antipattern | 1 | locomo_30q | locomo_30q |
| query_rewrite:cosine_only | 1 | puzzle_16q | puzzle_16q |
| self:self_cot | 1 | advanced_23q,locomo_30q,puzzle_16q,synthetic_19q | synthetic_19q |
| tree:tree_actual_v15_plus_gaps_reranked_k10_nr1 | 1 | locomo_30q | locomo_30q |
| tree:tree_tree_explicit_d4_s8_k10_nr1_beam | 1 | beam_30q | beam_30q |
| tree:tree_v15_multi_sub_k10_nr1 | 1 | locomo_30q | locomo_30q |

### Architectures on the Pareto Frontier for ALL Datasets They Were Tested On

- **chain:embedding_explore** — Pareto on 2 datasets: advanced_23q, puzzle_16q

### Architectures Strictly Dominated On Every Dataset They Were Tested On

- adaptive:adaptive (tested on: locomo_30q)
- agent:agent_adaptive_strategy (tested on: locomo_30q)
- agent:agent_agentic_loop (tested on: beam_30q, locomo_30q)
- agent:agent_context_bootstrapping (tested on: locomo_30q)
- agent:agent_dual_perspective (tested on: locomo_30q)
- agent:agent_focused_agentic (tested on: locomo_30q)
- agent:agent_hypothesis_driven (tested on: beam_30q, locomo_30q)
- agent:agent_orient_then_v15 (tested on: locomo_30q)
- agent:agent_v15_control (tested on: beam_30q, locomo_30q)
- agent:agent_v15_variable_cues (tested on: beam_30q, locomo_30q)
- agent:agent_v15_with_stop (tested on: locomo_30q)
- agent:agent_working_memory_buffer (tested on: locomo_30q)
- arch:agent_working_set (tested on: beam_30q, locomo_30q)
- arch:arch_agent_working_set (tested on: beam_30q, locomo_30q)
- arch:arch_centroid_walk (tested on: beam_30q, locomo_30q)
- arch:arch_cluster_diversify (tested on: beam_30q, locomo_30q)
- arch:arch_hybrid_gap_fill (tested on: beam_30q, locomo_30q)
- arch:arch_multi_query_fusion (tested on: beam_30q, locomo_30q)
- arch:arch_negative_space (tested on: beam_30q, locomo_30q)
- arch:arch_retrieve_summarize_retrieve (tested on: beam_30q, locomo_30q)
- arch:arch_segment_as_query (tested on: beam_30q, locomo_30q)
- arch:centroid_walk (tested on: beam_30q, locomo_30q)
- arch:cluster_diversify (tested on: beam_30q, locomo_30q)
- arch:hybrid_gap_fill (tested on: beam_30q, locomo_30q)
- arch:multi_query_fusion (tested on: beam_30q, locomo_30q)
- arch:negative_space (tested on: beam_30q, locomo_30q)
- arch:segment_as_query (tested on: beam_30q, locomo_30q)
- bestshot:decompose_then_retrieve (tested on: locomo_30q)
- bestshot:flat_multi_cue (tested on: locomo_30q)
- bestshot:frontier_v2_iterative (tested on: locomo_30q)
- bestshot:interleaved (tested on: locomo_30q)
- bestshot:meta_v2f (tested on: locomo_30q, synthetic_19q)
- bestshot:retrieve_then_decompose (tested on: locomo_30q)
- bestshot:v15_control (tested on: locomo_30q, synthetic_19q)
- chain:hybrid_v15_term (tested on: advanced_23q, puzzle_16q)
- chain:iterative_chain (tested on: advanced_23q, puzzle_16q)
- chain:iterative_chain_nostop (tested on: advanced_23q, puzzle_16q)
- cot:cot_chain_of_thought (tested on: advanced_23q, locomo_30q, puzzle_16q, synthetic_19q)
- entity:entity_entity_per_segment (tested on: locomo_30q, puzzle_16q, synthetic_19q)
- fairbackfill:v15_control (tested on: advanced_23q, locomo_30q, puzzle_16q, synthetic_19q)
- frontier:double_v15 (tested on: locomo_30q)
- frontier:frontier_double_v15 (tested on: locomo_30q)
- frontier:frontier_hybrid_v15_constrained (tested on: locomo_30q)
- frontier:frontier_hybrid_v15_frontier (tested on: locomo_30q)
- frontier:frontier_hybrid_v15_frontier_3gap (tested on: locomo_30q)
- frontier:frontier_hybrid_v15_frontier_deep (tested on: locomo_30q)
- frontier:frontier_iterative_frontier (tested on: locomo_30q)
- frontier:frontier_priority_frontier (tested on: locomo_30q)
- frontier:frontier_retrieval_grounded_decomp (tested on: locomo_30q)
- frontier:frontier_simple_frontier (tested on: locomo_30q)
- frontier:frontier_triple_v15 (tested on: locomo_30q)
- frontier:hybrid_v15_constrained (tested on: locomo_30q)
- frontier:hybrid_v15_frontier (tested on: locomo_30q)
- frontier:hybrid_v15_frontier_3gap (tested on: locomo_30q)
- frontier:hybrid_v15_frontier_deep (tested on: locomo_30q)
- frontier:iterative_frontier (tested on: locomo_30q)
- frontier:priority_frontier (tested on: locomo_30q)
- frontier:retrieval_grounded_decomp (tested on: locomo_30q)
- frontier:simple_frontier (tested on: locomo_30q)
- frontier:triple_v15 (tested on: locomo_30q)
- memindex:index_only (tested on: advanced_23q, locomo_30q, puzzle_16q, synthetic_19q)
- memindex:v15_with_index (tested on: advanced_23q, locomo_30q, puzzle_16q, synthetic_19q)
- memindex:v2f_without_index (tested on: advanced_23q, locomo_30q, puzzle_16q, synthetic_19q)
- meta:meta_v1_strategist_searcher (tested on: locomo_30q)
- meta:meta_v2_strategist_only (tested on: locomo_30q)
- meta:meta_v2b_improved_strategist (tested on: locomo_30q)
- meta:meta_v3_domain_knowledge (tested on: locomo_30q)
- meta:meta_v4_iterative (tested on: locomo_30q)
- meta:meta_v5_additive_v15 (tested on: locomo_30q)
- meta:v1_strategist_searcher (tested on: locomo_30q)
- meta:v2_strategist_only (tested on: locomo_30q)
- meta:v2b_improved_strategist (tested on: locomo_30q)
- meta:v3_domain_knowledge (tested on: locomo_30q)
- meta:v4_iterative (tested on: locomo_30q)
- meta:v5_additive_v15 (tested on: locomo_30q)
- optim:meta_v2h_diverse_cues (tested on: locomo_30q)
- optim:optim_double_v2f (tested on: locomo_30q)
- optim:optim_meta_v2a_strategist_format (tested on: locomo_30q)
- optim:optim_meta_v2b_v2_vocab (tested on: locomo_30q)
- optim:optim_meta_v2c_completeness (tested on: locomo_30q)
- optim:optim_meta_v2c_v15_plus_completeness (tested on: locomo_30q)
- optim:optim_meta_v2d_minimal (tested on: locomo_30q)
- optim:optim_meta_v2e_length (tested on: locomo_30q)
- optim:optim_meta_v2g_no_boolean (tested on: locomo_30q)
- optim:optim_meta_v2h_diverse_cues (tested on: locomo_30q)
- optim:optim_meta_v2i_anti_question_only (tested on: locomo_30q)
- optim:optim_meta_v2j_completeness_only (tested on: locomo_30q)
- optim:optim_v15_control (tested on: locomo_30q)
- optim:optim_v2f_general (tested on: locomo_30q)
- self:self_v3 (tested on: advanced_23q, locomo_30q, puzzle_16q, synthetic_19q)
- tree:tree_actual_v15_control_k10_nr1 (tested on: locomo_30q)
- tree:tree_actual_v15_plus_gaps_k10_nr1 (tested on: locomo_30q)
- tree:tree_decompose_then_retrieve_k10_nr1 (tested on: locomo_30q)
- tree:tree_interleaved_k10_nr1 (tested on: locomo_30q)
- tree:tree_interleaved_prioritized_k10_nr1 (tested on: locomo_30q)
- tree:tree_rerank_pool_k10_nr1 (tested on: locomo_30q)
- tree:tree_retrieve_then_decompose_k10_nr1 (tested on: locomo_30q)
- tree:tree_v15_boosted_tail_k10_nr1 (tested on: locomo_30q)
- tree:tree_v15_branch_cues_k10_nr1 (tested on: locomo_30q)
- tree:tree_v15_maxsim_rerank_k10_nr1 (tested on: locomo_30q)
- tree:tree_v15_plus_tree_k10_nr1 (tested on: locomo_30q)
- tree:tree_v15_rrf_rerank_k10_nr1 (tested on: locomo_30q)
- tree:tree_v15_targeted_second_hop_k10_nr1 (tested on: locomo_30q)
- tree:tree_v15_then_gaps_k10_nr1 (tested on: locomo_30q)

## Cost-Tier Recall Plateau Analysis

For each dataset, best r@20 and best r@50 achievable at each cost tier.

### advanced_23q

| Cost tier | Best r@20 | Best r@50 | Example arch |
|-----------|-----------|-----------|--------------|
| <=1 LLM, <=4 embed (v2f/v15 class) | 0.598 (memindex:v2f_v2_with_index) | 0.921 (entity:entity_entity_simple) | |
| <=2 LLM, <=8 embed | 0.598 (memindex:v2f_v2_with_index) | 0.923 (fairbackfill:hybrid_v2f_gencheck) | |
| <=3 LLM, <=12 embed | 0.598 (memindex:v2f_v2_with_index) | 0.923 (fairbackfill:hybrid_v2f_gencheck) | |
| <=5 LLM, any embed | 0.598 (memindex:v2f_v2_with_index) | 0.923 (fairbackfill:hybrid_v2f_gencheck) | |
| any cost | 0.598 (memindex:v2f_v2_with_index) | 0.923 (fairbackfill:hybrid_v2f_gencheck) | |

### beam_30q

| Cost tier | Best r@20 | Best r@50 | Example arch |
|-----------|-----------|-----------|--------------|
| <=1 LLM, <=4 embed (v2f/v15 class) | 0.636 (arch:mmr_diversified) | 0.743 (arch:arch_mmr_diversified) | |
| <=2 LLM, <=8 embed | 0.650 (arch:retrieve_summarize_retrieve) | 0.764 (agent:agent_v15_conditional_hop2) | |
| <=3 LLM, <=12 embed | 0.650 (arch:retrieve_summarize_retrieve) | 0.764 (agent:agent_v15_conditional_hop2) | |
| <=5 LLM, any embed | 0.667 (tree:tree_tree_explicit_d4_s8_k10_nr1_beam) | 0.764 (agent:agent_v15_conditional_hop2) | |
| any cost | 0.667 (tree:tree_tree_explicit_d4_s8_k10_nr1_beam) | 0.764 (agent:agent_v15_conditional_hop2) | |

### locomo_30q

| Cost tier | Best r@20 | Best r@50 | Example arch |
|-----------|-----------|-----------|--------------|
| <=1 LLM, <=4 embed (v2f/v15 class) | 0.756 (adaptive:v2f) | 0.858 (optim:optim_meta_v2f_antipattern) | |
| <=2 LLM, <=8 embed | 0.772 (agent:agent_v15_rerank) | 0.933 (self:self_v2) | |
| <=3 LLM, <=12 embed | 0.772 (agent:agent_v15_rerank) | 0.933 (self:self_v2) | |
| <=5 LLM, any embed | 0.772 (agent:agent_v15_rerank) | 0.933 (self:self_v2) | |
| any cost | 0.772 (agent:agent_v15_rerank) | 0.933 (self:self_v2) | |

### puzzle_16q

| Cost tier | Best r@20 | Best r@50 | Example arch |
|-----------|-----------|-----------|--------------|
| <=1 LLM, <=4 embed (v2f/v15 class) | 0.549 (chain:v15_reference) | 0.974 (chain:v15_reference) | |
| <=2 LLM, <=8 embed | 0.564 (chain:terminology_discovery) | 0.974 (chain:terminology_discovery) | |
| <=3 LLM, <=12 embed | 0.693 (chain:chain_of_thought) | 0.979 (chain:hybrid_full) | |
| <=5 LLM, any embed | 0.693 (chain:chain_of_thought) | 0.979 (chain:hybrid_full) | |
| any cost | 0.693 (chain:chain_of_thought) | 0.979 (chain:hybrid_full) | |

### synthetic_19q

| Cost tier | Best r@20 | Best r@50 | Example arch |
|-----------|-----------|-----------|--------------|
| <=1 LLM, <=4 embed (v2f/v15 class) | 0.660 (entity:entity_entity_v2f) | 0.851 (fairbackfill:meta_v2f) | |
| <=2 LLM, <=8 embed | 0.660 (entity:entity_entity_v2f) | 0.887 (fairbackfill:hybrid_v2f_gencheck) | |
| <=3 LLM, <=12 embed | 0.660 (entity:entity_entity_v2f) | 0.888 (self:self_cot) | |
| <=5 LLM, any embed | 0.660 (entity:entity_entity_v2f) | 0.888 (self:self_cot) | |
| any cost | 0.660 (entity:entity_entity_v2f) | 0.888 (self:self_cot) | |
