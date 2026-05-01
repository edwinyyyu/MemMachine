# R1 + R2 Representation Experiments

Utterance anchor (R1) + dual-score aggregation (R2). See `REPRESENTATION_EXPERIMENTS.md` for spec.

## Reference baselines
- Current base hybrid (T+S): R@5 0.555, MRR 0.918, NDCG@10 0.652 (on base 55 queries).

## Variants — all subsets, R@5

| Variant | all | base | disc | utt | utt_utterance | utt_referent |
|---|---:|---:|---:|---:|---:|---:|
| R1a_anchor_only | 0.030 | 0.013 | 0.000 | 0.200 | 0.400 | 0.000 |
| R1b_referent_only | 0.463 | 0.453 | 0.567 | 0.200 | 0.000 | 0.400 |
| R1c_union_max | 0.450 | 0.409 | 0.533 | 0.400 | 0.400 | 0.400 |
| R2a_union_sum | 0.450 | 0.409 | 0.533 | 0.400 | 0.400 | 0.400 |
| R2b_union_max | 0.450 | 0.409 | 0.533 | 0.400 | 0.400 | 0.400 |
| R2c_union_w_a03_b07 | 0.447 | 0.405 | 0.533 | 0.400 | 0.400 | 0.400 |
| R2d_union_w_a07_b03 | 0.470 | 0.423 | 0.567 | 0.420 | 0.440 | 0.400 |
| R2e_union_routed | 0.468 | 0.423 | 0.567 | 0.400 | 0.400 | 0.400 |
| R2f_union_w_a09_b01 | 0.472 | 0.429 | 0.567 | 0.400 | 0.400 | 0.400 |
| R2g_union_sumw_a1_b03 | 0.474 | 0.429 | 0.567 | 0.420 | 0.440 | 0.400 |
| HYBRID_R2g_union_sumw_a1_b03 | 0.549 | 0.506 | 0.567 | 0.707 | 0.480 | 0.933 |
| HYBRID_R1b_referent_only | 0.528 | 0.513 | 0.567 | 0.487 | 0.040 | 0.933 |

## Variants — all subsets, MRR

| Variant | all | base | disc | utt | utt_utterance | utt_referent |
|---|---:|---:|---:|---:|---:|---:|
| R1a_anchor_only | 0.035 | 0.020 | 0.002 | 0.212 | 0.425 | 0.000 |
| R1b_referent_only | 0.423 | 0.559 | 0.267 | 0.211 | 0.022 | 0.400 |
| R1c_union_max | 0.444 | 0.584 | 0.244 | 0.340 | 0.279 | 0.400 |
| R2a_union_sum | 0.439 | 0.574 | 0.245 | 0.342 | 0.283 | 0.400 |
| R2b_union_max | 0.444 | 0.584 | 0.244 | 0.340 | 0.279 | 0.400 |
| R2c_union_w_a03_b07 | 0.434 | 0.553 | 0.244 | 0.412 | 0.425 | 0.400 |
| R2d_union_w_a07_b03 | 0.449 | 0.568 | 0.282 | 0.353 | 0.307 | 0.400 |
| R2e_union_routed | 0.455 | 0.568 | 0.280 | 0.412 | 0.425 | 0.400 |
| R2f_union_w_a09_b01 | 0.442 | 0.555 | 0.284 | 0.348 | 0.295 | 0.400 |
| R2g_union_sumw_a1_b03 | 0.450 | 0.571 | 0.282 | 0.353 | 0.307 | 0.400 |
| HYBRID_R2g_union_sumw_a1_b03 | 0.701 | 0.796 | 0.582 | 0.579 | 0.259 | 0.900 |
| HYBRID_R1b_referent_only | 0.702 | 0.813 | 0.582 | 0.504 | 0.109 | 0.900 |

## Variants — all subsets, NDCG@10

| Variant | all | base | disc | utt | utt_utterance | utt_referent |
|---|---:|---:|---:|---:|---:|---:|
| R1a_anchor_only | 0.031 | 0.012 | 0.000 | 0.223 | 0.445 | 0.000 |
| R1b_referent_only | 0.378 | 0.434 | 0.339 | 0.210 | 0.020 | 0.400 |
| R1c_union_max | 0.383 | 0.426 | 0.320 | 0.354 | 0.309 | 0.400 |
| R2a_union_sum | 0.380 | 0.422 | 0.320 | 0.354 | 0.309 | 0.400 |
| R2b_union_max | 0.383 | 0.426 | 0.320 | 0.354 | 0.309 | 0.400 |
| R2c_union_w_a03_b07 | 0.381 | 0.413 | 0.320 | 0.403 | 0.405 | 0.400 |
| R2d_union_w_a07_b03 | 0.393 | 0.425 | 0.349 | 0.367 | 0.335 | 0.400 |
| R2e_union_routed | 0.397 | 0.425 | 0.349 | 0.403 | 0.405 | 0.400 |
| R2f_union_w_a09_b01 | 0.397 | 0.431 | 0.351 | 0.366 | 0.331 | 0.400 |
| R2g_union_sumw_a1_b03 | 0.399 | 0.436 | 0.349 | 0.367 | 0.335 | 0.400 |
| HYBRID_R2g_union_sumw_a1_b03 | 0.558 | 0.543 | 0.567 | 0.612 | 0.317 | 0.907 |
| HYBRID_R1b_referent_only | 0.551 | 0.557 | 0.567 | 0.475 | 0.043 | 0.907 |

## Hard case: q_utt_0 ("What did I write 2 years ago?") → doc_utt_0

Doc_utt_0 was written 2024-04-23 (= 2y before today) saying "Back in the 90s my dad taught me to fish." Should be retrieved via the utterance anchor, NOT via the 90s referent.

| Variant | rank_of_doc_utt_0 | top5 |
|---|---:|---|
| R1a_anchor_only | 1 | doc_utt_0 |
| R1b_referent_only | not-ranked | doc_multi_3 |
| R1c_union_max | 1 | doc_utt_0, doc_multi_3 |
| R2a_union_sum | 1 | doc_utt_0, doc_multi_3 |
| R2b_union_max | 1 | doc_utt_0, doc_multi_3 |
| R2c_union_w_a03_b07 | 1 | doc_utt_0, doc_multi_3 |
| R2d_union_w_a07_b03 | 1 | doc_utt_0, doc_multi_3 |
| R2e_union_routed | 1 | doc_utt_0, doc_multi_3 |
| R2f_union_w_a09_b01 | 1 | doc_utt_0, doc_multi_3 |
| R2g_union_sumw_a1_b03 | 1 | doc_utt_0, doc_multi_3 |

## Intents assigned (utterance queries)

| QID | text | assigned_intent | expected_intent |
|---|---|---|---|
| q_utt_0 | What did I write 2 years ago? | utterance | utterance |
| q_utt_1 | What messages did I write in 2024? | utterance | utterance |
| q_utt_2 | What did I write in spring 2024? | utterance | utterance |
| q_utt_3 | What did I say in summer 2024? | utterance | utterance |
| q_utt_4 | What did I write in early 2025? | utterance | utterance |
| q_utt_5 | What happened in the 90s? | referent | referent |
| q_utt_6 | What happened in 1995? | referent | referent |
| q_utt_7 | What happened in the 1960s? | referent | referent |
| q_utt_8 | What happened in 2010? | referent | referent |
| q_utt_9 | What happened in the 80s? | referent | referent |

## Cost
- Extraction: $0.0000 (0 in / 0 out tokens)
- Intent classifier: $0.0000 (0 in / 0 out tokens)
- **Total**: $0.0000

## Best variant (by all R@5 → tiebreak all MRR)
- **R2g_union_sumw_a1_b03**
