# Round 6 -- Compaction Strategies for Append-Only Topic Logs

Model: `gpt-5-mini`  Embeddings: `text-embedding-3-small`

Budget used: 182 LLM + 6 embed (~$0.46)


## Profile `P1_stable_user` (60 entries)

| Strategy | Correct | Acc | Avg compacted chars | Compaction LLM calls |
|----------|---------|-----|----------------------|----------------------|
| `C1_trunc_last20` | 4/8 | 50% | 1712 | 0 |
| `C2_middle_elision` | 5/8 | 62% | 2455 | 0 |
| `C3_hierarchical` | 4/8 | 50% | 2316 | 2 |
| `C4_query_gated` | 8/8 | 100% | 1687 | 0 |
| `C5_active_consolidate` | 7/8 | 88% | 3673 | 1 |
| `C6_relation_compact` | 6/8 | 75% | 3334 | 0 |
| `C7_hybrid_C6_C3` | 5/8 | 62% | 2282 | 2 |

## Profile `P2_evolving_user` (60 entries)

| Strategy | Correct | Acc | Avg compacted chars | Compaction LLM calls |
|----------|---------|-----|----------------------|----------------------|
| `C1_trunc_last20` | 5/8 | 62% | 1785 | 0 |
| `C2_middle_elision` | 7/8 | 88% | 2501 | 0 |
| `C3_hierarchical` | 7/8 | 88% | 2332 | 2 |
| `C4_query_gated` | 8/8 | 100% | 1678 | 0 |
| `C5_active_consolidate` | 7/8 | 88% | 3546 | 1 |
| `C6_relation_compact` | 7/8 | 88% | 3357 | 0 |
| `C7_hybrid_C6_C3` | 8/8 | 100% | 2419 | 2 |

## Profile `P3_detail_dense_user` (60 entries)

| Strategy | Correct | Acc | Avg compacted chars | Compaction LLM calls |
|----------|---------|-----|----------------------|----------------------|
| `C1_trunc_last20` | 4/8 | 50% | 1856 | 0 |
| `C2_middle_elision` | 7/8 | 88% | 2665 | 0 |
| `C3_hierarchical` | 7/8 | 88% | 2379 | 2 |
| `C4_query_gated` | 8/8 | 100% | 1810 | 0 |
| `C5_active_consolidate` | 6/8 | 75% | 3911 | 1 |
| `C6_relation_compact` | 4/8 | 50% | 1964 | 0 |
| `C7_hybrid_C6_C3` | 4/8 | 50% | 1906 | 2 |

## Aggregate (across profiles)

| Strategy | Total Correct / N | Acc | Avg chars | Total compaction LLM |
|----------|-------------------|-----|-----------|----------------------|
| `C1_trunc_last20` | 13/24 | 54% | 1784 | 0 |
| `C2_middle_elision` | 19/24 | 79% | 2540 | 0 |
| `C3_hierarchical` | 18/24 | 75% | 2342 | 6 |
| `C4_query_gated` | 24/24 | 100% | 1725 | 0 |
| `C5_active_consolidate` | 20/24 | 83% | 3710 | 3 |
| `C6_relation_compact` | 17/24 | 71% | 2885 | 0 |
| `C7_hybrid_C6_C3` | 17/24 | 71% | 2202 | 6 |

## Per-question detail -- `P1_stable_user`

| Q | type | C1_trunc_last20 | C2_middle_elision | C3_hierarchical | C4_query_gated | C5_active_consolidate | C6_relation_compact | C7_hybrid_C6_C3 |
|---|------|---|---|---|---|---|---|---|
| Q1 | current | Y | Y | Y | Y | Y | Y | Y |
| Q2 | current | Y | Y | Y | Y | Y | Y | Y |
| Q3 | current | Y | Y | Y | Y | Y | Y | Y |
| Q4 | historical | N | N | N | Y | Y | Y | N |
| Q5 | historical | N | N | N | Y | Y | Y | Y |
| Q6 | current | N | Y | Y | Y | Y | N | N |
| Q7 | relationship | N | N | N | Y | N | Y | Y |
| Q8 | relationship | Y | Y | N | Y | Y | N | N |

- Q1: What is the user's current favorite coffee order?
- Q2: How many dogs does the user have and what are their names?
- Q3: What is the user's current job title?
- Q4: Did the user ever take yoga classes, and if so when did they...
- Q5: What was the user's finishing time in the 3M half marathon?
- Q6: What medications does the user currently take?
- Q7: Is the user's cilantro aversion an allergy?
- Q8: What cholesterol medication change happened recently, and di...

## Per-question detail -- `P2_evolving_user`

| Q | type | C1_trunc_last20 | C2_middle_elision | C3_hierarchical | C4_query_gated | C5_active_consolidate | C6_relation_compact | C7_hybrid_C6_C3 |
|---|------|---|---|---|---|---|---|---|
| Q1 | current | Y | Y | Y | Y | Y | Y | Y |
| Q2 | current | Y | Y | Y | Y | Y | Y | Y |
| Q3 | current | Y | Y | Y | Y | Y | Y | Y |
| Q4 | historical | N | Y | Y | Y | Y | N | Y |
| Q5 | current | Y | Y | Y | Y | Y | Y | Y |
| Q6 | relationship | N | Y | Y | Y | N | Y | Y |
| Q7 | historical | N | N | N | Y | Y | Y | Y |
| Q8 | relationship | Y | Y | Y | Y | Y | Y | Y |

- Q1: Where does the user currently live?
- Q2: What is the user's current job title and company?
- Q3: Who is the user's current partner/spouse?
- Q4: Did the user ever work at Zephyr Inc, and why did they leave...
- Q5: What pets does the user currently have?
- Q6: What was the user's relationship status at the start of the ...
- Q7: What climbing gyms has the user been a member of?
- Q8: Does the user have pets with health issues, and what are the...

## Per-question detail -- `P3_detail_dense_user`

| Q | type | C1_trunc_last20 | C2_middle_elision | C3_hierarchical | C4_query_gated | C5_active_consolidate | C6_relation_compact | C7_hybrid_C6_C3 |
|---|------|---|---|---|---|---|---|---|
| Q1 | current | N | Y | Y | Y | Y | N | N |
| Q2 | current | Y | Y | Y | Y | Y | Y | Y |
| Q3 | current | N | Y | Y | Y | N | N | N |
| Q4 | current | Y | Y | Y | Y | Y | Y | Y |
| Q5 | current | N | Y | Y | Y | Y | Y | Y |
| Q6 | historical | Y | Y | Y | Y | Y | Y | Y |
| Q7 | historical | N | N | N | Y | N | N | N |
| Q8 | relationship | Y | Y | Y | Y | Y | N | N |

- Q1: What type of diabetes does the user have and what devices do...
- Q2: What is the user's current CGM (continuous glucose monitor) ...
- Q3: Who is the user's endocrinologist and where?
- Q4: What are the user's hobbies?
- Q5: What company does the user work at and what project are they...
- Q6: Did the user travel internationally recently, and how did it...
- Q7: What family event has the user attended recently and what di...
- Q8: What is the user's current TIR (time in range) and has it im...