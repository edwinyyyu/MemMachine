# T_causal — 2-step plan for causal_relative queries

## 1. R@1 on causal_relative — before vs after

| variant                            | R@1   | (count)   | R@5   | MRR   |
|---                                 |---:   |---        |---:   |---:   |
| rerank_only (semantic+CE)          | 0.467 | 7/15      | 1.000 | 0.656 |
| fuse_T_R + recency_additive (BASE) | 0.467 | 7/15      | 1.000 | 0.656 |
| **causal_mask**                    | **0.733** | 11/15 | 0.867 | 0.801 |
| **causal_signed (λ=0.5)**          | **0.733** | 11/15 | 0.933 | 0.828 |

**+0.267 R@1** over baseline (7 → 11 hits). Both strategies agree on R@1; signed wins on R@5 (0.933 vs 0.867) and MRR (0.828 vs 0.801).

## 2. Regression check (R@1, all 11 benchmarks)

| Benchmark            |  n | n_causal | n_resolved | rerank_only | fuse_T_R+rec | mask  | signed | Δ mask | Δ signed |
|---                   |---:|---:      |---:        |---:         |---:          |---:   |---:    |---:    |---:      |
| hard_bench           | 75 |  0       |  0         | 0.640       | 0.893        | 0.893 | 0.893  | +0.000 | +0.000   |
| temporal_essential   | 25 |  0       |  0         | 0.920       | 1.000        | 1.000 | 1.000  | +0.000 | +0.000   |
| tempreason_small     | 60 | 22       | 20         | 0.650       | 0.733        | 0.733 | 0.783  | +0.000 | **+0.050** |
| conjunctive_temporal | 12 |  0       |  0         | 1.000       | 1.000        | 1.000 | 1.000  | +0.000 | +0.000   |
| multi_te_doc         | 12 |  0       |  0         | 1.000       | 1.000        | 1.000 | 1.000  | +0.000 | +0.000   |
| relative_time        | 12 |  0       |  0         | 0.250       | 0.917        | 0.917 | 0.917  | +0.000 | +0.000   |
| era_refs             | 12 |  1       |  0         | 0.250       | 0.417        | 0.417 | 0.417  | +0.000 | +0.000   |
| latest_recent        | 15 |  0       |  0         | 0.133       | 0.667        | 0.667 | 0.667  | +0.000 | +0.000   |
| open_ended_date      | 15 |  0       |  0         | 0.267       | 0.400        | 0.400 | 0.400  | +0.000 | +0.000   |
| **causal_relative**  | 15 | 15       | 14         | 0.467       | 0.467        | 0.733 | 0.733  | **+0.267** | **+0.267** |
| negation_temporal    | 15 |  0       |  0         | 0.000       | 0.000        | 0.000 | 0.000  | +0.000 | +0.000   |

**No regressions.** open_ended_date / latest_recent / hard_bench all unaffected because the open-ended cue gate suppresses "after 2020"-style firings before the causal step runs. tempreason_small fires the cue 22 times (e.g., "Who was the head of state of Austria after Theodor Körner?"); signed *helps* (+0.050 R@1) and mask is neutral. R@5 on tempreason_small drops 1.000 → 0.867 with mask (-0.133) but only to 0.967 with signed (-0.033).

## 3. Anchor resolution accuracy

- 14/15 causal queries resolved (1 fell below the cosine threshold of 0.30: cr_q_008 "the move").
- 9/15 resolved to the *exact* `_a` doc (60% strict accuracy).
- 11/15 R@1 wins → resolving to the `_g` / `_n0` / `_wd` *near* the right cluster is often good enough for direction filtering, because all docs in a cluster are close in time and the anchor doc just needs to land on the correct *side* relative to the gold.

| qid | phrase | anchor doc | expected `_a` | sim | exact OK |
|---|---|---|---|---:|---:|
| cr_q_000 | the migration | cr_000_a | cr_000_a | 0.441 | Y |
| cr_q_001 | the launch | cr_001_a | cr_001_a | 0.482 | Y |
| cr_q_002 | the last review | cr_002_a | cr_002_a | 0.487 | Y |
| cr_q_003 | the offsite | cr_003_a | cr_003_a | 0.503 | Y |
| cr_q_004 | the merger | cr_004_a | cr_004_a | 0.577 | Y |
| cr_q_005 | the funding round | cr_005_a | cr_005_a | 0.622 | Y |
| cr_q_006 | the keynote | cr_006_g (gold) | cr_006_a | 0.534 | N |
| cr_q_007 | the audit | cr_007_a | cr_007_a | 0.487 | Y |
| cr_q_008 | the move | UNRESOLVED | cr_008_a | <0.30 | - |
| cr_q_009 | the cutover | cr_009_a | cr_009_a | 0.439 | Y |
| cr_q_010 | the design summit | cr_012_n0 | cr_010_a | 0.604 | N |
| cr_q_011 | the marathon | cr_006_n0 | cr_011_a | 0.494 | N |
| cr_q_012 | the relocation | cr_012_a | cr_012_a | 0.342 | Y |
| cr_q_013 | the client onsite | cr_003_n1 | cr_013_a | 0.429 | N |
| cr_q_014 | the promotion | cr_014_wd | cr_014_a | 0.364 | N |

## 4. Mask vs signed — which works?

**Signed wins overall** (same R@1, better R@5/MRR, lifts tempreason_small).

The tradeoff:
- **Mask** (drop wrong-side docs) is decisive — when anchor is right, it always pushes the gold to rank 1. When anchor is *wrong* (e.g. cr_q_006 resolved to gold doc itself, dated 10/25/2023, and the *actual* anchor `_a` is 11/9/2023), mask zeroes out the true gold and tanks the rank to 10.
- **Signed** (subtract λ=0.5 from wrong-side docs, suppress anchor itself) is forgiving — wrong-side docs can still surface if their topical score is high. cr_q_006: signed keeps the gold at rank 1; mask drops it to rank 10. Signed also lifts tempreason from 0.733 → 0.783 because wrong-side docs are penalized but not killed when the anchor resolution is noisy.

Per-query: mask wins 1 query that signed ties (cr_q_001), signed wins 1 query mask drops (cr_q_006). Net: signed > mask on this benchmark.

## 5. Limitations — patterns this still misses

1. **Anchor resolution failure for short noun phrases without distinguishing tokens.** cr_q_008 "the move" fell below sim threshold 0.30; "the move" is too generic to embed-match against "Yuki moved from Seattle to Singapore". Fix: lower threshold OR enrich anchor phrase with subject from query ("Yuki's move").
2. **Anchor resolution that lands on the wrong cluster but high-confidence.** cr_q_010 "the design summit" resolved to `cr_012_n0` (Hannah attended the design summit on 9/8/2023) instead of `cr_010_a` (Hannah attended the Lisbon design summit on 6/15/2023, the explicit anchor). Two anchor candidates in the corpus, the *wrong* one is a higher-cosine match. Fix: prefer the doc closest to query subject + phrase; or use top-K resolution with disambiguation by subject mention.
3. **Anchor that resolves to the gold itself.** cr_q_006 collapsed onto `cr_006_g` because gold also mentions "keynote". Mask then erases the actual gold. Signed handles this; mask doesn't.
4. **Multi-cue / nested anchors.** "after X but before Y" — current regex stops at the first cue and would only constrain one side. None of the 15 benchmark queries use this pattern.
5. **Causal cue without a noun anchor (counterfactuals).** "What did X say after that?" / "before then" / "right after" — pronoun anchors that refer to context. No fix without dialogue context.
6. **Cue overfiring on tempreason proper-noun queries** ("after Theodor Körner") — these are not causal_relative semantically; they are open-ended-event-relative. Signed is robust here because the anchor doc resolves to the named person and the signed penalty is small enough to not displace correct-direction docs (and even helps tempreason +0.050 because it nudges out wrong-direction noise). Mask is exactly neutral. Neither is a regression, but signaling could be cleaner with a "named-event" classifier.

## Recommendation

Ship **causal_signed** (λ=0.5) as the gate after open_ended_date and before fuse_T_R + recency_additive. Anchor resolution: cosine top-1 over the noun phrase following the cue, with a minimum-similarity floor of 0.30. On causal_relative: 0.467 → 0.733 R@1 (+0.267); on tempreason_small: 0.733 → 0.783 (+0.050); zero regressions on the other 9 benchmarks.
