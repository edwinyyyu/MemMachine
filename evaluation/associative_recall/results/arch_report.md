# Architecture Comparison Report

## Summary

Evaluated 9 alternative retrieval architectures against properly normalized baselines on BEAM (30q) and LoCoMo (30q). All comparisons are normalized: baseline gets the same segment budget as the architecture.

Reference: **v15** (self-monitoring prompt + hop-ordered retrieval + neighbor expansion)
- LoCoMo 30q: **+33.9pp** at r@20, 13W/17T/0L
- BEAM 30q: **+6.6pp** at r@20, 6W/22T/2L

## Results Table

| Architecture | Bench | B-r@20 | A-r@20 | Delta r@20 | W/T/L | Avg Ret | Emb/q | LLM/q |
|---|---|---|---|---|---|---|---|---|
| segment_as_query | BEAM | 0.563 | 0.512 | -0.051 | 4/20/6 | 65 | 14.0 | 0 |
| segment_as_query | LoCoMo | 0.383 | 0.222 | -0.161 | 1/22/7 | 65 | 14.0 | 0 |
| cluster_diversify | BEAM | 0.563 | 0.496 | -0.067 | 2/23/5 | 100 | 2.0 | 0 |
| cluster_diversify | LoCoMo | 0.383 | 0.400 | +0.017 | 2/27/1 | 100 | 2.0 | 0 |
| **multi_query_fusion** | BEAM | 0.563 | 0.621 | **+0.058** | 4/26/0 | 46 | 6.7 | 1 |
| multi_query_fusion | LoCoMo | 0.383 | 0.333 | -0.050 | 0/28/2 | 42 | 7.0 | 1 |
| **retrieve_summarize_retrieve** | BEAM | 0.563 | 0.650 | **+0.087** | 5/24/1 | 40 | 4.0 | 2 |
| retrieve_summarize_retrieve | LoCoMo | 0.383 | 0.378 | -0.006 | 4/23/3 | 40 | 4.0 | 2 |
| agent_working_set | BEAM | 0.563 | 0.417 | -0.146 | 1/21/8 | 26 | 3.6 | 2.5 |
| **agent_working_set** | LoCoMo | 0.383 | 0.575 | **+0.192** | 9/18/3 | 35 | 4.5 | 3.2 |
| hybrid_gap_fill | BEAM | 0.563 | 0.563 | +0.000 | 0/30/0 | 40 | 4.0 | 1 |
| hybrid_gap_fill | LoCoMo | 0.383 | 0.383 | +0.000 | 0/30/0 | 40 | 4.0 | 1 |
| centroid_walk | BEAM | 0.563 | 0.560 | -0.003 | 0/29/1 | 40 | 2.0 | 0 |
| centroid_walk | LoCoMo | 0.383 | 0.217 | -0.167 | 0/24/6 | 40 | 2.0 | 0 |
| **negative_space** | BEAM | 0.563 | 0.621 | **+0.057** | 4/25/1 | 45 | 2.0 | 0 |
| **negative_space** | LoCoMo | 0.383 | 0.450 | **+0.067** | 2/28/0 | 45 | 2.0 | 0 |
| **mmr_diversified** | BEAM | 0.563 | 0.636 | **+0.073** | 7/21/2 | 60 | 2.0 | 0 |
| **mmr_diversified** | LoCoMo | 0.383 | 0.450 | **+0.067** | 3/26/1 | 60 | 2.0 | 0 |
| *v15 reference* | BEAM | 0.563 | 0.629 | *+0.066* | *6/22/2* | *52* | *~4* | *1* |
| *v15 reference* | LoCoMo | 0.383 | 0.722 | *+0.339* | *13/17/0* | *58* | *~4* | *1* |

## Cost Comparison

| Architecture | LLM calls/q | Embed calls/q | Cost tier |
|---|---|---|---|
| segment_as_query | 0 | 14.0 | Cheap (embedding only) |
| cluster_diversify | 0 | 2.0 | Cheapest |
| centroid_walk | 0 | 2.0 | Cheapest |
| negative_space | 0 | 2.0 | Cheapest |
| mmr_diversified | 0 | 2.0 | Cheapest |
| multi_query_fusion | 1 | 6.7 | Moderate |
| hybrid_gap_fill | 1 | 4.0 | Moderate |
| retrieve_summarize_retrieve | 2 | 4.0 | Moderate |
| agent_working_set | 2.5-3.2 | 3.6-4.5 | Expensive |
| v15 (reference) | 1 | ~4 | Moderate |

## Architecture Analysis

### Winner: MMR Diversified (no LLM, zero cost)

MMR (Maximal Marginal Relevance) is the best architecture tested:
- **BEAM: +7.3pp** at r@20, W/T/L=7/21/2 (better than v15's +6.6pp)
- **LoCoMo: +6.7pp** at r@20, W/T/L=3/26/1
- **Zero LLM calls**, only 2 embedding calls per question (just the query)
- Achieves this by pulling from a large candidate pool (top-150 by cosine) then selecting for diversity using the MMR criterion: balance relevance to query with dissimilarity to already-selected segments

**Why it works**: The cosine top-k baseline tends to retrieve many near-duplicate or highly overlapping segments. MMR explicitly penalizes selecting segments similar to ones already selected, spreading coverage across the conversation. This diversity is exactly what multi-hop questions need.

### Runner-up: Negative Space (no LLM, zero cost)

- **BEAM: +5.7pp**, W/T/L=4/25/1 (0 losses on LoCoMo!)
- **LoCoMo: +6.7pp**, W/T/L=2/28/0
- Zero LLM calls, only 2 embedding calls per question
- Pushes the query embedding away from the centroid of already-found segments, exploring new territory

**Why it works**: Similar insight to MMR but achieved differently. Instead of selecting diverse segments from a pool, it shifts the query vector to point away from what's already been found. On LoCoMo, it achieved zero losses (never hurt performance).

### Interesting: Retrieve-Summarize-Retrieve (best BEAM single metric)

- **BEAM: +8.7pp** at r@20, W/T/L=5/24/1 -- highest single delta on BEAM
- LoCoMo: -0.6pp (essentially neutral)
- Costs 2 LLM calls + 4 embedding calls per question

**Why it works on BEAM**: The summary creates an embedding "centroid" that captures the gist of found content. On BEAM's focused conversations (single topic like "triangles"), this centroid is meaningful and lands near related content. On LoCoMo's diverse multi-topic conversations, the summary centroid is too generic to help.

### Interesting: Agent Working Set (best LoCoMo single metric)

- BEAM: -14.6pp (terrible)
- **LoCoMo: +19.2pp** at r@20, W/T/L=9/18/3
- Costs 2.5-3.2 LLM calls per question

**Why the split**: The agent's ability to SEARCH with custom queries helps on LoCoMo's diverse conversations (targeted queries find specific topics). But on BEAM's focused conversations, the agent stops too early (avg 26 segments vs 35 on LoCoMo) because it thinks it has enough coverage for the narrow topic -- but it doesn't have enough turns for multi-source questions.

### Strictly Worse Architectures

1. **Segment-as-query**: -5.1pp BEAM, -16.1pp LoCoMo. Walking through embedding space by using segments as queries drifts AWAY from the question. The "walk" is an aimless random walk, not directed by the question.

2. **Cluster-diversify**: -6.7pp BEAM, +1.7pp LoCoMo. K-means clustering of the top-100 and round-robin selection loses the ranking signal. The clusters are arbitrary geometric divisions, not semantically meaningful.

3. **Centroid walk**: -0.3pp BEAM, -16.7pp LoCoMo. Drifting the query toward the centroid of found content is the OPPOSITE of what you want -- it reinforces what's already found instead of exploring new territory.

4. **Hybrid gap-fill**: +0.0pp on both. The gap-fill cues land in the same regions as the baseline top-20. At r@20 the architecture returns the baseline top-20 exactly (the LLM-retrieved segments are appended beyond position 20).

## Key Findings

### 1. The LLM is NOT necessary for the retrieval improvement

MMR (+7.3pp BEAM) and Negative Space (+5.7pp BEAM, +6.7pp LoCoMo) achieve results competitive with v15 (+6.6pp BEAM) using ZERO LLM calls. The improvement comes from **diversification**, not from "understanding" the question better.

### 2. The real problem is duplicate/overlapping retrieval

Baseline cosine top-k wastes its budget on segments that are semantically near-identical. Any method that enforces diversity -- MMR, negative space, or LLM-generated cues that happen to target different topics -- captures the gains.

### 3. v15's massive LoCoMo advantage (+33.9pp) is NOT replicated by any architecture

No architecture comes close to v15's +33.9pp on LoCoMo. The best is agent_working_set at +19.2pp, but it's inconsistent (terrible on BEAM). v15's iterative cue generation with neighbor expansion is doing something special on LoCoMo -- likely the NEIGHBOR EXPANSION is the key factor, as it captures surrounding context that simple retrieval misses.

### 4. Benchmark-specific behavior is real and important

Every architecture shows different performance on BEAM vs LoCoMo. BEAM conversations are focused (single topic), LoCoMo conversations are diverse (many topics). This means:
- Diversity-focused methods (MMR, negative space) help on both but moderately
- LLM-guided search helps more on LoCoMo's diverse conversations (agent_working_set +19.2pp)
- Summary-based methods help on BEAM's focused conversations (retrieve_summarize +8.7pp)

### 5. Hybrid gap-fill is a failure mode

The "practical" architecture of baseline top-20 + LLM gap-fill produces exactly zero improvement. The gap-fill cues don't find anything new within the first 20 positions. This suggests the value of LLM cues is in REPLACING baseline retrieval (pushing low-quality segments out of the top-k), not in APPENDING to it.

## Assessment: Is the current architecture the right shape?

**The iterative LLM-cue loop (v15) is the right shape for LoCoMo but overkill for BEAM.**

For BEAM-like tasks (focused conversations), MMR diversification achieves comparable results at zero LLM cost. The optimal architecture would be:

1. **MMR-diversified retrieval** as the base layer (free, always-on diversity)
2. **Conditional LLM cue generation** only when the conversation is long/diverse enough to warrant it
3. **Neighbor expansion** as the key addition -- this is likely the main driver of v15's LoCoMo advantage over all tested architectures

The most promising direction would be combining MMR as the base retrieval with v15's neighbor expansion. This would capture the diversity benefit for free and add the neighbor-context benefit that drives LoCoMo performance.

### Recommended next experiment

Test: `MMR top-60 + neighbor expansion (radius=1)` -- this would be a no-LLM architecture that gets the diversity benefit of MMR plus the context-fill benefit of neighbor expansion. If it approaches v15's LoCoMo numbers, the LLM cue generation is unnecessary overhead.
