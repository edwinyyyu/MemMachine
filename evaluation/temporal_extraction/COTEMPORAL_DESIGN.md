# Co-Temporal Retrieval — Document-to-Document Temporal Graph

Current pipeline retrieves documents that match the QUERY's temporal references. This extends it: documents that SHARE temporal ground with each other form a graph, and queries can traverse that graph to reach items the direct retrieval misses.

## Motivating examples

1. **S8 adversarial** — "What year did I meet my wife?"
   - Doc A: "I met my wife at the 2012 Boulder retreat" (has date, no 'wife')
   - Doc B: "My wife and I adore hiking" (has 'wife', no date)
   - Doc C: "The Boulder retreat in 2012 changed me" (has date + retreat)
   
   Direct retrieval: Doc B matches on "wife", Doc A matches on "year" loosely. Doc B has no temporal signal; Doc A has no wife mention. Neither alone answers the question.
   
   Co-temporal expansion: Doc A and Doc C co-mention 2012 / Boulder retreat → linked. Query pivots through the shared temporal ground to gather both — and cosine rerank within the expanded set prefers Doc A for "wife+year".

2. **Cross-doc event linking** — "What did Alice say at the conference?"
   - Doc A: "The AI conference was Apr 12-15, 2024"
   - Doc B: "Alice's keynote was about memory systems" (written 2024-04-13)
   - Doc B's utterance anchor falls inside Doc A's interval → co-temporal link → Doc B retrieved when query mentions "the conference" even though Doc B itself doesn't name it.

3. **Paraphrase-diverged same-time** —
   - Doc A: "On March 15, 2024, I..."
   - Doc B: "The Ides of March 2024 were memorable"
   - Neither may embed-match the other; but both extract to the same day → co-temporal.

## Algorithm

### Graph construction (at ingest)

```
for each doc d in corpus:
    for each time_expr t in d:
        candidates = SELECT doc_id FROM intervals
                     WHERE earliest < t.latest
                     AND latest > t.earliest
                     AND doc_id != d.id
        for each neighbor c in candidates:
            compute edge_weight = multi_axis_score(t, c.time_expr)
            if edge_weight > THRESHOLD:
                add edge (d, c, weight=edge_weight, shared_ref=t)
```

Cap edges per node at top-M by weight (default M=20) to prevent dense decades/years from exploding the graph.

Store as adjacency list: `cotemporal_edges(doc_id, neighbor_id, weight, shared_interval_id)`.

### Retrieval (at query time)

```
first_hop   = T.retrieve(query)                       # direct temporal
expansion   = {}
for doc in first_hop.top_k(K_seed):
    neighbors = SELECT neighbor_id, weight 
                FROM cotemporal_edges 
                WHERE doc_id = doc.id
                ORDER BY weight DESC 
                LIMIT M_neighbors
    for n, w in neighbors:
        expansion[n] += first_hop.score(doc) * w * decay

combined = first_hop UNION expansion
rank combined by:
    final = α · first_hop_score + β · expansion_score + γ · semantic_cosine
```

Parameters:
- `K_seed = 20` (top-N from direct retrieval feed the expansion)
- `M_neighbors = 10` (per-seed neighbor budget)
- `decay = 0.5` (expansion scores discounted vs direct hits)
- `α = 0.6, β = 0.25, γ = 0.15`

### Fuzziness

Edge threshold uses the multi-axis scorer, not just overlap. Two docs co-mention if:
- their time-expressions have `multi_axis_score > 0.3`, OR
- they fall in the same (year, month) axis-tag bucket, OR
- one is a recurrence instance of the other's explicit date

Fuzzier than a hard interval equality — so "March 2024" and "March 15, 2024" count as co-mention, but "the 90s" and "March 2024" don't.

## Evaluation plan

### New synthetic data — designed for co-mention retrieval

30-40 cross-linked documents where the answer requires traversing temporal edges. Example:
- `cot_1_event` = "I went to the 2012 Boulder retreat" (year-specific event)
- `cot_1_connected` = "My wife and I met there" (no date, needs link)
- `cot_1_distractor` = "Boulder is a city in Colorado" (no date, shouldn't link)

Queries that require co-mention:
- "When did I meet my wife?" — needs `cot_1_event` + `cot_1_connected`
- "What happened at the retreat?" — needs all `cot_1_*` linked docs

Write to `data/cotemporal_*.jsonl`. Gold specifies for each query which docs should appear in top-K AND the intermediate link that justifies it.

### Metrics

Compare base pipeline (V7 SCORE-BLEND) vs +co-temporal expansion on:
- Base 55 queries (regression check)
- Adversarial S8 subset (target: close the cross-doc gap)
- New cotemporal queries (target: co-mention path retrieves gold)

Metrics: R@5, R@10, MRR, NDCG@10 per subset.

Also:
- **Graph density** — nodes, edges, average degree, max degree (catch runaway cases)
- **Expansion lift / noise ratio** — of the expansion candidates, how many are truly relevant vs. topic drift

## Risks

- **Topic drift**: if a doc mentions a very common date (today, "2020"), its neighborhood explodes. Mitigation: weight edges by granularity specificity; rare temporal references produce stronger edges.
- **Doubly-retrieving irrelevant content**: expansion amplifies docs that share time but nothing semantic. Mitigation: semantic cosine γ term in final scoring.
- **Graph stale-ness**: new docs require edge recomputation. Mitigation: incremental updates at ingest.

## Deliverables

- `cotemporal_graph.py` — graph build + query
- `cotemporal_retrieval.py` — expansion-aware retrieval integrating with V7 SCORE-BLEND
- `cotemporal_synth.py` — new synthetic linked data
- `cotemporal_eval.py` — orchestrate comparison
- `results/cotemporal.md` + `.json`
