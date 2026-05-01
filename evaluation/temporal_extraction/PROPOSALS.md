# Temporal Retrieval — Proposal Brainstorm

Ideas organized by category. Top 4 picked for empirical testing. The
ablation already running covers bracket width, scoring function shape, and
aggregation — this doc covers orthogonal directions.

## Extraction-layer ideas

1. **Event-time binding** ★ — Instead of extracting times as an independent list, extract `(event_span, time_expr)` pairs. A doc saying "I had dinner at 6pm, then met Alice at 8pm" produces two bound pairs, not two floating times. Retrieval then joint-matches event semantics AND temporal overlap. Addresses the case where a doc has multiple events with multiple times and only ONE is relevant to the query.
2. **Era / named-period resolution** ★ — Extend extraction to handle "during college", "after COVID", "in my 20s", "before the kids were born". Map to concrete intervals via a per-user era table (LLM-inferable from broader context) or a global era table (decades, well-known events). Major gap in long-form narratives.
3. **Tense + aspect as features** — Past/present/future + completed/ongoing. A query "was I going to the gym?" is different from "did I go to the gym?". Store as payload, filter at retrieval.
4. **Discourse-linked time** — "Then we went to...", "afterward", "meanwhile" — extract implicit temporal ordering relations, not just absolute times. Enables "what came after X" queries.
5. **Multi-hop anchor chains** — Pronouns and deictics resolve through prior discourse. "That meeting" → what meeting? Chain through preceding sentences before bracketing.
6. **Implicit time from world knowledge** — "Grandma's 70th birthday" implies grandma.birth + 70y. "The Obama years" = 2009-2017. Needs LLM knowledge plus a lightweight entity-date table.

## Representation-layer ideas

7. **Time-of-day vs date separation** — Index (date_interval, tod_interval) pairs. Query "Thursday afternoon" should match "Thursday 3pm" but not "Thursday morning". Currently collapsed into a single interval.
8. **Hierarchical / multi-resolution index** — Store intervals at multiple granularities (year, month, week, day, hour). Fast filtering at coarse level, precise scoring at fine level. Like a multi-level spatial index.
9. **Duration as an indexable feature** — "Long meetings" queries need duration. Store duration separately from anchor.
10. **Temporal topology (Allen's algebra)** — "Before the meeting", "during the trip", "after the conference". 13 qualitative relations; all expressible as simple inequalities on (earliest, latest).

## Retrieval / query-side ideas

11. **Pre-materialized temporal probes at ingest** ★ — At ingest time, LLM generates paraphrases per time expression: "March 15 2026" → also embedded as "mid-March", "3/15", "Q1 2026", "the Ides of March". Each paraphrase inherits parent doc-id. Query cosine matches any paraphrase → doc retrieved. Analogous to the v2f dual-view pattern from our associative_recall work (+4.2pp R@20 win on LoCoMo).
12. **Temporal query decomposition** — "What happened in 2023?" → 12 monthly sub-queries, union of retrievals. Covers months the coarse query misses.
13. **Multi-interpretation query expansion** — "Last summer" could be ref.year−1.summer or ref.summer if ref is in autumn. Retrieve for both interpretations, let ranking decide.
14. **Pivot-through-entity retrieval** — Query "when was my wedding?" → find event "wedding" → return its time. Different access pattern from "what happened on June 3?".

## Scoring-layer ideas

15. **LLM-as-relevance-judge** ★ — Given (query time-expr, doc time-expr), ask gpt-5-mini "do these refer to overlapping times?". Use as an upper-bound baseline: if our hand-crafted scorer comes within 5% of the LLM judge, we're calibrated. If not, we have headroom.
16. **Sparsity-aware weighting** — A doc where time is central (calendar entry, diary date) should outweigh a doc where time is incidental. Normalize by doc's temporal density.
17. **Uncertainty-aware ranking** — Low-confidence extractions (LLM.confidence < threshold) down-weight in aggregation. Currently ignored.
18. **Learned scoring** — Train a small model on LLM-judge labels. Replaces hand-crafted formula. Probably overkill for research; useful for production.
19. **Temporal-semantic score interaction** — `score = f(temporal_overlap, semantic_cos)` where f is learned, not a hard filter-then-rerank. Captures "temporal mismatch OK if semantics are strong enough" cases.

## Efficiency ideas (low urgency at prototype scale)

20. **Qdrant payload range filter** — Store `earliest_us, latest_us` as Qdrant payload; combine range filter with vector search in one query. Unifies substrate with the existing vector store.
21. **Interval tree in-memory** — Python `intervaltree` lib for O(log n + k) overlap queries; replaces two B-tree indexes.
22. **Pre-bucketed granularity index** — Separate indexes per granularity level; queries at year-grain search year-bucket, avoid scanning day-grain.

## Higher-level abstractions (future)

23. **Event chains with implicit ordering** — "Store, then pharmacy, then home" — extract sequence, index temporal order.
24. **Timeline reconstruction** — Reconstruct full event timeline for a doc; support event-relative queries ("before she left").
25. **Cross-doc event identity** — "Yesterday's meeting" and "the 3pm sync" might be the same event across two docs. Merge into single event node.

---

## Picked 4 for empirical test

Criteria: high expected lift, orthogonal to running ablation, testable on the
existing synthetic corpus with ≤$2 additional LLM spend.

### E1 — Event-time binding (★)

**Hypothesis**: Joint (event, time) extraction + matching beats independent
extraction. Especially on docs with 2+ events and queries that name one
specific event.

**Method**:
- New extractor emits `list[(event_span, time_expr)]` per doc + query.
- Storage: add `event_span` column + embed event_span with
  text-embedding-3-small into an `event_vec` payload.
- Retrieval score: `α · cosine(q_event_vec, d_event_vec) + β · temporal_overlap(q_time, d_time)`,
  summed across all (q, d) pair matches.
- Tune α/β on a held-out split.

**Expected**: wins on disambiguation-heavy subset; parity elsewhere.

### E2 — Pre-materialized temporal probes at ingest (★)

**Hypothesis**: Ingest-time paraphrase generation of time expressions
improves retrieval via more ways for a query to match, analogous to the
validated v2f dual-view pattern (+4.2pp R@20 on LoCoMo).

**Method**:
- For each extracted TimeExpression, LLM generates 3-5 paraphrase surfaces:
  date-string variants, granularity-shifted variants, named-era variants.
- Each paraphrase is embedded independently; all point to parent doc_id.
- Retrieval runs cosine over the expanded index.

**Expected**: +3-8pp on queries that use different phrasings than the doc.

### E3 — LLM-as-relevance-judge (★)

**Hypothesis**: Establishes the headroom between our hand-crafted scoring
and a gpt-5-mini judge. Answers "how much better could a smarter scorer be?".

**Method**:
- For 20 queries × top-20 hybrid candidates each, gpt-5-mini judges
  "is the temporal content compatible?" → 0-1.
- Use judge scores as a relevance label, compute Recall@5/@10 and compare
  to base hybrid.
- Report gap. If <5%, hand-crafted scorer is optimal; if >20%, significant
  headroom.

**Expected**: headroom is 5-15pp; suggests learned scoring would help.

### E4 — Named-era / implicit-time extraction (★)

**Hypothesis**: Extending extractor to recognize named eras and implicit
world-knowledge times ("during college", "after COVID", "in my 20s")
materially improves recall on long-form personal narratives.

**Method**:
- Extend Pass 1 prompt: include named-era detection class.
- Extend Pass 2 prompt: resolve era → fuzzy interval using LLM world
  knowledge + optional per-doc context (e.g., author's birth year inferred
  from other signals).
- Add ~15 new synthetic docs with era references; re-run eval subset.

**Expected**: +5-10pp recall on era-heavy subset; parity on clock-time
subset.

## Deferred / not picked

- Time-of-day separation (#7): useful but synthetic corpus doesn't stress it;
  would need specific synthetic data.
- Hierarchical index (#8): efficiency play, not accuracy play.
- Tense + aspect (#3): interesting but small expected lift.
- Learned scoring (#18): depends on LLM-judge labels from E3; would be E5.
- Qdrant payload (#20): substrate migration, not a scoring decision.
- Cross-doc event identity (#25): promising but complex; needs its own eval.
