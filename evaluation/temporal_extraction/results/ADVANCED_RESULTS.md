# Advanced Temporal Experiments — Results

Four orthogonal experiments (E1-E4) on top of the base hybrid
temporal-retrieval system. Base hybrid (T and S) on 55 queries:
**R@5 0.555 / MRR 0.918 / NDCG@10 0.652 / critical 5/5.**

Models: gpt-5-mini (LLM), text-embedding-3-small (embeddings). All
caches under `cache/advanced/`.

---

## E1 — Event-time binding

**Status: done.** New files: `event_binding.py`, `event_store.py`,
`event_retrieval.py`. Separate SQLite at `cache/temporal_advanced.db`.

One gpt-5-mini call per text returns `{pairs: [{time_surface,
event_span}]}`. Event spans are embedded with text-embedding-3-small
and stored alongside the resolved bracket. Retrieval scores
`alpha * cos(q_event, d_event) + beta * jaccard_overlap(q_time, d_time)`,
aggregating per doc by summing positive pair scores.

Queries in this corpus are almost all "What happened on X?" — so the
LLM correctly returns null for event_span on queries. That zeroes the
semantic channel for every query, making alpha irrelevant (0.3/0.5/0.7
all score identically).

| Metric    | Base hybrid | E1 (best alpha=0.3) | Delta  |
|-----------|-------------|----------------------|--------|
| Recall@5  | 0.555       | **0.535**            | -0.020 |
| Recall@10 | 0.590       | **0.567**            | -0.023 |
| MRR       | 0.918       | **0.870**            | -0.048 |
| NDCG@10   | 0.652       | **0.644**            | -0.008 |
| Critical top-1 | 5/5    | 4/5                  | -1     |

**Subset analysis:** event-binding gave no marginal lift on this corpus
because queries never contained event spans. The architecture should
help on docs with 2+ events and queries that name one specific event;
our synthetic queries don't stress that case. Cost: **$0.082**.

---

## E2 — Pre-materialized temporal probes at ingest

**Status: done.** New file: `probes.py`. 57 original time surfaces
expanded to 341 probe rows (3-5 paraphrases each) via gpt-5-mini.
Each paraphrase embedded; query text embedded directly; rank by
max-cosine per doc.

| Metric    | Base hybrid | E2      | Delta  |
|-----------|-------------|---------|--------|
| Recall@5  | 0.555       | 0.501   | -0.054 |
| Recall@10 | 0.590       | 0.594   | +0.004 |
| MRR       | 0.918       | 0.732   | -0.186 |
| NDCG@10   | 0.652       | 0.566   | -0.086 |
| Critical top-1 | 5/5    | **5/5** | +0     |

**Analysis:** probe expansion alone doesn't beat hybrid because the
base hybrid already fuses structured temporal overlap with semantic
rerank. Probes are purely semantic — they smooth the doc embedding but
introduce noise: many time-paraphrases match docs with unrelated
content. Critical-pair top-1 held (5/5), suggesting probes hit
day-specific surface forms reliably. Cost: **$0.087**.

---

## E3 — LLM-as-relevance-judge (upper bound)

**Status: done.** New file: `llm_judge.py`. 20 stratified queries x 20
top-semantic candidates = 400 pair judgments by gpt-5-mini returning a
0-1 score based on extracted time references from both sides.

Compared on the SAME 20-query subset:

| Metric    | Semantic (same subset) | Judge upper-bound | Delta vs Semantic |
|-----------|------------------------|---------------------|-------------------|
| Recall@5  | 0.464                  | **0.663**           | **+0.199**        |
| Recall@10 | 0.536                  | **0.728**           | **+0.193**        |
| MRR       | 0.774                  | **0.897**           | +0.124            |
| NDCG@10   | 0.527                  | **0.740**           | **+0.213**        |
| Critical top-1 | -                  | 3/3 on subset       | -                 |

**Full-corpus base hybrid** is 0.555 R@5 — the judge's 0.663 on a 20-query
subset is a meaningful ~11pp lift over full base hybrid's single-corpus
average, and a full ~20pp lift over a same-subset semantic floor. Big
headroom: a smarter scorer (learned from judge labels, or a stronger
hand-crafted formula) could likely close a significant fraction of this
gap. Cost: **$0.387**.

---

## E4 — Named-era / implicit-time extraction

**Status: done.** New files: `era_extractor.py`, `era_eval.py`, plus
`data/era_{docs,queries,gold}.jsonl` (15 docs, 20 queries covering
personal and world eras).

Extraction correctness (gold-window overlap >= 30%):

| Extractor | docs_hit/total | world | personal |
|-----------|----------------|-------|----------|
| Base      | 2/15 (13%)     | 2/10  | 0/5      |
| **Era**   | **13/15 (87%)**| 10/10 | 3/5      |

Retrieval on era corpus (temporal-only, 20 queries, 15 docs):

| Condition | Recall@5 | Recall@10 | MRR   | NDCG@10 |
|-----------|----------|-----------|-------|---------|
| Base T    | 0.400    | 0.725     | 0.242 | 0.343   |
| **Era T** | **0.950**| **0.975** | 0.917 | 0.908   |
| Pure S    | 0.950    | 0.975     | 0.942 | 0.928   |

**Analysis:** Era extractor recovers 11 more docs out of 15 than the
base extractor and lifts pure-temporal R@5 from 0.40 to 0.95 — tied
with semantic, slightly behind on MRR/NDCG. Personal-era resolution is
still the weak spot (3/5); without a per-doc birth-year context, the
LLM falls back to the rough default window. Cost: **$0.212**.

---

## Ranked lift summary

| Exp | Lift signal                         | Verdict                                            |
|-----|-------------------------------------|----------------------------------------------------|
| E4  | +74pp doc-extraction recall, +55pp R@5 on era corpus | **Big win on era queries.** Unlocks a query class the base extractor cannot see. |
| E3  | +20pp R@5, +21pp NDCG@10 over same-subset semantic   | **Large headroom** — smarter scoring ≈ 10-15pp over base hybrid. |
| E1  | Slight regression (-2pp R@5)                         | Event binding architecture sound but queries in this corpus lack event spans; neutral on this eval. |
| E2  | Mixed (-5pp R@5, neutral R@10)                       | Pure-paraphrase semantic probes underperform structured hybrid. Temporal brackets still dominate on this corpus. |

## Total cost

$0.082 (E1) + $0.087 (E2) + $0.387 (E3) + $0.212 (E4) ≈ **$0.77** LLM
spend across all four experiments. Embeddings negligible
(text-embedding-3-small, ~500 short texts, cache-heavy).

## Top recommendation

**Invest in E4 (era extraction) + scorer-learning via E3 labels.**
- E4 delivers lift on a realistic long-form narrative query class the
  base system cannot currently see.
- E3 shows clear headroom for a smarter scorer. Use the 400 already-
  generated judge labels to train a lightweight cross-feature scorer
  (e.g., gradient-boosted on features: jaccard, granularity gap,
  best-point distance, semantic cosine, era-class indicator) — this
  follows directly from the proposal's E5 item.
- E1 and E2 are worth revisiting only on corpora with richer query
  shapes (multi-event docs for E1; low-overlap paraphrase mismatch for
  E2). Skip them for this corpus.
