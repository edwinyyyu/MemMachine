# RAG Integration — Fusion Strategies

Temporal retrieval is now a credible standalone channel (multi-axis + utterance
anchor + Allen). Next: integrate it into a real RAG pipeline alongside a
semantic retriever. Test fusion strategies empirically.

## Retrievers available

1. **T-multi-axis** — multi-axis scorer (α=0.5/β=0.35/γ=0.15) over structured time extractions, plus utterance anchor. Ships as current temporal default.
2. **S-cosine** — semantic cosine over full doc text, text-embedding-3-small.
3. **A-allen** — Allen-relation channel (before/after/during/overlaps/contains). Fires only on queries with non-null `relation` + event anchor.
4. **E-era** — era-extractor channel. Fires on queries with era surface.

Each retriever produces a ranked list with scores. The fusion problem: combine them into a single ranking.

## Fusion strategies to test

### V1 — CASCADE (current system)
Temporal filter produces candidate set → semantic cosine reranks within.
Simple, what we have today.

### V2 — TEMPORAL-ONLY (baseline)
Skip semantic; rank purely by temporal structure.

### V3 — SEMANTIC-ONLY (baseline)
Skip temporal; rank purely by cosine.

### V4 — RRF-ALL
`score(doc) = Σ_i 1 / (k + rank_i(doc))` summed across T + S + A + E.
Classical IR fusion. Score-scale invariant.

### V5 — ROUTED-SINGLE
LLM router classifies query into one of {temporal, semantic, relational, era}. Invoke the chosen retriever only. Risk: router mistakes.

### V6 — ROUTED-MULTI
Router picks 1-N retrievers to invoke. Fuse chosen retrievers via RRF.

### V7 — SCORE-BLEND
Normalize each retriever's scores to [0,1] via min-max within top-K. Combine linearly: `score = 0.4·T + 0.4·S + 0.1·A + 0.1·E`. Sweep weights.

### V8 — LLM-RERANK (upper bound)
Gather top-20 from each retriever, dedupe, submit to gpt-5-mini for pairwise relevance scoring. Rerank by LLM score. Expensive; sets the ceiling.

### V9 — HYBRID-CASCADE-RRF
CASCADE for the temporal-rich queries, RRF for the ambiguous ones, pure
semantic when temporal extractor finds nothing.

## Metrics

Measure per variant on each query subset:
- base 55, discriminator 30, utterance 10, era 20, axis 20, allen 20, adversarial 50+
- R@5, R@10, MRR, NDCG@10
- Per-subset winner + overall

## Expected

- **RRF robust winner** across mixed subsets (proven in IR literature)
- **ROUTED-SINGLE fastest** but worst on multi-intent queries
- **LLM-RERANK best** but 10-50× cost of RRF
- **CASCADE strong** on temporal-rich, weak on loosely-temporal
- **SCORE-BLEND sensitive** to weight tuning

Target: find a variant that stays within 3pp of LLM-RERANK on every subset at
small fraction of the cost.

## Deliverables

- `rag_router.py` — LLM intent classifier
- `rag_fusion.py` — RRF + score-blend implementations
- `rag_pipeline.py` — 9 variants
- `rag_eval.py` — orchestration
- `results/rag_integration.md` + `.json`
