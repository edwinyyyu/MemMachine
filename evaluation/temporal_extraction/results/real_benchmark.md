# Real-Benchmark Generalization — TempReason-derived

## TL;DR

Adapted **TempReason** (test_l2 + test_l3, Wikidata-derived QA) into a 202-doc /
70-query retrieval benchmark. **Semantic-only (text-embedding-3-small) cosine
already gets R@5 = R@10 = 1.000 on this benchmark.** There is nothing for
the temporal pipeline to improve on R@5/R@10. The only headroom is in
ranking quality among the top few (MRR=0.62, NDCG@10=0.72), and even there,
the failure modes that drive the gap are **structural (gold-doc shares the
exact entity tokens with the query)** rather than **temporal**. Our
synthetic adversarial categories (A1-A9, S1-S8) **do not fire** on this
distribution. The full v2pp + V7L pipeline ran into time-budget limits
before completing extraction; results below cover the semantic-only run.

## 1. Benchmark

**Source.** `tonytan48/TempReason` test_l2.json + test_l3.json (downloaded
from HuggingFace Hub; primary HF dataset registration was broken).

**Adaptation.** Each query's `fact_context` field is a list of Wikidata
sentences:
> `Jaroslav Pelikan works for Concordia Seminary from Jan, 1949 to Jan, 1953.`

We split each fact_context into individual fact lines, capped at 3 facts per
query (gold + 2 distractors), pooled across queries to build a shared
corpus. For L2 queries, gold = the unique fact line whose entity matches
the answer AND whose date range covers the query date. For L3 queries
(`before/after <entity>`), gold = the fact line containing the answer
entity (relational reasoning over the corpus).

**Size.** 202 docs / 70 queries (40 L2 + 30 L3); gold cardinality = 1 per
query. Build script: `real_benchmark_build.py`.

**How it differs from our synthetic data.**

| dimension | synthetic (v2'' adversarial / lattice / etc.) | TempReason-derived |
|---|---|---|
| Vocabulary | hand-authored personal narratives | real Wikidata entity names (people, orgs, sports teams) |
| Doc length | ~30-80 tokens | ~12 tokens (one sentence) |
| Date language | fuzzy modifiers, holidays, narrative anchors, embedded quotes, named eras, tense shifts | `from <Mon, YYYY> to <Mon, YYYY>` only |
| Adversarial categories firing | A1-A9, R1-R7, S1-S8 by construction | none — explicit ranges everywhere |
| L2 query temporal cue | various | always `in <Mon>, <YYYY>` |
| L3 query temporal cue | various | `before/after <entity>` (entity-relational, no date in query) |

In other words, the real benchmark is roughly the **easy interior of the
extraction-quality space** that our synthetic suite was designed to stress.

## 2. Per-system metrics

Semantic-only (text-embedding-3-small cosine), full corpus + queries:

| subset | n | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| all | 70 | **1.000** | **1.000** | 0.624 | 0.720 |
| L2  | 40 | 1.000 | 1.000 | 0.683 | 0.765 |
| L3  | 30 | 1.000 | 1.000 | 0.544 | 0.660 |

The temporal pipeline (V7L SCORE-BLEND) was not completed: at concurrency
12 with gpt-5-mini, v2pp Pass-1 + Pass-2 averaged ~5-8s wall per doc and
extracted ~3 TimeExpressions per doc. With 202 docs the projected
extraction wall time was ~30+ min, and the projected LLM cost on the full
1051-doc corpus was ~$14 (over the $5 cap). A scaled-down attempt
(50 docs / 20 queries) was started but stopped at 10/50 docs to stay
within the 30-min wall cap.

## 3. Did the lift hold?

**No, and there's no headroom on R@5/R@10 to demonstrate one.** The
benchmark structure (1-sentence docs, gold doc shares the answer entity's
exact tokens with the query) makes semantic retrieval saturate at 1.000.
Whatever the temporal pipeline does, it can only **change the rank of the
gold doc within the top-5** (i.e. lift MRR from 0.62 → ?).

Note: the lift our synthetic suite reports is dominated by R@5 gains on
the adversarial subset (e.g., +13-15 points on Allen / lattice subsets in
recent ablation runs). With baseline R@5 = 1.000 on the real distribution,
that pathway is closed.

## 4. Failure-mode classification

**SEMANTIC-ONLY had ZERO failures (R@5 = 1.0 across all 70 queries).**
The MRR/NDCG gap (0.62 vs ideal 1.0) comes from gold doc landing at rank
2-5 instead of rank 1. Spot-checking: when L2 query mentions "Vincent
Ostrom in Dec, 1978", several of the same person's employer-fact docs end
up in top-5 (since they share the entity name and the date string format),
and semantic embedding can't distinguish which one CONTAINS Dec 1978 from
which one is just LEXICALLY similar.

This is exactly where a temporal-interval channel SHOULD help — but to
prove it requires running the full pipeline, which we couldn't finish.

**Failure-mode categories on this benchmark vs synthetic A1-S8:**

| category (synthetic) | fires on TempReason? | comment |
|---|---|---|
| A1 self-anchored ref_time | no | docs/queries are atomic, no nested narrative |
| A2 compositional relative | no | only `from X to Y` |
| A3 fuzzy modifiers ("a couple of") | no | all dates explicit |
| A4 same-day/weekday ambiguity | no | only month-granularity |
| A5 unknown-entity refs ("since divorce") | no | only known entities |
| A6 non-standard recurrence | no | no recurrence |
| A7 fictional/hypothetical | no | all factual |
| A8 tense+aspect shifts | no | all simple-present biographical |
| A9 holidays/eras | no | calendar months only |
| R1 massive span vs point | partial | ~5-10 year intervals vs month query |
| R2 recurrence density | no | no recurrence |
| R5 paraphrastic | no | exact tokens shared |
| S2-S8 retrieval-layer | no | corpus is too uniform |

**New failure-mode categories specific to this benchmark** that our
synthetic suite did NOT cover:

- **L3 entity-relational** ("X plays for which team after Y?") — the
  query has no date token; the temporal answer requires looking up Y's
  date in the corpus, then choosing an X-fact whose start ≥ Y's end.
  Our pipeline has no notion of entity-conditional date lookup.
- **Same-entity polyfact ranking** — the gold and several distractors
  share the same person/org name and the same date-format pattern; the
  ranking decision depends on whether the date range *covers* the query
  date, not on lexical similarity.

## 5. Cost

Spent:
- Embeddings (text-embedding-3-small): ~$0.01 (negligible).
- v2pp diagnostic run on 20 docs: ~$0.23.
- v2pp partial mini run on 10 docs (full eval cancelled): ~$0.10.
- **Total spend: ~$0.35** (well under $5 cap).

Projected cost if completed:
- v2pp on the full benchmark (202 docs + 70 queries × ~2-3 LLM calls each
  × ~$0.012/item) ≈ $3.30. Achievable within budget.
- v2pp on the originally-targeted 1051 docs / 150 queries: ~$14
  (over budget — that's why the corpus was cut to 202).

The real bottleneck was **wall time, not dollars**. gpt-5-mini reasoning
mode adds substantial latency per call (~5-15s); doc extraction at
concurrency 12 still queued items long enough that the full corpus
exceeded the 30-min wall cap.

## 6. Generalization verdict

**The temporal pipeline cannot be evaluated on this benchmark because
semantic-only already saturates R@5.** That's a generalization finding
in itself: our synthetic suite was constructed by design to defeat
semantic retrieval (long narratives, ambiguous time language, multi-doc
distractors), and the resulting "lift" of the temporal pipeline is
specific to that adversarial regime. On a real Wikidata-derived QA
distribution where:
1. each doc is a single dated fact,
2. the gold doc lexically overlaps with the query at the entity level,
3. and the corpus has no fuzzy / metaphorical / narrative time language,

semantic retrieval is the right primary signal, and the temporal pipeline
adds at best a small re-ranking gain (untested).

**Generalization caveats:**

1. The complement is also true: TempReason is **also** a narrow
   distribution (Wikidata-derived, structured biographical facts). If you
   want to measure "does the temporal pipeline generalize", you need a
   benchmark where (a) docs are richer than one sentence, AND (b)
   semantic retrieval cannot exploit entity-token overlap. Multi-news /
   CNN-DailyMail with synthetic temporal questions is the right candidate
   in our fallback list and would be the natural next step.
2. Our v2pp + V7L pipeline is **costly and slow** (5-15s per doc-pass-1
   under reasoning mode). For a real recall@K benchmark with 1000+ docs
   it pushes against single-digit-dollar budgets. Cache reuse across
   benchmarks helps.
3. The L3 / entity-relational query type ("after Y") is a category our
   synthetic adversarial suite **did not** include. It requires
   entity-conditional date lookup over the corpus before scoring. This
   is a real-world failure pattern worth adding to the synthetic suite.

## 7. Files

- `data/real_benchmark_docs.jsonl` — 202 fact-line docs.
- `data/real_benchmark_queries.jsonl` — 70 queries (40 L2, 30 L3).
- `data/real_benchmark_gold.jsonl` — gold qrels.
- `real_benchmark_build.py` — TempReason ingest + adapter.
- `real_benchmark_semantic_eval.py` — semantic-only baseline (ran).
- `real_benchmark_eval.py` — full V7L pipeline (did not complete in time).
- `results/real_benchmark_semantic.json` — completed semantic-only metrics.
