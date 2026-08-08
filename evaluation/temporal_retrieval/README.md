# Temporal Retrieval (v5.1)

A clean reference implementation of the v5.1 temporal-aware retriever.
Drops in as a re-ranking layer on top of any RAG pipeline that already has
embeddings and a cross-encoder.

## What it does

Takes time-aware queries — explicit dates ("in March 2024"), deictic
phrases ("yesterday", "two weeks ago"), recurring periods ("in March",
"during Diwali"), event references ("after the launch"), and personal
eras ("back in college") — and returns documents that satisfy the
temporal constraint, scored with semantic relevance.

Verified reproduction of v5.1 production behavior:

| Bench                     | Macro R@1 | Macro R@5 |
|---------------------------|-----------|-----------|
| 12-bench standard         | 0.804     | 0.931     |
| ambiguous_year (basic)    | (any)     | 1.000 *all_recall@5* |
| ambiguous_year_adv        | 0.500     | 0.917 *all_recall@5* |

The `research/_smoke_e2e.py` test reproduces the basic ambiguity bench:
builds the retriever from this directory only (no `temporal_extraction/`
imports), indexes 108 docs, runs 12 queries, gets all_recall@5 = 1.000.
Run from `evaluation/`:

```bash
uv run python -m temporal_retrieval.research._smoke_e2e
```

## Architecture

```
                 Query
                   │
                   ▼
          ┌─────────────────┐
          │ DNF Planner     │   Outer-OR of inner-AND clauses
          │ (gpt-5-mini)    │   leaf = (phrase, direction)
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Phrase          │   Each leaf →
          │ Classifier      │     calendar_pin | recurring_period |
          │ (gpt-5-mini)    │     anaphoric_event | personal_era |
          └────────┬────────┘     generic_skip
                   │
                   ▼
          ┌─────────────────┐
          │ Anchor resolve  │
          │  • calendar_pin → 2-pass extractor on phrase
          │  • anaphoric_event → top-1 doc by phrase embedding
          │  • others → no-op (fuse via rerank)
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Hybrid pool     │   top-K/2 raw-semantic ∪ top-K/2 filter-pass
          │                 │   (with top-up to fill K)
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Cross-encoder   │
          │ rerank          │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ score = norm(rerank) × DNF_mask × (1 + α·recency)
          └─────────────────┘
```

## Files

The directory packages the v5.1 production stack verbatim plus a clean
orchestrator:

| File | Role | Source |
|---|---|---|
| `retriever.py` | `TemporalRetriever` orchestration | new |
| `core.py` | `Interval`, mask / filter / pool / recency / norm primitives | consolidated from `composition_eval_v3.py`, `_v3_q1_retrieval_ablation.py`, `_v3_q10_hybrid.py`, `scorer.py`, `negation.py`, `salience_eval.py` |
| `__init__.py` | Public API exports | new |
| `planner.py` | DNF planner (prompt v4.0) + `Constraint` + `evaluate_dnf_mask` | merged from `query_planner_v2.py` + `query_planner_v4.py` (v4.1 prompt registry dropped) |
| `classifier.py` | LLM phrase classifier (prompt v5.1) | verbatim from `phrase_classifier.py` |
| `extractor.py` | Two-pass LLM extractor + cache + post-process | merged from `extractor_v2.py` + `extractor_common.py` (REGEX_PATTERNS pre-pass dropped) |
| `resolver.py` | TimeExpression validator + auto-correct | verbatim |
| `expander.py` | Recurrence expansion | verbatim |
| `schema.py` | TimeExpression dataclass + ISO helpers | verbatim (with `parse_iso`/`to_us` overloads added) |
| `research/` | Ablation / optimization / smoke-test harnesses (not public API). See below. | new |

### Research harnesses

`research/` holds scripts that exercise the production stack for
ablation, prompt optimization, and validation. None are part of the
public API; they share helpers via `research/_common.py` (proxy/dotenv
setup, `make_embed_fn` / `make_rerank_fn`, bench loading, summary
printers, the standard 6-variant prompt-component matrix).

| Module | Role |
|---|---|
| `research/_smoke_e2e.py` | End-to-end test against ambiguous_year bench |
| `research/_ablation_proper.py` | Prompt-component ablation on saturated standard benches (era_refs, multi_te_doc, conjunctive_temporal, composition) |
| `research/_ablation_hard.py` | Prompt-component ablation on harder benches (composition, adversarial, mixed_cue, realq_v2) |
| `research/_sensitivity_curated_bench.py` | Build a per-query sensitivity-curated bench from `_ablation_hard` outputs |
| `research/_prompt_optimizer.py` | Hill-climbing prompt optimizer (gpt-5-mini as both subject and proposer) |
| `research/_validate_best_prompt.py` | Validate the optimizer's best prompt against the full bench suite |

Prompts and schemas in the merged files are byte-for-byte identical to
the production v5.1 versions. The merge dropped only dead code: the
unused v4.1 planner prompt + its substitution machinery, and the
`REGEX_PATTERNS` pre-pass that the v5.1 pipeline never invoked.

Tooling: `uv run ruff check`, `uv run ruff format --check`, and
`uv run ty check temporal_retrieval/` all pass cleanly.

## Usage

```python
import asyncio
import numpy as np
from temporal_retrieval import TemporalRetriever, Doc

# 1. Bring your own embedding and reranker
async def embed_fn(texts: list[str]) -> list[np.ndarray]:
    # OpenAI text-embedding-3-small is what we benchmarked.
    ...

async def rerank_fn(query: str, doc_texts: list[str]) -> list[float]:
    # ms-marco-MiniLM-L-6-v2 is what we benchmarked.
    ...

# 2. Construct
retriever = TemporalRetriever(
    embed_fn=embed_fn,
    rerank_fn=rerank_fn,
    pool_size=10,         # K for hybrid pool. Bigger helps multi-cue queries.
    recency_alpha=3.0,    # Recency boost: final = base × (1 + α·rec)
    confidence_floor=0.5, # Drop extracted intervals below this confidence
)

# 3. Index your corpus once
docs = [
    Doc(id="d1", text="Held the kickoff for the migration project.",
        ref_time="2023-08-14T10:00:00Z"),
    ...
]
await retriever.index(docs)

# 4. Query
results = await retriever.query(
    "What happened after the kickoff?",
    ref_time="2025-06-15T00:00:00Z",
    k=10,
)
for r in results:
    print(f"{r.doc_id}  score={r.score:.3f}  rerank={r.rerank:.3f} "
          f"mask={r.mask:.2f}  recency={r.recency:.2f}")

print(retriever.stats())   # cache hit rates, parse failures, model usage
```

## Integration patterns

### As a re-ranking layer
If your RAG already has its own embedding-based retrieval, slot
`TemporalRetriever.query()` as a final step that re-orders the candidate
set with temporal awareness. Either pass your existing top-N to it as
the indexed corpus per query, or index your full corpus once and let the
hybrid pool do the work.

### As a filter / scoring layer only
`evaluate_dnf_mask()` and `doc_passes_filter()` are exposed as standalone
functions. Pre-extract intervals once per doc with
`extractor.TemporalExtractor`, plan a query with `planner.QueryPlanner`,
classify each leaf with `classifier.PhraseClassifier`, and apply mask +
filter directly to your existing candidate set.

## Classifier kinds

| Kind                | Behavior                                 | Examples                          |
|---------------------|------------------------------------------|-----------------------------------|
| `calendar_pin`      | Hard mask on extracted intervals         | "March 2024", "Q4 2023", "yesterday", "two weeks ago" |
| `recurring_period`  | No mask (fuse via rerank)                | "March", "summer", "Q1", "Lunar New Year", "Ramadan" |
| `anaphoric_event`   | Mask on top-1 doc's intervals            | "the launch", "the migration", "the offsite" |
| `personal_era`      | No mask                                  | "grad school", "during the pandemic", "back in college" |
| `generic_skip`      | No mask                                  | bare entity names, topical references without scoping |

`recurring_period` is the key insight for ambiguity handling — when the
user says "in March" without a year, the system doesn't pick one March.
It surfaces candidates from any year and lets rerank choose.

## Caching

LLM stages own their own caches under `temporal_retrieval/cache/`:
- `planner/` — DNF plans by (model, version, query, ref_time)
- `phrase_classifier/` — kinds by (query, ref_time, phrase, direction)
- `extractor_v2/` — pass-1 ref lists keyed by (model, system_hash, user)
- `extractor_shared_pass2/` — pass-2 resolutions (shared across pass-1 versions)

Each cache key includes the prompt version so prompt changes invalidate cleanly.

To share caches with the production research codebase, copy or symlink the
caches under `temporal_extraction/cache/` into `temporal_retrieval/cache/`.

## Costs (v5.1, gpt-5-mini)

- Per-doc index cost: 1 LLM call (pass-1 extractor) + N pass-2 calls
  (one per detected surface, typically N=1) + 1 embedding
- Per-query cost: 1 planner call + N classifier calls (one per leaf,
  typically 1-2) + N pass-1+pass-2 calls for `calendar_pin` leaves
  + 1 query embedding + reranker on K=10 docs

In steady state with cache populated, 0 LLM calls per query → cost is
just embedding + rerank + cosine sweeps.

## Verified ablations

`research/_ablation_proper.py` runs the pipeline end-to-end with parts of
the pass-1 extractor prompt removed. The earlier (saturated) variant ran
on `ambiguous_year` + `relative_time`: all five variants (baseline, drop
few-shot examples, drop trigger gazetteer, drop verbose ref-context, drop
all three) **tied baseline** on both benches:

| variant         | ambiguous_year R@1 / R@5 | relative_time R@1 / R@5 |
|-----------------|--------------------------|-------------------------|
| baseline        | 0.917 / 1.000            | 1.000 / 1.000           |
| no_few_shot     | 0.917 / 1.000            | 1.000 / 1.000           |
| no_gazetteer    | 0.917 / 1.000            | 1.000 / 1.000           |
| no_ref_context  | 0.917 / 1.000            | 1.000 / 1.000           |
| no_all          | 0.917 / 1.000            | 1.000 / 1.000           |

For typical corpora — single calendar-concrete or deictic phrase per doc
— the ~1300-token pass-1 prompt enrichment is dispensable; gpt-5-mini
handles "yesterday", "two months ago", "March 14, 2024" correctly from
a bare ISO ref_time without scaffolding.

**Caveats**: 12 queries each, 1 phrase per doc. Denser passages and
multi-cue queries (hard_bench, composition) might surface different
load-bearing claims. The gazetteer + few-shots are kept in the
production default until verified on those harder benches.

## Known bottlenecks

Documented from the v5.1 development:

1. **Multi-cue queries (composition R@5 = 0.640)** — when a query has
   2+ leaves AND the topical signal is weak ("What did I do after 2020
   but not in 2023?"), pure cosine ranks gold poorly (sometimes bottom
   of corpus). The filter correctly accepts gold but the K=10 hybrid
   pool can't surface it. **Fix candidates**: plan-conditional
   `pool_size` (K=20-50 when ≥2 leaves), or full filter-survivor
   channel without semantic-top-K capping.

2. **Anaphoric cross-cluster contamination** — top-1 corpus anchor for
   "the launch" can match a doc in a different cluster when phrase
   similarity is weak. **Fix candidate**: top-K corpus anchor with
   score-weighted intervals, or restrict anchor lookup to docs whose
   TEs overlap a co-occurring `calendar_pin`'s window. The
   `anaphoric_topk_in_pool` parameter is reserved for this; not yet
   wired.

3. **Era references can't be grounded without user-specific knowledge**
   — "back when I worked at Acme" is correctly classified as
   `personal_era` and skipped from masking, so retrieval falls back to
   pure rerank. R@5 = 1.000 on the era_refs bench (gold always in top
   5), but R@1 = 0.250 — rerank can't disambiguate without knowing
   when "Acme" actually was.

4. **Set-valued queries are graded as single-gold** — composition's
   R@5 = 0.640 is partly an artifact of grading: queries like "What
   did I do after 2020 but not in 2023?" describe a SET of valid
   answers but the bench picks one as gold. A multi-gold re-grade
   would likely lift composition closer to its real performance. See
   discussion in the parent `temporal_extraction/` research notes.

## Provenance

Distilled from the temporal_extraction research line at
`evaluation/temporal_extraction/`. The reference here imports the same
production modules (with relative imports rewired) and reproduces v5.1's
metrics exactly on the ambiguous_year bench — see `_smoke_e2e.py`.

Memory: `~/.claude/projects/.../memory/project_v5_phrase_classifier.md`,
`project_ambiguous_year_bench.md`.
