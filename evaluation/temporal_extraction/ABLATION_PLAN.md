# Scoring + Bracket-Width Ablations

Follow-up empirical study after the base DESIGN.md implementation lands. These
ablations test four questions the user raised — all unverified conjectures
until we run them against synthetic ground truth.

## Hypotheses

### H1 — Bracket width for counted-relative expressions

Natural language "N weeks ago" is semantically fuzzy. How fuzzy should we
bracket it for retrieval?

- **A — narrow** (original table): "2 weeks ago" = [ref−15d, ref−13d], width ~2 days
- **B — ±25%** (DESIGN.md current): "2 weeks ago" = [ref−21d, ref−7d], width ~14 days
- **C — ±50%** (widest): "2 weeks ago" = [ref−28d, ref−0d], width ~28 days
- **D — full-surrounding-unit**: "2 weeks ago" = the entire week containing
  ref−14d, i.e. [start-of-week(ref−14d), end-of-week(ref−14d)]

Prediction: B or D wins on retrieval-by-relevance when queries are also fuzzy
relative expressions. C may over-retrieve and hurt precision. A is right only
when precision anchors are present.

### H2 — Scoring function

Given a query interval Q and a stored interval S (both FuzzyInstants with
earliest/latest/best):

- **Jaccard-composite** (current DESIGN.md):
  `0.5·Jaccard(Q,S) + 0.3·proximity(Q.best, S.best) + 0.2·granularity_compat`
- **Gaussian overlap (closed form)**:
  Model each as `N(best, σ²)` with σ = (latest − earliest) / 4.
  Score = `exp(−(Q.best − S.best)² / (2(σ_q² + σ_s²)))`.
  Smooth, no hard edges, naturally combines two uncertainties.
- **Gaussian integrated overlap** (normalized):
  `∫ N(t; Q) · N(t; S) dt / max` — same form as above up to constant;
  equivalent under normalization. Test both unnormalized and normalized.
- **Hard overlap indicator**: 1 if overlap, 0 if not. Baseline only.

Prediction: Gaussian outperforms Jaccard-composite when centers matter more
than bracket endpoints. Critical for "best" point alignment.

### H3 — Recurrences as convolution of spikes with a Gaussian

Current design expands a recurrence into discrete instance intervals, each
indexed independently. User's suggestion: represent the recurrence as a sum
of Gaussians (one per instance) and query-score it as an integrated density.

Mathematically:
- Instance train: `R(t) = Σ_i δ(t − μ_i)`
- Convolved with Gaussian kernel of width σ_instance: `R(t) = Σ_i N(t; μ_i, σ_instance²)`
- Query Gaussian: `Q(t) = N(t; μ_q, σ_q²)`
- Score: `∫ Q(t) · R(t) dt = Σ_i exp(−(μ_q − μ_i)² / (2(σ_q² + σ_i²)))`

Under expand-to-instances + sum-aggregation this is exactly what you get.
Under expand-to-instances + max-aggregation, you get the nearest-instance
only. Test both.

### H4 — Aggregation across (query-expr × stored-expr) pairs

When a query has multiple temporal expressions and a doc has multiple:

- **Sum**: total score = sum of all pair scores (current — favors docs with
  many near-matches)
- **Max**: total = best single pair (favors docs with one very close match)
- **Sum of top-K**: favors docs with a few strong matches, ignores noise
- **Log-sum**: compromise — rewards multiple matches but with diminishing
  returns

Prediction: depends on query shape. Sum helps when both sides densely
populated; max helps sparse queries. Sum-top-3 is my guess for most robust.

### H5 — Multiple-match bonus vs saturation

Related to H4. Does a doc with 3 overlapping intervals beat one with 1
near-perfect interval? At what point does adding more matches saturate?

## Experimental design

Same 30 docs + 60 queries as the base eval, with these additions:

1. **Add 10 "wide vs narrow" discriminator queries** where the correct answer
   requires wide brackets — e.g., doc says "3 weeks ago we..." and query says
   "any time last month...". Under narrow (H1-A) these miss; under wide
   (H1-C,D) they hit.

2. **Add 10 "center matters more than overlap" queries** where two docs
   overlap the query bracket equally but differ in center. Under Jaccard
   (H2-A) these tie; under Gaussian (H2-B) they separate.

3. **Add 10 "recurrence density" queries** where recurring events should
   rank relative to one-time events. Does sum-aggregation over-rank
   recurrences? Does top-K aggregation calibrate?

## Metrics

Run every combination of {H1 ∈ A,B,C,D} × {H2 ∈ Jaccard, Gaussian} × {H4 ∈
Sum, Max, TopK, Log-sum} = 32 configurations. For each:

- Recall@5, Recall@10, MRR, NDCG@10 on the full 60+30 query set
- Subset metrics on the discriminator sets (H1, H2, H3 subsets)
- Critical-case accuracy on the "2 weeks ago ↔ 2 weeks from now" pairs
  (should hold across all reasonable settings)

## Decision gate

Ship the best combination, documented with numbers. If no single configuration
dominates, default to the combination that wins the most subset metrics, or
the simpler one that ties.

## Implementation

Reuse the cached extractions from the base eval — don't re-extract. Modify
only `resolver.py` (bracket-width variants) and `scorer.py` (score function
and aggregation variants). Add `ablation.py` orchestrator.

Output `results/ABLATION_RESULTS.md` with a matrix of 32 rows × 4 metrics.
