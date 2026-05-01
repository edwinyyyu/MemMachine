# T_recency — Recency-decay scoring channel

Recency cue regex (`latest`, `most recent`, `last`, `recently`, `newly`, `just`, `current(ly)`, `present`, `now`) gates an exponential-decay channel:

```
recency_score(d, q) = exp(-lambda * |q.ref_time - d.anchor| / 1 day)
```

Doc anchor = MAX over all extracted-TE `best_us`; falls back to doc `ref_time` if no TE. Cue gate is regex-based with verb-form `present` / `presented` / `presenting` suppressed and explicit-date phrases (`last week`, `last Monday`, `current 2024`) suppressed.

## Headline — `latest_recent` R@1 (n=15, cued=15/15)

| Recipe | R@1 | hits |
|---|---:|---:|
| rerank_only (baseline) | 0.133 | 2/15 |
| T_lblend | 0.000 | 0/15 |
| T_v4 | 0.000 | 0/15 |
| recency_only (full corpus, topic-blind) | 0.000 | 0/15 |
| T_lblend + recency replace | 0.000 | 0/15 |
| T_v4 + recency replace | 0.000 | 0/15 |
| **rerank * recency (mult, h=90d)** | **0.867** | **13/15** |
| **rerank + recency (additive α=0.5, h=90d)** | **0.867** | **13/15** |
| rerank > recency (replace within rerank top-50) | 0.000 | 0/15 |

Δ vs rerank_only baseline = **+0.733**.

## R@1 across all benches (half-life=90d)

| Benchmark | n | cued | rerank_only | T_lblend | T_v4 | rer*rec | rer+rec add | rer>rec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| latest_recent | 15 | 15 | 0.133 | 0.000 | 0.000 | **0.867** | **0.867** | 0.000 |
| hard_bench | 75 | 0 | 0.640 | 0.000 | 0.027 | 0.640 | 0.640 | 0.640 |
| temporal_essential | 25 | 0 | 0.920 | 0.280 | 0.240 | 0.920 | 0.920 | 0.920 |
| tempreason_small | 60 | 0 | 0.650 | 0.283 | 0.300 | 0.650 | 0.650 | 0.650 |
| conjunctive_temporal | 12 | 0 | 1.000 | 0.917 | 0.667 | 1.000 | 1.000 | 1.000 |
| multi_te_doc | 12 | 0 | 1.000 | 0.750 | 0.667 | 1.000 | 1.000 | 1.000 |
| era_refs | 12 | 0 | 0.250 | 0.083 | 0.167 | 0.250 | 0.250 | 0.250 |
| open_ended_date | 15 | 0 | 0.267 | 0.000 | 0.067 | 0.267 | 0.267 | 0.267 |
| causal_relative | 15 | 1 | 0.467 | 0.000 | 0.000 | 0.400 | **0.467** | 0.400 |
| negation_temporal | 15 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Half-life sweep — `latest_recent`

| half-life | rer*rec | rer+rec add | rer>rec |
|---:|---:|---:|---:|
| 7d  | 0.467 | 0.667 | 0.000 |
| 21d | 0.667 | 0.800 | 0.000 |
| 90d | 0.867 | 0.867 | 0.000 |

Half-life sweep on T-only / lblend-only recipes was uniformly 0.000 because T_v4 returns 0 for every doc when the query has no temporal extraction (the `latest_recent` regime). Adding recency to a dead T channel cannot recover topic information; only multiplying or adding recency to the cross-encoder rerank score works.

## Findings

### 1. Did recency module fix `latest_recent`?

**Yes.** R@1 went from 0.133 (rerank_only baseline) to 0.867 with `rerank + recency` (α=0.5 additive, half-life=90d) — Δ = **+0.733** (11 additional queries answered correctly out of 15).

T_lblend and T_v4 alone score 0/15 because the queries have no extracted query interval, so the overlap-based primitive returns 0 for every candidate. Pure recency over the full corpus is also 0/15 because it is topic-blind — among 75 docs the closest-by-date doc is rarely in the relevant cluster. The lift requires combining recency with the cross-encoder's topic signal.

### 2. Did recency hurt any other benchmark?

**`rer + rec add` (additive, α=0.5): zero regressions** across 9 non-recency benchmarks.

**`rer * rec` (multiplicative gate): one regression** — `causal_relative` drops 0.467 → 0.400 (-0.067). The single cued query in this bench (*"What did Maya report since the last review?"*) has multiple candidate docs at similar dates, so multiplying by recency suppresses the topically-correct doc and elevates a slightly-newer but topically-wrong one.

**`rer > recency replace` (replace within rerank top-50): one regression** — same `causal_relative` -0.067.

The regression-check column (`cued` count) confirms the cue gate worked: 8/9 non-`latest_recent` benches have cued=0, so the recency channel never activated. Hard_bench had a 5/75 false-positive on the verb form `presented` in the first run; the fix (verb-form `present` suppression in the gate) brought hard_bench cued count to 0/75 with zero regression.

### 3. Half-life tuning

**Half-life = 90 days won.** The `latest_recent` cluster has 5 docs spaced roughly monthly across a 5-month window, so a 7d or 21d half-life decays too aggressively — the gold (closest doc) only modestly outweighs the second-closest, and the rerank-score noise dominates. With 90d half-life the decay curve has enough slope at month-scale spacing to consistently rank the most recent doc first while still distinguishing within-cluster docs.

Half-life sweep for `rer + rec add` on `latest_recent`: 7d → 0.667, 21d → 0.800, 90d → 0.867. None of the half-lives caused a regression on any non-`latest_recent` bench (additive recipe), so 90d is the safe default.

### 4. Recommendation

**Ship recency as a separate channel added to the cross-encoder rerank score, additive α=0.5, half-life=90d.** This is the only recipe with zero regressions across the test bench while delivering the full +0.733 lift on `latest_recent`.

```python
if has_recency_cue(query_text):
    norm_rerank = (rerank_score - r_min) / (r_max - r_min)
    final_score = 0.5 * norm_rerank + 0.5 * exp(-ln(2)/90 * days_diff)
else:
    final_score = rerank_score  # unchanged
```

Why additive over multiplicative:

- **Multiplicative gate** zeroes out high-rerank docs whose dates are slightly off, flipping otherwise-correct rankings (causal_relative -0.067).
- **Additive blend** preserves the rerank's topic verdict when both signals agree and shifts ranking only when recency genuinely matters; **zero regressions** across 9 non-recency benchmarks at h=90d.
- Both recipes hit the same ceiling (13/15) on `latest_recent` at h=90d; additive matches it without the regression risk.

The "replace recency only when T is dead" strategy from the original task description does not work at the corpus level (recency_only / lb+rec replace / v4+rec replace all 0/15) because pure-recency is topic-blind. The right place to gate is at the **rerank level** — the rerank already filters to topic-similar candidates, so adding recency on top adds the missing temporal signal.

### 5. Limitations / what this does NOT handle

- **Anti-decay queries** (`earliest X`, `first time I X`, `originally`, `started`): need an *inverse* decay (small Δ is bad, large Δ is good). The fix is a parallel `anti_recency_score = exp(-λ * (Δ_max - Δ_doc) / DAY)` keyed off a separate `earliest`/`first` cue list. Untested here.
- **Future-anchored queries** (`upcoming meeting`, `next appointment`): need decay around a *future* reference, not `ref_time`. Detect with a `next`/`upcoming` cue and treat negative Δ (future docs) as preferable.
- **Date-bearing recency phrases** (`the last meeting before March`): the suppress rule disables recency because the explicit date should be handled by T_v4. If T_v4 returns a tie among multiple cluster members, recency could still tiebreak — but the current code drops the recency channel entirely when an explicit date is present.
- **Ambiguous cue words**: `last` and `present` carry both recency and non-recency senses (`last week` vs `last appointment`, `the present situation` vs `presented at`). Verb-form `present` and date-phrase `last <unit>` are suppressed but a regex cannot catch every false positive — a small classifier would be more robust.
- **Cluster size dependency**: the 90d half-life is calibrated to the `latest_recent` cluster spacing (~30d). Tighter clusters (sub-week) or wider clusters (years) need different half-lives. This could be made adaptive by estimating typical Δ from the rerank candidate set and setting λ to give half-life = median(Δ).
- **Tail docs (no extracted TE)**: we fall back to `ref_time`, which the synthesizer sets uniformly. In production, docs without good time anchors will all decay the same — recency will only differentiate them by their `ref_time` field. Fine when ingestion stamps every doc but degrades on docs with stale or default timestamps.
- **Multi-cue conjunctions** (`latest project finished after Q1`): recency and overlap both contribute information; the additive blend (rerank + recency) handles this naturally because the rerank score retains topic signal while recency biases toward newer. The replace recipes drop one signal entirely, which loses precision on these queries.
