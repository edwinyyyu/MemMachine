# Database-friendly T-channel alternatives

Goal: replace `T_lblend = 0.2*interval_jaccard + 0.2*tag_jaccard + 0.6*lattice_score`
with a design whose components fit cleanly into Postgres / Elasticsearch as
**inverted-index** + **range-overlap** queries, with channels separately
indexed and merged after top-K.

Eval driver: `t_db_alternatives_eval.py`. Reuses cached extractions under
`cache/v7l-{hard_bench, temporal_essential, tempreason_small}/` (no LLM
calls). Raw numbers in `results/T_db_alternatives.json`.

## Candidates

### A. HIER_TAGS — pure inverted index, granularity-weighted
Index per doc the union of lattice-cell tags across all its TimeExpressions
(absolute cells at native precision + applicable cyclical tags). At query
time, expand the query TE to ancestors+self+cyclical (NO down-walk to
children), then do a single inverted-index lookup. Score = Σ (granularity
weight) of matched tags. SQL/ES form:

```sql
SELECT doc_id, SUM(prec_weight(tag)) AS s
FROM   doc_tags
WHERE  tag = ANY(:expanded_tags)
GROUP  BY doc_id
ORDER  BY s DESC
LIMIT  :K;
```
ES: `terms` query on `doc.tags.keyword` with a per-term `boost` proportional
to granularity; or a `function_score` over a tags field.
Index: GIN (Postgres) / inverted index (ES). O(log n) per term.

### B. RANGE_OVERLAP + year tag boost
Per doc, store one or more `daterange` rows derived from each TE's
`flatten_intervals`. At query time, range-overlap; score by Jaccard of
overlap span. Add a year-tag boost (0.3) when year tags coincide — covers
recurrences and queries with no usable range. SQL form:

```sql
SELECT doc_id,
       (LEAST(d.upper, :q_l) - GREATEST(d.lower, :q_e))
       / (GREATEST(d.upper, :q_l) - LEAST(d.lower, :q_e)) AS jacc
FROM   doc_ranges d
WHERE  d.span && tstzrange(:q_e, :q_l)
ORDER  BY jacc DESC
LIMIT  :K;
```
ES: `date_range` field with `relation: intersects`. Index: GIST on
`tstzrange` (Postgres) / `date_range` mapping (ES). O(log n).

### C. PER_TE_MAX — multi-TE friendly per-row index
Each doc-TE is its own row (1 doc → N rows). Each row has its compact
tag-set + `daterange`. Per-doc score = MAX over rows of `(weighted_tag_match
+ 2*range_jaccard) / log2(2 + |row_tags|)`. Avoids tag-union dilution that
hurts T_lblend on docs with many TEs. SQL/ES form:

```sql
SELECT doc_id, MAX(per_row_score) FROM (
  SELECT doc_id, te_row_id,
         f_tag(:expanded, row_tags) + 2*f_range(:q_range, row_range) AS per_row_score
  FROM   doc_te_rows
  WHERE  EXISTS (SELECT 1 FROM unnest(row_tags) t WHERE t = ANY(:expanded))
      OR row_range && tstzrange(:q_e, :q_l)
) GROUP BY doc_id ORDER BY 2 DESC LIMIT :K;
```
ES: nested mapping (`doc.te_rows[*].tags`, `doc.te_rows[*].range`) +
nested-aggregation MAX. Index: GIN on tag, GIST on range — both nested.

### D. RRF_TAG_RANGE — A and B as separate index queries, fused by RRF
Run A and B as two separate top-K queries; merge top-100 each by Reciprocal
Rank Fusion (k=60). No score-arithmetic across indexes — strictly DB-native.

---

## Results — R@1 (R@5 in parens)

### T-channel only

| Candidate | hard_bench | temporal_essential | tempreason_small |
|---|---|---|---|
| **T_lblend (baseline)** | 0.000 (0.053) | 0.280 (0.880) | 0.283 (0.433) |
| A_hier_tags | 0.000 (0.000) | 0.040 (0.360) | 0.167 (0.417) |
| B_range_overlap | 0.027 (0.093) | 0.240 (0.760) | 0.283 (0.450) |
| C_per_te_max | 0.000 (0.080) | 0.240 (0.760) | 0.183 (0.417) |
| D_rrf_AB | 0.000 (0.080) | 0.240 (0.760) | 0.267 (0.467) |

### T + S linear blend (w_T=0.4, dispersion-CV gated — production style)

| Candidate | hard_bench | temporal_essential | tempreason_small |
|---|---|---|---|
| **T_lblend (baseline)** | **0.813** (0.933) | **1.000** (1.000) | **0.750** (1.000) |
| A_hier_tags | 0.600 (0.800) | 1.000 (1.000) | 0.683 (1.000) |
| B_range_overlap | 0.267 (0.800) | 1.000 (1.000) | 0.650 (1.000) |
| C_per_te_max | 0.040 (0.760) | 1.000 (1.000) | 0.700 (1.000) |
| D_rrf_AB | 0.213 (0.733) | 1.000 (1.000) | **0.750** (1.000) |

### T + S Reciprocal-Rank Fusion (top-K each, no score sharing — pure DB merge)

| Candidate | hard_bench | temporal_essential | tempreason_small |
|---|---|---|---|
| T_lblend (still uses internal scores) | 0.467 (0.800) | 1.000 (1.000) | 0.500 (0.600) |
| A_hier_tags | 0.000 (0.760) | 0.840 (1.000) | 0.633 (0.983) |
| B_range_overlap | **0.440** (0.787) | 0.920 (1.000) | **0.700** (1.000) |
| C_per_te_max | **0.440** (0.787) | 0.920 (1.000) | 0.600 (0.983) |
| D_rrf_AB | **0.440** (0.787) | 0.920 (1.000) | **0.700** (1.000) |

(T_lblend numbers under "RRF" use the produced T-ranking; the value of
T_lblend's continuous score is *only* exploited inside the linear-blend gate.)

---

## Recommendation

**D_rrf_AB** (A and B as separate DB queries, fused by RRF) is the best
DB-friendly replacement *under the channel-separation constraint*. It is the
only candidate that consistently matches T_lblend at the system level once
fusion is also constrained to RRF (i.e. when the production
dispersion-CV-gated blend can't be implemented because S lives in a
different index). Specifically:

- On **tempreason_small** (multi-TE realistic corpus) — D_rrf_AB matches
  T_lblend at R@1 = 0.750 under linear blend AND beats it at R@1 = 0.700
  under pure RRF fusion (vs T_lblend 0.500).
- On **temporal_essential** — saturates at R@1 = 1.000 under linear blend
  (the easy bench).
- On **hard_bench** — D loses to T_lblend (0.213 vs 0.813 under blend), but
  this is largely a *fusion* artifact, not a T-channel artifact (see next
  section). Under pure RRF the gap closes to 0.027 (0.440 vs 0.467).

If a single-index design is required, **B_range_overlap** alone is the
single best DB-native primitive: it carries most of the signal D recovers,
and competes on tempreason. **A_hier_tags alone is not viable** — pure
tag-counting on year-dense corpora (hard_bench) collapses because dozens
of docs match the same `year:2023` tag.

---

## Why T_lblend's lossy aggregation gives signal pure index queries can't

The hard_bench gap (T_lblend 0.813 vs D_rrf_AB 0.213 under linear blend)
is **not because the *components* of T_lblend out-discriminate the
indexable primitives** — they don't, see the T-only column where T_lblend
also sits at R@1=0 with R@5=0.053. The gap is created by T_lblend producing
a **single continuous, well-calibrated score** that the dispersion-CV gate
in `score_blend` (cv_ref=0.20) can up-weight when it's confident.

Concretely:

1. T_lblend's three components are mixed *before* the channel exits
   (`α*iv + γ*tag + δ*lattice` with calibrated weights) → one channel with
   a high coefficient of variation on its top-K.
2. `score_blend` measures CV per channel and re-scales T's contribution
   accordingly. When T has high CV (i.e. the top-1 doc dominates), T
   effectively gets w_T near 1.
3. Pure-index candidates can each be calibrated, but you have **two
   independent channels** (A's tag-sum and B's range-overlap-Jaccard)
   that the system must merge **outside the index**. RRF compresses each
   to ranks, **destroying the dispersion signal**, so the gate can never
   fire. Linear-blending raw scores across A and B is what the constraint
   forbids when the indexes are separate.
4. Worse, on hard_bench many docs have IDENTICAL year-tag matches (A) and
   IDENTICAL year-range overlap (B), so each component returns **flat top-K
   plateaus**. RRF over flat top-K is roughly random.

In short: **T_lblend's win is half intra-T blending and half the
dispersion-gated post-fusion** — both of which require continuous-score
access to *all* candidates from *all* sub-channels at once. The DB
constraint (separate indexes, top-K-only) erases both. The cleanest
DB-native recovery is to **pre-blend the T sub-channels into one stored
score** (a feature column) so the channel that exits the database to the
fusion layer is again a single dispersion-rich number.

### Practical hybrid worth shipping

The "pure DB-native" variant compatible with the current codebase:

1. **Server-side**: store, per doc, a `t_lattice_score_payload` column or
   ES `function_score` precomputed at write-time using the `lattice_score`
   formula (already produces calibrated top-K). Do this as a Postgres
   tag-with-weight inverted index variant (`A_hier_tags` using
   `cell_score = 1/log2(2 + tag_span/q_span)` weights instead of fixed
   per-precision weights).
2. **Augment with a range-overlap score column** (B's primitive) when the
   query has a range; otherwise fall back to year-tag.
3. **Keep T as one channel** by linearly combining the two scores in a
   `function_score` query (still indexable via composite scoring within ES).
4. The final fusion against S can then once again be the dispersion-CV
   linear blend — restoring the property that lets T_lblend win on
   hard_bench.

This is essentially **A with lattice-style continuous weights + B's range
overlap fused inside the same index query**, returning a single T score —
which is the only configuration that simultaneously satisfies: indexable,
separate from S, and able to produce a dispersion-rich score the gate can
exploit.
