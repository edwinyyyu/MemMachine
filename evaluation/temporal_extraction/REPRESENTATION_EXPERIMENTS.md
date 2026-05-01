# Representation Experiments

Distinct from scoring (ablation) and extraction (improvements). These test
structural changes to the temporal index.

## R1 — Utterance-time anchor (primary)

**Hypothesis**: Indexing each doc's `ref_time` as a first-class interval,
separate from any extracted expressions, strictly improves recall for
queries that ask "when was this said?" even when the doc's content is
about a different time.

**Key case**: doc written at T, content says "back in the 90s". Query
at now asks "what happened 2 years ago?" — if T ≈ now − 2y, the utterance
anchor matches the query even though the referent (the 90s) does not.

**Method**:
- Extend `intervals` schema (or add a parallel `utterance_anchors` table) with
  one row per doc: `(doc_id, earliest_us, latest_us, best_us, granularity)`
  where the row represents the doc's own creation time.
- Granularity of the anchor depends on doc type:
  - Timestamped log message: second
  - Chat turn: minute
  - Diary entry: day
  - Unclear source: day (default)
- Retrieval: OR-query against both expression-intervals and utterance-anchors,
  union candidate doc_ids, merge scores.

**Discriminator queries to add**: 10 queries that test utterance-by-content
divergence. Example:
- Doc created 2024-04-23: "Back in 1995 my dad taught me to fish."
- Query at 2026-04-23: "What did I write about 2 years ago?"
- Should retrieve via utterance anchor, NOT via referent.

**Expected**: major lift on utterance-shape queries, parity elsewhere.

## R2 — Dual-score (anchor + referent) aggregation

**Hypothesis**: How to combine utterance-anchor score with referent scores?
- Sum: "doc is relevant if ANY evidence is in window" — floods on long docs
- Max: "best-matching interval" — favors single strong match
- Weighted: α · best_referent_score + β · anchor_score
- Hybrid by query-intent: if query is about "when was this said", weight
  anchor; if about "what happened on X", weight referent.

**Method**: test all four combinations, measure on discriminator set + full
corpus. Ablate α/β.

## R3 — Relative-offset representation (hybrid absolute + relative)

**Hypothesis**: Store each referent as BOTH absolute-resolved and
(reference_anchor, offset_duration, offset_direction). When the reference
changes (e.g., re-parsing with a different ref_time, or relative-to-event
chains), we can re-resolve without re-running the LLM.

**Method**:
- Extend `FuzzyInstant` to optionally carry `(anchor, offset)` for
  expressions whose original form was relative ("N weeks ago", "last X",
  "before Y").
- Pre-compute absolute form at extraction (as now); keep relative form for
  rewrites.
- For THIS eval, primarily a robustness check — does maintaining both forms
  create inconsistency? Test by perturbing ref_time and re-resolving.

**Expected**: No lift on current corpus; useful for future multi-hop queries
("what did I discuss the week before my birthday?").

## R4 — Doc-level temporal density (continuous representation)

**Hypothesis**: Instead of indexing intervals, compute a per-doc density
function d(t) = Σᵢ Kernelᵢ(t − μᵢ, σᵢ) over all the doc's time-refs.
Query density = similar function. Score = ∫ d_query(t) · d_doc(t) dt.

Mathematically this is a different aggregation pattern than the per-pair
Gaussian we already tested. Here density is computed PER DOC, normalized,
and compared as a whole.

**Method**:
- For each doc, compute density on a discretized time grid (e.g., 1 bucket
  per day over ±20y).
- Query similarly.
- Compute dot product or overlap coefficient.

**Expected**: Smoother ranking than per-pair; might help when docs have
multiple related times clustered in a region (matches query centered on that
region even if no single pair is a strong match). Possible regression due to
normalization flattening strong signals.

## R5 — Two-tier (granularity-bucketed) index for efficiency

**Hypothesis**: Pre-partition intervals by granularity. Query at year-grain
searches only year-grain and coarser buckets.

**Method**: N separate SQLite tables, one per granularity. Query matches
up-to-coarser-or-equal to query's own granularity.

**Expected**: efficiency lift (less scanning on large corpora), minimal
ranking change. Deprioritize — we're not at scale yet.

## Priorities

Run **R1 and R2** — direct fix for the utterance/referent divergence.
Skip R3 (no current payoff), R4 (similar to Gaussian ablation already done),
R5 (efficiency not scale-bound yet).

## Success criteria

R1+R2 ship-best configuration beats the current base hybrid on:
- New utterance-divergence discriminator subset (target +10pp R@5)
- Full corpus (must not regress >1pp on overall R@5)

If R1 alone hits both, skip R2. If R1 misses the overall criterion, R2's
weighting is needed to keep non-utterance queries from being diluted.
