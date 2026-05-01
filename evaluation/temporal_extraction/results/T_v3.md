# T_v3 — Density-correlation temporal scoring

Per-pair primitive: `0.40 * iv_score + 0.50 * tags_score + 0.10 * containment` where
- `iv_score`: range overlap as ∩/∪ over the best (q_iv, d_iv) pair (continuous, normalized).
- `tags_score`: hierarchical-chain match. For each q absolute tag, walk its ancestor chain (inclusive); count weighted matches against the union of d's ancestor chains. Per-level weight = `1 / log2(2 + breadth_seconds_of_level)` (finer levels weighted more). Cyclical tags contribute via Jaccard at year-level weight.
- `containment`: open-ended fallback. Reward `d ⊆ q` (1.0 when q is meaningfully wider), weak `q ⊆ d` (0.3 when d is meaningfully wider).

Aggregation: per-anchor MAX over doc TEs, geomean across query anchors with floor 1e-6. (Same as T_v2.)

## R@1 table

| Benchmark | n | T_lblend R@1 | T_v2 R@1 | T_v3 R@1 | Δ(v3−v2) | Δ(v3−lblend) |
|---|---:|---:|---:|---:|---:|---:|
| conjunctive_temporal | 12 | 0.917 (11/12) | 0.917 (11/12) | 0.917 (11/12) | +0.000 | +0.000 |
| multi_te_doc | 12 | 0.750 (9/12) | **0.833 (10/12)** | 0.750 (9/12) | **−0.083** | +0.000 |
| relative_time | 12 | 0.417 (5/12) | 0.417 (5/12) | 0.417 (5/12) | +0.000 | +0.000 |
| era_refs | 12 | 0.083 (1/12) | 0.083 (1/12) | **0.167 (2/12)** | **+0.083** | **+0.083** |
| hard_bench | 75 | 0.000 (0/75) | 0.027 (2/75) | 0.027 (2/75) | +0.000 | +0.027 |
| temporal_essential | 25 | 0.280 (7/25) | **0.360 (9/25)** | 0.240 (6/25) | **−0.120** | −0.040 |
| tempreason_small | 60 | **0.283 (17/60)** | 0.250 (15/60) | 0.233 (14/60) | −0.017 | −0.050 |

## R@5 table

| Benchmark | n | T_lblend R@5 | T_v2 R@5 | T_v3 R@5 |
|---|---:|---:|---:|---:|
| conjunctive_temporal | 12 | 0.917 (11/12) | 1.000 (12/12) | 1.000 (12/12) |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| relative_time | 12 | 0.917 (11/12) | 0.917 (11/12) | 0.833 (10/12) |
| era_refs | 12 | 0.250 (3/12) | 0.250 (3/12) | 0.250 (3/12) |
| hard_bench | 75 | 0.053 (4/75) | 0.053 (4/75) | **0.120 (9/75)** |
| temporal_essential | 25 | 0.880 (22/25) | 0.880 (22/25) | 0.760 (19/25) |
| tempreason_small | 60 | 0.433 (26/60) | 0.383 (23/60) | 0.400 (24/60) |

## Headline verdict

**T_v3 fails its two main targets.** It did NOT fix the tempreason regression (Δ vs T_lblend = **−0.050**, worse than T_v2's −0.033) and it LOST T_v2's two confirmed wins on temporal_essential (Δ vs T_v2 = **−0.120**) and multi_te_doc (Δ = **−0.083**). It earns one principled win on era_refs (+0.083) and a secondary R@5 win on hard_bench (+0.067 R@5). Net: T_v2 remains the best per-anchor scorer; T_v3's hierarchical-chain primitive is over-rewarding ancestor-only matches and displacing exact-bracket gold.

## Per-failure diagnosis

### temporal_essential: −0.120 vs T_v2 (8 queries lost, 5 new wins)
Concrete failure mode (all 8 lost queries are `n_q_tes=1`, single anchor):
- te_q_003, te_q_005, te_q_011, te_q_012, te_q_015, te_q_016, te_q_020, te_q_022 — in every case T_v2 placed gold at rank 1; T_v3 places a non-gold doc at top with the actual gold at rank 3–7.
- Mechanism: the gold doc's TE has the EXACT same year/quarter as the query, but several non-gold docs share the SAME YEAR ANCESTOR. T_v3's `tags_score` chain-walk credits all of those equally at the year-level (weight ~0.10), and the gold's exact-quarter tag (weight ~0.13) only adds one small increment on top. T_v2's pure Jaccard rewarded only the exact tag overlap and so concentrated the score on the gold.
- The hierarchical primitive trades discrimination for partial credit. Cost > benefit when the bench has many same-year-different-month distractors (which temporal_essential does).

### multi_te_doc: −0.083 vs T_v2 (1 query lost: mte_q_006)
- Single-anchor query, gold doc lost rank 1 → 2; T_v3's top-1 is `mte_010_d3` (a peripheral multi-TE doc).
- Same mechanism: chain-walk credits coarse-ancestor matches in the peripheral doc's many TEs more than the focal-TE alignment in the gold doc. The 0.50 weight on `tags_score` (relative to T_v2's 0.40 on a stricter Jaccard) compounds the issue.

### tempreason_small: −0.050 vs T_lblend, −0.017 vs T_v2 (NOT FIXED)
- The hypothesis was that hierarchical containment would recover open-ended `<2010` matching. It does help SOME queries — tempreason_small's R@5 actually improves from T_v2's 23/60 → 24/60 — but at top-1 the partial-credit chain-walk generates ties / displacements at the same rate it solves, netting −0.017 vs T_v2.
- Root cause specific to tempreason: the queries are complex multi-fact questions ("Who was X's spouse before 2010?"). The `<2010` open-ended containment IS picked up correctly by T_v3's `containment` channel (weight 0.10), but the `tags_score` chain-walk simultaneously rewards every doc with an ancestor tag in the 2000s/decade, flattening the ranking among many candidates. T_lblend's lattice-score channel uses a discriminative ancestor-walk **at retrieval time** (lattice_retrieve_multi) which only fires on actual stored matches; T_v3 reproduces this poorly inside a per-pair scalar.
- **Mitigation**: lower `w_tags` to 0.30 and raise `w_iv` to 0.50; or weight the `tags_score` only on the FINEST matched level (winner-take-all per chain) instead of summing across the whole chain.

### era_refs: +0.083 (1 new win, principled)
- era_q_008: query targets a coarse era; gold doc has month-precision TE inside that era. T_v2's pure Jaccard scored 0 (no exact-tag match). T_v3's chain-walk credits the year/decade ancestor match.
- This is the win condition the design was built for; it works. But era_refs is small (12q) so the absolute movement is small.

### hard_bench: +0.067 R@5 (R@1 unchanged)
- T_v3 lifts R@5 from 4/75 to 9/75 by spreading partial credit through ancestor chains (more docs land in top-5 even though no extra docs land at top-1).
- Operationally consistent with the failure-mode analysis: T_v3 is "softer" than T_v2 at the top.

## Salience weighting

**Skipped.** TimeExpression (schema.py:111) does not expose a `role` / focality field — only `confidence`. Per-TE focal-vs-peripheral weighting would require:
- (a) a separate salience pass via `salience_extractor.SalienceExtractor` per-TE, OR
- (b) a heuristic from `te.confidence` (extractor confidence is not the same as focality but might correlate).

Out of scope for this round. Strong recommendation: tag the focal d_TE explicitly during extraction (one new field on TimeExpression) and downweight non-focal d_TEs by 0.5 in `t_v3_doc_scores` — this would directly target the multi_te_doc / temporal_essential displacement failures observed above (peripheral d_TEs are precisely the ones whose chain-walks displaced gold).

## Why the principled view didn't pan out

The "density correlation = inner product of measures" framing is mathematically clean, but the discrete approximation it suggests (sum-of-weights along the matching ancestor chain) maps onto a real failure mode of retrieval ranking: it converts an exact-match signal (which discriminates) into a soft partial-credit signal (which compresses ranks). When the bench has many same-ancestor distractors, the soft signal ties them to the gold instead of separating them.

T_v2's pure Jaccard was effectively a "winner-take-all at the finest level" rule. That's a feature, not a bug, for top-1 retrieval — it concentrates score on exact matches and discards the noise.

The era_refs win demonstrates the chain-walk has real value, but only when the gold's tag IS a strict ancestor/descendant of the query tag and exact match would score 0. That's a small slice of the corpus.

## Suggested follow-up experiments

1. **Winner-take-all chain match (most likely to recover the regressions)**: replace
   ```python
   for level_prec, level_tag in q_chain:
       if level_tag in d_chain_tags:
           chain_hit += _gran_weight(level_prec)
   ```
   with: take the FINEST matched level only:
   ```python
   for level_prec, level_tag in q_chain:  # finest first by construction
       if level_tag in d_chain_tags:
           chain_hit = _gran_weight(level_prec)
           break
   ```
   Predicted: recovers most of temporal_essential / multi_te_doc loss while keeping the era_refs win (still scores at the matched ancestor level).

2. **Weight sweep (w_iv, w_tags, w_cont)**: the 0.40/0.50/0.10 split is a guess. Run a 6-point simplex over `{0.2, 0.4, 0.6}` summing to 1.0 jointly across temporal_essential + multi_te_doc + tempreason_small.

3. **Asymmetric containment with polarity**: T_v3 currently rewards `d ⊆ q` symmetrically. If we can detect query polarity ("after 2010" vs "before 2010") from the extractor, only reward the matching side. Likely small lift but principled.

4. **Salience flag on extraction**: add a `focal: bool` field to TimeExpression (or a separate salience pass) and downweight non-focal d_TEs by 0.5 in the per-pair max. Target: multi_te_doc dilution (this is the same failure that motivated T_v2 originally).

5. **Hybrid v2+v3 with bench-shape gate**: T_v2 wins exact-match-rich benches (temporal_essential, multi_te_doc), T_v3 wins coarse-era benches (era_refs, hard_bench R@5). A per-query gate on `n_q_tes` + the dispersion of `tags_score` distribution across docs may pick the right scorer per query.

6. **Bigger era_refs bench**: 12 queries is too few to confirm the era_refs win. Resynth at 50+ to actually measure the effect size of the chain-walk's principled benefit.
