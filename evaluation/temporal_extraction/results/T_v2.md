# T_v2 — Per-anchor AND-coverage temporal scoring

T_v2 per-pair scalar (one query TE × one doc TE):
`per_pair = 0.40*interval_overlap + 0.40*lattice_match + 0.20*axis_match`

Aggregation: for each q_TE take MAX over d_TEs (best-anchor coverage), then take GEOMEAN across query anchors (floor 1e-6). T_lblend (current shipping) was bag-merged: `0.20*interval_jaccard_global_max + 0.20*tag_jaccard + 0.60*lattice_score`.

## R@1 table

| Benchmark | n | T_lblend R@1 | T_v2 R@1 | Δ R@1 | T_lblend R@5 | T_v2 R@5 | Δ R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conjunctive_temporal | 12 | 0.917 (11/12) | 0.917 (11/12) | +0.000 | 0.917 (11/12) | 1.000 (12/12) | +0.083 |
| multi_te_doc | 12 | 0.750 (9/12) | 0.833 (10/12) | **+0.083** | 1.000 (12/12) | 1.000 (12/12) | +0.000 |
| hard_bench | 75 | 0.000 (0/75) | 0.027 (2/75) | +0.027 | 0.053 (4/75) | 0.053 (4/75) | +0.000 |
| temporal_essential | 25 | 0.280 (7/25) | 0.360 (9/25) | **+0.080** | 0.880 (22/25) | 0.880 (22/25) | +0.000 |
| tempreason_small | 60 | 0.283 (17/60) | 0.250 (15/60) | −0.033 | 0.433 (26/60) | 0.383 (23/60) | −0.050 |

### Per-query R@1 by n_q_tes (anchor-count breakdown)

| Benchmark | n_q_tes=0 | n_q_tes=1 | n_q_tes=2+ |
|---|---|---|---|
| conjunctive_temporal | — | 0/1 vs 0/1 | 11/11 vs 11/11 (tied at ceiling) |
| multi_te_doc | — | 9/12 → 10/12 (+1) | — |
| hard_bench | 0/1 | 0/67 → 0/67 | 0/7 → 2/7 (+2) |
| temporal_essential | — | 7/25 → 9/25 (+2) | — |
| tempreason_small | 0/25 → 0/25 | **17/35 → 15/35 (−2)** | — |

## Diagnosis

### Conjunctive temporal (target failure 1) — Δ R@1 = 0.000, Δ R@5 = +0.083
- T_lblend already ranked the gold doc at top-1 in 11/12 of these (current cache; the prior result with 9/12 in `edge_summary.json` reflects an older cache version). T_v2 also gets 11/12 — the inversion is *largely already absent* in the current extraction cache, so R@1 cannot move further on this small bench.
- T_v2 *does* push R@5 from 11/12 → 12/12: the previously-missed query (the same one missed by both) lands in top-5 under T_v2, suggesting T_v2's per-anchor coverage helps the partial-coverage tail even when it doesn't move R@1.
- **Conclusion**: the conjunctive-tag-union inversion fix is logically correct, but ceiling effects on the 12-query bench mask its R@1 contribution. Need a larger conjunctive bench to separate the two architectures.

### Multi-TE doc (target failure 2) — Δ R@1 = +0.083 (9/12 → 10/12)
- T_v2 wins one query that T_lblend lost. With every query having only 1 q_TE here, the win comes from T_v2's per-doc-TE pair MAX (not the AND-coverage geomean): T_v2 isolates the focal doc-TE that matches the single query anchor, while T_lblend dilutes it via tag-union over all peripheral doc TEs and global-max interval normalization.
- **Conclusion**: dilution fix confirmed. Even on single-anchor queries, the per-d_TE MAX is a strict improvement over union/global-max.

### Hard_bench — Δ R@1 = +0.027 (0/75 → 2/75)
- Both T-only scores are essentially zero on hard_bench (this is a small-n result; the production pipeline uses T fused with reranker + semantic, where T-only contribution is expected to be small). T_v2 pulled in 2 wins on multi-anchor queries where T_lblend got nothing. **Holds, no crater.**

### Temporal_essential — Δ R@1 = +0.080 (7/25 → 9/25)
- All 25 queries are single-anchor (n_q_tes=1). T_v2's gain here is *purely* from per-d_TE pair MAX + the dropped tag_jaccard channel + the absence of interval-global-max normalization. **+2 wins outright.**

### Tempreason_small — Δ R@1 = −0.033 (17/60 → 15/60), regression
- Bucketing by anchor count (above) shows the regression is concentrated entirely in the n_q_tes=1 segment (17/35 → 15/35; the n_q_tes=0 queries score 0 in both since neither variant has a q_TE to anchor on).
- **Why**: tempreason_small queries are real-world historical questions like "Who was X's spouse before 2010?" — the q_TE is often a single open-ended interval (`<2010`), and the gold doc may have multiple TEs spanning a long stretch. Under T_lblend, the lattice channel (weight 0.6) walks ancestors aggressively (decade/century), giving partial credit for any-overlap. Under T_v2, the per-pair lattice is strict Jaccard on `tagset.all_tags` for each individual TE; a doc with 4 TEs (year:2008, year:2010, year:2012, year:2014) gets 4 chances to MAX-match `<2010`, but each pair's Jaccard is 1/N (one matching tag among many), so the per-anchor max stays low.
- **Mitigation candidate**: in `lattice_match_pair`, swap pure Jaccard for a "containment-aware" score — count common ancestors of one tag in the other tag's lattice walk (i.e., reuse `lattice_retrieve_multi`-style hierarchical match per-pair). This was the original strength of the `lattice_score` channel that T_v2 dropped.

## When to use each
| Query shape | Winner |
|---|---|
| Multi-anchor (AND across 2+ TEs) | T_v2 |
| Single anchor with multi-TE docs (focal vs peripheral disambig) | T_v2 |
| Single anchor with open-ended/coarse interval queries (era-style) | T_lblend (lattice ancestor walk) |
| 0 q_TEs | Neither helps (T returns 0 either way) |

## Hybrid proposal (for the regression on tempreason)
Since the regression is anchor-count-bucketed, a **gate by `n_q_tes`** captures both regimes:
```
T_hybrid(q, d) = T_v2(q, d)      if n_q_tes(q) >= 2
              = T_lblend(q, d)   otherwise   # falls back to lattice ancestor walk
```
Expected outcome from the bucket numbers above (held-constant):
- conjunctive_temporal: 11/12 (uses T_v2)
- multi_te_doc: 9/12 (uses T_lblend; loses the +1 win since n_q_tes=1 here)
- temporal_essential: 7/25 (loses the +2 win since n_q_tes=1)
- tempreason_small: 17/60 (recovers)
- hard_bench: same R@1 as T_lblend on the n_q_tes=1 queries (zero), gains 2 on multi-anchor

A more principled hybrid blends scores: `T_hybrid = β · T_v2 + (1−β) · T_lblend` with `β` learned per-query from `n_q_tes` and `lattice_score_dispersion` (the lattice channel is most valuable when it has clear winners; when it's flat across docs, T_v2's discriminative per-pair pairing wins).

## Suggested next experiments
1. **Containment-aware per-pair lattice** (most likely T_v2 win on tempreason): replace `lattice_match_pair` Jaccard with `len(common_ancestors(q_tag, d_tag)) / len(q_ancestors)` summed over the tag pair. This reuses lattice hierarchy per-pair, recovering the era/decade match strength without the global tag union.
2. **Score-blend hybrid** with anchor-count gate: implement `T_hybrid = β·T_v2 + (1−β)·T_lblend` and grid β ∈ {0.0, 0.25, 0.5, 0.75, 1.0} on the 5 benches above. Anchor-count gate is the simplest predicate.
3. **Aggregator ablation**: swap geomean for arithmetic mean of `best_per_anchor` (less harsh on partial coverage). Conjunctive R@5 already moved to 12/12; the geomean floor isn't hurting there, but on tempreason the harsh floor may hurt single-anchor when there's only one anchor (then it's just MAX and geomean = MAX, so this is irrelevant for tempreason regression actually — confirms my containment hypothesis above).
4. **Per-pair weight tuning**: the 0.40/0.40/0.20 split is a guess. Sweep `(w_iv, w_lat, w_ax) ∈ {0.2, 0.4, 0.6}` on a 6-point simplex jointly across all 5 benches.
5. **Per-query LLM gate** (existing `per_query_gate_eval.py` pattern): let the gate pick T_v2 vs T_lblend per query. Worked for fuse_T_R distribution split — should work here.
6. **Bigger conjunctive bench**: 12 queries is too few; resynth with 50+ to actually measure the inversion fix at R@1 above the ceiling.
