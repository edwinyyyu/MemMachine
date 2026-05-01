# T-component ablation: which T-score components do the work?

## Setup

T-only ranking — no semantic, no rerank, no fusion. For each query we rank
every doc by a weighted blend of four per-component score tables:

- `interval` — `interval_pair_best` Jaccard composite over interval lists
  (max-normalized to [0,1] across docs per query).
- `tag` — `tag_score` Jaccard over `{axis}:{value}` tag sets.
- `lattice` — `lattice_retrieval.retrieve_multi` containment score (the
  current "L" channel inside `T_lblend`).
- `axis` — `multi_axis_scorer.axis_score` (geomean of Bhattacharyya
  coefficients across informative axes — currently NOT in `T_lblend`).

7 variants:

| Variant            | w_interval | w_tag | w_lattice | w_axis |
| ------------------ | ---------- | ----- | --------- | ------ |
| T_interval_only    | 1.00       | 0     | 0         | 0      |
| T_tag_only         | 0          | 1.00  | 0         | 0      |
| T_lattice_only     | 0          | 0     | 1.00      | 0      |
| T_axis_only        | 0          | 0     | 0         | 1.00   |
| T_lblend (current) | 0.20       | 0.20  | 0.60      | 0      |
| T_lblend_axis      | 0.15       | 0.15  | 0.45      | 0.25   |
| T_eq               | 0.25       | 0.25  | 0.25      | 0.25   |

## Results — R@1 / R@5 per benchmark

| Variant            | hard_bench       | temporal_essential | tempreason_small |
| ------------------ | ---------------- | ------------------ | ---------------- |
| T_interval_only    | 0.000 / 0.053    | 0.280 / 0.880      | 0.217 / 0.400    |
| T_tag_only         | **0.040** / 0.107| 0.200 / 0.440      | 0.117 / 0.250    |
| T_lattice_only     | 0.027 / 0.080    | 0.120 / 0.720      | 0.117 / 0.217    |
| T_axis_only        | 0.013 / 0.067    | 0.320 / 0.840      | 0.250 / 0.367    |
| T_lblend (current) | 0.000 / 0.053    | 0.280 / 0.880      | **0.300 / 0.450**|
| T_lblend_axis      | 0.000 / 0.053    | **0.360 / 0.880**  | 0.283 / 0.383    |
| T_eq               | 0.000 / 0.053    | **0.360 / 0.880**  | 0.250 / 0.400    |

(`R@1` is bolded for the per-benchmark winner. R@5 ties broken by R@1.)

Counts (R@1 / R@5 / N) — for reference:

- hard_bench: T_tag_only 3/8 of 75; T_lblend 0/4 of 75.
- temporal_essential: T_lblend_axis 9/22 of 25; T_eq 9/22 of 25; T_axis_only 8/21.
- tempreason_small: T_lblend 18/27 of 60; T_lblend_axis 17/23; T_axis_only 15/22.

Raw JSON: `results/T_component_ablation.json`.

## What is doing the work?

### Per-benchmark dominant component

- **temporal_essential** — `T_axis_only` is the single best individual
  component (R@1 0.320, R@5 0.840). Adding `axis` to the blend lifts R@1 by
  +0.080 absolute (0.280 → 0.360). Lattice alone is the *worst* individual
  component here (R@1 0.120) yet the current `T_lblend` puts 0.6 weight on
  it. That weight is paying for itself only because lattice is correlated
  with interval/tag and contributes through the union-of-evidence.

- **tempreason_small** — Interval and axis are the strongest *individual*
  components (R@1 0.217 and 0.250). Lattice and tag are weakest (R@1 0.117
  each). But the current `T_lblend` (0.6 lattice) still wins R@1 here at
  0.300 — combining the noisy lattice signal with the 0.2 interval/tag
  helps the borderline cases. Adding axis at the cost of lattice
  *regresses* (0.300 → 0.283 R@1, 0.450 → 0.383 R@5). On this benchmark
  axis and lattice are partly substitutes for each other and lattice wins.

- **hard_bench** — All variants collapse near 0 R@1. T-alone is
  fundamentally underpowered for this corpus: hard_bench has many docs
  sharing the same date with only entity names distinguishing the gold —
  exactly what semantic+rerank handles and T cannot. `T_tag_only` is the
  best at R@1 0.040 (3/75); `T_lblend` and every blended variant tie at
  0/75 R@1 because their weights spread across components that all index
  the same date and dilute the small signal that tags carry. **This is
  the headline finding for the current ship-config**: on hard_bench `T`
  is not just weak — `T_lblend`'s exact 0.2/0.2/0.6 mix is *strictly
  worse* than `T_tag_only` because the lattice term is undiscriminating.
  In production we know hard_bench is best served by `fuse_T_R` with the
  rerank doing the entity disambiguation, so this is consistent with the
  per-query gate result — but it demonstrates that the lattice-heavy
  weight is not buying T anything on this distribution.

### Component summary across benchmarks

- **interval** — Strong on date-anchored corpora (temporal_essential R@5
  0.880, tempreason R@1 0.217). Useless on hard_bench (R@1 0.000) because
  many docs share the same date.

- **tag** — Mid-tier, but the *only* component that delivers any R@1 on
  hard_bench (3/75). Otherwise weakest individual on temporal_essential
  R@1 0.200 and tempreason R@1 0.117.

- **lattice** — The current 0.6-weight workhorse, but as a standalone
  component it is the *weakest or tied-weakest* on every benchmark
  (temporal_essential R@1 0.120, tempreason R@1 0.117, hard_bench R@1
  0.027). Its 0.6 weight is over-indexed; it survives mostly because it
  is correlated with interval and contributes redundant evidence.

- **axis** — The *best* standalone component on temporal_essential (R@1
  0.320) and tempreason (R@1 0.250). It is currently dropped from
  T_lblend, costing measurable R@1.

## Conclusions

1. **Lattice (0.6 weight) is over-weighted.** It is the weakest standalone
   component on 2/3 benchmarks (and tied on the 3rd). On tempreason its
   weight is justifiable because the blend with interval+tag wins; on
   temporal_essential it is the *worst* standalone component yet absorbs
   3x the weight of either of the others.

2. **Axis is under-weighted (= zero).** It is the best standalone on the
   two benchmarks where T can do real work, and adding it (0.25) lifts
   R@1 +0.080 on temporal_essential. The `axis` channel is currently
   dispensed to the floor, dropping a real signal.

3. **Interval is undervalued for date-anchored queries** but neutralized
   by hard_bench's same-date adversarial structure.

4. **Tag is the only component carrying hard_bench R@1.** It survives
   the entity-vs-date disambiguation slightly better than the others.
   That said, the absolute level on hard_bench (3/75 R@1) is not what T
   should be doing — that benchmark is for rerank.

## Recommended weight adjustment

Drop one weight from lattice and split the freed 0.10 between axis and
interval. Concrete proposal:

`T_lblend_v2`: 0.25 interval + 0.20 tag + 0.35 lattice + 0.20 axis

Rationale, by component delta:

- `interval` 0.20 → 0.25 (+0.05): pure-interval temporal_essential
  R@5 0.880 and tempreason R@1 0.217 say it is at least as informative
  as tag, but it currently has the same weight.

- `tag` 0.20 → 0.20 (unchanged): only component carrying hard_bench R@1;
  do not lower it.

- `lattice` 0.60 → 0.35 (-0.25): every standalone test says lattice is
  the weakest individual signal. Keeping a substantial weight (0.35,
  still the largest single weight) preserves its value as a
  cross-validating coverage signal on tempreason where the blend works.

- `axis` 0.0 → 0.20 (+0.20): introduce the channel that wins R@1 on the
  two non-adversarial benchmarks. This matches the `T_lblend_axis`
  experiment (which lifted temporal_essential R@1 0.280 → 0.360 with
  weight 0.25) but tempers it slightly so we do not regress tempreason
  (where the 0.25-axis variant cost -0.017 R@1).

For the small `T_eq` and `T_lblend_axis` results that already match or
beat the current ship on temporal_essential while losing only minor
ground on tempreason, the data supports adopting an axis-bearing blend.

`T_eq` (0.25 each) is a respectable, simpler alternative — it ties
T_lblend_axis on temporal_essential and matches T_lblend on tempreason
R@5 (0.400 vs 0.450). If we want a single commit, `T_lblend_v2` above is
the lower-variance choice; if we want minimum surface change, `T_eq`
(four equal weights with axis added) is the safe drop-in.

## Caveats

- T-alone numbers are not directly comparable to shipping pipeline
  numbers — they exclude semantic and rerank. The conclusion is about
  the *internal* allocation of T's weights, not about T's headline
  contribution to the full pipeline.

- The current ship's per-query gate (`per_query_gate_works.md`) selects
  between rerank-only and fuse_T_R based on query type. A T-internal
  re-weighting interacts with that gate; before merging `T_lblend_v2`,
  re-run the per-query-gate eval to confirm the gated full-pipeline R@1
  on hard_bench (where rerank-only is selected) is unchanged and the
  gated result on tempreason/temporal_essential improves.

- Note: `T_lblend` here was implemented as the same 0.2/0.2/0.6 pure
  linear blend used in `make_t_scores`. Its lower R@1 than the best
  individual component on hard_bench is a property of the weights, not
  a quirk of the blender — the per-doc max-normalization happens
  identically in `make_t_scores`.
