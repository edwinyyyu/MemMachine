# T_v4 — Asymmetric containment ratio temporal scoring

Single primitive: `|q_iv ∩ d_iv| / |d_iv|` (asymmetric, normalized by doc duration). MAX across (q_iv, d_iv) pairs per (q_te, d_te); MAX across d_tes per q_te; geomean across q anchors with floor 1e-6. No tag_jaccard, no axis, no lattice.

## R@1 table

| Benchmark | n | T_lblend R@1 | T_v2 R@1 | T_v3 R@1 | T_v4 R@1 | Δ(v4−v2) | Δ(v4−lblend) |
|---|---:|---:|---:|---:|---:|---:|---:|
| conjunctive_temporal | 12 | 0.917 (11/12) | 0.917 (11/12) | 0.917 (11/12) | 0.667 (8/12) | -0.250 | -0.250 |
| multi_te_doc | 12 | 0.750 (9/12) | 0.833 (10/12) | 0.750 (9/12) | 0.667 (8/12) | -0.167 | -0.083 |
| relative_time | 12 | 0.417 (5/12) | 0.417 (5/12) | 0.417 (5/12) | 0.417 (5/12) | +0.000 | +0.000 |
| era_refs | 12 | 0.083 (1/12) | 0.083 (1/12) | 0.167 (2/12) | 0.167 (2/12) | +0.083 | +0.083 |
| hard_bench | 75 | 0.000 (0/75) | 0.027 (2/75) | 0.027 (2/75) | 0.027 (2/75) | +0.000 | +0.027 |
| temporal_essential | 25 | 0.280 (7/25) | 0.360 (9/25) | 0.240 (6/25) | 0.240 (6/25) | -0.120 | -0.040 |
| tempreason_small | 60 | 0.283 (17/60) | 0.250 (15/60) | 0.233 (14/60) | 0.300 (18/60) | +0.050 | +0.017 |
| causal_relative | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | +0.000 | +0.000 |
| latest_recent | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.067 (1/15) | 0.000 (0/15) | +0.000 | +0.000 |
| open_ended_date | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.067 (1/15) | 0.067 (1/15) | +0.067 | +0.067 |
| negation_temporal | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | +0.000 | +0.000 |

## R@5 table

| Benchmark | n | T_lblend R@5 | T_v2 R@5 | T_v3 R@5 | T_v4 R@5 |
|---|---:|---:|---:|---:|---:|
| conjunctive_temporal | 12 | 0.917 (11/12) | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| relative_time | 12 | 0.917 (11/12) | 0.917 (11/12) | 0.833 (10/12) | 0.833 (10/12) |
| era_refs | 12 | 0.250 (3/12) | 0.250 (3/12) | 0.250 (3/12) | 0.250 (3/12) |
| hard_bench | 75 | 0.053 (4/75) | 0.053 (4/75) | 0.120 (9/75) | 0.120 (9/75) |
| temporal_essential | 25 | 0.880 (22/25) | 0.880 (22/25) | 0.760 (19/25) | 0.760 (19/25) |
| tempreason_small | 60 | 0.433 (26/60) | 0.383 (23/60) | 0.400 (24/60) | 0.450 (27/60) |
| causal_relative | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) |
| latest_recent | 15 | 0.067 (1/15) | 0.067 (1/15) | 0.067 (1/15) | 0.000 (0/15) |
| open_ended_date | 15 | 0.333 (5/15) | 0.200 (3/15) | 0.267 (4/15) | 0.333 (5/15) |
| negation_temporal | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) |

## Headline

- **tempreason regression**: FIXED — T_v4 R@1 = 0.300, T_lblend = 0.283, T_v2 = 0.250.
- **temporal_essential win**: LOST (Δ=-0.120 vs T_v2) — T_v4 R@1 = 0.240, T_v2 = 0.360.
- **multi_te_doc win**: LOST (Δ=-0.167 vs T_v2) — T_v4 R@1 = 0.667, T_v2 = 0.833.

## Per-bench v4 vs v2 swap counts

| Benchmark | v4-only top1 (gain) | v2-only top1 (loss) | both | neither |
|---|---:|---:|---:|---:|
| conjunctive_temporal | 1 | 4 | 7 | 0 |
| multi_te_doc | 0 | 2 | 8 | 2 |
| relative_time | 1 | 1 | 4 | 6 |
| era_refs | 1 | 0 | 1 | 10 |
| hard_bench | 2 | 2 | 0 | 71 |
| temporal_essential | 5 | 8 | 1 | 11 |
| tempreason_small | 6 | 3 | 12 | 39 |
| causal_relative | 0 | 0 | 0 | 15 |
| latest_recent | 0 | 0 | 0 | 15 |
| open_ended_date | 1 | 0 | 0 | 14 |
| negation_temporal | 0 | 0 | 0 | 15 |

## Per-failure diagnosis

### tempreason_small

**Losses (v4 missed, v2 hit, up to 5):**
- `q_l2_0010` (n_q_tes=1): v4_top1=`d_00088`, v4_rank=2, gold=['d_00030']
  - q: Which position did Henri Madelin hold in Dec, 1985?
- `q_l2_0011` (n_q_tes=1): v4_top1=`d_00198`, v4_rank=7, gold=['d_00033']
  - q: Which position did Michael Fallon hold in Dec, 2001?
- `q_l2_0028` (n_q_tes=1): v4_top1=`d_00114`, v4_rank=3, gold=['d_00082']
  - q: Which team did Katarina Kolar play for in Feb, 2013?
**Gains (v4 hit, v2 missed, up to 5):**
- `q_l2_0003` (n_q_tes=1): v4_top1=`d_00011`, v2_top1=`d_00015`, gold=['d_00011']
  - q: Where was Hans Bethe educated in Aug, 1929?
- `q_l2_0024` (n_q_tes=1): v4_top1=`d_00069`, v2_top1=`d_00080`, gold=['d_00069']
  - q: Which employer did Caroline C. Hunter work for in Aug, 2005?
- `q_l2_0030` (n_q_tes=1): v4_top1=`d_00088`, v2_top1=`d_00030`, gold=['d_00088']
  - q: Which team did Domenico Maggiora play for in Apr, 1986?
- `q_l2_0032` (n_q_tes=1): v4_top1=`d_00092`, v2_top1=`d_00119`, gold=['d_00092']
  - q: Who was the head coach of the team K.V. Kortrijk in Mar, 2020?
- `q_l2_0034` (n_q_tes=1): v4_top1=`d_00099`, v2_top1=`d_00198`, gold=['d_00099']
  - q: Which team did Steve Jones play for in Jan, 2002?

### temporal_essential

**Losses (v4 missed, v2 hit, up to 5):**
- `te_q_003` (n_q_tes=1): v4_top1=`te_010_s1`, v4_rank=6, gold=['te_003_g']
  - q: When did Kim Patel complete the kitchen remodel in March 2024?
- `te_q_005` (n_q_tes=1): v4_top1=`te_002_s3`, v4_rank=3, gold=['te_005_g']
  - q: When did Olivia Roberts sign up for yoga class in early January 2025?
- `te_q_011` (n_q_tes=1): v4_top1=`te_008_g`, v4_rank=3, gold=['te_011_g']
  - q: When did Layla Smith complete the tax filing in early April 2024?
- `te_q_012` (n_q_tes=1): v4_top1=`te_018_s3`, v4_rank=3, gold=['te_012_g']
  - q: When did Maya Singh make the code freeze announcement in December 2024?
- `te_q_015` (n_q_tes=1): v4_top1=`te_006_s1`, v4_rank=7, gold=['te_015_g']
  - q: When did Mira Khan run the data migration cutover in May 2024?
**Gains (v4 hit, v2 missed, up to 5):**
- `te_q_002` (n_q_tes=1): v4_top1=`te_002_g`, v2_top1=`te_023_s0`, gold=['te_002_g']
  - q: When did Priya Johnson lead the team retrospective in early May 2024?
- `te_q_008` (n_q_tes=1): v4_top1=`te_008_g`, v2_top1=`te_011_g`, gold=['te_008_g']
  - q: When did Quinn Reeves do the lease signing in early April 2024?
- `te_q_013` (n_q_tes=1): v4_top1=`te_013_g`, v2_top1=`te_005_s0`, gold=['te_013_g']
  - q: When did Alice Liu host the garage sale in mid-April 2024?
- `te_q_018` (n_q_tes=1): v4_top1=`te_018_g`, v2_top1=`te_013_s1`, gold=['te_018_g']
  - q: When did Vera Lin do the moving day in late January 2024?
- `te_q_021` (n_q_tes=1): v4_top1=`te_021_g`, v2_top1=`te_001_s3`, gold=['te_021_g']
  - q: When did Eric Hall give the keynote speech in February 2023?

### multi_te_doc

**Losses (v4 missed, v2 hit, up to 5):**
- `mte_q_006` (n_q_tes=1): v4_top1=`mte_010_d3`, v4_rank=2, gold=['mte_006_g']
  - q: What did Henry Ford do on September 9, 2023?
- `mte_q_010` (n_q_tes=1): v4_top1=`mte_001_g`, v4_rank=2, gold=['mte_010_g']
  - q: What did Tom Reed do on April 15, 2024?

### conjunctive_temporal

**Losses (v4 missed, v2 hit, up to 5):**
- `conj_q_000` (n_q_tes=2): v4_top1=`conj_001_o0`, v4_rank=2, gold=['conj_000_g']
  - q: What were Sarah Park's her dental appointments in Q3 2023 and Q1 2024?
- `conj_q_005` (n_q_tes=2): v4_top1=`conj_001_g`, v4_rank=5, gold=['conj_005_g']
  - q: What were Olivia Roberts's yoga classes in both spring and fall 2024?
- `conj_q_009` (n_q_tes=2): v4_top1=`conj_001_g`, v4_rank=2, gold=['conj_009_g']
  - q: What were Sara Lee's puppy training sessions in March and June 2024?
- `conj_q_010` (n_q_tes=2): v4_top1=`conj_001_o0`, v4_rank=2, gold=['conj_010_g']
  - q: What were Tom Reed's marathons in April 2023 and October 2024?
**Gains (v4 hit, v2 missed, up to 5):**
- `conj_q_001` (n_q_tes=1): v4_top1=`conj_001_g`, v2_top1=`conj_009_g`, gold=['conj_001_g']
  - q: What were Marcus Davis's his quarterly reviews in between March and August 2024?

### relative_time

**Losses (v4 missed, v2 hit, up to 5):**
- `rel_q_004` (n_q_tes=1): v4_top1=`rel_008_g`, v4_rank=2, gold=['rel_004_g']
  - q: When did Aiden Park give his TED talk, two months ago?
**Gains (v4 hit, v2 missed, up to 5):**
- `rel_q_001` (n_q_tes=2): v4_top1=`rel_001_g`, v2_top1=`rel_004_d0`, gold=['rel_001_g']
  - q: When did Marcus Davis have his quarterly check-in, last week?

### era_refs

**Gains (v4 hit, v2 missed, up to 5):**
- `era_q_008` (n_q_tes=1): v4_top1=`era_008_g`, v2_top1=`era_006_d0`, gold=['era_008_g']
  - q: When did Quinn Reeves adopt his dog during the pandemic year?

### open_ended_date

**Gains (v4 hit, v2 missed, up to 5):**
- `oe_q_014` (n_q_tes=2): v4_top1=`oe_014_g0`, v2_top1=`oe_009_d1`, gold=['oe_014_g0']
  - q: What apartments did I tour before I signed the lease in October 2023?

### hard_bench

**Losses (v4 missed, v2 hit, up to 5):**
- `q_easy_008` (n_q_tes=2): v4_top1=`hd_0005`, v4_rank=3, gold=['hd_0210']
  - q: When did Priya Nguyen deliver the quarterly review in 2022?
- `q_hard_012` (n_q_tes=2): v4_top1=`hd_0007`, v4_rank=None, gold=['hd_0582']
  - q: When did someone on the team deliver the quarterly review in Q2 2024?
**Gains (v4 hit, v2 missed, up to 5):**
- `q_medium_001` (n_q_tes=1): v4_top1=`hd_0004`, v2_top1=`hd_0245`, gold=['hd_0004']
  - q: When did Priya lead the project kickoff in Q2 2023?
- `q_medium_007` (n_q_tes=1): v4_top1=`hd_0003`, v2_top1=`hd_0558`, gold=['hd_0003']
  - q: When was Marcus promoted in Q3 2024?

## Verdict

Macro-average R@1 across 11 benches: T_lblend=0.248, T_v2=0.262, T_v3=0.262, **T_v4=0.232**.

**T_v4 LOSES overall by −0.030 vs T_v2/T_v3 macro.** The single-primitive containment ratio fixes the tempreason regression cleanly (+0.050 vs T_v2; +0.017 vs T_lblend, the first variant to actually beat the shipping scorer there) and earns small but principled wins on `era_refs` (+0.083), `open_ended_date` (+0.067), and `hard_bench` R@5 (+0.067). But it loses big on `temporal_essential` (−0.120 vs T_v2), `multi_te_doc` (−0.167 vs T_v2), and especially `conjunctive_temporal` (**−0.250** vs all baselines).

The tempreason win confirms the open-ended-containment claim (T_lblend never beat itself there before; T_v4 does). Era_refs / open_ended_date wins confirm the broad-q-narrow-d cases. The losses confirm a different failure mode below.

## Root cause of T_v4's losses: tie-collapse at 1.0

The asymmetric ratio `|q ∩ d| / |d|` saturates at **1.0 whenever d ⊆ q**. When many distractor docs have a TE that lies inside the query interval (which is *every* exact-month or fuzzy-month query in temporal_essential / multi_te_doc / conjunctive_temporal), they all get 1.0 and the ranking breaks down to alphabetical/insertion order.

### temporal_essential (−0.120 vs T_v2): year-month query, many docs share that month
- `te_q_003` ("kitchen remodel in March 2024"): T_v4 top-1 is `te_010_s1` (a sibling distractor with a March 2024 TE), gold is `te_003_g` at rank 6. T_v2's pure Jaccard scored only the doc whose TE was closest to "March 2024" exactly; T_v4 ties every doc whose TE is *anywhere inside* March 2024.
- 8 of 8 losses on temporal_essential are single-anchor queries where multiple docs match the query month at 1.0.
- Mechanism: T_v2's interval Jaccard (with the proximity term in `score_jaccard_composite`) discriminates by best_us; T_v4's pure containment doesn't see best_us at all.

### multi_te_doc (−0.167 vs T_v2): peripheral TEs falsely score 1.0
- `mte_q_006` ("September 9, 2023"): gold is `mte_006_g`; v4 picks `mte_010_d3` (a multi-TE doc whose peripheral TE happens to fall on Sep 9, 2023). v2's Jaccard penalized via the global-min normalization; v4 awards 1.0 to any d_te that is a delta on that day.
- T_v4 has no notion of focal-vs-peripheral, AND awards equal credit to any matching TE inside a multi-TE doc.

### conjunctive_temporal (−0.250 vs all): geomean amplifies tie-collapse
- All 4 losses are 2-anchor queries ("Q3 2023 AND Q1 2024", "March AND June 2024", etc.). Multiple docs hit BOTH anchors at 1.0 → geomean = 1.0 for many docs → arbitrary winner.
- v2/v3 used Jaccard, so each anchor scores ~0.6–0.9 depending on exactness, geomean spreads, gold's exact-pair wins.
- v4 collapses both anchors to 1.0 for any doc whose TEs are inside both query intervals, even if those TEs are entirely peripheral.

### Why tempreason/era_refs/open_ended_date WIN
These benches have *open-ended* or *broad* queries (`in Dec 1985`, `during the pandemic year`, `before October 2023`). T_v2's interval Jaccard scored 0 when the query was much wider than the doc TE (small overlap / huge union). T_v4's asymmetric ratio gives 1.0 in exactly those cases. Per-pair comparisons:
- tempreason `q_l2_0030` ("Apr 1986"): gold `d_00088` has Apr 1986 TE, query is Apr 1986 — both 1.0, but v2's Jaccard ties tons of distractors with the same month and v4's containment plus geomean still discriminates by single-anchor coverage.
- era_refs `era_q_008` ("during the pandemic year"): query is broad (~2020–2021), gold has a delta inside that period. v2's Jaccard = tiny (broad q, narrow d); v4's containment = 1.0. Same in open_ended_date.

## Per-failure root cause table

| Bench | failure type | mechanism |
|---|---|---|
| conjunctive_temporal | tie-collapse | both anchors ⇒ both saturate to 1.0 ⇒ geomean=1.0 for many docs |
| temporal_essential | distractor-saturation | year-month query, many docs share month ⇒ all hit 1.0 |
| multi_te_doc | peripheral-saturation | multi-TE doc's peripheral TE hits 1.0, no focality penalty |

## Why this isn't fixable inside T_v4's primitive

The asymmetric containment is mathematically correct for "is this doc inside the query window". But retrieval ranking needs a tighter, off-by-zero discriminator: among docs that ARE inside the window, which one is the best match? T_v4 throws away that signal by saturating. To get it back inside the same primitive you'd need to either:

1. Switch back to symmetric overlap (Jaccard) which discriminates by tightness — but that loses the open-ended win.
2. Add a tie-breaker — but then it's not a single primitive anymore.

This is a fundamental limitation: **a single saturating primitive cannot distinguish "fully inside, exact match" from "fully inside, fuzzy match"**. T_v2 had this (proximity term in `score_jaccard_composite`); T_v3 had it (granularity-weighted chain match); T_v4 deleted it.

## Proposal: T_v5 — saturated containment + proximity tiebreak

Single-formula primitive (still no tag/lattice/axis fan-out):

```python
def pair_score_v5(q_iv, d_iv):
    inter = max(0, min(qi.latest_us, di.latest_us) - max(qi.earliest_us, di.earliest_us))
    d_dur = max(1, di.latest_us - di.earliest_us)
    q_dur = max(1, qi.latest_us - qi.earliest_us)
    cont  = inter / d_dur                           # T_v4 primitive: 1.0 when d ⊆ q
    if q.best_us and d.best_us:
        span = max(q_dur, d_dur, 1_000_000)
        prox = max(0.0, 1.0 - abs(q.best_us - d.best_us) / span)
    else:
        prox = 0.5
    # Multiplicative tiebreak: cont dominates the magnitude, prox breaks ties.
    # When cont=1.0 (d ⊆ q), score in [0.5, 1.0] from prox.
    # When cont<1.0, score is strictly less than 1.0.
    return cont * (0.5 + 0.5 * prox)
```

Predictions:
- **conjunctive_temporal**: should restore. Both anchors at 1.0×prox, gold's exact-best wins.
- **temporal_essential**: should restore most of the loss. Same-month distractors all get 1.0, but only the one whose best_us aligns with the query's best_us gets max prox.
- **multi_te_doc**: still tricky — peripheral TE in a multi-TE doc can have best_us inside query. Need salience flag (orthogonal to T_v5).
- **tempreason / era_refs / open_ended_date**: keep wins. cont=1.0 dominates; prox at 0.5–1.0 still ranks above non-contained docs whose cont<1.0.
- **conjunctive geomean tie**: prox tiebreaker enters the geomean, so gold's per-anchor prox compounds.

This is essentially T_v2's `score_jaccard_composite` rewritten with `cont` replacing `jaccard`. Cleaner derivation: `cont * f(prox)` where `f` is monotone in [0.5, 1.0]. Single primitive, but it has a structural reason for the tiebreak (saturation needs prox to disambiguate). Worth one more round.

If T_v5 still fails on multi_te_doc, the problem is salience/focality — orthogonal to scoring formula, requires extractor change (focal flag on TE).

## Headline summary

- **T_v4 fixes tempreason** (the original target of this whole T_vN line). Δ vs T_lblend = +0.017, first variant to beat shipping.
- **T_v4 loses on saturation-prone benches** (conjunctive, temporal_essential, multi_te_doc). Macro R@1 drops to 0.232 vs T_v2's 0.262.
- **Single-primitive design is incompatible with tie-breaking** in this benchmark suite. T_v5 should multiply by a proximity factor.
- **Per-query gate** (already-built winner from project_per_query_gate_works) could pick T_v2 vs T_v4 per query — T_v4 dominates broad-query distributions, T_v2 dominates exact-match distributions.
