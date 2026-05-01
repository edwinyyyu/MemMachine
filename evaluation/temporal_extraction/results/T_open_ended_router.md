# T_open_ended_router — switched T-channel for open-ended date queries

Per-query gate: `has_open_ended_cue(query)` regex on side-keyword (after/before/since/until/prior-to/post/pre) + YYYY or Month-YYYY anchor; OR `from YYYY onwards`; OR symbolic `<YYYY`/`>YYYY`. When fired, T-channel = T_v5; else T_lblend.

Top-line stack: `fuse_T_router_R + recency_additive(α=0.5 when recency_cue)`. Baseline is the same recipe with T_lblend pinned.


## R@1 — Baseline vs Router

| Benchmark | n | oe_act | rec_act | rerank_only | baseline (T_lblend) | **router** | v5_only (no gate) | Δ(router − baseline) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0 | 0 | 0.640 (48/75) | 0.893 (67/75) | **0.893 (67/75)** | 0.907 (68/75) | **+0.000** |
| temporal_essential | 25 | 0 | 0 | 0.920 (23/25) | 1.000 (25/25) | **1.000 (25/25)** | 1.000 (25/25) | **+0.000** |
| tempreason_small | 60 | 2 | 0 | 0.650 (39/60) | 0.733 (44/60) | **0.733 (44/60)** | 0.733 (44/60) | **+0.000** |
| conjunctive_temporal | 12 | 0 | 0 | 1.000 (12/12) | 1.000 (12/12) | **1.000 (12/12)** | 1.000 (12/12) | **+0.000** |
| multi_te_doc | 12 | 0 | 0 | 1.000 (12/12) | 1.000 (12/12) | **1.000 (12/12)** | 1.000 (12/12) | **+0.000** |
| relative_time | 12 | 0 | 1 | 0.250 (3/12) | 0.917 (11/12) | **0.917 (11/12)** | 0.917 (11/12) | **+0.000** |
| era_refs | 12 | 0 | 0 | 0.250 (3/12) | 0.417 (5/12) | **0.417 (5/12)** | 0.333 (4/12) | **+0.000** |
| open_ended_date | 15 | 15 | 0 | 0.267 (4/15) | 0.400 (6/15) | **0.533 (8/15)** | 0.533 (8/15) | **+0.133** (router better) |
| causal_relative | 15 | 0 | 1 | 0.467 (7/15) | 0.467 (7/15) | **0.467 (7/15)** | 0.467 (7/15) | **+0.000** |
| latest_recent | 15 | 0 | 15 | 0.133 (2/15) | 0.667 (10/15) | **0.667 (10/15)** | 0.733 (11/15) | **+0.000** |
| negation_temporal | 15 | 0 | 0 | 0.000 (0/15) | 0.000 (0/15) | **0.000 (0/15)** | 0.000 (0/15) | **+0.000** |

## Macro-average R@1 across 11 benches

- rerank_only:     0.507
- baseline (T_lblend): **0.681**
- **router**:      **0.693**
- v5_only (no gate): 0.693
- Δ(router − baseline): **+0.012**
- Δ(v5_only − baseline): +0.012

## Switch firing pattern

| Benchmark | n | oe_active | %  | expected? |
|---|---:|---:|---:|---|
| hard_bench | 75 | 0 | 0% | no (closed-range 'in YYYY') |
| temporal_essential | 25 | 0 | 0% | no |
| tempreason_small | 60 | 2 | 3% | varies |
| conjunctive_temporal | 12 | 0 | 0% | no |
| multi_te_doc | 12 | 0 | 0% | no |
| relative_time | 12 | 0 | 0% | no |
| era_refs | 12 | 0 | 0% | no (event refs, no date) |
| open_ended_date | 15 | 15 | 100% | yes (15/15 ideal) |
| causal_relative | 15 | 0 | 0% | no (event refs, no date) |
| latest_recent | 15 | 0 | 0% | no (recency, no anchor) |
| negation_temporal | 15 | 0 | 0% | no (mostly closed-range) |

## Switch correctness audit

- False positives across non-open-ended benches: **0**
- False negatives on `open_ended_date`: **0** / 15

## Did the router capture T_v5's win on open_ended_date?

- rerank_only:  0.267 (4/15)
- baseline (T_lblend): 0.400 (6/15)
- **router**:    **0.533** (8/15)
- v5_only:      0.533 (8/15)
- Δ(router − baseline) = +0.133  (target ≥ +0.133)

**YES** — router captured T_v5's open-ended-date advantage.

## Regression check (other benches)

No regressions on non-open-ended benches.

## Recommendation

**SHIP** the open-ended router. Captures T_v5's open_ended_date win without regressing other benches.

## Sample firings (router fired on these)

### tempreason_small
- `q_l3_0005`: Which team did Marco Piccinni play for after F.B. Brindisi 1912?
- `q_l3_0010`: Which team did Daniele Delli Carri play for before Delfino Pescara 1936?

### open_ended_date
- `oe_q_000`: What did I work on after 2022?
- `oe_q_001`: What did I work on before the pandemic (January 2020)?
- `oe_q_002`: What's my activity since I moved in June 2023?
- `oe_q_003`: What did I do before I joined Acme in March 2022?
- `oe_q_004`: What courses did I complete after graduating in May 2020?
- `oe_q_005`: What investments did I make before retirement (December 2023)?
- `oe_q_006`: What did I publish since I started the blog in March 2024?
- `oe_q_007`: What trips did I take after my child was born in August 2023?
