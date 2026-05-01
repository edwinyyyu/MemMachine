# T_v5 — Containment × bounded proximity

Per-pair primitive: `pair = (|q∩d|/|d|) * (0.5 + 0.5 * prox)`, where `prox = max(0, 1 - |q_best - d_best| / q_span)`. MAX over (q_iv, d_iv) pairs per (q_te, d_te); MAX over doc TEs per query anchor; geomean across query anchors with floor 1e-6. Bounded multiplier in [0.5, 1.0] preserves T_v4's containment correctness while restoring dispersion that T_lblend gets from the lattice channel.

## R@1 in fusion — lead with deltas

| Benchmark | n | rerank_only | fuse_T_lblend_R | fuse_T_v4_R | fuse_T_v5_R | Δ(v5 − lblend) | Δ(v5 − v4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0.640 (48/75) | 0.893 (67/75) | 0.640 (48/75) | **0.893 (67/75)** | **+0.000** | +0.253 |
| temporal_essential | 25 | 0.920 (23/25) | 1.000 (25/25) | 1.000 (25/25) | **1.000 (25/25)** | **+0.000** | +0.000 |
| tempreason_small | 60 | 0.650 (39/60) | 0.733 (44/60) | 0.733 (44/60) | **0.733 (44/60)** | **+0.000** | +0.000 |
| conjunctive_temporal | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | **1.000 (12/12)** | **+0.000** | +0.000 |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | **1.000 (12/12)** | **+0.000** | +0.000 |
| relative_time | 12 | 0.250 (3/12) | 1.000 (12/12) | 1.000 (12/12) | **1.000 (12/12)** | **+0.000** | +0.000 |
| era_refs | 12 | 0.250 (3/12) | 0.417 (5/12) | 0.333 (4/12) | **0.333 (4/12)** | **-0.083** (lblend better) | +0.000 |
| open_ended_date | 15 | 0.267 (4/15) | 0.400 (6/15) | 0.467 (7/15) | **0.533 (8/15)** | **+0.133** (v5 better) | +0.067 |
| causal_relative | 15 | 0.467 (7/15) | 0.467 (7/15) | 0.467 (7/15) | **0.467 (7/15)** | **+0.000** | +0.000 |
| latest_recent | 15 | 0.133 (2/15) | 0.267 (4/15) | 0.133 (2/15) | **0.067 (1/15)** | **-0.200** (lblend better) | -0.067 |
| negation_temporal | 15 | 0.000 (0/15) | 0.000 (0/15) | 0.000 (0/15) | **0.000 (0/15)** | **+0.000** | +0.000 |

## Macro-average R@1 across 11 benches

### Fusion
- rerank_only:      0.507
- fuse_T_lblend_R:  **0.652**
- fuse_T_v4_R:      0.616
- fuse_T_v5_R:      **0.639**
- Δ(v5 − lblend):   **-0.014**
- Δ(v5 − v4):       **+0.023**

## R@1 standalone (no fusion)

| Benchmark | n | T_lblend | T_v4 | T_v5 | Δ(v5 − v4) | Δ(v5 − lblend) |
|---|---:|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0.000 (0/75) | 0.027 (2/75) | **0.013 (1/75)** | -0.013 | +0.013 |
| temporal_essential | 25 | 0.280 (7/25) | 0.240 (6/25) | **0.280 (7/25)** | +0.040 | +0.000 |
| tempreason_small | 60 | 0.283 (17/60) | 0.300 (18/60) | **0.283 (17/60)** | -0.017 | +0.000 |
| conjunctive_temporal | 12 | 0.917 (11/12) | 0.667 (8/12) | **0.500 (6/12)** | -0.167 | -0.417 |
| multi_te_doc | 12 | 0.750 (9/12) | 0.667 (8/12) | **0.750 (9/12)** | +0.083 | +0.000 |
| relative_time | 12 | 0.417 (5/12) | 0.417 (5/12) | **0.417 (5/12)** | +0.000 | +0.000 |
| era_refs | 12 | 0.083 (1/12) | 0.167 (2/12) | **0.083 (1/12)** | -0.083 | +0.000 |
| open_ended_date | 15 | 0.000 (0/15) | 0.067 (1/15) | **0.000 (0/15)** | -0.067 | +0.000 |
| causal_relative | 15 | 0.000 (0/15) | 0.000 (0/15) | **0.000 (0/15)** | +0.000 | +0.000 |
| latest_recent | 15 | 0.000 (0/15) | 0.000 (0/15) | **0.000 (0/15)** | +0.000 | +0.000 |
| negation_temporal | 15 | 0.000 (0/15) | 0.000 (0/15) | **0.000 (0/15)** | +0.000 | +0.000 |

### Standalone macro
- T_lblend: 0.248
- T_v4:     0.232
- T_v5:     **0.212**
- Δ(v5 − v4):     -0.020
- Δ(v5 − lblend): -0.037

## R@5 in fusion

| Benchmark | n | rerank_only | fuse_T_lblend_R | fuse_T_v4_R | fuse_T_v5_R |
|---|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0.853 (64/75) | 0.960 (72/75) | 0.853 (64/75) | 0.960 (72/75) |
| temporal_essential | 25 | 1.000 (25/25) | 1.000 (25/25) | 1.000 (25/25) | 1.000 (25/25) |
| tempreason_small | 60 | 1.000 (60/60) | 1.000 (60/60) | 1.000 (60/60) | 1.000 (60/60) |
| conjunctive_temporal | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| relative_time | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| era_refs | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) |
| open_ended_date | 15 | 0.733 (11/15) | 0.800 (12/15) | 0.800 (12/15) | 0.800 (12/15) |
| causal_relative | 15 | 1.000 (15/15) | 1.000 (15/15) | 1.000 (15/15) | 1.000 (15/15) |
| latest_recent | 15 | 1.000 (15/15) | 1.000 (15/15) | 1.000 (15/15) | 1.000 (15/15) |
| negation_temporal | 15 | 0.933 (14/15) | 0.467 (7/15) | 0.400 (6/15) | 0.467 (7/15) |

## Hypothesis verdicts

### 1. Did T_v5 close T_v4's standalone saturation gap?
(T_v4 standalone losses on these benches were the original motivation.)

| Benchmark | T_v4 R@1 | T_v5 R@1 | Δ(v5 − v4) | T_lblend R@1 | Closed gap? |
|---|---:|---:|---:|---:|---|
| temporal_essential | 0.240 | 0.280 | +0.040 | 0.280 | YES |
| multi_te_doc | 0.667 | 0.750 | +0.083 | 0.750 | YES |
| conjunctive_temporal | 0.667 | 0.500 | -0.167 | 0.917 | NO (still -0.417 vs lblend) |

### 2. Did T_v5 in fusion match or beat T_lblend in fusion?

**NO.** T_v5 in fusion regresses by Δ = -0.014 macro R@1 vs T_lblend.

### 3. Did T_v5 preserve T_v4's win on `open_ended_date`?

- Standalone:  T_lblend=0.000  T_v4=0.067  T_v5=0.000
- Fusion:      T_lblend=0.400  T_v4=0.467  T_v5=0.533
- **PRESERVED** — T_v5 in fusion beats T_lblend in fusion by Δ=+0.133.

### 4. Recommendation: T_v5 as drop-in?

**Do NOT drop in.** T_v5 fusion regresses by Δ = -0.014 macro R@1. Stick with T_lblend; consider hybrid (T_lblend + T_v5 only on open_ended_date).

### 5. Per-bench regression diagnosis (where v5 underperforms T_lblend in fusion)

| Benchmark | Δ(v5 − lblend) | v5-only top1 | lblend-only top1 | both | neither |
|---|---:|---:|---:|---:|---:|
| hard_bench | +0.000 | 1 | 1 | 66 | 7 |
| temporal_essential | +0.000 | 0 | 0 | 25 | 0 |
| tempreason_small | +0.000 | 0 | 0 | 44 | 16 |
| conjunctive_temporal | +0.000 | 0 | 0 | 12 | 0 |
| multi_te_doc | +0.000 | 0 | 0 | 12 | 0 |
| relative_time | +0.000 | 0 | 0 | 12 | 0 |
| era_refs | -0.083 | 0 | 1 | 4 | 7 |
| open_ended_date | +0.133 | 2 | 0 | 6 | 7 |
| causal_relative | +0.000 | 0 | 0 | 7 | 8 |
| latest_recent | -0.200 | 0 | 3 | 1 | 11 |
| negation_temporal | +0.000 | 0 | 0 | 0 | 15 |

### Detailed losses on regressing benches (v5 missed top1, lblend hit, up to 5)

#### latest_recent  (Δ = -0.200)
- `lr_q_005` (n_q_tes=1): v5_top1=`lr_005_d3`, v5_rank=4, gold=['lr_005_g']
  - q: When did I last renew the office lease?
- `lr_q_011` (n_q_tes=1): v5_top1=`lr_011_d3`, v5_rank=2, gold=['lr_011_g']
  - q: When was my most recent blood test?
- `lr_q_012` (n_q_tes=1): v5_top1=`lr_012_d1`, v5_rank=2, gold=['lr_012_g']
  - q: When was the last pull request I reviewed from the platform team?

#### era_refs  (Δ = -0.083)
- `era_q_006` (n_q_tes=1): v5_top1=`era_006_d3`, v5_rank=2, gold=['era_006_g']
  - q: When did Henry Ford start learning piano during my parental leave?
