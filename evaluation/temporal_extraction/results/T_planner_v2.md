# T_planner_v2 — simplified TimeWindow planner

Schema collapse: absolute_anchor + open_ended + negation -> single TimeWindow list with op/open_lower/open_upper. Drops causal (multi-hop), drops earliest_intent (folded into extremum), drops normalize_plan post-processor (no more absolute_anchor leak).

## Composition R@1 — lead

| Type | n | rerank_only | regex_stack | **planner_v2_stack** | Δ(planner_v2 − regex) |
|---|---:|---:|---:|---:|---:|
| A: recency × absolute | 5 | 0.200 (1/5) | 0.200 (1/5) | **0.800** (4/5) | **+0.600** |
| B: negation × absolute | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.000** (0/5) | **-0.200** |
| C: causal × recency | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.200** (1/5) | **+0.000** |
| D: causal × absolute | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.200** (1/5) | **+0.000** |
| E: open_ended × negation | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.400** (2/5) | **+0.200** |
| ALL | 25 | 0.040 (1/25) | 0.200 (5/25) | **0.320** (8/25) | **+0.120** |

## Single-cue regression check (R@1)

| Benchmark | n | rerank_only | regex_stack | planner_v2_stack | Δ(p2 − regex) |
|---|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0.640 (48/75) | 0.893 (67/75) | 1.000 (75/75) | **+0.107** (+) |
| temporal_essential | 25 | 0.920 (23/25) | 1.000 (25/25) | 1.000 (25/25) | **+0.000** |
| tempreason_small | 60 | 0.650 (39/60) | 0.783 (47/60) | 0.733 (44/60) | **-0.050** (regress!) |
| conjunctive_temporal | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | **+0.000** |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | **+0.000** |
| relative_time | 12 | 0.250 (3/12) | 0.917 (11/12) | 0.833 (10/12) | **-0.083** (regress!) |
| era_refs | 12 | 0.250 (3/12) | 0.417 (5/12) | 0.333 (4/12) | **-0.083** (regress!) |
| open_ended_date | 15 | 0.267 (4/15) | 0.533 (8/15) | 0.733 (11/15) | **+0.200** (+) |
| causal_relative | 15 | 0.467 (7/15) | 0.733 (11/15) | 0.467 (7/15) | **-0.267** (regress!) |
| latest_recent | 15 | 0.133 (2/15) | 0.667 (10/15) | 0.800 (12/15) | **+0.133** (+) |
| negation_temporal | 15 | 0.000 (0/15) | 0.733 (11/15) | 0.733 (11/15) | **+0.000** |

### Macro-average R@1 across 12 benches

- rerank_only: 0.468
- regex_stack: 0.740
- **planner_v2_stack**: **0.746**
- Δ(p2 − regex): **+0.006**

## Planner cost

- Model: `gpt-5-mini`
- Total queries: 293
- Live calls: 293
- Cache hits: 0
- Hit rate: 0.0%
- Parse failures: 0

## Failure analysis

### composition

#### comp_q_A_004 — Type A, p2 rank=2
- Query: `Latest workout I did in January 2025`
- Gold: `['comp_A_004_g0']`
- Plan: `{"constraints": [{"phrase": "January 2025", "direction": "in"}], "extremum": "latest"}`
- p2 top5: `['comp_B_008_d0', 'comp_A_004_g0', 'comp_A_004_d2', 'comp_A_002_d1', 'comp_B_008_d1']`
- regex top5 (regex_rank=1): `['comp_A_004_g0', 'comp_A_004_d2', 'comp_B_007_n0', 'comp_C_013_g0', 'comp_A_004_d0']`

#### comp_q_B_005 — Type B, p2 rank=3
- Query: `What client meetings did I have in 2024 not in summer?`
- Gold: `['comp_B_005_g0']`
- Plan: `{"constraints": [{"phrase": "2024", "direction": "in"}, {"phrase": "summer", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_B_005_d0', 'comp_B_005_d1', 'comp_B_005_g0', 'comp_C_014_d2', 'comp_C_014_a']`
- regex top5 (regex_rank=3): `['comp_B_005_d0', 'comp_B_005_d1', 'comp_B_005_g0', 'comp_B_005_d2', 'comp_C_014_d2']`

#### comp_q_B_006 — Type B, p2 rank=None
- Query: `Meetings outside Q3 2023`
- Gold: `['comp_B_006_g0']`
- Plan: `{"constraints": [{"phrase": "Q3 2023", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_A_001_d0', 'comp_C_014_g0', 'comp_D_019_d2', 'comp_C_012_n0', 'comp_C_014_d0']`
- regex top5 (regex_rank=None): `['comp_A_001_d2', 'comp_B_005_g0', 'comp_B_005_d2', 'comp_B_005_d0', 'comp_B_005_d1']`

#### comp_q_B_007 — Type B, p2 rank=3
- Query: `What classes did I take in 2024 excluding the spring semester?`
- Gold: `['comp_B_007_g0']`
- Plan: `{"constraints": [{"phrase": "2024", "direction": "in"}, {"phrase": "spring semester", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_B_007_d1', 'comp_B_007_d0', 'comp_B_007_g0', 'comp_D_015_n0', 'comp_E_021_d0']`
- regex top5 (regex_rank=2): `['comp_B_007_d1', 'comp_B_007_g0', 'comp_B_007_d0', 'comp_B_007_d2', 'comp_C_014_d0']`

#### comp_q_B_008 — Type B, p2 rank=None
- Query: `What did I do in 2025 not in January?`
- Gold: `['comp_B_008_g0']`
- Plan: `{"constraints": [{"phrase": "2025", "direction": "in"}, {"phrase": "January", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_C_014_g0', 'comp_E_024_d0', 'comp_E_024_d2', 'comp_E_024_d1', 'comp_A_000_d1']`
- regex top5 (regex_rank=None): `['comp_C_014_g0', 'comp_E_021_g0', 'comp_A_004_d0', 'comp_A_001_d0', 'comp_E_024_g0']`

#### comp_q_B_009 — Type B, p2 rank=3
- Query: `Trips I took in 2023 outside of December`
- Gold: `['comp_B_009_g0']`
- Plan: `{"constraints": [{"phrase": "2023", "direction": "in"}, {"phrase": "December", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_B_009_d0', 'comp_B_009_d1', 'comp_B_009_g0', 'comp_D_015_d1', 'comp_A_003_g0']`
- regex top5 (regex_rank=1): `['comp_B_009_g0', 'comp_B_009_d1', 'comp_B_009_d0', 'comp_B_009_d2', 'comp_A_002_n0']`

#### comp_q_C_010 — Type C, p2 rank=4
- Query: `My most recent update after the migration`
- Gold: `['comp_C_010_g0']`
- Plan: `{"constraints": [], "extremum": "latest"}`
- p2 top5: `['comp_C_012_a', 'comp_D_016_a', 'comp_C_010_a', 'comp_C_010_g0', 'comp_A_000_d1']`
- regex top5 (regex_rank=2): `['comp_C_012_a', 'comp_C_010_g0', 'comp_B_007_n0', 'comp_C_010_a', 'comp_C_013_g0']`

#### comp_q_C_011 — Type C, p2 rank=None
- Query: `The latest thing I did since the launch`
- Gold: `['comp_C_011_g0']`
- Plan: `{"constraints": [], "extremum": "latest"}`
- p2 top5: `['comp_D_015_d2', 'comp_C_011_a', 'comp_C_011_d1', 'comp_D_015_a', 'comp_D_015_g0']`
- regex top5 (regex_rank=None): `['comp_D_015_d2', 'comp_B_007_n0', 'comp_C_011_a', 'comp_C_013_g0', 'comp_D_015_g0']`

#### comp_q_C_013 — Type C, p2 rank=3
- Query: `Most recent change since the redesign shipped`
- Gold: `['comp_C_013_g0']`
- Plan: `{"constraints": [], "extremum": "latest"}`
- p2 top5: `['comp_C_013_a', 'comp_C_012_a', 'comp_C_013_g0', 'comp_C_013_d1', 'comp_C_013_d0']`
- regex top5 (regex_rank=2): `['comp_B_007_n0', 'comp_C_013_g0', 'comp_C_010_g0', 'comp_A_001_d0', 'comp_C_014_g0']`

#### comp_q_C_014 — Type C, p2 rank=None
- Query: `Latest meeting I had after the offsite`
- Gold: `['comp_C_014_g0']`
- Plan: `{"constraints": [], "extremum": "latest"}`
- p2 top5: `['comp_C_014_d1', 'comp_C_014_d2', 'comp_C_014_a', 'comp_A_001_d0', 'comp_A_001_d1']`
- regex top5 (regex_rank=5): `['comp_C_014_d1', 'comp_B_007_n0', 'comp_C_013_g0', 'comp_A_001_d0', 'comp_C_014_g0']`

#### comp_q_D_015 — Type D, p2 rank=2
- Query: `What did I do in Q3 2023 after the launch?`
- Gold: `['comp_D_015_g0']`
- Plan: `{"constraints": [{"phrase": "Q3 2023", "direction": "in"}], "extremum": null}`
- p2 top5: `['comp_D_015_a', 'comp_D_015_g0', 'comp_D_015_d0', 'comp_A_001_n0', 'comp_D_016_d2']`
- regex top5 (regex_rank=3): `['comp_D_015_a', 'comp_D_015_d0', 'comp_D_015_g0', 'comp_A_001_d0', 'comp_D_019_d2']`

#### comp_q_D_016 — Type D, p2 rank=None
- Query: `What happened in 2024 after the migration?`
- Gold: `['comp_D_016_g0']`
- Plan: `{"constraints": [{"phrase": "2024", "direction": "in"}], "extremum": null}`
- p2 top5: `['comp_D_016_a', 'comp_C_010_a', 'comp_D_019_a', 'comp_D_019_d2', 'comp_D_018_d1']`
- regex top5 (regex_rank=None): `['comp_D_016_a', 'comp_C_010_a', 'comp_D_019_a', 'comp_B_009_d2', 'comp_C_011_a']`

#### comp_q_D_017 — Type D, p2 rank=2
- Query: `What I worked on in March 2024 after the freeze`
- Gold: `['comp_D_017_g0']`
- Plan: `{"constraints": [{"phrase": "March 2024", "direction": "in"}], "extremum": null}`
- p2 top5: `['comp_D_017_a', 'comp_D_017_g0', 'comp_D_017_d0', 'comp_D_018_d2', 'comp_B_005_g0']`
- regex top5 (regex_rank=2): `['comp_D_017_a', 'comp_D_017_g0', 'comp_D_017_d0', 'comp_D_018_d2', 'comp_B_005_g0']`

#### comp_q_D_019 — Type D, p2 rank=6
- Query: `What I did in May 2024 since the kickoff`
- Gold: `['comp_D_019_g0']`
- Plan: `{"constraints": [{"phrase": "May 2024", "direction": "in"}], "extremum": null}`
- p2 top5: `['comp_D_019_a', 'comp_D_019_d0', 'comp_C_011_a', 'comp_D_017_d1', 'comp_C_011_d0']`
- regex top5 (regex_rank=6): `['comp_D_019_a', 'comp_D_019_d0', 'comp_C_011_a', 'comp_D_017_d1', 'comp_C_011_d0']`

#### comp_q_E_020 — Type E, p2 rank=None
- Query: `What did I do after 2020 but not in 2023?`
- Gold: `['comp_E_020_g0']`
- Plan: `{"constraints": [{"phrase": "2020", "direction": "after"}, {"phrase": "2023", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_A_004_d2', 'comp_E_024_d0', 'comp_A_004_g0', 'comp_A_004_d1', 'comp_A_004_d0']`
- regex top5 (regex_rank=None): `['comp_E_021_d3', 'comp_A_003_n0', 'comp_E_021_d0', 'comp_E_021_g0', 'comp_E_021_d2']`

#### comp_q_E_022 — Type E, p2 rank=None
- Query: `Things I did before 2024 not in summer 2022`
- Gold: `['comp_E_022_g0']`
- Plan: `{"constraints": [{"phrase": "2024", "direction": "before"}, {"phrase": "summer 2022", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_B_009_g0', 'comp_A_001_n0', 'comp_B_005_n0', 'comp_B_006_n0', 'comp_D_016_n0']`
- regex top5 (regex_rank=None): `['comp_B_009_d1', 'comp_D_015_d0', 'comp_A_003_n0', 'comp_B_009_d0', 'comp_E_021_n0']`

#### comp_q_E_023 — Type E, p2 rank=None
- Query: `What I did since 2022 outside of Q1 2023`
- Gold: `['comp_E_023_g0']`
- Plan: `{"constraints": [{"phrase": "2022", "direction": "after"}, {"phrase": "Q1 2023", "direction": "not_in"}], "extremum": null}`
- p2 top5: `['comp_A_000_d1', 'comp_A_001_d0', 'comp_C_014_g0', 'comp_D_019_d2', 'comp_D_018_g0']`
- regex top5 (regex_rank=None): `['comp_C_011_n0', 'comp_A_003_n0', 'comp_B_005_n0', 'comp_C_014_g0', 'comp_B_006_n0']`
