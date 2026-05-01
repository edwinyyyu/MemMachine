# T_llm_planner — LLM-based structured query planner

Replace the regex parsers (recency / negation / open_ended / causal) with a single gpt-5-mini call returning a structured QueryPlan. Composition is multiplicative (S2 from prior eval). New: `absolute_anchor` adds a hard interval filter on the doc set.

## Composition R@1 — lead

| Type | n | rerank_only | regex_stack | **llm_planner_stack** | Δ(llm − regex) |
|---|---:|---:|---:|---:|---:|
| A: recency × absolute | 5 | 0.200 (1/5) | 0.200 (1/5) | **0.200** (1/5) | **+0.000** |
| B: negation × absolute | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.000** (0/5) | **-0.200** |
| C: causal × recency | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.400** (2/5) | **+0.200** |
| D: causal × absolute | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.200** (1/5) | **+0.000** |
| E: open_ended × negation | 5 | 0.000 (0/5) | 0.200 (1/5) | **0.400** (2/5) | **+0.200** |
| ALL | 25 | 0.040 (1/25) | 0.200 (5/25) | **0.240** (6/25) | **+0.040** |

## Single-cue regression check (R@1)

| Benchmark | n | rerank_only | regex_stack | llm_planner_stack | Δ(llm − regex) |
|---|---:|---:|---:|---:|---:|
| hard_bench | 75 | 0.640 (48/75) | 0.893 (67/75) | 1.000 (75/75) | **+0.107** (llm better) |
| temporal_essential | 25 | 0.920 (23/25) | 1.000 (25/25) | 1.000 (25/25) | **+0.000** |
| tempreason_small | 60 | 0.650 (39/60) | 0.783 (47/60) | 0.783 (47/60) | **+0.000** |
| conjunctive_temporal | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | **+0.000** |
| multi_te_doc | 12 | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | **+0.000** |
| relative_time | 12 | 0.250 (3/12) | 0.917 (11/12) | 1.000 (12/12) | **+0.083** (llm better) |
| era_refs | 12 | 0.250 (3/12) | 0.417 (5/12) | 0.417 (5/12) | **+0.000** |
| open_ended_date | 15 | 0.267 (4/15) | 0.533 (8/15) | 0.533 (8/15) | **+0.000** |
| causal_relative | 15 | 0.467 (7/15) | 0.733 (11/15) | 0.733 (11/15) | **+0.000** |
| latest_recent | 15 | 0.133 (2/15) | 0.667 (10/15) | 0.800 (12/15) | **+0.133** (llm better) |
| negation_temporal | 15 | 0.000 (0/15) | 0.733 (11/15) | 0.733 (11/15) | **+0.000** |

### Macro-average R@1 across 12 benches

- rerank_only: 0.468
- regex_stack: 0.740
- **llm_planner_stack**: **0.770**
- Δ(llm − regex): **+0.030**

## Per-composition-type fix audit

| Type | n | regex misses | llm fixes | llm breaks | Net |
|---|---:|---:|---:|---:|---:|
| A (recency × absolute) | 5 | 4 | 0 | 0 | **+0** |
| B (negation × absolute) | 5 | 4 | 0 | 1 | **-1** |
| C (causal × recency) | 5 | 4 | 1 | 0 | **+1** |
| D (causal × absolute) | 5 | 4 | 0 | 0 | **+0** |
| E (open_ended × negation) | 5 | 4 | 1 | 0 | **+1** |

## Cost

- Planner model: `gpt-5-mini`
- Total queries seen: 293
- Live LLM calls: 0
- Cache hits: 293
- Cache hit rate: **100.0%**
- Parse failures: 0
- LLM calls per UNIQUE query: 1 (single planner call); cache eliminates rerun cost.

## Failure analysis (composition queries the LLM stack still misses)

### comp_q_A_000 — Type A, llm rank=2
- Query: `What's my latest project Alpha update from Q4 2023?`
- Gold: `['comp_A_000_g0']`
- Plan: `{"absolute_anchor": "Q4 2023", "open_ended": null, "recency_intent": true, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=True neg=False oe=False causal=False abs=True
- llm top5: `['comp_A_000_d2', 'comp_A_000_g0', 'comp_C_011_d2', 'comp_C_012_n0', 'comp_A_003_g0']`
- regex top5 (regex_rank=2): `['comp_A_000_d2', 'comp_A_000_g0', 'comp_B_007_n0', 'comp_A_000_d0', 'comp_C_014_g0']`

### comp_q_A_001 — Type A, llm rank=2
- Query: `Most recent meeting in March 2024`
- Gold: `['comp_A_001_g0']`
- Plan: `{"absolute_anchor": "March 2024", "open_ended": null, "recency_intent": true, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=True neg=False oe=False causal=False abs=True
- llm top5: `['comp_A_001_d2', 'comp_A_001_g0', 'comp_B_005_g0', 'comp_D_018_d2', 'comp_D_017_g0']`
- regex top5 (regex_rank=2): `['comp_A_001_d2', 'comp_A_001_g0', 'comp_B_005_g0', 'comp_B_007_n0', 'comp_A_001_d0']`

### comp_q_A_002 — Type A, llm rank=2
- Query: `My latest budget review in Q2 2024`
- Gold: `['comp_A_002_g0']`
- Plan: `{"absolute_anchor": "Q2 2024", "open_ended": null, "recency_intent": true, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=True neg=False oe=False causal=False abs=True
- llm top5: `['comp_A_002_d2', 'comp_A_002_g0', 'comp_C_010_d1', 'comp_B_006_g0', 'comp_C_013_d2']`
- regex top5 (regex_rank=2): `['comp_A_002_d2', 'comp_A_002_g0', 'comp_B_007_n0', 'comp_C_014_g0', 'comp_A_001_d0']`

### comp_q_A_003 — Type A, llm rank=5
- Query: `The most recent design review in 2023`
- Gold: `['comp_A_003_g0']`
- Plan: `{"absolute_anchor": "2023", "open_ended": null, "recency_intent": true, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=True neg=False oe=False causal=False abs=True
- llm top5: `['comp_B_006_d0', 'comp_B_006_d1', 'comp_A_003_d2', 'comp_B_006_d2', 'comp_A_003_g0']`
- regex top5 (regex_rank=9): `['comp_B_007_n0', 'comp_A_001_d0', 'comp_B_006_d0', 'comp_C_013_g0', 'comp_B_006_d1']`

### comp_q_B_005 — Type B, llm rank=3
- Query: `What client meetings did I have in 2024 not in summer?`
- Gold: `['comp_B_005_g0']`
- Plan: `{"absolute_anchor": "2024", "open_ended": null, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "summer"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=False causal=False abs=True
- llm top5: `['comp_B_005_d0', 'comp_B_005_d1', 'comp_B_005_g0', 'comp_C_014_d2', 'comp_C_014_a']`
- regex top5 (regex_rank=3): `['comp_B_005_d0', 'comp_B_005_d1', 'comp_B_005_g0', 'comp_B_005_d2', 'comp_C_014_d2']`

### comp_q_B_006 — Type B, llm rank=None
- Query: `Meetings outside Q3 2023`
- Gold: `['comp_B_006_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": null, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "Q3 2023"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=False causal=False abs=False
- llm top5: `['comp_A_001_d0', 'comp_C_014_g0', 'comp_D_019_d2', 'comp_C_012_n0', 'comp_C_014_d0']`
- regex top5 (regex_rank=None): `['comp_A_001_d2', 'comp_B_005_g0', 'comp_B_005_d2', 'comp_B_005_d0', 'comp_B_005_d1']`

### comp_q_B_007 — Type B, llm rank=3
- Query: `What classes did I take in 2024 excluding the spring semester?`
- Gold: `['comp_B_007_g0']`
- Plan: `{"absolute_anchor": "2024", "open_ended": null, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "spring semester"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=False causal=False abs=True
- llm top5: `['comp_B_007_d1', 'comp_B_007_d0', 'comp_B_007_g0', 'comp_D_015_n0', 'comp_E_021_d0']`
- regex top5 (regex_rank=2): `['comp_B_007_d1', 'comp_B_007_g0', 'comp_B_007_d0', 'comp_B_007_d2', 'comp_C_014_d0']`

### comp_q_B_008 — Type B, llm rank=None
- Query: `What did I do in 2025 not in January?`
- Gold: `['comp_B_008_g0']`
- Plan: `{"absolute_anchor": "2025", "open_ended": null, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "January"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=False causal=False abs=True
- llm top5: `['comp_C_014_g0', 'comp_E_024_d0', 'comp_E_024_d2', 'comp_E_024_d1', 'comp_A_000_d1']`
- regex top5 (regex_rank=None): `['comp_C_014_g0', 'comp_E_021_g0', 'comp_A_004_d0', 'comp_A_001_d0', 'comp_E_024_g0']`

### comp_q_B_009 — Type B, llm rank=3
- Query: `Trips I took in 2023 outside of December`
- Gold: `['comp_B_009_g0']`
- Plan: `{"absolute_anchor": "2023", "open_ended": null, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "December"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=False causal=False abs=True
- llm top5: `['comp_B_009_d0', 'comp_B_009_d1', 'comp_B_009_g0', 'comp_D_015_d1', 'comp_A_003_g0']`
- regex top5 (regex_rank=1): `['comp_B_009_g0', 'comp_B_009_d1', 'comp_B_009_d0', 'comp_B_009_d2', 'comp_A_002_n0']`

### comp_q_C_010 — Type C, llm rank=3
- Query: `My most recent update after the migration`
- Gold: `['comp_C_010_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": null, "recency_intent": true, "earliest_intent": false, "negation": null, "causal": {"anchor_phrase": "the migration", "direction": "after"}, "cyclical_intent": false}`
- LLM cues: rec=True neg=False oe=False causal=True abs=False
- LLM resolved causal anchor: `comp_D_016_a`
- llm top5: `['comp_C_012_a', 'comp_C_010_a', 'comp_C_010_g0', 'comp_C_010_d1', 'comp_A_000_d1']`
- regex top5 (regex_rank=2): `['comp_C_012_a', 'comp_C_010_g0', 'comp_B_007_n0', 'comp_C_010_a', 'comp_C_013_g0']`

### comp_q_C_011 — Type C, llm rank=None
- Query: `The latest thing I did since the launch`
- Gold: `['comp_C_011_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": null, "recency_intent": true, "earliest_intent": false, "negation": null, "causal": {"anchor_phrase": "the launch", "direction": "after"}, "cyclical_intent": false}`
- LLM cues: rec=True neg=False oe=False causal=True abs=False
- LLM resolved causal anchor: `comp_D_015_a`
- llm top5: `['comp_D_015_d2', 'comp_C_011_a', 'comp_D_015_g0', 'comp_C_011_d1', 'comp_A_000_d2']`
- regex top5 (regex_rank=None): `['comp_D_015_d2', 'comp_B_007_n0', 'comp_C_011_a', 'comp_C_013_g0', 'comp_D_015_g0']`

### comp_q_C_014 — Type C, llm rank=10
- Query: `Latest meeting I had after the offsite`
- Gold: `['comp_C_014_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": null, "recency_intent": true, "earliest_intent": false, "negation": null, "causal": {"anchor_phrase": "the offsite", "direction": "after"}, "cyclical_intent": false}`
- LLM cues: rec=True neg=False oe=False causal=True abs=False
- LLM resolved causal anchor: `comp_C_014_a`
- llm top5: `['comp_C_014_d1', 'comp_A_001_d0', 'comp_A_001_d1', 'comp_C_014_d2', 'comp_E_024_d1']`
- regex top5 (regex_rank=5): `['comp_C_014_d1', 'comp_B_007_n0', 'comp_C_013_g0', 'comp_A_001_d0', 'comp_C_014_g0']`

### comp_q_D_015 — Type D, llm rank=3
- Query: `What did I do in Q3 2023 after the launch?`
- Gold: `['comp_D_015_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": {"side": "after", "anchor": "Q3 2023"}, "recency_intent": false, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=False oe=True causal=False abs=False
- llm top5: `['comp_D_015_a', 'comp_D_015_d0', 'comp_D_015_g0', 'comp_A_001_d0', 'comp_D_019_d2']`
- regex top5 (regex_rank=3): `['comp_D_015_a', 'comp_D_015_d0', 'comp_D_015_g0', 'comp_A_001_d0', 'comp_D_019_d2']`

### comp_q_D_016 — Type D, llm rank=None
- Query: `What happened in 2024 after the migration?`
- Gold: `['comp_D_016_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": {"side": "after", "anchor": "2024"}, "recency_intent": false, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=False oe=True causal=False abs=False
- llm top5: `['comp_D_016_a', 'comp_C_010_a', 'comp_D_019_a', 'comp_B_009_d2', 'comp_C_011_a']`
- regex top5 (regex_rank=None): `['comp_D_016_a', 'comp_C_010_a', 'comp_D_019_a', 'comp_B_009_d2', 'comp_C_011_a']`

### comp_q_D_017 — Type D, llm rank=2
- Query: `What I worked on in March 2024 after the freeze`
- Gold: `['comp_D_017_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": {"side": "after", "anchor": "March 2024"}, "recency_intent": false, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=False oe=True causal=False abs=False
- llm top5: `['comp_D_017_a', 'comp_D_017_g0', 'comp_D_017_d0', 'comp_D_018_d2', 'comp_B_005_g0']`
- regex top5 (regex_rank=2): `['comp_D_017_a', 'comp_D_017_g0', 'comp_D_017_d0', 'comp_D_018_d2', 'comp_B_005_g0']`

### comp_q_D_019 — Type D, llm rank=6
- Query: `What I did in May 2024 since the kickoff`
- Gold: `['comp_D_019_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": {"side": "after", "anchor": "May 2024"}, "recency_intent": false, "earliest_intent": false, "negation": null, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=False oe=True causal=False abs=False
- llm top5: `['comp_D_019_a', 'comp_D_019_d0', 'comp_C_011_a', 'comp_D_017_d1', 'comp_C_011_d0']`
- regex top5 (regex_rank=6): `['comp_D_019_a', 'comp_D_019_d0', 'comp_C_011_a', 'comp_D_017_d1', 'comp_C_011_d0']`

### comp_q_E_020 — Type E, llm rank=None
- Query: `What did I do after 2020 but not in 2023?`
- Gold: `['comp_E_020_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": {"side": "after", "anchor": "2020"}, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "2023"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=True causal=False abs=False
- llm top5: `['comp_A_004_d2', 'comp_E_024_d0', 'comp_A_004_g0', 'comp_A_004_d1', 'comp_A_004_d0']`
- regex top5 (regex_rank=None): `['comp_E_021_d3', 'comp_A_003_n0', 'comp_E_021_d0', 'comp_E_021_g0', 'comp_E_021_d2']`

### comp_q_E_022 — Type E, llm rank=None
- Query: `Things I did before 2024 not in summer 2022`
- Gold: `['comp_E_022_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": {"side": "before", "anchor": "2024"}, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "summer 2022"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=True causal=False abs=False
- llm top5: `['comp_E_021_d1', 'comp_D_017_d2', 'comp_A_004_d2', 'comp_B_009_d2', 'comp_E_021_d0']`
- regex top5 (regex_rank=None): `['comp_B_009_d1', 'comp_D_015_d0', 'comp_A_003_n0', 'comp_B_009_d0', 'comp_E_021_n0']`

### comp_q_E_023 — Type E, llm rank=None
- Query: `What I did since 2022 outside of Q1 2023`
- Gold: `['comp_E_023_g0']`
- Plan: `{"absolute_anchor": null, "open_ended": {"side": "since", "anchor": "2022"}, "recency_intent": false, "earliest_intent": false, "negation": {"excluded_phrase": "Q1 2023"}, "causal": null, "cyclical_intent": false}`
- LLM cues: rec=False neg=True oe=True causal=False abs=False
- llm top5: `['comp_A_000_d1', 'comp_A_001_d0', 'comp_C_014_g0', 'comp_D_019_d2', 'comp_D_018_g0']`
- regex top5 (regex_rank=None): `['comp_C_011_n0', 'comp_A_003_n0', 'comp_B_005_n0', 'comp_C_014_g0', 'comp_B_006_n0']`

## Recommendation

**STAY WITH PATCHED REGEX.** Composition Δ +0.040; single-cue Δ +0.030. Lift is too small to justify the extra LLM call; better ROI is patching parse_negation_query (Type B) and the open_ended/causal gate conflict (Type D).
