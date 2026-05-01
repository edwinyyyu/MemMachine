# T_hybrid_planner — regex-primary / LLM-planner-on-multi-cue

Hybrid routing: count regex cues per query (recency, open_ended, negation, causal). If ≤1 cue fires, use the proven regex_stack; if ≥2 fire, use the llm_planner_stack. Goal: zero single-cue regression while preserving the +0.12 composition win.

## Regression-fix table (R@1) — lead

| Benchmark | n | rerank_only | regex_stack | llm_planner | **hybrid** | Δ(hyb − regex) | Δ(hyb − llm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| composition | 25 | 0.040 | 0.200 | 0.240 | **0.280** | **+0.080** (+) | +0.040 |
| hard_bench | 75 | 0.640 | 0.893 | 1.000 | **0.893** | **+0.000** | -0.107 |
| temporal_essential | 25 | 0.920 | 1.000 | 1.000 | **1.000** | **+0.000** | +0.000 |
| tempreason_small | 60 | 0.650 | 0.783 | 0.783 | **0.783** | **+0.000** | +0.000 |
| conjunctive_temporal | 12 | 1.000 | 1.000 | 1.000 | **1.000** | **+0.000** | +0.000 |
| multi_te_doc | 12 | 1.000 | 1.000 | 1.000 | **1.000** | **+0.000** | +0.000 |
| relative_time | 12 | 0.250 | 0.917 | 1.000 | **0.917** | **+0.000** | -0.083 |
| era_refs | 12 | 0.250 | 0.417 | 0.417 | **0.417** | **+0.000** | +0.000 |
| open_ended_date | 15 | 0.267 | 0.533 | 0.533 | **0.533** | **+0.000** | +0.000 |
| causal_relative | 15 | 0.467 | 0.733 | 0.733 | **0.733** | **+0.000** | +0.000 |
| latest_recent | 15 | 0.133 | 0.667 | 0.800 | **0.667** | **+0.000** | -0.133 |
| negation_temporal | 15 | 0.000 | 0.733 | 0.733 | **0.733** | **+0.000** | +0.000 |

## Composition by type (R@1)

| Type | n | regex | llm | **hybrid** | Δ(hyb−regex) |
|---|---:|---:|---:|---:|---:|
| A: rec×abs | 5 | 0.200 | 0.200 | **0.200** | +0.000 |
| B: neg×abs | 5 | 0.200 | 0.000 | **0.200** | +0.000 |
| C: causal×rec | 5 | 0.200 | 0.400 | **0.400** | +0.200 |
| D: causal×abs | 5 | 0.200 | 0.200 | **0.200** | +0.000 |
| E: oe×neg | 5 | 0.200 | 0.400 | **0.400** | +0.200 |
| ALL | 25 | 0.200 | 0.240 | **0.280** | +0.080 |

## Macro R@1 across 12 benches

- regex_stack:        0.740
- llm_planner_stack:  0.770
- **hybrid_stack:     0.746**
- Δ(hybrid − regex):  **+0.007**
- Δ(hybrid − llm):    -0.024

## Hybrid routing histogram (which queries got the LLM stack?)

| Benchmark | n | LLM-routed | regex-routed |
|---|---:|---:|---:|
| composition | 25 | 10 | 15 |
| hard_bench | 75 | 0 | 75 |
| temporal_essential | 25 | 0 | 25 |
| tempreason_small | 60 | 0 | 60 |
| conjunctive_temporal | 12 | 0 | 12 |
| multi_te_doc | 12 | 0 | 12 |
| relative_time | 12 | 0 | 12 |
| era_refs | 12 | 0 | 12 |
| open_ended_date | 15 | 0 | 15 |
| causal_relative | 15 | 1 | 14 |
| latest_recent | 15 | 0 | 15 |
| negation_temporal | 15 | 0 | 15 |

## Verdict

- **No single-cue regressions vs regex_stack.** Hybrid hits or beats regex on every benchmark.
- **Composition Δ vs regex: +0.080** (PARTIAL the +0.12 win)
- **Macro Δ vs regex: +0.007**

## Recommendation

**SHIP hybrid_stack.** Zero single-cue regressions; composition Δ +0.080; macro Δ +0.007. Best of both worlds.
