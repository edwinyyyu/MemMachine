# T_llm_planner_fixed — open_ended_date regression eliminated

## Regression-fix lead row

| Bench | regex_stack | llm_planner_BEFORE | llm_planner_FIXED | Δ(fixed − regex) |
|---|---:|---:|---:|---:|
| **open_ended_date** | 0.533 | 0.267 (REGRESS −0.267) | **0.533** | **+0.000** (regression eliminated) |
| **tempreason_small** | 0.783 | 0.733 (REGRESS −0.050) | **0.783** | **+0.000** (regression eliminated) |

**No regressions remain anywhere.** macro R@1 = 0.767 (regex 0.740, +0.027).

## Diagnosis

Two distinct root causes, both surfaced by inspecting `cache/planner/llm_plan_cache.json` against `data/open_ended_date_queries.jsonl` and the `causal_llm`/`abs_active_llm` flags in `results/T_llm_planner.json`:

1. **(open_ended_date) Schema-leakage on event-attached date bounds.** gpt-5-mini ignored prompt rule 3 (open_ended ⊥ causal). On 7/15 open_ended queries the plan set BOTH `open_ended` (correct) and `causal` with the embedded event phrase (e.g. "after I joined Acme in March 2022" → `open_ended.anchor="March 2022"` AND `causal={"I joined Acme","before"}`). The downstream multiplicative `dir_match` mask then zeroed the correct docs. On 4/15 queries the plan inverted entirely: `absolute_anchor=<date>` and `open_ended=null` (e.g. "after my child was born in August 2023" → `absolute_anchor="August 2023"`). The hard-window absolute filter then masked the gold which is *outside* that interval by construction.

2. **(tempreason_small) Multiplicative causal mask is too aggressive vs signed-additive.** On 3/60 entity-anchored queries ("Who was the chair … before Lawrence Eagleburger?") regex and LLM agreed on activations and resolved the same anchor doc, but `regex_stack`'s `causal_signed_scores` (additive λ=0.5 penalty) found the gold while `llm_planner_stack`'s `dir_factor`-multiplier (`anchor→0.0`, `wrong-side×0.1`) collapsed the fuzzy ranking and dropped gold below the cutoff.

## Fix

Two surgical changes to `composition_eval_v2.py` (no prompt change, no cache invalidation needed beyond one truncated entry):

1. **`normalize_plan(plan, query_text)` post-processor.** (a) If `has_open_ended_cue(query)` regex fires AND the plan has `absolute_anchor` but no `open_ended`, flip: parse the side keyword from the query, move the date phrase from `absolute_anchor` to `open_ended.anchor`, zero `absolute_anchor`. (b) If both `open_ended` and `causal` are set, drop `causal` (prompt rule 3 mutual exclusion).
2. **Replace causal multiplier with `causal_signed_scores`** in `llm_planner_stack` so it uses the same signed-additive composition the regex stack uses. The other dimensions (recency, negation, absolute_anchor) keep multiplicative composition.

Also bumped `max_output_tokens` 2048→3072 in `query_planner.py` and dropped 1 truncated cache entry.

## Full R@1 table (post-fix)

| Benchmark | n | rerank_only | regex_stack | llm_planner_FIXED | Δ vs regex |
|---|---:|---:|---:|---:|---:|
| composition | 25 | 0.040 | 0.200 | 0.200 | +0.000 |
| hard_bench | 75 | 0.640 | 0.893 | **1.000** | **+0.107** |
| temporal_essential | 25 | 0.920 | 1.000 | 1.000 | +0.000 |
| tempreason_small | 60 | 0.650 | 0.783 | 0.783 | +0.000 |
| conjunctive_temporal | 12 | 1.000 | 1.000 | 1.000 | +0.000 |
| multi_te_doc | 12 | 1.000 | 1.000 | 1.000 | +0.000 |
| relative_time | 12 | 0.250 | 0.917 | **1.000** | **+0.083** |
| era_refs | 12 | 0.250 | 0.417 | 0.417 | +0.000 |
| open_ended_date | 15 | 0.267 | 0.533 | 0.533 | +0.000 |
| causal_relative | 15 | 0.467 | 0.733 | 0.733 | +0.000 |
| latest_recent | 15 | 0.133 | 0.667 | **0.800** | **+0.133** |
| negation_temporal | 15 | 0.000 | 0.733 | 0.733 | +0.000 |
| **MACRO** | — | 0.468 | 0.740 | **0.767** | **+0.027** |

## Verification

- **Zero regressions vs regex_stack**: every cell ≥ regex. Confirmed.
- **LLM-planner wins preserved**: hard_bench 0.893→1.000, latest_recent 0.667→0.800, relative_time 0.917→1.000. All three preserved.
- **Composition win partially lost**: 0.200 (was 0.320 with the multiplicative-causal version). Trade-off — recovering tempreason via signed-additive gave back 0.05pp there at the cost of 0.12pp on composition. The composition win lived inside the multiplicative formula's hard-zero of the anchor doc; replacing it with `λ=0.5` signed penalty surrenders some Type B/C edge cases. **Composition is still ≥ regex (= 0.200), so no regression.**

## Macro R@1

- regex_stack: **0.740**
- llm_planner_FIXED: **0.767**
- Δ: **+0.027** macro lift, zero regressions.

## What's next (optional)

The remaining 0.12pp composition headroom is recoverable by routing causal composition by query shape:
- Signed-additive (current) when causal fires alone or with open_ended/absolute (entity-anchored or date-bound questions).
- Multiplicative hard-zero when causal fires with `recency_intent` (the original Type C win). The recency-causal queries are exactly where multiplicative dominated.

A two-line dispatch (`if causal and rec_active_llm and not (oe or abs): use multiplicative else signed-additive`) would likely restore composition to 0.32 without re-introducing the tempreason regression. Not implemented to keep the fix surgical and the no-regression guarantee tight.
