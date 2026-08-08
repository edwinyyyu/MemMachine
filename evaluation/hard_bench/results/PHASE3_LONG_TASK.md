# Phase 3 — temporal_filter applied to long-task family

## Result

| metric | em_cosine alone | em_cosine + temporal_filter |
|---|---|---|
| total addressed | 126/147 (85.7%) | 121/147 (82.3%) |
| total plant_retrieved | 138/147 (93.9%) | 136/147 (92.5%) |
| cross_domain_pattern addressed | 38/45 | 39/45 |
| episodic addressed | 57/62 | 54/62 (-3) |
| temporal addressed | 31/40 | 28/40 (-3) |

**Filter regresses on long-task family** by 5 absolute (-3.4pp).

## Why

Long-task scenarios have task_prompts that mostly DON'T have explicit temporal anchors ("plan our team retreat", "investigate latency regression" — no calendar pin, no anaphoric event reference, no deictic phrase). The temporal_retrieval planner sometimes still extracts speculative anchors from these prompts (e.g., "regression" might trigger a "since regression started" interpretation), and those incorrect anchors then over-filter the eligible set.

The hybrid k/2+k/2 mitigates this — the unfiltered half preserves cosine's coverage — but the filtered half occupies slots that would otherwise be cosine's other rank-3-5 hits. For tasks without true temporal anchors, those displaced cosine hits would have been more useful than the speculatively-filtered ones.

## Architectural conclusion

**temporal_filter should be query-routed, not always-on.** Specifically:
- Apply when task_prompt contains an explicit anchor (calendar_pin, anaphoric_event, deictic_relative, recurring_period, date_range)
- Skip when no anchor is present — fall back to em_cosine alone

Detection signal: `_compute_temporal_eligibility()` already runs the planner+classifier on task_prompt. If `valid_includes` and `valid_excludes` are both empty after classification, return None → no filtering. This gate IS in place. But the planner is producing low-confidence pseudo-anchors that pass through.

Fix candidates:
1. Tighten the planner prompt: only pin anchors when the query EXPLICITLY mentions time
2. Route on classifier output: only filter if at least one classified leaf is calendar_pin / anaphoric_event / deictic_relative (drop generic_skip and personal_era from triggering)
3. Confidence threshold on the resolver's intervals — drop anchors below a threshold

For now: the insight is that hybrid k/2+k/2 is **architecturally right for temporal-anchored queries (Phase 3 main result: temporal family +6pp)** but **regresses on non-anchored queries (long-task family -3.4pp)**. Routing is the next layer.

## Per-channel summary across 4 families on em_cosine baseline

| family | em_cosine | em_cosine + temporal_filter | best |
|---|---|---|---|
| guideline | 20/20 surfaced (saturated) | (untested) | em_cosine |
| qa | 18/19 correct | (untested) | em_cosine |
| long | 126/147 addressed (85.7%) | 121/147 (82.3%) | **em_cosine** |
| temporal | 16/19 correct (84%) | 18/20 (90%) | **em_cosine + temporal_filter** |

**Routing rule for production**: apply temporal_filter only when task has explicit anchor (detect via planner output). em_cosine + spreading-activation agent saturates the rest.

## Files

- `results/long_em_cosine_temporal_filter_results.json`
- `results/long_em_cosine_results.json` (Phase 1 baseline)

Phase 3 long-task cost: ~$0.40 OpenAI for the 20-scenario run.
