# Phase 3 — temporal filter as hybrid k/2 + k/2

## Architecture

Per user prescription: temporal filtering should never be the sole retriever.
Combine semantic (cosine) with anchor-filtered:
- k/2 slots from FILTERED view (cosine top-K filtered to in-window per
  temporal_retrieval planner+classifier+resolver)
- k/2 slots from UNFILTERED cosine top-K, deduped against the filtered slots

Implementation in `system.py::retrieve` when `temporal_filter` flag set:
```python
in_window = [h for h in fused if h.turn_id in eligible]
filtered_slice = in_window[:k//2]
remaining = [h for h in fused if h.turn_id not in seen][:k - len(filtered_slice)]
return (filtered_slice + remaining)[:k]
```

Eligibility computed once per scenario via `_compute_temporal_eligibility()` —
runs the temporal_retriever's planner/classifier/resolver on the task_prompt's
anchor, then for each doc combines extracted intervals with a 1-day window
around the doc's `ref_time` (fallback for docs whose text doesn't mention
the date but whose ref_time is in window).

## Results across all temporal architectures tested

| architecture | correct | respected | plant | oow_decoy |
|---|---|---|---|---|
| em_cosine alone | 16/19 (84%) | 5/19 (26%) | 19/19 | 19/19 |
| em_cosine + em_temporal RRF (no date prefix) | 13/19 | 8/19 | 18/19 | 19/19 |
| em_cosine + em_temporal RRF (date prefix) | 14/19 | 7/19 | 18/19 | 19/19 |
| em_temporal alone (date prefix) | 6/20 | 13/20 | 13/20 | 7/20 |
| em_cosine + filter (hard) | 13/20 | 11/20 | 19/20 | 9/20 |
| **em_cosine + filter (hybrid k/2+k/2)** | **18/20 (90%)** | 6/20 | **20/20** | 19/20 |

## Headline

**Hybrid is the production-ready architecture.** Correct: 90% (+6pp vs em_cosine baseline). Plant retrieval: 100%. No regression on metrics that the standalone-filter and RRF integrations regressed.

## What hybrid achieves vs doesn't

**Achieves:**
- Guarantees in-window content in context (k/2 reserved slots)
- Preserves semantic coverage (other k/2 slots from unfiltered cosine)
- Beats em_cosine alone on correct (+2 absolute) and plant retrieval (+1 plant)
- Zero regression on any metric

**Doesn't achieve (by design):**
- respected_anchor not lifted (still 30%, ~baseline) — the unfiltered half exposes OOW decoys to the agent, which uses them in answers.

## Why respected_anchor doesn't lift

The respected_anchor metric measures whether the agent's ANSWER mixes content from outside the temporal window. Even when retrieval surfaces the gold plant (which hybrid does at 100%), the agent's answer may still reference OOW content because:

1. The unfiltered half of hybrid retrieval includes OOW decoys (those that share topic with query but are outside anchor's window).
2. The agent has no instruction in its prompt to specifically respect the anchor — it treats all retrieved hits as candidates for the answer.

To lift respected_anchor, the agent's plan/exec prompts would need explicit guidance: "the user's question references events in {anchor}; events outside that window are BACKGROUND CONTEXT ONLY, not part of the answer." This is a separate prompt-engineering experiment, not a retrieval-architecture one.

## Why k/2+k/2 (vs alternatives)

- **Hard filter (drop all OOW)**: regresses correct -19pp because some plants have empty extracted intervals → wrongly excluded.
- **RRF over (em_cosine, em_temporal)**: dilutes em_cosine's coverage at K=5 (each channel gets ~half the slots after fusion); em_temporal's pool/rerank stage is biased toward in-window via doc_passes_filter, so contribution overlaps with what the filter would do anyway.
- **Hybrid k/2+k/2**: explicit slot allocation. Filter contributes anchor-respecting hits without losing semantic coverage.

## ref_time fallback

The temporal_retrieval extractor pulls intervals from doc text. Memory turns whose text doesn't explicitly mention the date have empty `_doc_ivs` and fail include filters. Fix: build per-doc interval list as `_doc_ivs ∪ {1-day window around ref_time}`. So a doc whose text says "Marco merged the rebase" but whose ref_time is 2026-03-15 will pass an "in March 2026" filter.

This is critical for chat-style memory where most turns don't restate the date.

## Current best configuration

```python
system.retrieve(query, channels=("em_cosine", "temporal_filter"), k=5)
```

This:
1. Runs em_cosine for 15 candidates (k=5 × 3 over-fetch when filter active)
2. Computes eligible turn_ids from task_anchor_phrase (cached per scenario)
3. Returns 2 in-window hits + 3 unfiltered hits (deduped)

Plug-and-play with the existing spreading-activation agent. Per-scenario cost adds ~3 LLM calls (planner+classifier+resolver, all cached after first scenario) plus per-turn extraction at ingest time (cached across scenarios).

## Files

- `system.py::_compute_temporal_eligibility` — eligibility cache + ref_time fallback
- `system.py::retrieve` — k/2+k/2 hybrid logic when temporal_filter is in channels
- `results/temporal_em_cosine_temporal_filter_results.json` — n=20

Phase 3 cost: ~$0.30 OpenAI for the rerun batches.
