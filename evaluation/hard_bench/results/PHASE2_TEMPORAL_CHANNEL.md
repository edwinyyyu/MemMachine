# Phase 2 — em_temporal channel integration

## What was integrated

Real `temporal_retrieval` v5.1 (DNF planner + classifier + 2-pass extractor + hybrid pool + cross-encoder rerank) wired as the `em_temporal` retrieval channel. Per-scenario index built from the same memory turns as EventMemory; query-time uses `current_time` as ref_time and the original task_prompt as the anchor query (spreading-activation probes drop temporal anchors as they refine concepts, so we use task_prompt verbatim instead).

## Result on temporal family (n=19-20)

| condition | correct | respected_anchor | plant_retrieved | oow_decoy_retr |
|---|---|---|---|---|
| **em_cosine alone (Phase 1)** | 16/19 (84%) | **5/19 (26%)** | 19/19 (100%) | 19/19 (100%) |
| em_cosine + em_temporal RRF | 13/19 (68%) | 8/19 (42%) | 18/19 (95%) | 19/19 (100%) |
| **em_temporal alone** | 6/20 (30%) | **13/20 (65%)** | 13/20 (65%) | 7/20 (35%) |

The temporal channel works (anchor respect doubled standalone, +16pp in RRF) but trade-offs are sharp:

- **em_temporal alone** filters OOW decoys aggressively (100% → 35%) and lifts anchor respect (26% → 65%), but plant retrieval drops (100% → 65%) — the filter is too restrictive in some regimes (deictic_relative: 1/5 correct, 5/5 respected — temporal filter excludes even the gold plants because the deictic window is narrow).

- **em_cosine + em_temporal RRF** at fixed K=5 dilutes em_cosine's coverage. respected_anchor lifts +3 but correct drops -3. The two channels fight for slots; em_temporal's anchor-filtered hits replace em_cosine's topically-rich ones.

## Per-anchor breakdown

| anchor type | em_cosine alone | em_cosine+temporal | em_temporal alone |
|---|---|---|---|
| anaphoric_event (5) | corr=5, resp=2 | corr=4, resp=3 | corr=2, resp=4 |
| calendar_pin (5) | corr=4, resp=0 | corr=4, resp=2 | corr=3, resp=2 |
| deictic_relative (5) | corr=5, resp=1 | corr=5, resp=1 | corr=1, **resp=5** |
| recurring_period (3) | corr=1, resp=1 | corr=0, resp=1 | corr=0, resp=1 |
| date_range (1-2) | corr=1, resp=1 | corr=0, resp=1 | corr=0, resp=1 |

**deictic_relative** is the sharpest tradeoff: em_temporal alone gets 5/5 anchor respect (perfect filter) but 1/5 correct (filter excludes plants). For "last week" queries with narrow windows, the filter is too tight.

**calendar_pin** is the cleanest win for em_temporal: 0/5 → 2/5 respected_anchor, with correct staying at 4/5 in RRF.

## Architectural diagnosis

Two issues compounded:

1. **Spreading-activation agent fires many probes during Phase 1.** Each probe goes through every channel. For em_temporal, I use `task_anchor_phrase` (= task_prompt) as the query regardless of probe text — but this means em_temporal returns the SAME small set of in-window hits every probe, while em_cosine returns diverse hits per probe. The agent's accumulated context is dominated by em_cosine's diversity when both channels are active; only the anchor-respecting hits that em_temporal's pool overlaps with em_cosine's get RRF-rewarded.

2. **RRF at fixed K=5 across 2 channels with disparate roles.** em_cosine retrieves 5 topically-relevant; em_temporal retrieves 5 anchor-respecting. RRF fuses to 5 top — pulls 2-3 from each. Net loss of em_cosine's topical coverage that drove correct=84% in Phase 1.

## Fixes to test (Phase 3)

1. **Use em_temporal as a FILTER, not a separate retriever**: take em_cosine's top-K (larger K, e.g. 15), drop hits that em_temporal would exclude (have the planner check each hit's interval against query plan), keep top 5. Preserves em_cosine's recall + applies anchor.

2. **Wider retrieval budget when temporal channel is active**: K_per_probe=6-8 instead of 3 so each channel keeps more representation in the RRF pool.

3. **Per-query routing**: detect from task_prompt whether the query has an explicit anchor (calendar_pin, anaphoric_event, deictic_relative, etc.). Route to em_temporal-dominant for those; route to em_cosine-only for others.

4. **Loosen the deictic filter window**: the v5.1 deictic resolver may be too narrow on "last week"-style queries. Inspect specific TT11-TT15 cases.

## Headline takeaway

**em_temporal channel does what it's supposed to** (filters OOW decoys 100% → 35%, lifts anchor respect 26% → 65% standalone). But at fixed K=5 context budget with RRF, integrating it costs more than it gains. The next architectural step is filter-style integration (apply em_temporal AFTER em_cosine retrieval) rather than independent-channel RRF.

This validates the underlying design intuition (anchored retrieval needs anchor filtering) and identifies the specific integration mechanism that must change.

## Files

- `system.py` — em_temporal channel real implementation, `make_temporal_infra()` builder
- `results/temporal_em_cosine_em_temporal_results.json` — RRF run (n=19)
- `results/temporal_em_temporal_results.json` — em_temporal alone (n=20)
- `cache/temporal_retrieval/` — extractor + planner + classifier caches (LLM, persistent across runs)

Cost: ~$0.50 for the 2 Phase 2 batches (39 runs through gpt-5-mini + cross-encoder).
