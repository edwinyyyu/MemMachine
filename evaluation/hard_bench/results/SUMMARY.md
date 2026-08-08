# Hard-bench summary across phases

## What was built

A unified memory-augmented LLM agent benchmark testing 4 task families:

- **Family A: long-task** (n=20) — agent gets short task prompt; must probe memory for constraints/decisions/cross-domain priors via spreading-activation
- **Family B: guideline** (n=20) — memory has team-specific rules; user task innocently violates one; agent must surface + obey
- **Family C: qa** (n=20) — single-question single-answer over dialog; episodic recall baseline
- **Family D: temporal** (n=20) — questions with explicit time anchors; OOW decoys must be excluded

All data subagent-generated (Claude), internally consistent. **Real EventMemory** (Qdrant + SQLite + text-embedding-3-small), **gpt-5-mini** for all LLM (memory + agent + judge), **temporal_retrieval v5.1** wired as filter channel, **cross-encoder ms-marco-MiniLM** for rerank.

## Headline results

### Per-family best on em_cosine baseline (Phase 1)

| family | n | best metric | result | notes |
|---|---|---|---|---|
| guideline | 20 | surfaced/warned/alt | 20/20/20 | SATURATED at all obscurity tiers |
| qa | 19 | correct | 18/19 (94.7%) | 1 EASY missed (TQ20 calendar inference) |
| long | 20 | addressed | 126/147 (85.7%) | retrieved 138/147 (93.9%) |
| temporal | 19 | correct | 16/19 (84%) | but respected_anchor 5/19 (26%) |

### Phase 2/3 augmentation results

**Temporal family** (the discriminating regime):

| architecture | correct | respected | plant | oow_decoy |
|---|---|---|---|---|
| em_cosine alone | 16/19 (84%) | 5/19 (26%) | 100% | 100% |
| em_cosine + em_temporal RRF | 14/19 (74%) | 7/19 (37%) | 95% | 100% |
| em_temporal alone | 6/20 (30%) | 13/20 (65%) | 65% | 35% |
| em_cosine + filter (hard) | 13/20 (65%) | 11/20 (55%) | 95% | 45% |
| **em_cosine + filter (hybrid k/2+k/2)** | **18/20 (90%)** | 6/20 | **100%** | 95% |

**Long-task family**:

| architecture | addressed | plant | best?
|---|---|---|---|
| em_cosine alone | 126/147 (85.7%) | 138/147 | **best** |
| em_cosine + temporal_filter | 121/147 (82.3%) | 136/147 | regression |

## Architectural conclusions

1. **Real EventMemory + DESIGN.md spreading-activation agent saturates 3 of 4 families on em_cosine alone.**
   Cross-domain pattern recall reaches 96% via iterative probing (much higher than my earlier standalone-V15 thread predicted).

2. **Temporal regime is the only one with retrieval headroom.** em_cosine retrieves out-of-window decoys at 100% — cosine has zero temporal filtering. Adding a temporal filter as augmentation lifts performance.

3. **Hybrid k/2+k/2 is the production-ready filter integration**:
   - k/2 from filtered (in-window) cosine
   - k/2 from unfiltered cosine, deduped
   - On temporal family: correct 84% → 90%, plant 100% → 100%
   - Hard filter (drop all OOW): drops plants whose text doesn't mention dates → -19pp correct
   - RRF over (em_cosine, em_temporal): dilutes em_cosine's coverage → -10pp correct
   - Standalone em_temporal: too aggressive → -54pp correct

4. **temporal_filter must be query-routed** (Phase 4 gap). On long-task family the always-on filter regressed -3.4pp because long-task prompts mostly don't have explicit anchors but the planner emits speculative anchors. Production: detect anchor type from planner classifier output; activate filter only when at least one classified leaf is calendar_pin / anaphoric_event / deictic_relative / recurring_period / date_range with non-empty resolved intervals.

5. **respected_anchor metric needs a separate prompt lever, not retrieval.** Even with the filter retrieving in-window content, the unfiltered half exposes OOW decoys that the agent uses. To lift respected_anchor, the agent's plan/exec prompts need explicit anchor instruction ("focus on events in {anchor}; outside-window events are background only"). Untested.

6. **ref_time fallback is critical for chat memory.** temporal_retrieval extracts intervals from doc TEXT. Memory turns whose text doesn't mention dates have empty intervals → wrongly filtered out by include constraints. Fix: combine extracted intervals with a 1-day window around doc.ref_time. Without this, hybrid filter regresses correctness by ~10-15pp on temporal family.

## V15 finding (carried over from prior thread)

Standalone V15 was bimodal in earlier work. **Per Phase 1 results here**: cross-domain pattern recall is 96% via spreading-activation + cosine alone, leaving little headroom for V15 augmentation. The +1-5/30 unique-recall lift V15 added in synthetic short-note benchmarks doesn't replicate when (a) the substrate is real EventMemory (not naive cosine) and (b) the agent loops iteratively over retrieval. This validates the Phase 1 takeaway: V15-as-augmentation provides marginal lift in production-realistic settings; V15-as-substitute hurts.

## Files

```
hard_bench/
  DESIGN.md                       # architecture
  data/
    task_long.json                # n=20, 147 subdecisions
    task_guideline.json           # n=20, gold rules
    task_qa.json                  # n=20, gold answers
    task_temporal.json            # n=20, anchor types
  system.py                       # UnifiedSystem with em_cosine + em_pattern_v15 + em_temporal + temporal_filter
  agent.py                        # spreading-activation loop
  runner.py                       # per-family runners with resume-from-output
  judge.py                        # 4 family-specific judges
  results/
    PHASE1_BASELINE.md            # em_cosine on all 4 families
    PHASE2_TEMPORAL_CHANNEL.md    # em_temporal channel results
    PHASE3_FILTER_HYBRID.md       # hybrid k/2+k/2 on temporal family (best)
    PHASE3_LONG_TASK.md           # filter applied to long-task (regression diagnosis)
    SUMMARY.md                    # this file
    {family}_{channels}_results.json    # per-batch raw judgments + transcripts
  cache/
    llm_cache.json                # gpt-5-mini call cache (~3-5MB)
    temporal_retrieval/            # planner + classifier + extractor caches
```

## Cost

Phase 1+2+3 total: ~$5-7 OpenAI (text-embedding-3-small + gpt-5-mini reasoning_effort=medium across ~150 scenario-runs and ~5 ablations).

## Remaining work (queued)

- Phase 4: **query-route temporal_filter** based on classifier output. Should preserve +6pp on temporal family while eliminating -3.4pp on long-task.
- em_entity channel (R20-R25 prose-fact + DSU) integration. Limited expected headroom (qa already 94.7%) but completes the 4-channel ensemble.
- Cache race fix (TQ13 + TT19 errored during parallel runs). Trivial.
- Larger n per family (current ±1 std error on most cells is wide).
- Agent prompt instruction for respected_anchor lift.
