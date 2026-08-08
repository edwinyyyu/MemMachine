# Hard-bench Phase 1 baseline (em_cosine, gpt-5-mini)

**System under test**: real EventMemory (Qdrant + SQLite + text-embedding-3-small), spreading-activation agent (Phase 1 planning probes + Phase 2 plan + Phase 3 per-step exec), 1 retrieval channel (em_cosine), gpt-5-mini for all LLM operations.

**Corpus**: 79 scenarios across 4 families (guideline 20, long 20, qa 19, temporal 19). All data subagent-generated (Claude), internally consistent.

## Results

| family | n | metric | result | notes |
|---|---|---|---|---|
| guideline | 20 | surfaced / warned / rec_alt | **20 / 20 / 20** | SATURATED across all obscurity tiers |
| long | 20 | subdecisions addressed | 126/147 (**85.7%**) | spread across 3 capabilities |
| long | 20 | plants retrieved | 138/147 (**93.9%**) | retrieval is high; bottleneck is agent's use of context |
| qa | 19 | answer correct | 18/19 (**94.7%**) | 1 missed on EASY (likely TQ20 calendar inference) |
| qa | 19 | plant retrieved | 18/19 (**94.7%**) | |
| temporal | 19 | answer correct | 16/19 (**84%**) | |
| temporal | 19 | **respected_anchor** | **5/19 (26%)** | KEY GAP — agent's answers mix out-of-window content |
| temporal | 19 | plant retrieved | 19/19 (100%) | em_cosine pulls everything topic-similar |
| temporal | 19 | **oow_decoy retrieved** | **19/19 (100%)** | em_cosine has zero temporal filtering |

## Long-task by capability

| capability | n | addressed | retrieved |
|---|---|---|---|
| episodic | 62 | 57 (92%) | 57 (92%) |
| cross_domain_pattern | 45 | 38 (84%) | **43 (96%)** |
| temporal | 40 | 31 (77.5%) | 38 (95%) |

Cross-domain pattern recall is **96%** with naive em_cosine + spreading-activation — much higher than the v9-v25 thread's standalone-V15 results suggested. The iterative probing loop compensates for cosine's surface-vocab bias by surfacing different aspects of the abstract pattern across rounds.

Temporal subdecisions: retrieval 95%, addressed 77.5% — agent gets the relevant turns but doesn't always handle the temporal constraint correctly.

## Temporal by anchor type

| anchor | n | correct | respected | retr | oow_decoy_retr |
|---|---|---|---|---|---|
| anaphoric_event | 5 | 5/5 | 2/5 | 5/5 | 5/5 |
| calendar_pin | 5 | 4/5 | **0/5** | 5/5 | 5/5 |
| deictic_relative | 5 | 5/5 | 1/5 | 5/5 | 5/5 |
| recurring_period | 3 | 1/3 | 1/3 | 3/3 | 3/3 |
| date_range | 1 | 1/1 | 1/1 | 1/1 | 1/1 |

**calendar_pin 0/5 respected_anchor** is the sharpest failure: agent gets correct answers 4/5 but ALWAYS includes out-of-window content. Adding temporal_retrieval as a retrieval channel (filtering by anchor window before passing to agent) would directly address this.

## Architectural takeaways (Phase 1)

1. **Real EventMemory + spreading-activation agent saturates guideline + qa families on em_cosine alone.** No headroom for V15/temporal augmentation here. This contradicts my earlier standalone-V15 findings on synthetic short-note corpora — when combined with proper agent looping over real EM, cosine alone is sufficient for surface-vocab-aligned tasks.

2. **Cross-domain pattern recall is 96%** at the retrieval layer with em_cosine + iterative probing. V15's structural pattern abstraction is unlikely to add unique recall here. (Earlier findings showed V15-as-augmentation adds 1-5/30 unique queries; that lift will be tiny here because the spreading-activation agent already covers the cross-domain space.)

3. **Temporal regime is the discriminating bench.** em_cosine retrieves OOW decoys at 100%; agent must filter by reading timestamps in `[date, time]`-formatted hits. It mostly answers correctly (16/19, 84%) but anchor-respect is only 5/19 (26%). **temporal_retrieval channel as augmentation is expected to lift respected_anchor substantially** by filtering out-of-window candidates before retrieval. UNTESTED in Phase 1.

4. **Long-task temporal subdecisions: retrieval 95%, addressed 77.5%.** Same gap shape — agent gets the right plants but doesn't always honor the temporal anchor. Could lift with temporal_retrieval channel.

## Phase 1 limitations

- **n=20 per family** — small. Confidence intervals on most numbers are wide.
- **gpt-5-mini for all LLM** — agent reasoning is the smaller model. Higher-effort or larger model could shift numbers.
- **Single judge pass** — judge variance unmeasured.
- **No augmentation channels tested** — only em_cosine. Phase 2 will add em_pattern_v15, em_temporal, em_entity.
- **TQ13 + TT19 errored** during parallel runs (LLM cache file race). Reduces n by 1 each.
- **Spreading-activation agent params not tuned** (max_phase1_rounds=4 for short-mem, 6 for long; per_step_probe_rounds=1-2). Could be optimized.

## Phase 2 priorities (next session)

In rough priority order based on Phase 1 baseline:

1. **Add em_temporal channel and re-run temporal family.** Expected lift: respected_anchor 5/19 → ~15+/19. Sharpest expected win.
2. **Add em_temporal to long-task family.** Expected lift: temporal cap 31/40 addressed → ~36+/40.
3. **Add em_pattern_v15 to long-task family**. Expected lift: cross_domain_pattern 38/45 addressed → maybe +2-4 from V15-unique cross-domain bridges.
4. **Add em_entity channel.** Currently a stub. R20-R25 prose-fact + DSU expected to help with multi-entity scenarios in qa family. Limited headroom (qa already 18/19).
5. **Fix LLM cache race.** Per-process cache file or file locking. Trivial.
6. **Re-run TQ13 + TT19** to fill missing data points.

## Files

- `data/task_{guideline,long,qa,temporal}.json` — 79 scenarios, subagent-generated
- `system.py` — UnifiedSystem with em_cosine working; em_pattern_v15 implemented but untested in this baseline; em_temporal + em_entity stubs
- `agent.py` — spreading-activation loop with current_time threading
- `runner.py` — per-family runners with resume-from-output, current_time per scenario
- `judge.py` — judges for each family (subdecision, guideline, qa, temporal)
- `results/{family}_em_cosine_results.json` — per-scenario judgments + transcripts
- `cache/llm_cache.json` — gpt-5-mini call cache (~3-5MB after Phase 1)

## Cost so far

~$2.5-3 in OpenAI API spend across all 79 scenarios on em_cosine. Includes ~80MB embedding calls + ~600 gpt-5-mini reasoning_effort=medium calls.
