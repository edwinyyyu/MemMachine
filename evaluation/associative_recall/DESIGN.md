# Mid-execution retrieval cues — current best architecture

## Problem

Tasks like "prepare a presentation for external clients" or "prepare a banquet for the high school track team" do not embed the cues for the sub-decisions an executor will face after working on them for a while (presentation → colors → brand guidelines; banquet → food → allergies). Pre-task decomposition is structurally insufficient — the agent only encounters many sub-decisions during execution. The user-given prompt is not a useful retrieval cue for those mid-task sub-decisions.

Constraint: build on EventMemory only. No model-provider swaps, no BM25, no cross-encoder rerank.

## Headline

The current best architecture on the 10 hard scenarios:

| variant | n | coverage | full_R@5 |
|---|---|---|---|
| **`spreading_activation_full` + multi-turn explicit thinking** | 10 | **0.876** | **0.646** |

vs the natural (no-cue) baseline cov 0.79 / full_R@5 0.50, this is +9pp coverage and +14pp R@5.

## Approach

End-to-end agent loop with three phases:

- **Phase 1 — planning-time spreading activation.** Agent gets `task_prompt` only. It probes EM with the task prompt as seed, then iteratively reads the surfaced snippets, decides what's still missing, and emits new probes. Up to 8 rounds; saturates when no new turn IDs are surfaced or the agent emits `STOP`. Within Phase 1, agent thinking is preserved as a multi-turn chat history (`mt_messages`), so each round sees the agent's prior reasoning AND the new snippets. This is the "concept → see → re-probe" loop, the cognitive analog of associative spreading activation.

- **Phase 2 — plan-only.** Agent gets the full accumulated context block (deduped chat snippets across all Phase 1 probes) and writes a numbered plan, with no execution. Decoupling plan generation from probe generation keeps the plan items concise and free of probe-text contamination.

- **Phase 3 — per-step execution-time spreading.** For each plan item, agent does up to 2 mid-step probe rounds (CUE generation → embed-probe → see snippets → optionally probe again), then writes the step content. The CUE prompt sees: the original plan, prior plan-context, exec-context-from-prior-steps, prior step outputs. The WRITE prompt sees: full deduped context + reasoning disciplines (world-knowledge framing, recency/supersession framing).

### What works (load-bearing)

1. **Memory access at planning time** is the dominant unlock (E3: 0.83 → 0.94 cov on the easier 10-scenario set). Pure prompting saturates around 0.83 because the bottleneck isn't "agent can't think of the sub-decision" but "agent doesn't know what's IN memory."
2. **Iterative spreading activation** beats single-pass RAG (E3: full_R@5 0.62 → 0.76). Each round surfaces concepts the next round can probe specifically. E.g., round 1 reveals "Diane's wedding is a Quaker silent ceremony" → round 2 probes "Quaker phone collection rules" → finds the specific plant.
3. **Multi-turn explicit thinking (mt_messages)** for Phase 1 working memory. The agent's prior reasoning STAYS in the conversation context across iterations — closer to human "where I left off" continuity than fresh prompts. Multi-turn variant beat single-turn prompted-reasoning by +3.6pp coverage / +9.5pp full_R@5.
4. **Per-step iterative probing at execution.** Round 2 sees round 1's chat hits and probes for what's still missing (chain assembly).
5. **Single flat-similarity index for the episodic modality (chat history) only.** What you DON'T put in it is as important as what you do — agent thinking, semantic facts, procedural traces should not flat-share an index with chat (see Substrate Experiments below for the empirical falsification).

### What was tested and rejected

- **Critic-pass review of plan.** Hurts retrieval despite lifting coverage (cond_R@5 0.50 vs 0.63). Critic plans are over-abstract (12-13 steps); per-step content gets diluted.
- **Multi-probe with max-score merge.** Worse than single cue (R@1 0.264 vs 0.424). One bad probe with high-score-but-wrong hits dominates.
- **LLM precision rerank.** User-rejected ("don't use LLM rerank").
- **Pre-execution one-shot DECOMPOSE+CUEGEN.** Actively hurts per-step retrieval (R@1 0.104 vs decision_text 0.347). Upfront probes are scenario-level, not step-specific — they add noise.
- **Iterative probe refinement via embedding vector arithmetic** (centroid pull or LLM add/subtract phrases). Consistently fails on text-embedding-3-small (saved as `project_no_vector_arithmetic`).

### Native vs explicit reasoning

Native reasoning (Responses API + `previous_response_id`) at `reasoning.effort="low"`: cov **0.924**, full_R@5 **0.608** on 9 hard scenarios. Higher coverage (cleaner step planning) but lower R@5 (less rich exec-time cue text — the explicit-thinking variant has the THINKING blocks bleeding vocabulary into subsequent prompts via mt_messages). Mixed signal at low effort. Untested at medium/high effort.

## Substrate experiments (informative negative results)

Tested whether ingesting the agent's own THINKING/PROBE outputs back into EM as retrievable events helps cross-phase reasoning continuity:

| variant | n | coverage | full_R@5 |
|---|---|---|---|
| explicit baseline (champion) | 10 | 0.876 | 0.646 |
| EM-ingest, no retrieval filter | 10 | 0.717 | **0.197** |
| EM-ingest + retrieval-time agent_*  filter | 10 | 0.876 | 0.646 |
| separate cognition partition + exec-time surfacing | 10 | 0.838 | 0.569 |

**Findings:**

- **Flat ingest catastrophically displaces gold** (-45pp R@5). The agent's exec-time cue is generated from the same conceptual neighborhood as its prior thinking; cosine similarity ranks self-similarity above chat-form plant content. Top-10 for `multi-hop-banquet-01` step 1: nine agent_thinking events, only one real plant.
- **Type-filter restores baseline bit-identical**. The substrate provides zero retrieval value once events stop competing. Working memory was already covered by `mt_messages` in Phase 1; adding it to EM was redundant.
- **Stratified version** (separate Qdrant collection + SQLAlchemy partition for cognition, probed independently with K=2, surfaced as labeled prompt section) is better than flat ingest but still net-negative. Wins 2/10 (supersession +11pp, negative-space +17pp — scenarios where Phase 1 reasoning provides a useful META-pattern); loses 7/10 to confirmation bias on wrong-anchored Phase 1 reasoning.

**Architectural takeaway:** stratification is *necessary* but *insufficient*. The "memory is cognition's substrate" framing is right at the storage layer, wrong if applied uniformly at the retrieval-scoring layer. Different memory types serve different cognitive roles and need different retrieval treatments. The cognition channel needs **gating** — agent-decided surfacing, confidence-based surfacing, or role-specific surfacing (cognition useful at planning, mostly noise at fact retrieval).

## Source files

### Entry point / orchestration

| File | Role |
|---|---|
| `evaluation/associative_recall/mid_execution_eval_e2.py` | Main eval. Defines the SA-full mode (planning + plan-only + per-step exec), all SPREADING_* prompts, LLM helpers (`_llm`, `_llm_multiturn`, `_llm_responses`), env toggles (`EXECUTOR_BACKEND`, `REASONING_MODE`, `EM_INGEST_THINKING`, `EM_RETRIEVAL_FILTER_AGENT`, `EM_COGNITION_CHANNEL`, `COGNITION_PROBE_K`, `SA_RERANK`), and the per-scenario driver `run_scenario_e2`. |
| `evaluation/associative_recall/mid_execution_eval.py` | E0 base. Defines `_scenario_collection` (Qdrant 32-byte collection name, hashed for long IDs), `ingest_scenario` (preamble plants + LoCoMo distractors + extra-conversation distractor packs), `Hit` dataclass, `probe()` (uses `EventMemory.string_from_segment_context` for `[date, time]` formatted hits), `triggered_recall`, `false_positive_rate`, scenario/segment loaders. |
| `evaluation/associative_recall/mid_execution_eval_e1.py` | E1 helpers. Provides `probe_multi` (multi-cue with score-merge, used by `cue_aware_multi` and combined modes — NOT used by SA-full but shared in the module). |

### Reasoning architecture (Phase 1 / Phase 2 / Phase 3)

| File / location | Role |
|---|---|
| `mid_execution_eval_e2.py` `SPREADING_PROBE_SYSTEM_MT` | Phase 1 system prompt for multi-turn explicit thinking. Sets the iterative concept-driven probing role; prompts the agent to emit `THINKING:` then `PROBE:` lines. |
| `mid_execution_eval_e2.py` `SPREADING_PROBE_USER_INITIAL_MT` / `SPREADING_PROBE_USER_FOLLOWUP_MT` | Phase 1 user-side templates for round 1 (seed snippets) and round N+1 (new snippets surfaced this round, total probes so far). |
| `mid_execution_eval_e2.py` `SPREADING_PROBE_SYSTEM_MT_NATIVE` | Phase 1 system prompt for the native-reasoning variant (Responses API + `previous_response_id`). Same role; no THINKING prefix because reasoning is server-side. |
| `mid_execution_eval_e2.py` `SPREADING_PLAN_ONLY_SYSTEM` | Phase 2 system prompt. Takes accumulated chat-snippet context, asks the agent to write a numbered plan with no execution. Includes world-knowledge and recency/supersession framing disciplines. |
| `mid_execution_eval_e2.py` `SPREADING_EXEC_STEP_SYSTEM` | Phase 3 system prompt for per-step CUE generation. Sees task prompt, full plan, plan-context, exec-context-from-prior-steps, prior outputs. Emits `CUE: none` or 1-3 CUE lines. |
| `mid_execution_eval_e2.py` `SPREADING_EXEC_WRITE_SYSTEM` | Phase 3 system prompt for per-step content writing. Sees full deduped context + reasoning disciplines (world-knowledge, recency). Writes 1-3 concrete sentences delivering the step's deliverable. |
| `mid_execution_eval_e2.py` `run_freelance_executor` | Driver. For mode `spreading_activation_full`: runs Phase 1 (multi-turn or native branch), Phase 2 plan-only, then Phase 3 per-step iterative probing + writing. Constants: `MAX_ITERS=8`, `PER_STEP_PROBE_ROUNDS=2`, `K_PER_PROBE=3`. |

### Retrieval

| File / location | Role |
|---|---|
| `mid_execution_eval.py` `probe(memory, query_text, K)` | Single-cue EM probe. Returns `Hit` list with `formatted_text` (uses `EventMemory.string_from_segment_context([seg], format_options=FormatOptions(date_style="medium", time_style="short"))` so `[2023-01-04, 09:30] alice: ...` prefixes are visible to the agent for recency reasoning). |
| `mid_execution_eval_e2.py` `_saf_probe(query, K)` | SA-full retrieval wrapper. Routes to `probe()` by default; under `SA_RERANK=1` routes to `probe_with_rerank`; under `EM_RETRIEVAL_FILTER_AGENT=1` over-fetches 3× and drops hits with `event_type ∈ {agent_thinking, agent_probes}` so substrate experiments don't pollute the chat retrieval. |
| `mid_execution_eval_e2.py` `probe_with_rerank` | LLM precision rerank wrapper. Implemented but currently unused by the champion architecture (user-rejected "don't use LLM rerank"). |

### Substrate / cognition channel (currently OFF in champion architecture)

| File / location | Role |
|---|---|
| `mid_execution_eval_e2.py` `_ingest_agent_round` | Writes the agent's THINKING block and emitted PROBE list back into EM as two events (`event_type="agent_thinking"` and `event_type="agent_probes"`, `speaker="self"`, turn_id ≥ 100000). Off by default. Activated by `EM_INGEST_THINKING=1`. |
| `mid_execution_eval_e2.py` `_create_cognition_memory` | Spins up a SECOND `EventMemory` instance per scenario, with its own collection (`<base>_cog`) and partition. Activated by `EM_COGNITION_CHANNEL=1`. Routes agent thinking to a separate retrieval channel that does not share a top-K with chat plants. |
| `mid_execution_eval_e2.py` `cognition_context_block` block in Phase 3 step loop | Probes `cognition_memory` with the step label (top-K = `COGNITION_PROBE_K`, default 2) and surfaces the result as a labeled "YOUR PRIOR REASONING" section in `SPREADING_EXEC_STEP_SYSTEM` and `SPREADING_EXEC_WRITE_SYSTEM`. |

### Ingestion

| File / location | Role |
|---|---|
| `mid_execution_eval.py` `ingest_scenario` | Per-scenario fresh EM. Plants (real `plant_id="p0"...`) and decoys (`plant_id=None`, sharing entities with plants for discrimination difficulty) at `turn_ids 0..N-1`; LoCoMo conversation distractor at `N..N+M-1`; optional `extra_distractor_runs` for multi-conversation scenarios. Timestamps `base_ts + 60s × turn_id`. |
| `evaluation/associative_recall/data/mid_execution_scenarios.json` | 20 scenarios across 5 difficulty tiers (6 v3 original, 4 hard-v2, 4 hard-v3, 3 v4 max, 3 v5 break). Each scenario has `task_prompt`, `preamble_turns` (plants + decoys), `subdecision_script` (gold sub-decisions with `gold_plant_ids`), `base_conversation` (LoCoMo conv ID for distractor), optional `extra_base_conversations`. |
| `evaluation/data/locomo_segments.npz` | Distractor source. LoCoMo conversations 26/30/41 (~419-663 turns each) loaded by `load_locomo_segments`. |
| `evaluation/associative_recall/data/locomo_speakers.json` | Speaker map for LoCoMo conv IDs. |

### Scoring / evaluation

| File / location | Role |
|---|---|
| `mid_execution_eval_e2.py` `judge_coverage` + `COVERAGE_JUDGE_PROMPT` | LLM judge (gpt-5-mini, low effort). Reads agent's full transcript + a single gold sub-decision's text + a representative plant. Outputs `addressed: true/false` and the agent step label that addressed it. |
| `mid_execution_eval_e2.py` per-gold scoring loop in `run_scenario_e2` (~lines 1780-1860) | For each addressed gold step: pulls the agent's step content (or evidence quote when judge couldn't pinpoint a step) as `cue_text`, probes EM with it, computes `triggered_recall_full@K = recall_given_covered@K = (gold plant_ids found ∩ top-K) / |gold plant_ids|`. For non-addressed gold steps, both metrics are 0. |
| `mid_execution_eval_e2.py` cross-scenario aggregator in `main()` | Means across scenarios for `coverage_rate` and per-K `triggered_recall_full@K` / `recall_given_covered@K`. |

### Run wrappers / harness

| File | Role |
|---|---|
| `evaluation/associative_recall/run_hard_set.py` | Iterates the 10 hard scenarios (`SCENARIOS[10:20]`) by spawning per-scenario subprocesses (`mid_execution_eval_e2.py --scenario <id> --out <path>`). Inherits env vars (`EM_INGEST_THINKING`, `REASONING_MODE`, `EM_COGNITION_CHANNEL`, etc.) so each architectural variant is a single env-var change. Aggregates results into `hard_<run_id>_SUMMARY.json`. |
| `evaluation/associative_recall/compare_runs.py` | 5-way per-scenario comparison table (explicit baseline / native / em_ingest / em_ing+filter / cog_chan). Auto-loads run timestamps via `EM_INGEST_RUN_ID` / `EM_INGEST_FILTER_RUN_ID` / `EM_COGNITION_CHANNEL_RUN_ID` constants. |
| `mid_execution_eval_e2.py` `__main__` block | Single-scenario or all-scenarios driver. Args: `--scenario`, `--K`, `--modes`, `--out`, `--no-overwrite`. Sets up `AsyncQdrantClient` (gRPC), `QdrantVectorStore`, `SQLAlchemySegmentStore` (shared SQLite at `RESULTS_DIR/eventmemory_mid_exec_e2.sqlite3`), `OpenAIEmbedder` (`text-embedding-3-small`, dim 1536). |

### Result artifacts

| Path pattern | Role |
|---|---|
| `evaluation/associative_recall/results/mid_execution_eval_e2_<ts>.json` | Per-scenario or per-batch result file. Contains `K_list`, `modes`, `n_scenarios`, full `scenarios[].per_mode[mode].per_gold[]` with per-step `top_hits` (rank, turn_id, plant_id, score) for forensic analysis. |
| `evaluation/associative_recall/results/hard_<run_id>_<sid>.json` | Per-scenario result from `run_hard_set.py`. |
| `evaluation/associative_recall/results/hard_<run_id>_SUMMARY.json` | Aggregated 10-scenario summary written by `run_hard_set.py`. |
| `evaluation/associative_recall/results/2026-04-24_mid_execution_cues_design.md` | Initial design + E0/E1/E2/E3 mechanism comparison log. |
| `evaluation/associative_recall/results/2026-04-28_em_ingest_substrate.md` | Substrate-experiment session log (native, EM-ingest, EM-ingest+filter, cognition channel). |

### Cache

| File | Role |
|---|---|
| `evaluation/associative_recall/cache/mid_exec_e2_executor_cache.json` | LLM call cache for executor calls (system+user → response). Cache-key includes a `cache_tag` like `gpt-5-mini:saf_plan_mt_round0`. Reproducible re-runs hit cache; architectural variants get fresh calls because their prompts differ. |
| `evaluation/associative_recall/cache/mid_exec_e2_judge_cache.json` | LLM judge cache (transcript + gold + plant → addressed/label/quote). |

## Tunable constants

`mid_execution_eval_e2.py` `run_freelance_executor`, mode `spreading_activation_full`:

- `MAX_ITERS = 8` — Phase 1 spreading-activation rounds. Bumped from 6 to give deeper chains room.
- `PER_STEP_PROBE_ROUNDS = 2` — Phase 3 mid-step iterative probes per step.
- `K_PER_PROBE = 3` — Hits accepted per probe (over-fetch is `K_PER_PROBE * 2`, then deduped by `turn_id`).
- `COGNITION_PROBE_K = 2` (env, when channel is on).

## Known unsolved gaps

- **world-knowledge-bridge** (cov 0.667): agent doesn't probe for facts that need world knowledge to bridge ("track team food" → "allergies of high-school athletes").
- **deductive-chain-procurement** (cov 0.667): agent doesn't chain multi-step inferences ("vendor X requires Y" + "we must do Y" → "we need vendor X").
- **negation-and-inference** (full_R@5 0.611 despite cov 1.0): cue text matches semantically-similar-but-wrong content; gold negation hint is sparse.
- **Cross-task procedural memory** (untested): completed task transcripts available to *future* tasks. The current single-task benchmark cannot test this; it is the most likely real win for substrate-style memory.

## Memory subsystems analog

| memory type | retrieval role | implementation |
|---|---|---|
| chat history (episodic) | flat-similarity probe | current EM main partition |
| agent thinking (working) | always-in-context, never probed by similarity | `mt_messages` (multi-turn) or `previous_response_id` (native) |
| agent thinking (procedural, cross-task) | type-filtered probe — surfaces "how I solved a similar task" | not yet tested |
| semantic facts (extracted) | type-filtered probe by entity/relation | adjacent to current EM (see `project_semantic_memory_research`) |

Stratification is necessary at retrieval; gating is necessary on top of stratification. A flat similarity index over a heterogeneous substrate inverts itself.
