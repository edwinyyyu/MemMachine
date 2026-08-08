# shared_harness_v4_mtmsg_twoloop_easy — build notes

Two-loop architecture (Option B, in-thread) on top of the
`subdec_split_easy` base. Outer loop coordinates the overall task;
inner loops are per-sub-decision sessions, each given a fresh turn-1
priming when the model spawns them with a marker.

## Mechanism

### New output types

- `DECISION_SPAWN: <id>: <one-line summary>` — model declares it is
  starting work on a NEW sub-decision and wants a fresh turn-1 priming.
- `DECISION_CLOSE: <id>` — model declares the sub-decision's
  STEP_OUTPUT is complete.

### Harness behavior

- On a **NEW** `DECISION_SPAWN` id (one not seen before), the harness
  appends `SPAWN_DIRECTIVE_TEMPLATE` to the next user followup. The
  template mirrors `USER_INITIAL`'s structure — full enumeration of
  further sub-decisions WITHIN the sub-task, the implicit-constraint
  question for each, and an instruction to probe both the obvious
  facts and the implicit-constraint facts.
- On `DECISION_CLOSE`, the harness appends `CLOSE_DIRECTIVE_TEMPLATE`
  ("[sub-task <id> closed; resume coordinating overall task]") to the
  next user followup.
- Both directives are appended via the `{spawn_close_directives}`
  slot in `USER_FOLLOWUP`. Multiple spawns/closes in a single turn
  produce concatenated directive blocks.
- Re-emitting an already-spawned id only re-adds it to the open stack
  (no duplicate directive injection). Closing pops it.

### Tracking state

- `spawned_decisions: set[str]` — every id ever spawned.
- `currently_open_decisions: list[str]` — stack of open ids
  (LIFO; the most recent is "currently_active_decision_id").
- `decision_summaries: dict[str, str]` — first-emitted summary per id.

## Difference from boundaries_easy (subdec_boundaries_easy)

| | boundaries_easy | twoloop_easy |
|---|---|---|
| Marker | `DECISION_OPEN` / `DECISION_CLOSE` | `DECISION_SPAWN` / `DECISION_CLOSE` |
| Injected text on open | One-line "[on opening: identify implicit constraints]" hint | FULL turn-1 enumeration directive scoped to sub-decision |
| Injected text on close | (none) | "[sub-task closed; resume coordinating overall task]" |
| Hypothesis | Brief boundary nudge sufficient to surface implicit facts | Each sub-decision deserves its OWN fresh turn-1 priming, uncontaminated by accumulated retrieval context |

The renamed marker (`SPAWN` vs `OPEN`) signals to the LLM that the
boundary is **a fresh starting point**, not just a checkpoint —
matching the system-prompt language about inner loops.

## Difference from pergate / cogonly (rejected)

- pergate added per-emission gating (an `IMPLICIT_CONSTRAINTS_PROBE`
  output type). Two-loop deliberately does NOT add new probe types;
  the only addition is the marker + harness-side directive.
- cogonly added per-turn cog reflection in `USER_FOLLOWUP`. Two-loop
  leaves `USER_FOLLOWUP` unchanged except for the
  `{spawn_close_directives}` slot — directives only appear AT
  boundaries, not every turn.

## Key trace fields to inspect

Per-turn (in `trace[i]`):
- `decision_spawns` — all spawn lines emitted this turn.
- `new_decision_spawns` — only those whose ids are NEW (the only
  ones that triggered a directive injection).
- `decision_closes` — close lines emitted this turn.
- `spawn_directives_injected` — list of decision ids whose spawn
  directive was injected into the NEXT followup.
- `close_directives_injected` — same for close directives.
- `currently_active_decision_id_at_turn_start` — top of the open-stack
  at the moment this turn's prompt was sent. None if no open inner
  loop.

Per-scenario (top-level `agent_result`):
- `n_decision_spawns_total` — total spawn-line emissions.
- `n_decision_closes_total` — total close-line emissions.
- `n_unique_decisions_spawned` — count of distinct ids spawned.
- `n_spawn_directives_injected` — should equal
  `n_unique_decisions_spawned`.
- `n_close_directives_injected` — total close-directive injections.
- `n_step_outs_orphan` / `orphan_step_labels` — step outputs whose
  raw_label was never spawned (compliance proxy).
- `currently_open_at_end` — ids the model never explicitly closed.
- `spawn_engagement_turns` — convenience list of `{decision_id,
  summary, spawn_turn}` per new spawn for offline engagement analysis.

Cross-scenario (`per_variant_means.mtmsg_capped`):
- `n_decision_spawns_total` / `n_decision_closes_total` /
  `n_unique_decisions_total` / `n_spawn_directives_injected_total` /
  `n_close_directives_injected_total` / `orphan_rate`.

## Open questions for runtime testing

1. **Does the model emit DECISION_SPAWN at all?**
   The system-prompt and turn-1 user prompt both reference it.
   Boundaries_easy got the model to emit `DECISION_OPEN` reasonably
   reliably; the renamed marker shouldn't change that, but watch
   `n_decision_spawns_total`.
2. **Does the FULL directive injection cause runaway nesting?**
   The injected directive itself instructs the model to enumerate
   "further sub-decisions WITHIN this sub-task". If the model treats
   each enumerated bullet as a candidate for ANOTHER `DECISION_SPAWN`,
   we could see exploding tree depth. Track
   `n_unique_decisions_spawned` per scenario — if it exceeds ~6-8,
   we may need to soften the inner directive.
3. **Does the FULL directive cause excessive token spend?**
   The spawn directive is ~150 tokens. With several spawns per
   scenario the followup payload grows. Watch
   `max_thread_tokens` and `truncate_pairs_total` versus
   subdec_split_easy baseline (max thread ~10k cap is shared).
4. **Does the inner turn-1 priming actually reach probe-generation?**
   Engagement proxy: at each new spawn turn, does the FOLLOWING
   turn's probes correlate to the sub-decision summary? Inspect
   per-turn `probes` field after each `new_decision_spawns` event.
5. **Does the model close inner loops, or rely on DONE?**
   Compare `n_decision_closes_total` to `n_unique_decisions_spawned`.
   `currently_open_at_end` non-empty signals lazy closing — possibly
   fine, but worth checking whether the model's outer-loop
   coordination suffers when many inner loops are nominally still open.
6. **Does cov vs R@5 split as expected?**
   subdec_split_easy n=2 baseline is cov 0.934 / full_R@5 0.776.
   Hypothesis: twoloop should at least match cov (sub-decision
   enumeration still happens) and improve R@5 if fresh inner
   priming yields better implicit-constraint probes.

## Env hooks

- `SH_RESULTS_SUBDIR` (default `results`) — per-seed results dir
  (e.g., `results_run2`).
- `SH_SCENARIO_INDICES` (default `0,1,...,9`) — comma-separated
  scenario indices for smoke runs.
- `SH_DB_SUFFIX` (default empty) — appended to SQLite filename so
  multiple seeds don't collide. Final path:
  `<RESULTS_DIR>/eventmemory_shared_harness_v4_mtmsg_twoloop<suffix>.sqlite3`.

## Recommended test order (smoke, then n=2)

1. **One scenario smoke** (e.g., `SH_SCENARIO_INDICES=4` —
   banquet/presentation-style): verify
   - The model actually emits `DECISION_SPAWN` lines.
   - Spawn directive is injected once per unique id.
   - The model engages with the injected directive (probes after
     spawn correlate to the sub-decision).
   - No regex/parser surprises (e.g., DECISION_SPAWN id collides
     with STEP_OUTPUT id parsing).
2. **3-scenario smoke** (`SH_SCENARIO_INDICES=1,4,8`): check that
   `n_unique_decisions_spawned` is bounded (< ~8), `orphan_rate` is
   reasonable (most STEP_OUTPUTs match a spawned id), and overall
   token budget hasn't exploded.
3. **n=1 full bench** (`SH_RESULTS_SUBDIR=results
   SH_DB_SUFFIX=_run1`): full 10-scenario sweep.
4. **n=2 full bench** (`SH_RESULTS_SUBDIR=results_run2
   SH_DB_SUFFIX=_run2`): seed isolation; compare against
   subdec_split_easy n=2 baseline (cov 0.934 / full_R@5 0.776).
