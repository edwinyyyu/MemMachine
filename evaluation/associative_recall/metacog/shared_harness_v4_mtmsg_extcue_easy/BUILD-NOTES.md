# shared_harness_v4_mtmsg_extcue_easy — BUILD NOTES

## Mechanism

External cue generator: a separate `gpt-5-mini` call invoked once per
agent turn, AFTER the main agent emits its turn output and BEFORE
retrieval runs. Its prompt scope is decoupled from the main agent's
conversation thread — it sees only:

1. The original task prompt.
2. The agent's most recent turn output (used as "current focus").
3. A short rolling abbreviation of the last ~4 user/assistant messages
   from the main thread (so it has minimal continuity context).
4. The list of probes the agent already issued this turn (so it can
   avoid duplicates).

Output is structured JSON `{cues: [str, ...]}` validated via
`response_format=json_schema` with `reasoning_effort="low"`. On parse or
LLM error the function returns `[]` and the turn proceeds with agent
probes alone (graceful degradation).

The system prompt, USER_INITIAL, and USER_FOLLOWUP for the main agent
are IDENTICAL to `shared_harness_v4_mtmsg_subdec_split_easy`. The only
mechanism added is the external generator. This isolates the
external-generator effect from prompt-level confounds.

## Replacement vs addition policy

**REPLACEMENT (chosen for iteration 7).**

- Total per-turn probe budget held at `TOTAL_PROBE_BUDGET_PER_TURN = 4`
  (matches subdec_split's existing cap).
- Agent probes fill first, capped at 4.
- External cues fill remaining slots, capped at `EXT_CUE_BUDGET`
  (default 2).
- If the agent already saturates the budget (issues 4+ probes), no
  external cues are added that turn.

Rationale: tests "decoupled cognition" without expanding the probe
surface. Probe budget expansions confound architectural ablations
because more probes mean more retrieval surface, which can flood
top-K with distractors (per `project_q1_retrieval_ablation` and the
pergate failure mode).

External cues are tagged `[EXT]` in the snippet rendered to the agent
in the next user turn; the trace also records `ext_probes` and
`n_ext_probes` per turn for attribution.

## External cue generator prompt (principle-level, deployed)

System prompt (truncated for clarity; full text in `EXT_CUE_SYSTEM_PROMPT`):

> You generate retrieval queries for a memory-augmented agent. The
> agent is working on a multi-step task and consults a memory store of
> past chat turns where the user has shared specific facts
> (constraints, preferences, allergies, dates, numbers, identities)
> that materially shape correct deliverables.
>
> Your job: given the agent's current focus (recent thinking + most
> recent deliverable), propose 1-2 retrieval queries that target
> IMPLICIT USER-CONTEXT FACTS — facts the user has shared previously
> that the task description itself does not name, but which would
> change what a correct answer looks like for the agent's current
> focus.
>
> Discipline:
> - Target facts the agent's current working memory does NOT already
>   show. If the agent has already surfaced and is reasoning about a
>   relevant fact, do NOT re-probe for it.
> - Be concrete and specific. Prefer queries that name a plausible
>   value (e.g., "user prefers Thursday afternoons") over abstract
>   ones ("user's scheduling preferences"). Memory retrieves by
>   semantic similarity, so words in the query should resemble words a
>   stored fact would use.
> - Stay close to the agent's current focus.
> - Each query is a short retrieval probe, not a question to the user.
>
> Output ONLY a JSON object: `{"cues": ["<probe 1>", "<probe 2>"]}`.
> Maximum {budget} cues. Fewer is fine. Empty list is acceptable.

User template provides: original task, agent's current focus (latest
raw output), recent history snippet, agent probes already issued.

The principle-level framing follows the cue-worthiness classifier
result (`project_cue_worthiness_classifier`) — terse principle-only
prompts beat example-laden variants.

## Env hooks added

- `SH_RESULTS_SUBDIR` (default `results`) — per-run results subdir
  under `THIS_DIR`.
- `SH_SCENARIO_INDICES` (comma-separated) — subset of scenario indices
  from the easy bench. Empty → all 10.
- `SH_DB_SUFFIX` (default empty) — appended to the SQLite filename
  (`eventmemory_shared_harness_v4_mtmsg_extcue{suffix}.sqlite3`) so
  multiple seeds don't collide.
- `SH_EXT_CUE_BUDGET` (default `2`) — cap on external cues per turn.

## Open questions (for runtime testing)

1. **Replacement vs addition.** Iteration 7 starts with replacement
   (no budget expansion). If signal looks promising but cues are
   getting crowded out by saturated agent probes, retest with
   addition: `EXT_CUE_BUDGET=2` ON TOP of the existing 4 (i.e., raise
   `TOTAL_PROBE_BUDGET_PER_TURN=6`). That introduces a budget
   confound but separates "decoupled cognition" from "more retrieval
   surface."

2. **Cue budget calibration.** Default `EXT_CUE_BUDGET=2`. If the
   external generator returns garbage / shallow paraphrases of agent
   probes, drop to 1 (smaller perturbation). If it consistently emits
   high-quality misses-the-agent-didn't-cover, raise to 3 — but
   replacement policy already caps absolute extras at
   `TOTAL_PROBE_BUDGET - len(agent_probes)`.

3. **Cue quality.** Per-turn external generator runs even when agent
   probes are already strong. Need to inspect traces to see whether
   external cues redundantly mirror agent probes vs. genuinely fill
   gaps. The "do not duplicate" instruction depends on the LLM
   honoring it — empirically check.

4. **Latency cost.** One extra `gpt-5-mini` call per turn × ~10 turns
   per scenario × 10 scenarios = ~100 extra mini calls per run.
   Reasoning-effort=low keeps latency low, but per-scenario wall-clock
   will rise. Worth tracking.

5. **What does the generator see?** The recent-history window is
   abbreviated to ~4 messages × 600 chars each. If the agent's
   current focus is buried deeper in the thread, the generator may
   miss it. Consider passing the agent's last STEP_OUTPUT (most recent
   sub-decision deliverable) explicitly as a separate field.

6. **Failure mode: empty cue lists.** When generator returns `[]`,
   that turn behaves identically to subdec_split. If `[]` dominates
   (high refusal rate), mechanism is a no-op and we'll see noise
   parity. Check `n_ext_probes` distribution across the trace.

7. **Interaction with done_emitted.** External generator is suppressed
   on the turn the agent emits `DONE` (we don't probe further). This
   matches "no point spawning cognition if we're terminating."
