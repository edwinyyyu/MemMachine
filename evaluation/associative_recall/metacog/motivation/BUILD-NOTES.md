# motivation — BUILD NOTES

Standalone motivation generator for autonomous agents. Lives at
`evaluation/associative_recall/metacog/motivation/`. Imports nothing
from any harness; harnesses import this module.

## Mechanism

One `gpt-5-mini` call per invocation, `reasoning_effort="low"`,
structured `response_format=json_schema` (`MOTIVATION_SCHEMA`). The
generator picks one of six fixed categorical states and emits an
imperative `drive_directive` for injection into the agent's prompt.

A thin Python wrapper enforces dwell-time and forced-rotation guards
on top of the LLM's choice. Failure (network, parse, invalid enum) is
non-fatal: the prior state is returned unchanged.

## States

`curious | focused | anxious | restless | satisfied | bored`

Mutually exclusive; defined in `VALID_STATES` and enforced in the
schema as an `enum`.

## Prompt template (verbatim — see `motivation.py` for the live string)

System prompt (high-level structure):

1. Role + the six categories (one-line each).
2. Inputs description.
3. **Decay / transition rules** (six rules, one per state). E.g.:
   - `focused` → `restless` if no progress despite long dwell.
   - `anxious` → `focused` once urgency-source is no longer present.
   - `satisfied` → `focused` (goals open) or `bored` (no goals) when
     completion is no longer fresh.
   - `bored` → `curious` if a latent topic of interest surfaces.
   - `restless` → `curious`/`focused` on a fresh angle.
   - `curious` → `focused` when a concrete goal dominates.
4. **Dwell and rotation rules** (hard):
   - MIN dwell ~3 turns unless an external trigger fires
     (new-user-input, just-completed, sharp-new-urgency).
   - FORCED rotation if `turns_since_motivation_update > 8`.
5. **Anti-paraphrase rule**: default to shifting; if keeping the input
   state, the rationale MUST justify why no decay rule fires; the
   `drive_directive` MUST be imperative ("Because you're <state>
   about <object>, do <action> now."), forbid "you might consider"
   etc., require concrete object from inputs.
6. **Intensity calibration**: default 0.4–0.7; >0.8 only on clear
   peaks; cap drift at ~0.2 between calls.

User template (filled in per call): current_state block (state /
intensity / since_turn / prior_rationale), `current_turn`,
`turns_since_motivation_update`, `turns_since_last_user_input`,
`turns_since_last_completion`, `recent_activity_summary`, and
`unresolved_goals`.

## State-transition table (encoded in prompt + Python guards)

| From → To  | Trigger condition                                                  | Enforced where         |
|------------|--------------------------------------------------------------------|------------------------|
| focused → restless     | high `turns_since_last_completion`, no progress in summary       | prompt rule 1           |
| anxious → focused      | urgency-source absent from summary AND goals                      | prompt rule 2           |
| satisfied → focused    | completion no longer fresh AND goals non-empty                    | prompt rule 3           |
| satisfied → bored      | completion no longer fresh AND goals empty                        | prompt rule 3           |
| bored → curious        | new entity / question / anomaly in summary                        | prompt rule 4           |
| restless → curious/focused | new approach visible in summary                              | prompt rule 5           |
| curious → focused      | concrete goal dominates                                            | prompt rule 6           |
| any → any              | held > FORCED_ROTATION_TURNS (8) → MUST switch                     | prompt + Python fallback (`_force_rotation_if_stuck`) |
| any → switch blocked   | held < MIN_DWELL_TURNS (3) AND no external trigger                 | Python (`_enforce_dwell`) — reverts categorical state, keeps new intensity/rationale/directive |

External triggers that bypass the dwell minimum (Python
`_trigger_present`):
- `turns_since_last_user_input == 0` (fresh input may shift topic),
- `turns_since_last_completion == 0` (a completion just happened →
  natural switch toward `satisfied`),
- `turns_since_motivation_update > FORCED_ROTATION_TURNS` (forced
  rotation overrides dwell).

Forced-rotation fallback map (used only when the LLM ignores its own
prompt rule and the dwell ceiling is exceeded):
`curious→focused, focused→restless, anxious→focused, restless→curious,
satisfied→bored, bored→curious`.

## Anti-failure design — recap

| Failure mode                | Mitigation                                                        |
|-----------------------------|--------------------------------------------------------------------|
| Sticky motivation           | Forced-rotation prompt rule + Python ceiling guard                 |
| Wandering motivation        | Min-dwell prompt rule + Python `_enforce_dwell` floor              |
| Generator paraphrases input | "Default to shifting; justify any non-shift" + JSON enum forces categorical commitment |
| Acknowledged but not acted  | `drive_directive` schema description demands imperative shape; system prompt bans "you might consider" etc. |
| Domain bias in examples     | No domain-specific examples in the prompt — only abstract rules    |
| LLM call failure            | Returns prior state unchanged; logged at WARNING                   |

## Harness integration sketch (5 lines)

```python
from evaluation.associative_recall.metacog.motivation import (
    initial_motivation_state, update_motivation,
)

# Once at task start:
mot = initial_motivation_state(turn=0)

# Once per agent turn (or every K turns), BEFORE building the agent prompt:
mot = await update_motivation(
    openai_client,
    current_state=mot,
    recent_activity_summary=last_compaction_narrative or last_5_turns_summary,
    unresolved_goals=list_of_open_subdec_ids_or_descriptions,
    turns_since_last_user_input=turns_since_last_user_input,
    turns_since_last_completion=turns_since_last_completion,
    turns_since_motivation_update=current_turn - mot.since_turn,
    current_turn=current_turn,
)

# Inject the directive — either prepend to SYSTEM_PROMPT or add to per-turn user followup:
system_with_drive = SYSTEM_PROMPT + "\n\nMOTIVATION: " + mot.drive_directive
```

## Open questions for runtime testing

1. **Does it actually shift?** Across a multi-turn run, does the
   categorical `state` move through ≥3 distinct values, or does it
   collapse onto one ("focused" forever)?
2. **Does dwell/rotation behave?** Do early-switch attempts get
   suppressed by `_enforce_dwell`? Does `_force_rotation_if_stuck`
   ever fire, or does the LLM rotate on its own?
3. **Does the directive translate to behavior?** Does the agent's
   next-turn output visibly track the directive (does an "anxious
   about deadline X — close out X now" directive cause the agent to
   prioritize X), or does it get acknowledged-but-ignored?
4. **Intensity calibration.** Does intensity actually move with
   inputs, or does it flatline near the default 0.5?
5. **Cost / latency.** ~one extra `gpt-5-mini` call per turn at
   `reasoning_effort=low` — acceptable on long runs?
6. **Trigger-sourcing.** What is the right source for
   `recent_activity_summary` — last compaction's narrative, last K
   turns concatenated, or a separate summary call? This affects
   whether decay rules can actually fire.
7. **Goal-source.** Where do `unresolved_goals` come from in each
   harness — open sub-decision IDs, listener notes, planner state?
8. **Update cadence.** Every turn vs. every K turns vs. event-driven
   (only after a completion / new user input). Generator is cheap
   enough that per-turn is plausible, but per-K may be saner.
