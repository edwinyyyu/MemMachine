# shared_harness_v4_mtmsg_streaming_extcue — Build Notes

## Mechanism: extcue + streaming_v2 layered

Base: `shared_harness_v4_mtmsg_streaming_v2/main.py`. Preserves verbatim:
- Streaming-aware `SYSTEM_PROMPT` (open-ended endless loop, stream messages
  arrive embedded in user followups, premature DONE warning).
- All four user-followup templates (`USER_INITIAL_STREAM`,
  `USER_FOLLOWUP_PROBE_ONLY`, `USER_FOLLOWUP_WITH_STREAM`,
  `USER_FOLLOWUP_STREAM_CLOSED`, `USER_FOLLOWUP_PREMATURE_DONE`).
- Online ingest via `OnlineStreamIngester` (no future leakage; messages
  encoded to EM right before they're handed to the agent).
- LLM-judge DP coverage via `judge_dp_coverage` (gpt-5-mini structured JSON,
  per-emit per-eligible-DP).
- Premature-DONE handling: agent emitting DONE before stream close triggers
  `USER_FOLLOWUP_PREMATURE_DONE` re-prompt up to `MAX_PREMATURE_DONE_NUDGES=3`.

Layered from `shared_harness_v4_mtmsg_extcue_easy/main.py`:
- `external_cue_generator(...)` async function, plus helpers
  `_parse_ext_cues_json`, `_format_recent_history`, `EXT_CUE_SCHEMA`,
  `EXT_CUE_SYSTEM_PROMPT`, `EXT_CUE_USER_TEMPLATE`.
- Replacement-policy probe budgeting: `TOTAL_PROBE_BUDGET_PER_TURN=4`,
  `EXT_CUE_BUDGET=2` (env-overridable via `SH_EXT_CUE_BUDGET`).
- `[EXT]` tag prefix on external-cue hits in the next turn's snippets block.
- Per-turn trace fields: `n_agent_probes`, `n_ext_probes`, `ext_probes`.
  Aggregate fields on agent_result: `n_agent_probes_total`,
  `n_ext_probes_total`. Summary captures `ext_cue_budget` and
  `total_probe_budget_per_turn`.

## "Current focus" construction in streaming context

Different from easy-10. Easy-10 has a single up-front `task_prompt` in the
scenario JSON; the cue generator's user message includes that as the
ORIGINAL TASK. Streaming has no such fixed prompt — work is open-ended and
the agent's "current focus" can shift each turn as new stream messages
arrive.

Adaptation in this variant:
- `task_prompt_for_extcue` is sourced from `scenario.get("task_prompt")` if
  present, else `scenario.get("description")`, else a placeholder string
  `"(open-ended streaming task — no fixed up-front prompt)"`. The cue
  generator's prompt template still renders cleanly.
- `recent_agent_output` is the agent's just-emitted raw turn (THINKING /
  PROBE / STEP_OUTPUT / DONE). This is the same construction extcue uses.
- `recent_history` is `_format_recent_history(mt_messages, k=4)` — the last
  4 user/assistant messages. Critically, in the streaming variant the
  user-side messages embed the
  `--- INCOMING USER STREAM MESSAGE (turn N) ---` blocks. So the cue
  generator can see the most recent stream message the agent is reacting to,
  the prior assistant emit, and the prior probe-results-block. This is the
  intended fix for "current focus is more dynamic" — the generator anchors
  on what the user just said, not just the agent's reaction.
- Cue generation runs ONCE per turn after the agent emits and before
  retrieval, just as in extcue.

## Budget interactions

- Total probe cap: 4 (matches extcue and streaming_v2's `probes[:4]` cap).
- Agent probes fill first, capped at 4. If agent emits 0 probes,
  ext budget = min(2, 4) = 2. If agent emits 3 probes, ext budget = min(2, 1)
  = 1. If agent emits 4 probes, ext budget = 0 (no ext cue gen call).
- `ext_budget_this_turn` is computed BEFORE the cue generator is invoked.
  When 0, the generator call is skipped entirely (no cost).
- When the agent emits DONE this turn, ext cue gen is also skipped (the
  next turn is either premature-DONE re-prompt or stream-closed; ext cues
  wouldn't be useful in either path).

## Open questions for runtime

1. **Does the cue generator paraphrase the most recent stream message?**
   The recent_history slice now contains the verbatim incoming-stream
   block. The generator could degenerate into proposing cues that just
   rephrase the message that just arrived rather than surface IMPLICIT
   past-context facts. The system prompt says "stay close to the agent's
   current focus" but also "target IMPLICIT user-context facts not already
   surfaced". Watch for ext_probes whose semantic content matches the
   message that arrived this turn — if so, they're useless duplicates and
   we may need a "do not paraphrase the most recent stream message" guard
   in the system prompt.

2. **Does the cue generator generate cues when the stream still has future
   messages we don't know about?** The cue generator only sees what the
   agent currently knows. It cannot know the user is about to ask about
   topic X two stream messages from now. So it will generate cues for
   what's in scope right now — which may turn out to be useless if the
   user pivots. This isn't fixable architecturally (no oracle); the open
   question is whether ext cues fired EARLY in a stream still pay off
   later when a related DP arrives, or whether they only help when issued
   AT or AFTER the DP-triggering stream message. Trace inspection should
   correlate ext_probe turn vs DP after_turn.

3. **Cost interaction with premature-DONE loop**: a premature-DONE turn
   triggers a re-prompt; the agent's next emit could trigger another ext
   cue gen call. With `MAX_PREMATURE_DONE_NUDGES=3` and 28 max turns, ext
   cue gen could fire on every non-saturated turn. We currently skip ext
   gen when `done_this_turn` is True (since the next turn is the re-prompt
   reaction). Whether that's the right gating to validate at runtime.

4. **Online-ingest timing vs ext gen**: ext gen runs AFTER the agent's
   emit but BEFORE retrieval. The next stream message hasn't been
   delivered or ingested yet. So cues fired this turn search a memory
   that contains all stream messages up to and including the most recent
   one the agent has seen — this is correct (no future leakage).

## Files

- `main.py` — combined harness.
- `results/` — per-scenario JSON outputs (created on first run).
- `SUMMARY.json` — cross-scenario summary written at end of run.
- SQLite at `evaluation/associative_recall/results/eventmemory_shared_harness_v4_mtmsg_streaming_extcue.sqlite3`.

## Smoke-run command (for reference, NOT executed in this build)

```
uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_extcue/main.py --scenario family-dinner-stream-01
```
