# shared_harness_v4_mtmsg_compaction_motivation — BUILD-NOTES

Integrated harness combining three independently built components for
iteration-9 ablation runs on the perpetual-stream benchmark.

## Integration topology

Three components compose end-to-end:

1. **Perpetual benchmark** —
   `evaluation/associative_recall/data/perpetual_scenarios.json`
   - 1 scenario `extended-project-stream-01`, 40 stream messages, 10 decision
     points, ground-truth + distractor facts. Schema is byte-compatible with
     `streaming_scenarios.json` (`messages[*]={turn,role,text}`,
     `decision_points[*]={after_turn,sub_decision,required_facts,...}`,
     `ground_truth_facts[*]={id,text,from_turn}`).
   - The integrated harness selects this file via the `SH_SCENARIO_FILE` env
     var or `--scenarios-file` flag.

2. **Compaction-aware harness** —
   forked from `shared_harness_v4_mtmsg_compaction/main.py`
   - At turn-start, when `mt_messages` token count exceeds
     `SH_COMPACTION_THRESHOLD * MT_HARD_CAP`, the OLDEST half of evictable
     pairs is fed to a `gpt-5-mini` compactor LLM (structured `json_schema`)
     which emits `{narrative, facts, sub_decisions, retrieval_cues}`.
   - Each fact is written to `EventMemory` as a new `compacted_fact` event
     tagged with `compacted_from_turn_range`. The evicted span is replaced
     with one user-style summary message.
   - `SH_DISABLE_COMPACTION=1` disables the compactor and falls back to
     `truncate_thread()` (verbatim streaming-v2 hard-truncate).

3. **Motivation generator** —
   `evaluation/associative_recall/metacog/motivation/motivation.py`
   - One `gpt-5-mini` `json_schema` call returning
     `{state, intensity, rationale, drive_directive}` over a fixed six-state
     enum (`curious / focused / anxious / restless / satisfied / bored`).
   - Module is standalone — no harness imports — and protected by Python-side
     dwell-floor (`MIN_DWELL_TURNS=3`) and forced-rotation
     (`FORCED_ROTATION_TURNS=8`) guards.

### How the integrated harness composes them

In `run_agent_loop`:

- **Turn-start compaction check** (unchanged from compaction harness).
  If a compaction fires this turn, `last_compaction_turn = turn` and the
  compaction record is stashed.

- **Agent emits assistant turn**. Probes ran, snippets gathered, etc.

- **Motivation update decision** (NEW). After probes/snippets but before the
  next user followup is appended, decide whether to call
  `update_motivation`:
  - **Trigger A — post-compaction**: a compaction fired THIS turn.
    Treated as the "sleep consolidation" moment.
  - **Trigger B — periodic**: `turn - last_motivation_update_turn >=
    SH_MOTIVATION_PERIOD` (default 5).
  - If neither triggers, motivation state persists. (If both triggered on
    the same turn, we still update once — they are not double-counted.)

  When called, inputs are derived as:
  - `recent_activity_summary`: latest compaction's `narrative` if a
    compaction happened within the last `SH_MOTIVATION_PERIOD` turns,
    otherwise the joined last-3 `THINKING:` block bodies parsed from the
    agent's transcript.
  - `unresolved_goals`: empty list (LIMITATION — see below).
  - `turns_since_last_user_input`: counter incremented each turn, reset
    to 0 when a `INCOMING USER STREAM MESSAGE` is delivered.
  - `turns_since_last_completion`: counter incremented each turn, reset
    to 0 on any `STEP_OUTPUT` emission.
  - `turns_since_motivation_update`: `turn - last_motivation_update_turn`.
  - `current_turn`: agent loop's `turn`.

- **Followup injection** (NEW). When motivation is enabled, the next user
  followup body is prefixed with one motivation line:

  ```
  [CURRENT MOTIVATION: <state>, intensity <intensity>. <drive_directive>]

  <existing followup body, unchanged>
  ```

  The prefix is REFRESHED on every turn from the persisting `current_motivation`
  (so the directive is present each turn even between explicit updates).
  When `SH_DISABLE_MOTIVATION=1`, no prefix is injected.

- **Trace bookkeeping** (NEW): each `TurnLog` carries
  `motivation_updated_this_turn / motivation_state_after / motivation_intensity_after`.
  The agent_result returns `motivation_events` (list of
  `{at_turn, trigger, state, intensity, rationale, drive_directive, error}`)
  and `final_motivation`.

The SYSTEM_PROMPT and all USER_FOLLOWUP_* templates are **byte-equal** to
streaming v2 / the compaction harness — the only delta is the optional
prefix on followup `content` strings.

## Env vars exposed

| Env var                     | Default                                      | Purpose                                                                          |
| --------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------- |
| `SH_SCENARIO_FILE`          | (unset → `data/streaming_scenarios.json`)    | Path to scenarios JSON. Relative paths resolved against repo cwd then `data/`.   |
| `SH_RESULTS_SUBDIR`         | `results`                                    | Subdir under harness dir for per-scenario JSON output.                           |
| `SH_DB_SUFFIX`              | `""`                                         | Suffix on the SQLite DB filename (parallel-run isolation).                       |
| `SH_DISABLE_COMPACTION`     | `0`                                          | When `1/true/yes`, disables compaction → hard-truncate baseline.                 |
| `SH_DISABLE_MOTIVATION`     | `0`                                          | When `1/true/yes`, disables motivation generator and prefix injection.           |
| `SH_MOTIVATION_PERIOD`      | `5`                                          | Periodic motivation-update cadence (in agent turns).                             |
| `SH_COMPACTION_THRESHOLD`   | `0.85`                                       | Trigger compaction when `messages_tokens > threshold * MT_HARD_CAP`.             |
| `SH_COMPACTION_KEEP_RECENT` | `3`                                          | User/assistant pairs kept intact at end of thread (eligible for compaction = older half above this tail). |
| `OPENAI_API_KEY`            | (must be set; loaded from repo `.env`)       | OpenAI auth.                                                                     |
| `QDRANT_HOST` / `_PORT` / `_GRPC_PORT` | `localhost` / `6333` / `6334`     | Qdrant vector store endpoint.                                                    |

CLI flags `--scenarios-file`, `--scenario`, `--stream-interval`, `--max-turns`,
`--no-overwrite`, `--sqlite-suffix` are also supported.

Distinct EM/SQLite namespaces vs sibling harnesses:
- vector-store namespace: `arc_em_compmot`
- collection prefix: `arc_cm`
- sqlite filename: `eventmemory_shared_harness_v4_mtmsg_compaction_motivation<SH_DB_SUFFIX>.sqlite3`

## Recommended ablation matrix (iteration 9)

Target: 1 perpetual scenario × 3 configurations × N seeds.

The perpetual scenario has 40 stream messages — at the default
`STREAM_INTERVAL=2`, the agent needs ~80 agent turns to receive the full
stream. Bump `--max-turns 80` (or higher) for perpetual runs; the default
28 is tuned for short streaming scenarios and will only deliver ~14 stream
messages.

### A — BASELINE (hard-truncate, no motivation)

```bash
SH_SCENARIO_FILE=evaluation/associative_recall/data/perpetual_scenarios.json \
SH_RESULTS_SUBDIR=results_baseline SH_DB_SUFFIX=_baseline \
SH_DISABLE_COMPACTION=1 SH_DISABLE_MOTIVATION=1 \
uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_compaction_motivation/main.py \
  --max-turns 80
```

Isolates: pure streaming-v2 retrieval-only behavior over the long stream.
Expected to suffer once the front of the thread is hard-dropped — early
ground-truth facts only recoverable via PROBE.

### B — COMPACTION (compaction on, motivation off)

```bash
SH_SCENARIO_FILE=evaluation/associative_recall/data/perpetual_scenarios.json \
SH_RESULTS_SUBDIR=results_compaction SH_DB_SUFFIX=_compaction \
SH_DISABLE_MOTIVATION=1 \
uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_compaction_motivation/main.py \
  --max-turns 80
```

Isolates pillar 2 (compaction). Compactor extracts facts and seeds them
into EM with `event_type=compacted_fact`; agent should be able to recover
them via PROBE.

### C — FULL (compaction on, motivation on)

```bash
SH_SCENARIO_FILE=evaluation/associative_recall/data/perpetual_scenarios.json \
SH_RESULTS_SUBDIR=results_full SH_DB_SUFFIX=_full \
uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_compaction_motivation/main.py \
  --max-turns 80
```

Composes pillars 2 + 3. Tests whether the motivation directive actually
shifts agent behavior on the perpetual stream, and whether it composes
constructively with compaction.

Recommended order: A → B → C, comparing pairwise (B−A isolates compaction;
C−B isolates motivation; C−A is full lift). Run ≥2 seeds per cell to
establish a noise floor (gpt-5-mini stochasticity, see
project_extractor_stochasticity memory).

## Limitations / open implementation notes

- **`unresolved_goals` is empty.** The compaction-fork harness does not
  itself maintain an open-sub-decisions tracker. The motivation generator's
  prompt expects `unresolved_goals` to be a list of strings; we pass `[]`.
  The compactor DOES emit `sub_decisions` with state `opened|closed|active`
  on each compaction event — wiring those into a rolling
  `unresolved_goals` set is a natural next iteration. For iteration 9 we
  rely on `recent_activity_summary` + the compaction narrative to give the
  motivation generator enough context.

- **Motivation directive is REFRESHED every turn**, even when no new update
  was made. This means once the generator picks a state, that state's
  directive line keeps appearing on every followup until the next update
  changes it. Alternative: only inject on turns when the state was just
  updated. Iteration 9 should observe whether constant-prefix is too
  invasive.

- **Anti-paraphrase.** The motivation prompt aggressively rotates state.
  Combined with compaction-triggered updates, the agent may see frequent
  state changes early in a perpetual run when token-cap pressure is high.

- **Default `MAX_TURNS=28`** is tuned for short streaming scenarios. For
  perpetual you want ≥80.

## Open questions for iteration-9 runtime testing

1. **Does the motivation directive actually change agent behavior on the
   perpetual stream?** Compare PROBE selection, STEP_OUTPUT phrasing, and
   coverage between B and C. Look for diffs that correlate with the
   `motivation_state_after` field per turn.

2. **Does the compaction narrative help future probe selection?** Compare
   the agent's PROBE strings before vs after a compaction event in B.
   Specifically: do `retrieval_cues` from the compaction surface as
   subsequent PROBE queries?

3. **Sub-additive, additive, or super-additive composition?** Compute
   `n_dp_covered_C - n_dp_covered_A` vs
   `(n_dp_covered_B - n_dp_covered_A) + (n_dp_covered_C - n_dp_covered_B)`.
   If composition is sub-additive, motivation may be displacing
   probe-budget. If super-additive, motivation directives may be steering
   the agent toward exactly the slots compaction surfaced.

## Quick verification recipe

Schema/scenario load (no API calls):

```bash
python3 -c "
import json
from pathlib import Path
s = json.loads(Path('evaluation/associative_recall/data/perpetual_scenarios.json').read_text())[0]
print(f'id={s[\"id\"]} n_msgs={len(s[\"messages\"])} n_dps={len(s[\"decision_points\"])}')"
```

Syntax check:

```bash
python3 -m py_compile evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_compaction_motivation/main.py
```

Both have been run successfully at build time.
