# shared_harness_v4_mtmsg_streaming_v2 — build notes

Second-iteration streaming harness. Inherits everything from
`shared_harness_v4_mtmsg_streaming/main.py` (system prompt, parsers,
truncation, probe rendering, scheduler skeleton) and changes three
mechanisms.

## What changed vs v1

### 1. Online stream ingest

- v1: `ingest_streaming_scenario` pre-encoded ALL stream messages into
  Event Memory before the agent loop started. PROBES at turn 1 could
  surface text the agent had not yet been shown.
- v2: `open_empty_memory` creates a fresh empty EM collection. An
  `OnlineStreamIngester` (constructed via `online_stream_ingester(scenario,
  em_session)`) holds a reference to the memory and exposes
  `await ingester.ingest(stream_turn)`. The agent loop calls this exactly
  once per stream message — at the same turn the message is delivered to
  the agent, and BEFORE the LLM call that handles that message. This
  guarantees: at every PROBE the memory contains exactly the stream
  messages the agent has seen so far, never more.
- Event property layout matches v1 (`scenario_id`, `turn_id`, `speaker`,
  `event_type=stream_message`, `plant_id=stream_turn_<N>`, `from_turn`).
  Probe-result rendering is unchanged.

### 2. LLM-judge DP coverage

- v1: `annotate_step_output` matched STEP_OUTPUT label/content tokens
  against `decision_point.sub_decision` strings (>=1 token overlap). This
  over-credited generic content that happened to share a stem with the
  sub-decision label.
- v2: `judge_dp_coverage(step_output_text, required_facts,
  gold_text_for_facts)` calls `gpt-5-mini` with `response_format=
  json_schema` (`DP_JUDGE_SCHEMA`) and `reasoning_effort="low"`. The
  prompt asks whether the deliverable text **reflects** the gold
  required-fact content while addressing the sub-decision area. Multiple
  required facts must ALL be reflected for `covered=true`.
- The judge runs at every STEP_OUTPUT emit, against every DP that is
  *eligible at emit time* (`after_turn <= stream_turns_delivered`).
  Per-emit judgements are stored on the step (`per_dp_judgements`) and
  aggregated at run-end into `decision_point_coverage[*].covered` (true
  iff ANY emit was judged covered for that DP — the latest-emit revision
  wins, but earlier-covered also counts so the agent isn't penalized for
  later edits).
- Older deployments that reject `reasoning_effort` are handled with a
  retry that strips the kwarg.

### 3. Premature-DONE handling

- v1: `done_emitted` broke the agent loop unconditionally (with a
  `premature` flag in the trace).
- v2: when DONE arrives before `stream_closed_announced`, the harness
  emits a `USER_FOLLOWUP_PREMATURE_DONE` user message instead of
  breaking. The nudge says "a new user message is still expected — keep
  working on what you've heard so far and watch for the next message"
  and reminds the agent not to DONE until told the stream is closed.
  Stream advancement is paused on premature-DONE turns so the nudge has
  a chance to land before the next stream message arrives.
- A safety cap (`MAX_PREMATURE_DONE_NUDGES=3`) prevents infinite loops
  if the agent keeps insisting on DONE.

The system prompt is byte-identical to v1 (verified at build time via a
side-by-side import). No prompt redesign in this iteration.

## Output schema additions vs v1

Per scenario JSON file in `results/<scenario_id>.json`:

- `agent_result.dp_coverage_log` — every per-(emit, DP) judgement row.
- `agent_result.decision_point_coverage[*]` now reports `covered`
  (bool — was the DP ever judged covered), `n_emits_judged_eligible`,
  `n_emits_covered`, `first_covered_*` and `latest_covered_*` instead of
  v1's `n_matching_emits` / `matching_emit_step_ids`.
- `agent_result.n_premature_done_nudges` — count of premature DONE nudges.
- `agent_result.done_at_close` — bool: agent's final action was DONE
  AFTER stream-close.
- `agent_result.n_stream_messages_ingested` — should equal
  `n_stream_messages_delivered` in healthy runs.
- Per-turn `stream_msg_ingested_this_turn` — confirms ingestion happened
  on the same turn as delivery.

`SUMMARY.json` reports per-scenario `n_dp_covered`, `n_dp_total`,
`n_premature_done_nudges`, `stream_msgs_delivered`, `stream_msgs_ingested`.

## How to invoke

Required env (same as v1):

- `OPENAI_API_KEY` — judge + agent both call gpt-5-mini.
- `QDRANT_HOST` (default localhost), `QDRANT_PORT`, `QDRANT_GRPC_PORT`.
- `.env` at repo root is auto-loaded.

Default run (uses v1 scenarios file):

```
uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_v2/main.py
```

Run only the new scenario from `streaming_scenarios_v2.json`:

```
uv run python evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_v2/main.py \
    --scenarios-file streaming_scenarios_v2.json \
    --scenario home-renovation-stream-01
```

Useful flags:

- `--scenarios-file` — relative to `evaluation/associative_recall/data/`,
  or absolute path. Default: `streaming_scenarios.json`.
- `--scenario` — single scenario id; default runs all.
- `--stream-interval` — agent turns between stream-message deliveries
  (default 2).
- `--max-turns` — total agent-turn budget (default 28; bumped from v1's
  24 because premature-DONE nudges can extend runs).
- `--no-overwrite` — reuse existing EM collections (faster reruns).

Outputs land in `results/<scenario_id>.json` and `SUMMARY.json` in this
directory.

## Open questions for runtime testing

These need an actual run to resolve and were intentionally not tested
during the build:

1. **Judge calibration on multi-fact DPs.** The judge prompt insists
   ALL required facts be reflected for `covered=true`. With 5-fact DPs
   like `whole_job_contractor_brief` the judge may be either too strict
   (deliverables that hit 4/5 are still real progress) or too lax (LLM
   coalesces "respects budget + accessibility" into a single
   evidence-quote without checking each fact). May need either a
   per-fact judge call, or a per-fact `covered_facts: [fid, ...]` field
   in the schema.
2. **Premature-DONE prevalence.** Unknown how often the agent will fire
   DONE early in v2 vs v1. The cap is 3 nudges before honoring DONE; if
   it's hitting that often, either the prompt is leaking DONE-bias or
   the cap is too low.
3. **Online ingest latency.** Each delivery now blocks on
   `memory.encode_events([single_event])` before the LLM call. Probably
   negligible (<0.5s embedding + insert) but worth confirming.
4. **DP eligibility rule under premature-DONE.** When the agent emits
   STEP_OUTPUT during a premature-DONE turn, that emit IS judged
   against eligible DPs. That's intentional (the work still ships), but
   may interact oddly with later revisions if the cap fires.
5. **`home-renovation-stream-01` calibration.** The scenario has not
   been run end-to-end. Difficulty distribution (DP1 medium-early, DP4
   hardest at end) is by design; whether it's actually harder than
   `wedding-planning-stream-01` depends on judge behavior on the
   contractor-brief 5-fact integration.
6. **Recall-vs-coverage divergence.** The judge measures whether the
   deliverable *reflects* the fact, not whether the agent *retrieved*
   the fact via PROBE. Side-by-side per-DP judge-coverage and
   PROBE-recall stats are not currently aggregated in `SUMMARY.json`;
   add if useful.
