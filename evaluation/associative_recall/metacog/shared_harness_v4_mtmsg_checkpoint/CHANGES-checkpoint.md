# shared_harness_v4_mtmsg_checkpoint

Forked from `shared_harness_v4_mtmsg_revisit`. Replaces the
single-oldest-commitment revisit scheduler with a **batch checkpoint
review** scheduler: every K stream messages delivered, the harness pauses
and asks the agent to triage ALL currently-open commitments at once.

## Why a new variant

After 11 iterations of perpetual-execution exploration the
bottleneck is "commitment abandonment" — the agent gets pulled by new
stream messages and forgets earlier-opened sub-decisions. Two
scheduling primitives are now under comparison:

| Variant                                 | Primitive                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `shared_harness_v4_mtmsg_revisit`       | Re-surface the SINGLE oldest commitment every few turns ("nag, by recency").                                              |
| `shared_harness_v4_mtmsg_checkpoint`    | Pause every K stream messages, review ALL open commitments in one structured pass ("planning meeting, by batch").         |
| `shared_harness_v4_mtmsg_compaction`    | Baseline: no scheduling at all; open commitments may simply linger.                                                       |

Compaction is ON. Motivation is OFF by default (per project learnings;
iter10 motivation injection regressed coverage).

## Mechanism

### Trigger

`SH_CHECKPOINT_PERIOD` (default 10) — every N stream messages delivered.
Internally the harness tracks `stream_msgs_delivered_count` (starts at 1
because turn 1 delivers the first stream message). It also tracks
`last_checkpoint_multiple_fired`. A checkpoint fires the first turn at
which `stream_msgs_delivered_count // SH_CHECKPOINT_PERIOD` exceeds
`last_checkpoint_multiple_fired`.

`SH_DISABLE_CHECKPOINT=1` ablates the scheduler entirely.

Eligibility constraints:
- Scheduler enabled
- At least one currently-open commitment to review (if none, the
  multiple-marker is still advanced so we don't refire every turn until
  something opens)
- Not piggybacked on a `PREMATURE_DONE` followup (don't pile on)
- Allowed to coexist with new-stream-message delivery or stream-closed
  followups; the checkpoint is a batch triage moment, not a fallback.

### Injection (the exact prompt template)

When a checkpoint fires, the harness appends this block to the next
followup, after the optional motivation prefix and the body:

```
[CHECKPOINT REVIEW (turn N): Below are all currently-open commitments. For each, classify and act:
1. <label1> (opened turn X1, age Y1 turns): <summary1>
2. <label2> (opened turn X2, age Y2 turns): <summary2>
...

For each commitment listed above, in this turn's response you MUST include exactly one line of the form:
  CHECKPOINT_DECISION: <label>: READY_TO_DELIVER | STILL_WAITING | ABANDON

If READY_TO_DELIVER, also emit a `STEP_OUTPUT: <label>: ...` addressing it this same turn. If ABANDON, the harness will close the commitment in its tracker. If STILL_WAITING, briefly state in THINKING what info is needed before it can be delivered.]
```

Commitments are listed chronologically (oldest first) using their
`opened_at_compaction_turn` field. Items with no summary print only the
label and `(opened turn X, age Y turns)`.

### Parsing the response

On the agent's NEXT turn the harness:

1. Parses `CHECKPOINT_DECISION: <label>: <verdict>` lines via
   `CHECKPOINT_DECISION_LINE_RE`. Verdict is upper-cased.
2. Matches each parsed label against the listed labels with a tolerant
   match (exact case-insensitive, then substring either direction). This
   covers the common case where the agent paraphrases or truncates a
   label.
3. For each match:
   - **READY_TO_DELIVER**: look at this turn's `STEP_OUTPUT`s. If any
     matches the ready label (exact / substring on raw_label OR
     substring of the label inside the content), close the commitment in
     the tracker with `reason=closed_via_checkpoint_ready_step_output`.
   - **STILL_WAITING**: no harness-side action; the commitment stays
     open.
   - **ABANDON**: close the commitment with
     `reason=abandoned_via_checkpoint`.
4. Unrecognized labels (the agent invents a label not in the list) are
   counted in `checkpoint_n_unrecognized_label` but don't otherwise act.

### Trace fields added

Per turn (`TurnLog`):

- `checkpoint_injected_after_this_turn`: bool — did we inject a
  checkpoint at the END of this turn?
- `checkpoint_n_commitments_listed`: int — number of commitments printed
  in the injection
- `checkpoint_response_this_turn`: bool — was this turn classified as a
  response to a pending checkpoint?
- `checkpoint_decisions_parsed`: list of `{label, verdict}` from the
  agent's output
- `checkpoint_n_ready`, `_n_waiting`, `_n_abandon`: counts per verdict
- `checkpoint_n_unrecognized_label`: count of decisions whose label
  didn't match any listed commitment
- `checkpoint_n_ready_with_step_output`: count of READY_TO_DELIVER
  labels for which a matching STEP_OUTPUT was emitted this turn

Per-event (`checkpoint_events`):

- `at_turn`, `n_commitments_reviewed`, `listed_labels`,
  `stream_msgs_delivered_at_trigger`
- `agent_responded`, `responded_at_turn`
- `n_classified`, `n_ready`, `n_waiting`, `n_abandon`,
  `n_unrecognized_label`
- `ready_labels`, `waiting_labels`, `abandon_labels`
- `n_actually_delivered` (matches the trace's
  `checkpoint_n_ready_with_step_output`)
- `n_abandoned` (matches `n_abandon`)
- `classifications`: list of `{label, verdict}` parsed

Aggregated in `agent_result`:

- `checkpoint_events`, `n_checkpoint_total`, `n_checkpoint_responded`
- `n_commitments_closed_via_checkpoint` (READY_TO_DELIVER turned into
  STEP_OUTPUT)
- `n_commitments_abandoned_via_checkpoint` (ABANDON verdicts)
- `stream_msgs_delivered_count_final`

## How it differs

### vs `mtmsg_revisit` (single-oldest)

Revisit:
- Asks about ONE commitment at a time, picked by `opened_at_compaction_turn` (oldest first).
- Fires every `SH_REVISIT_PERIOD` turns (default 4) on turns without
  stream delivery. Tight, frequent nag.
- Three valid responses: STEP_OUTPUT for that label, `DEFER: <label>`,
  or `CLOSE: <label>`.
- The agent can give up by deferring; the tracker closes it.

Checkpoint:
- Asks about ALL currently-open commitments in one batch.
- Fires every `SH_CHECKPOINT_PERIOD` *stream messages delivered*
  (default 10) — sparser, decoupled from turn count.
- For each commitment the agent must pick one of three verdicts and
  (for READY_TO_DELIVER) emit a STEP_OUTPUT this turn.
- This loads more triage work onto a single turn but reduces the
  total number of nag-turns and lets the agent compare commitments
  side by side ("X is ready; Y still blocked; Z no longer matters").

### vs `mtmsg_compaction` (baseline / no scheduling)

The baseline has compaction's narrative + facts + sub-decision list but
never explicitly asks the agent "what's open, what's actionable?". The
agent must self-prompt against the compaction narrative.

Both revisit and checkpoint variants extract `open_sub_decisions` from
compaction events and inject prompts referencing them; only checkpoint
forces a batch classification.

## Other notes

- Compaction is still on (`SH_DISABLE_COMPACTION=0` default). The
  open-sub-decisions tracker is populated by the compactor's emitted
  `sub_decisions[*]` array via `_refresh_open_sub_decisions`.
- Motivation generator is off by default (`SH_DISABLE_MOTIVATION=1`).
  It can be re-enabled but is decoupled from the checkpoint mechanism.
- Namespace / collection-prefix / sqlite-db filename are renamed
  (`arc_em_checkpoint`, `arc_cp`,
  `eventmemory_shared_harness_v4_mtmsg_checkpoint.sqlite3`) so a
  parallel checkpoint run can't clobber a revisit run's EM state.
- The variant tag in `agent_result["variant"]` is `mtmsg_checkpoint`.
- The harness output JSON files land in
  `SH_RESULTS_SUBDIR` (default `results`) under this variant's
  directory.

## What runtime testing should validate

1. **Compliance with the classification protocol** — for each listed
   commitment, does the agent emit a single
   `CHECKPOINT_DECISION: <label>: <verdict>` line? Check
   `checkpoint_n_commitments_listed` vs `checkpoint_n_ready +
   checkpoint_n_waiting + checkpoint_n_abandon` and
   `checkpoint_n_unrecognized_label`.

2. **READY → STEP_OUTPUT follow-through** — when the agent says
   READY_TO_DELIVER, does a matching STEP_OUTPUT actually appear in the
   same turn? Compare `checkpoint_n_ready` to
   `checkpoint_n_ready_with_step_output` per event.

3. **ABANDON correctly cleans up the tracker** — for each
   `n_commitments_abandoned_via_checkpoint`, the commitment should be
   absent from `open_sub_decisions` with state="closed".

4. **Net effect on DP coverage** — does forcing the batch triage move
   `n_dp_covered` up vs the no-scheduling baseline? Does it match or
   exceed the revisit variant on long streams (where many commitments
   accumulate)?
