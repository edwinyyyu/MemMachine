# shared_harness_v4_mtmsg_compaction — BUILD NOTES

## Forked from
`evaluation/associative_recall/metacog/shared_harness_v4_mtmsg_streaming_v2/main.py`.

## Why this variant exists

For PERPETUAL execution the bottleneck isn't per-turn prompting — it's working-
memory bounding. Streaming v2 hard-truncates the oldest user/assistant pairs
verbatim when the thread crosses `MT_HARD_CAP=10_000` tokens. That permanently
loses any fact only present in the dropped span (the agent has to PROBE for it
later, which fails when the wording / semantic surface diverges from the live
probe).

This variant replaces hard-truncate with **compaction**: a separate LLM extracts
load-bearing facts from the about-to-be-evicted span, writes them to
EventMemory as new events, and replaces the span in-thread with a single
compact `[COMPACTED MEMORY ...]` summary message.

**System prompt and follow-up templates are byte-equal to streaming v2** per
build instructions. Notably `SYSTEM_PROMPT` still says "OLDEST user/assistant
turns get hard-dropped"; we deliberately did NOT update this. Whether the
prompt mismatch matters in practice is one of the runtime questions.

## Mechanism

### Trigger (turn-start)
At the top of each agent loop iteration — BEFORE the model is invoked — the
harness measures `messages_tokens(mt_messages)`. If
`tokens_before > SH_COMPACTION_THRESHOLD * MT_HARD_CAP`, it invokes
`maybe_compact(...)`.

### Span selection
`_identify_compaction_span(...)` walks `mt_messages` (skipping the system
prompt at index 0), groups remaining messages into user/assistant pairs, and:
1. Preserves the most-recent `SH_COMPACTION_KEEP_RECENT` pairs intact at the
   tail (default 3).
2. Computes the oldest half (rounded up) of the remaining "eligible" pairs.
3. Returns `(start_idx, end_idx)` — a Python slice of `mt_messages` to
   compact. Returns `(1, 1)` (empty span) if no compaction is possible.

This means the FIRST compaction event compresses roughly 1/2 the
not-most-recent pairs; subsequent events compress half of what's left, etc.
Compaction is gated on the threshold check; we don't compact until pressure
builds again.

### Compactor LLM call
`call_compactor(...)` runs `gpt-5-mini` at `reasoning_effort=low` with
`response_format=json_schema` (schema `COMPACTION_SCHEMA`). The prompt
(`COMPACTION_PROMPT`) is the **only new prompt in this harness** and asks for:
1. `narrative` — 3-5 sentence summary
2. `facts` — concrete facts, each `{text, source_turn}`
3. `sub_decisions` — `{label, state ∈ opened|closed|active, summary}`
4. `retrieval_cues` — short cue strings (entities, topics, dates)

### EM write
For each non-empty `fact.text`, `write_compacted_fact(...)` encodes one
`Event` into the same `EventMemory` collection as the stream messages. Each
fact event carries:
- `event_type: "compacted_fact"` (vs `"stream_message"` for raw user msgs)
- `plant_id: "compacted_fact_<uuid12>"` so probe-result matching counts these
  the same way as stream-message hits
- `compaction_at_turn: <int>` and `compacted_from_turn_range: "X-Y"` for
  attribution
- `from_turn: <source_turn or compaction_at_turn>` (chronological anchor)
- `timestamp: base_ts + 60s * compaction_at_turn` (so retrieval ordering keeps
  the compaction-time ordering, not the original turn — see open question (b))

### Span replacement
The evicted span in `mt_messages` is replaced with a SINGLE user-style message:

```
[COMPACTED MEMORY (covering turns X-Y): <narrative>. Key facts written to
long-term memory (N fact(s)); you can probe for them as needed. Active
sub-decisions: <inline list>. Open retrieval cues: <comma list>.]
```

This is a single message (one slot in the list) so the user/assistant
alternation is not broken — the assistant message that follows it is the
agent's response to the followup that was already in tail.

### Hard-truncate safety belt
After compaction (and even when compaction is disabled) we still call
`truncate_thread(mt_messages, MT_HARD_CAP)`. When compaction is working this
is a no-op. When `SH_DISABLE_COMPACTION=1`, hard-truncate IS the bounding
mechanism (matches streaming v2 baseline). It also guards against the
pathological case where the compactor returns a summary big enough that the
thread is still over cap.

## Trace fields added

Per-turn:
- `compaction_fired_this_turn: bool`

Per-result:
- `n_compactions: int`
- `compaction_events: list[{at_turn, evicted_pair_count, evicted_msg_count,
  evicted_token_count, tokens_before, tokens_after, n_facts_written,
  n_facts_failed, narrative, narrative_length, sub_decisions, retrieval_cues,
  facts_written, stream_turns_in_span, turn_start, turn_end, summary_msg,
  compactor_error}]`
- `compacted_facts: list[{fact_text, fact_from_turn, compaction_at_turn,
  fact_id, error}]`
- `n_compacted_facts_written: int`

## Config knobs

| Env var | Default | Meaning |
|---|---|---|
| `SH_COMPACTION_THRESHOLD` | `0.85` | Fraction of `MT_HARD_CAP` at which compaction fires (turn-start). |
| `SH_COMPACTION_KEEP_RECENT` | `3` | User/assistant pairs at the tail to leave intact. |
| `SH_DISABLE_COMPACTION` | `0` (false) | `1`/`true`/`yes` falls back to hard-truncate (ablation baseline). |

CLI:
- `--sqlite-suffix <s>`: salt the EM SQLite filename (e.g., for parallel
  ablation runs).

## EM/Qdrant namespace

- SQLite path: `eventmemory_shared_harness_v4_mtmsg_compaction{suffix}.sqlite3`
- Qdrant `NAMESPACE = "arc_em_compaction"`
- Qdrant collection prefix: `arc_cmp` (vs `arc_sv2` for streaming v2)

These are distinct from streaming v2's namespaces, so the two harnesses can
coexist without overwriting each other's collections.

## Design choices worth flagging

1. **Compaction fires at TURN START, not after the assistant's emit.** This
   means the agent reads the compacted message BEFORE responding for the
   first time at that turn — so it doesn't have to navigate a freshly
   compacted span as part of the same response cycle.

2. **Half-eviction, not full-eviction.** When pressure hits, we compact the
   oldest *half* of eligible pairs, not all of them. Reasoning: leaves more
   recent context still verbatim, which the agent can reason against
   directly. Trade-off: compaction fires more often but each fires less
   aggressively.

3. **Facts are written with `plant_id` set** so the existing `new_facts`
   probe-counter machinery in the agent loop counts compacted-fact hits the
   same way as stream-message hits.

4. **No retrieval-side awareness of compacted vs raw events.** The agent
   probes; the retriever returns the top-k by embedding similarity; it does
   not currently distinguish or filter `event_type=compacted_fact` vs
   `event_type=stream_message`. This is an explicit open question (b).

5. **Summary message uses user role**, not system or assistant. User role is
   "the harness", which already wraps stream messages. Putting it in user
   role keeps the alternation valid for the chat completions API.

6. **Compaction is opt-out via env, not a separate harness directory.** This
   keeps the ablation baseline (hard-truncate) bit-equivalent at the
   trace-emit level when `SH_DISABLE_COMPACTION=1`, modulo one trace field
   (`compaction_fired_this_turn` always false).

7. **Stream-turn integers in the span** are extracted via regex over the
   embedded `INCOMING USER STREAM MESSAGE (turn N)` and `Turn N.` tags. This
   lets the compaction summary reference `turns X-Y` accurately rather than
   forcing the LLM to guess.

## Open questions for runtime testing

(a) **Fact granularity.** Does `gpt-5-mini` at low reasoning_effort extract
facts at the right granularity? Risks observed in past EM-writer experiments
include over-emission (one paragraph → 12 redundant facts) and
under-aggregation (multi-claim facts that defeat retrieval). The
COMPACTION_PROMPT explicitly asks "one self-contained statement per fact"
and "constrain future decisions" but the writer's behavior under load is
unknown.

(b) **EM retrieval pollution.** Compacted facts are written to the same
collection as raw stream messages. If the LLM extracts 10 facts per
compaction and we compact 3-5 times in a long run, the EM may have
30-50 paraphrased fact events competing for top-k against the
~20-50 raw stream messages. This could:
  - Push raw-stream context off the top-k for some probes (recall regression).
  - Help when the raw wording diverges from the probe but the compacted
    paraphrase aligns (recall gain).

  Worth measuring per-probe whether the top-k contains stream messages,
  compacted facts, or both, and whether compacted-fact hits correlate with
  DP coverage.

(c) **Compacted summary fidelity.** Does the agent recognize, after seeing
the `[COMPACTED MEMORY ...]` block, that the original messages are gone and
that PROBES are needed to retrieve specific phrasing? The system prompt is
unchanged and still says "hard-dropped" — the model may be confused by
seeing a placeholder it doesn't expect, OR may treat it correctly as a
hint to probe more aggressively. We may need a follow-up pass with prompt
edits if the model loses track of pending sub-decisions.

(d) **Wall-clock cost.** Each compaction adds one `gpt-5-mini` call with
~1.5-3k tokens of span input. In a 28-turn run that fires 2-4 times, this
adds ~5-15 seconds. Compare against streaming v2's no-extra-call baseline.
If compaction events become frequent (>50% of turns) we should either raise
the threshold or compress more aggressively per event.

(e) **Trigger frequency.** With `keep_recent=3` and threshold=0.85 the
expected steady-state is: pressure builds → compact half of older content →
pressure drops to ~50-70% of cap → grow back up over 4-6 turns → compact
again. If thresholds are tuned wrong we either compact every turn (cost)
or never compact (effectively hard-truncate). The trace lets us measure
this directly via `compaction_fired_this_turn`.

(f) **`SYSTEM_PROMPT` stale.** The unchanged system prompt still claims
hard-truncation. The agent will see compacted-summary messages it didn't
expect. Consider a follow-up A/B with a prompt-aligned variant to isolate
the prompt-mismatch effect from the compaction effect itself.
