# Semantic Memory Ingestion Pipeline -- Architecture

End-to-end pipeline that sits on top of event memory and produces per-topic
append-only logs that a downstream agent can query. Seven design questions,
each answered with a concrete default and rationale.

```
  event stream (raw turns: ts, source, text)
        |
        v
  salience prefilter (OFF by default)  <--- cheap keyword rules; default
        |                                   off because it strips
        v                                   conversational context around
  batcher (5-turn window OR silence-gap     corrections. LLM's own noop
         OR document-flush)                  discipline is good enough.
        |
        v
  EXTRACTION LLM (gpt-5-mini, reasoning=low)
    input: batch + known topics + known entities + recent log entries
    output: array of {append|append_ref|noop} commands
        |
        v
  router (embedding alias index; collapses
         "User/Cats/Luna" ~ "User/Pets/Luna")
        |
        v
  APPEND-ONLY LOG STORE (per topic)
        |
        v
  BACKGROUND CONSOLIDATION (per-batch trigger:
   when a topic exceeds 10 live entries, one
   LLM call summarizes -> 'consolidate' entry)
        |
        v
  QUERY INTERFACE:
    free text -> topic-embedding top-K +
    root-entity prefix-union -> render selected
    topic logs -> reader LLM answers
```

## 1. Trigger

**Chosen: batch-on-flush (rolling window of N=5 OR silence-gap 30min OR
end-of-stream OR long-chunk).**

- **Per-turn** is pathologically expensive (3x the calls of per-5) for no
  quality gain (see ablation: 9 calls per_turn, 9 per_5_turns, identical 8/8).
- **Per-conversation** fails in persistent chat where "conversation" has no
  clear end.
- **Silence-gap only** undershoots in long same-session conversations.
- The hybrid is the floor of all three: flush whenever any criterion fires.

## 2. Batching

**Chosen: 5-turn rolling window.**

- Small enough that the LLM can attend the full batch in context.
- Big enough that multi-turn patterns (correction, set-addition) land in
  the same call. Specifically, a correction needs both the original claim
  AND the correction in-context; per-turn splits them.
- Document ingest: any single turn >=500 chars flushes immediately so the
  extractor sees the whole paragraph at once. Dense paragraphs yielded
  24 separate facts from a single MBARI bio (S4), answering 9/9 rubric
  points.

## 3. Extraction

**Chosen: multi-append per batch + explicit noop.**

- The extractor emits `[{append|append_ref|noop, topic, text, ...}, ...]`.
- Grounded in round-5's finding that append-log schemas beat row-family
  (12/14 vs 8/14) with the same LLM.
- Explicit noop command avoids the "must emit something" bias.
- For correction batches: the LLM emits an `append_ref` with
  `relation: "supersede"` referencing the prior entry_id (provided in the
  prompt). Refs are integer entry ids, not text-match.

## 4. Routing

**Chosen: LLM-chosen topic name + embedding alias index.**

- The extractor picks the topic slash-path in the same LLM call (no second
  routing call).
- Prompt includes `KNOWN TOPICS` list to drive reuse.
- After extraction, a fuzzy alias pass: if the chosen topic's embedding has
  cosine >= 0.78 with any existing topic, collapse to the existing topic.
  Catches "User/Cats/Luna" vs "User/Pets/Luna" drift.
- Threshold 0.78 was picked by eyeballing proposed-vs-existing cosines on
  S1-S3; above 0.82 semantically distinct topics start to collide, below
  0.74 paraphrases stop collapsing.

Cites Round-6A thread-routing research by assumption: its recommendation
(when complete) could replace the LLM-embedded-in-extraction choice with a
separate routing step; this prototype inlines it to save a call.

## 5. Consolidation

**Chosen: threshold-trigger at 10 live entries per topic, single-LLM
consolidation that emits a `consolidate`-relation entry referencing rolled-up
ids.**

- Prior entries stay in the log (append-only invariant holds) but are
  marked `consolidated=True` so the live view skips them.
- The consolidate entry itself is NOT marked consolidated when a later
  `supersede` references it -- narrow supersedes of single facts within
  a consolidate would otherwise evict the whole summary. Observed in S2
  Partner/Riley path: a gender-reveal supersede was collapsing the entire
  pregnancy/wedding history. Fix: preserve consolidate entries; let the
  reader LLM resolve later supersedes at read time.
- The consolidation pass runs in-line at the end of each batch ingest. In
  production this could go background/async; cost is bounded (one extra
  LLM call per 10-entry threshold crossing).

## 6. Query interface

**Chosen: free-text query -> embedding top-6 topics UNION root-entity
prefix match -> render selected topic logs -> reader LLM answers.**

- Embedding retrieval alone was losing broad "tell me about the user"
  queries because top-K was biased toward most-recently-updated topics
  (e.g. User/Music bumped User/Partner/Riley out of top-6).
- The prefix-union covers this: if the query mentions (or implies via
  "me/my/i/you") the `User` root, include every User/ topic.
- Answer LLM sees numbered log entries with relation + refs annotations
  and resolves state.

## 7. Entity linking

**Chosen: LLM-at-ingest with a running entities list in the prompt; NO
separate resolution step.**

- The extractor prompt includes `KNOWN ENTITIES (canonical names -- reuse)`.
- When "Luna" is mentioned turn after turn, the LLM reuses the same token.
- For paraphrase (e.g. "my cat" / "the kitten" / "her"), the LLM resolves
  it within the batch window (5-turn locality is sufficient). If the
  referent falls outside the window, the entities list carries the anchor.
- Embedding-based entity matching was not needed in these scenarios; the
  LLM at batch time is both cheaper and more accurate because it uses
  conversational context.
- Tradeoff: this fails for very large entity inventories (hundreds of
  people) where the prompt can't list them all. For those, a pre-retrieval
  step filtering the entities list by embedding similarity to the batch
  text would be the fix. Untested in this prototype.

## Integration with event memory

The pipeline sits **on top of** event memory; it does not replace or modify
it.

- Input: an ordered stream of (ts, source, text) events, matching the
  `Event` concept in `event_memory.py`. The pipeline is oblivious to the
  underlying segment/derivative storage.
- Provenance: each `LogEntry` stores `source_event_indexes` (positions in
  the event stream). In production these would be event UUIDs, enabling
  back-references from semantic claims to the raw turns that produced
  them. Useful for audit, correction, and rollback.
- The pipeline's `forget_events` analogue is `rollback_batch(batch_id)`,
  which deletes all log entries produced by a given batch and un-marks
  any consolidated flags set by removed entries. Tested: a batch with
  "My SSN is 123-..." ingested into 1 log entry, then rolled back ->
  0 entries remain.
- Event memory continues to serve raw-turn retrieval independently. This
  pipeline is additive: semantic memory answers "who is the user" in a
  single structured call, while event memory remains the fallback for
  "what did they say on March 15 at 10am".

## Ablations tested

| Question | Tested | Finding |
|----------|--------|---------|
| Trigger granularity | per-turn / per-5 / silence-only | All hit 8/8 on S1; per-5 is cheapest. |
| Salience prefilter | on vs off | OFF is better: 8/8 vs 7/8; the filter stripped context around a correction. LLM noop is sufficient. |
| Rollback primitive | ingest+forget | Works; 1 entry removed cleanly. |
| Consolidation threshold | 8 vs 10 | 8 fires too aggressively during long scenarios, creating a supersede cascade that loses facts. 10 + consolidate-survives-supersede is the fix. |

## Decisions I'd want to revisit

- The extractor chooses topic names. In heavy multi-user settings an
  entity-scoped routing pre-pass (Round-6A's direction) would reduce
  topic-name drift further.
- The consolidation strategy is single-pass-per-threshold. For very long
  timelines, we'd want chunked consolidation with an explicit "this
  consolidate entry spans dates X to Y" so that queries about specific
  time ranges can retrieve the right summary.
- Retrieval is topic-granular. A second-stage within-topic filter (entry-
  level embedding) would be needed once topics exceed ~30 live entries.
- Embedding-based entity resolution: untested against LLM-at-ingest.
  Hypothesis: LLM wins on conversational paraphrase, embedding wins on
  large-inventory lookup. Untested.
