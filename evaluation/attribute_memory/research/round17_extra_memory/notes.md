# Round 17 — agamemnon `extra_memory` vs round 15 + 16C v2

## TL;DR

`ExtraMemory` does **not** match the round 15 + 16C v2 baseline as wired
out-of-the-box. On a partial 265-turn ingest (29 non-first transitions in
range), it scored **ref_emission_rate=0.034, ref_correctness=0.034**, vs
round-15+16C v2's 0.91/0.72 on the full 743-turn run. The structural pieces
(deterministic_relink_batch, cluster_id, skip_clarify) are sound, but the
shipped writer prompt ("Source"-anchored generic-stream prompt) is
mis-targeted for the dense_chains scenario, which is a single first-person
diary stream centered on @User. The writer hallucinates entities ("@Big",
"@Pretty", "@Listening") from sentence-start tokens, drops @User from User
state changes, and uses ad-hoc predicate names (`@User.next_job`) instead of
canonical ones (`@User.employer`).

## Numbers

Hard-cap stop. The full 743-turn ingest didn't complete in time (~149
gpt-5-mini batches at ~10s each, far slower than the round-15 pipeline). Run
was halted at turn 265 (~36% of the stream) with 247 entries written;
results are partial.

| metric | round 15 cap=100 | round 16C v2 (relink) | round 17 extra_memory a=20 (partial 1-265) |
| --- | --- | --- | --- |
| ref_emission_rate (overall non-first) | 0.907 | 0.907 | **0.034** |
| ref_correctness (refs walk) | 0.686 | 0.721 | **0.034** |
| cluster_correctness_rate (cluster_id-based) | n/a | n/a | 0.034 |
| entry_emission_rate (non-first) | n/a | n/a | **0.069** |
| Q/A judge pass | 17/32 | **26/32** | not run (budget cut) |

Bucket curve (extra_memory partial, non-first transitions):
```
  (0,100]    8  emit=0.00 refs=0.00 cluster=0.00
  (100,200] 14  emit=0.00 refs=0.00 cluster=0.00
  (200,300]  7  emit=0.14 refs=0.14 cluster=0.14
```

## Cost / log size

- **LLM**: ~80 writer calls before halt (cache not saved in-memory; harness
  was killed mid-`ingest_turns()` — the LLM cache only persists at the end
  of the call, which is friction point #5 below).
- **Embed**: ~50 calls.
- **Cost (estimated)**: ~$0.24 LLM + ~$0.001 embed = ~$0.24 (well under
  $4 cap).
- **Log size**: 247 MemoryEntry rows / 303 KB SQLite file.

## Friction points (real bugs / API misalignments)

1. **`ResolutionRow.metadata` collides with SQLAlchemy `DeclarativeBase`.**
   Module fails to import as shipped. We patched at runtime by rewriting the
   attribute name. **This is a real bug in agamemnon and should be fixed.**

2. **Writer prompt is mis-targeted.** The `WRITE_SYSTEM_PROMPT` calls the
   first-person speaker `@Source` and lists no first-person heuristics. On
   dense_chains it produces:
   - Bogus entities from sentence-start tokens: `@Big`, `@Pretty`, `@Email`,
     `@Going`, `@Listening`, `@Mid` — these fill the active-cluster slots
     and crowd out real entities.
   - Wrong subject on User state-changes: "I'm starting at Notion" →
     `pred='@Big.next_job'` instead of `@User.employer`.
   - Predicate-name drift: `next_job`, `interest`, `manager` (with wrong
     subject) instead of canonical `employer`/`hobby`/`boss`. The
     `_PREDICATE_SYNONYMS` map covers `title|role|occupation|job_title` and a
     few others, but not `next_job` or `manager` (yes — `manager` is
     unmapped and gets routed to a new cluster every time).
   - User-relevant facts often emit without `@User` in mentions; the
     "Source" label is added to known_labels but the prompt doesn't strongly
     instruct the writer to use it on first-person utterances.

3. **No usage examples / wiring tests for `ExtraMemory`.** `EventMemory` has
   a server_tests conftest with an InMemoryVectorStoreCollection fake; there
   is no equivalent for `ExtraMemory`. We had to reach into
   `server_tests/.../in_memory_vector_store_collection.py` directly.

4. **`OpenAIChatCompletionsLanguageModel` doesn't interoperate with the
   round-7 evaluation cache.** Different cache-key shape, no
   reasoning_effort hook, no shared budget. We wrote a thin `LanguageModel`
   adapter around `_common.llm` and parsed JSON via `json_repair`. Not a
   blocker but a friction point.

5. **`ExtraMemory.ingest_turns` runs serially over batches with no
   incremental progress callback.** A 743-turn × 5-batch ingest is one
   ~25-minute call that holds the LLM cache only in memory; if it gets
   killed (or if the harness errors), all writer-LLM work is lost. There's
   no `flush` hook, no progress callback, no per-batch save. This is real
   shippability friction — for benchmark runs you want incremental save.
   Round-15's pipeline saved cache after every batch.

6. **Schema property names use leading underscore (`_partition_key` etc.).**
   The vector-store `validate_identifier` regex accepts them, but they
   don't round-trip with `normalize_filter_field`. Not a blocker for our
   run; downstream filter-field mapping has to use
   `extra_memory._to_vector_record_property`.

7. **`active_state_limit=20` was arguably too low for a dialog with 14
   chains plus the writer's bogus-entity inflation.** With 20 active heads
   and ~10 hallucinated entities ("@Coffee", "@Big", "@Pretty", ...) the
   real chain heads get evicted. Bumping to 50 or 100 would only paper
   over the underlying writer-prompt problem.

## Ship recommendation

**Don't ship `extra_memory` as the round-17 baseline yet.** The structural
core (cluster_id, deterministic_relink_batch, skip_clarify, resolution
records) is good and the right direction — but as wired:

- The default writer prompt regresses ref-emission from 0.91 to <0.05 on
  dense_chains. Until that prompt is hardened (anchor=`User`,
  predicate-discipline rules from round-16C v3, predicate-synonym
  expansion), it's not a drop-in replacement.
- The SQLAlchemy `metadata` collision is a hard-fail import bug — anyone
  trying to use the SQLAlchemy entry store today hits it on first import.
  This should be fixed in agamemnon.
- The LLM-cache-only-on-success ingest pattern is a benchmarking
  blocker. Add per-batch cache flush.

## What I'd do next (if continuing)

1. Patch agamemnon's `ResolutionRow.metadata`.
2. Replace `WRITE_SYSTEM_PROMPT` / `WRITE_USER_PROMPT` in `extra_memory.py`
   with the round-16C-v3 predicate-discipline prompt, anchored on the user
   label that the caller supplies (no more "@Source" generic). Add an
   explicit "ignore sentence-start capitalized words that aren't in
   KNOWN LABELS" rule.
3. Expand `_PREDICATE_SYNONYMS` to cover `manager→boss`, `next_job→
   employer`, `interest→hobby`, `gym|workout→gym`, etc. — match the
   dense_chains chains that round-15 covers.
4. Add a per-batch cache-flush callback to `ingest_turns`.
5. Re-run on the full 743-turn scenario and compare against round-16C v2
   directly.

Once those four changes land, the cluster_id-based grader (which is
strictly cleaner than refs-walk) should give a fair head-to-head — and I
expect `extra_memory` to win on ref_correctness because cluster_id is more
robust than chasing refs.
