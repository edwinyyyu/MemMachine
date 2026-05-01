# Semantic Memory Representation & Commands — Research Synthesis

Six research rounds, 2026-04-23:

- `storage_representations/` — 6 representations × 6-12 scenarios (round 1-2)
- `update_commands/` — 5 schemas × 3 framings × 12 scenarios (rounds 1-3, LLM-judge)
- `update_commands/` round 4 — fine-grained ops ablation (deterministic applier)
- `update_commands/round5/` — upsert / append / no-delete exploration (deterministic applier, 14 scenarios)
- `round6/threads/` — thread routing strategies (R1-R7, 32 facts, deterministic eval)
- `round6/compaction/` — context compaction strategies (C1-C7, 60-entry profiles, Q/A eval)
- `round6/pipeline/` — end-to-end ingestion prototype on top of event memory (4 scenarios, 93%)
- `prior_art/` — survey of mem0, Letta/MemGPT, Zep/Graphiti, Cognee, ChatGPT, Claude, Character.AI

All budgets within caps (~750 LLM calls + embeddings, ~$2.50 total).

---

## Headline findings (round 5 changed the recommendation)

1. **Append-log schemas beat row-family schemas by 4/14 on deterministic grading.** S4 `append` + `append_ref(clarify/refine/supersede/invalidate)` + `noop` scored 12/14 (86%) vs 8/14 (57%) for the round-4 row-family winner. S5 pure-prose append tied S4. Gap closes ~1 scenario with a bi-temporal row applier but is real.

2. **Intent-level relations work where surface-level fine ops don't.** Round 4: `patch` / `append_to` / `strengthen` / `weaken` fire 0-1× across 30 turns — dead weight. Round 5: `append_ref` relations fire 20× with clean distribution (10 clarify / 4 supersede / 4 invalidate / 2 refine). The difference is surface-transformation vs claim-to-claim-relationship. LLMs are good at the latter.

3. **The attribute-name slot is the dominant drift source.** Row schemas produce `cats|count|2` one turn and `pets|cat_names|Luna,Milo` the next, even under the winning prompt. Append has no attribute slot, so the drift disappears. This explains most of round 5's 4/14 gap.

4. **UPSERT is strictly not-worse than ADD+REVISE.** Collapses the add/revise choice without cost. Removing further (upsert-only, no member ops, no remove) drops to 43% — keep `upsert + add_member + remove_member + remove`.

5. **DELETE isn't needed at all in append-only.** Retractions are just new appends that negate prior claims. The log is monotone; "current state" is derived at read time from the full history. This matches event-sourcing, bi-temporal databases, git, and human episodic memory — you remember the correction, you don't erase the original. Round 5 evidence: S5 (pure append, no typed relations) tied S4 (with `invalidate`) at 12/14 including retraction scenarios, so not even an `invalidate` relation is load-bearing for LLM correctness. Typed relations are a materializer optimization (deterministic state resolution), not a LLM requirement. The "DELETE needs a replacement" caveat applied only to row-family S3 (upsert-only, no remove verb), where it failed because the reader didn't honor `confidence=negated` — a reader-implementation gap, not a fundamental need for a retraction verb.

6. **Framing beats schema for hedged-revise and noop.** Default editor framing noops on confidence-shift turns ("more aspirational than actual"); all schemas fail it. Archivist framing fixes that but breaks noop discipline on weather/joke. Unified fix (untested): default editor framing + DO-NOT-WRITE block + hedging-cue enumeration.

7. **Typed JSON hierarchies with rigid slots actively hurt.** Invent entities (`unnamed_pet_5`), produce double-negations (`NOT no`), fracture single facts into many leaves. Don't do it. (Round 2)

8. **LLM-as-judge has a blind spot on `delete(old)+add(new)`.** Round 1 LLM-judge gave the add/delete baseline 67%; round 4 deterministic applier gave the same baseline 30%. Always use deterministic grading for command research.

---

## Recommended design

### Preferred: append-log schema

**Storage**: each topic is a log of appended entries. Row-family tables become a materialized view derived from the log.

```
(partition_key, entry_uuid, topic, seq_no,
 text,                            # natural-language content
 refs: list[entry_uuid],          # prior entries this entry relates to
 relation: "clarify" | "refine" | "supersede" | "invalidate" | null,
 valid_from,                      # timestamp of ingestion
 source_event_uuids: list,        # provenance
 properties: json
)
```

**Commands**:
```ts
type Command =
  | { op: "append"; topic: string; text: string }
  | { op: "append_ref"; topic: string; refs: number[];
      relation: "clarify" | "refine" | "supersede" | "invalidate";
      text: string }
  | { op: "noop" };
```

**Author-time rendering**: numbered log entries for the relevant topic(s):
```
LOG for User/Medical (refer by [n]):
[1] (Mar 15) Has peanut allergy — severe reaction to raw peanuts [confirmed]
[2] (Mar 22) Tree nut allergy confirmed as well [clarify: 1]
[3] (Apr 5) Peanut allergy severity re-tested and confirmed [clarify: 1]
```

**Reader-time rendering**: derived current-state view via relation resolution (latest non-invalidated claim wins; supersede replaces; clarify adds detail; refine narrows). This is a pure function of the log — no LLM involvement.

**Relations**:
- `clarify` — adds detail to a prior entry without contradicting it
- `refine` — narrows or qualifies a prior entry
- `supersede` — replaces a prior entry with a new version (old stays in log, marked)
- `invalidate` — prior entry was wrong; retracts without replacement

### Row-family fallback (if migration cost is prohibitive)

Same flat rows as round-4 recommendation, with `upsert` collapsing add+revise:

```
(partition_key, uuid, topic, category, attribute, value,
 cardinality, confidence, valid_from, valid_to,
 source_event_uuids, prior_version_uuid, properties)
```

```ts
type Command =
  | { op: "upsert"; topic: string; category: string; attribute: string;
      value: string; cardinality: Cardinality; confidence: Confidence }
  | { op: "add_member"; index: number; member: string }       // set-only
  | { op: "remove_member"; index: number; member: string }    // set-only
  | { op: "remove"; index: number }
  | { op: "noop" };
```

5 ops. `upsert` subsumes `add` and `revise`; backend distinguishes create-vs-modify from whether the key exists.

### Author framing (both schemas)

- DO-NOT-WRITE enumeration (weather / generic platitudes / meta-chat / filler / jokes / repetitions) — **load-bearing for noop discipline, removing it regresses immediately**.
- Verbatim-on-revise rule (for row-family): "When emitting revise, copy the unchanged portion of the value verbatim. Do not rephrase, summarize, or shorten prior content."
- Hedging-cue enumeration (untested, round-5 open question): enumerate phrases like "more aspirational than actual", "I think I might", "trying to" that should trigger a `(hedged)` revise rather than noop.

---

## Anti-patterns observed

- **Typed-JSON hierarchies with rigid slots** → confabulation. Invented phantom set members, double-negations.
- **String-match `delete`** (round-1 baseline) → drift and duplicates; LLM can't reliably exact-match paraphrased values.
- **Surface-level fine ops** (`patch[old, new]`, `append_to`, `strengthen`, `weaken`) → never fire even on designed scenarios. Dead weight.
- **Separate `add_member`/`remove_member` without cardinality rendering** → LLM applies to scalar attrs. Fixed by showing `(n=K total)` in the sheet.
- **Archivist / minimal-diff framing without DO-NOT-WRITE block** → noop discipline collapses; weather chitchat becomes a preference row.
- **More verbs to cover more cases** → each new verb is a new bias source. Round-1 8-verb `intent_ops` was worse than 4-verb indexed_patch.
- **Free-form attribute names** → every turn spawns a new attribute string; `cats|count|2`, `pets|cat_names|...`, `household|hamster|existence|present`. Append avoids this; canonical vocabularies would mitigate it for rows.

---

## Open questions (priority order)

1. **Bi-temporal row applier.** Round 5's row schemas were graded without `valid_from`/`valid_to` — T13 retrieval probe is somewhat stacked against them. With the bi-temporal reader, row schemas likely recover ~1/14. Still behind append, but narrower.
2. **Long-conversation log growth.** Append rate is ~1 entry per non-noop turn; 100-turn behavior untested. Consolidation / compaction strategy is a real question for append-only.
3. **Canonical attribute-name vocabulary per category.** Would close T01 for row schemas (~1 more scenario). Untested.
4. **Hedging-cue + DO-NOT-WRITE combined framing.** Fixes T06 (confidence-shift noop) without regressing T07/T08 (noop discipline). Most likely untested framing win.
5. **Domain generalization.** Round 5 is all user-attributes. Relational and event-driven memory (meetings, decisions, news) untested.
6. **Model scale.** All experiments at gpt-5-mini. Append advantage may shift (either way) at larger models.
7. **Embedding retrievability** of append prose vs structured rows — proxy retrieval is representation-agnostic for LLM readers, but vector-index behavior untested.

---

---

## Round 6: Full pipeline design

### End-to-end architecture (from `round6/pipeline/`)

```
event memory (stateless, raw text, no LLM at ingest)
  ↓ batcher: 5-turn rolling window + silence-gap (30 min) + doc-flush (≥500 chars)
  ↓ extraction LLM — one call per batch, emits [append|append_ref|noop] array
  ↓ topic router — entity-first (R7 from 6A), 1 LLM call per extracted fact, embedding-dedup fallback
append-only per-entity logs (each topic = `<Entity>/<Category>`)
  ↓ background consolidation at ~10 live entries per topic
query interface
  ↑ embedding top-K ∪ entity-prefix filter → reader LLM
```

End-to-end: 93% (26/28) across simple chat, evolving user, multi-entity, novel chunk. Cost ~$0.25/day for a 500-turn/day user. Linear scaling.

### Thread routing (from `round6/threads/`)

**R7 entity-first with embedding dedup** won cleanly:
- 4/4 consistency, 97% entity-match, 15 topics, 1 LLM call/fact
- Subject-heuristic prompt is load-bearing (enumerates "pet/child/partner = that entity")
- Embedding dedup defense-in-depth for scale (didn't fire at 32-fact scale)
- **No cheap path**: no strategy below 1 LLM call/fact beat 90% of R7. Sub-1-call routing is not on the Pareto frontier.
- **Fixed taxonomy scored 53% entity-match** — jams Luna/Rex/Jamie into `User/Relationships`. Only acceptable when user is the sole entity.
- **Consistency-metric trap**: hybrid strategies got 100% consistency by over-merging. Entity-match + topic count are the real tie-breakers.

### Context compaction (from `round6/compaction/`)

**C4 query-gated retrieval** is the clear default:
- 100% Q/A accuracy, 1725 chars average, zero compaction LLM calls
- Just: top-K relevant (by embedding) ∪ last-K recent
- Size-invariant — doesn't degrade as log grows

**C2 middle-elision with literal `...`** is the no-embedding fallback:
- 79% accuracy, zero LLM calls
- Strictly beats C1 truncation (54%)
- **Strictly beats LLM-summarization (C3 hierarchical: 75%).** Paying LLM calls for prose summaries doesn't beat literal elision. Surprising.

**Do not ship:**
- C3 hierarchical (dominated by C2 at same char count)
- C5 active consolidation as currently designed (83%, but LLM silently drops retractions/hedges — "cilantro not an allergy" flipped to "Not in the log")
- C6 relation-aware structural compaction (71%, worse than C2 — "keep latest clarify per root" drops parallel orthogonal clarifies; fix: `(root, sub-attribute)` axis)
- C7 hybrid (inherits C6 bug)

---

## Round 6 discoveries / anti-patterns

- **Entity-first partitioning** (`Luna/Profile`, `Jamie/Relationships`) beats category-first under a single user topic. Fixed taxonomies collapse non-user entities.
- **Salience prefilter is net NEGATIVE.** Cheap keyword prefilters ahead of the extraction LLM fragment batches around corrections, stripping the referent from the correction. The extraction LLM's own noop discipline beats any pre-filter. Default: OFF.
- **Consolidates must NOT forward supersede flags to their constituents.** Without this rule, a narrow supersede ("baby is a girl") wipes a broad summary ("married + moved + pregnant + baby"). Single 2-line applier fix changed S2 from 0/5 to 4/5 and S4 from 4/9 to 9/9.
- **LLM-summarizing old entries drops retractions and hedges silently.** Active consolidation erases nuance. The fix is not "better summarization prompt" — it's "don't summarize; use retrieval instead" (C4 path).
- **Literal `...` elision works.** gpt-5-mini is not confused by `... (N entries elided) ...`. This removes pressure to summarize mid-log at all.
- **No cheap routing.** 1 LLM call per extracted fact is the floor. Embedding-only routers scored 53-69% entity-match; nothing below 1 LLM call matches R7.
- **5-turn batch window** is the right size. Per-turn costs 3× with no quality gain. Per-conversation has no boundary for persistent chat.
- **R7 is at a local optimum for single-call routing prompts** (world-state framing test). Four opening-paragraph variants (fact-extraction baseline, entity-state, state-change-multi-label, simulation) produced identical routes on 40/45 facts. gpt-5-mini with R7's scaffolding already treats entities as persistent objects; reframing is a no-op. Next improvement requires architectural change, not prompt engineering.
- **Within-turn aliases work; cross-turn anonymous→named consolidation fails for all framings.** "My boss wants me to lead Q3" (turn 3) → `User/Employment`. "His name is Marcus" (turn 18) → new `Marcus/Profile` topic, no merge. Embedding clusters aliases within a turn; nothing re-examines prior facts when a named entity appears later.
- **Multi-label routing emerges in state-change framing but needs gating.** F3 emitted multi-label on 5/45 facts; only 2/5 were correct. Over-eager splits ("User is nurse and diabetic" → two logs, wrong). Multi-label is valuable for "son got hamster" but requires a gate: emit only when fact introduces a new entity or is primarily bi-entity.

## What's still uncertain (round 6)

- **Topic explosion at 1000+ topics.** No ceiling. Needs periodic merge pass.
- **Long-horizon consolidation** (500+ turns, layered consolidate-of-consolidates) untested.
- **Multi-label routing** — state-change framing emits ~11% of facts as multi-label but 60% are over-eager. Needs a gate (only when fact introduces new entity or is primarily bi-entity). Promising but unreliable as shipped.
- **Cross-turn coreference (anonymous → named consolidation)** — distinct architectural gap. Requires a late-binding entity-resolution step that re-examines prior topics when a new named entity appears in a turn. Not solvable by prompt engineering.
- **Entity resolution at 50+ entities** — prompt-list approach won't scale; needs embedding-based matching.
- **Rollback granularity** — batch-level only in prototype; fine-grained fact retraction needs another LLM call.
- **Adversarial input** — paraphrase attacks that force entity-splitting over time.
- **C6 (root, sub-attribute) fix** — pure-Python structural compaction may still be viable; ~1 day to re-implement and test.

---

## Round 7: Entity identity and lifecycle

Integrated solutions for four interacting problems, all 100% on designed stress scenarios (43 LLM calls, $0.13):

### Multi-label routing gate (P1: 9/9)

Deterministic decision-tree at extraction time:
- **Multi-label IF**: fact introduces a new entity (not in known-entities list) OR fact is a relationship-making event with 2+ state-changing parties
- **Single-label otherwise** (including multi-attribute single-entity facts like "User is nurse and diabetic")
- Known-entities list passed into the prompt makes "new" a deterministic test

Fixes round-6's over-eager split rate (60% wrong in F3). Load-bearing: the known-entities list, not the prompt framing.

### Cross-turn coreference buffer (P2: 6/6)

Ring buffer of unresolved anonymous/descriptive mentions ("my boss", "this guy at work"), size 30 turns / 20 mentions. LLM coref-resolve call fires **only when** a turn introduces a named entity AND the buffer is non-empty. Resolves 28-turn-delay cases. Distractor names (Alice mentioned casually ≠ "my boss") correctly rejected.

Emits a `CoreferenceMerge` signal with metadata (`anonymous_topic`, `matched_mention_turn_idx`) for downstream retroactive log-rewrite. **The applier that actually moves entries between topic logs is not implemented** — flagged as the most important remaining gap.

Same-turn appositives ("Jamie, my partner") resolve automatically as a side-effect of buffering descriptors before processing same-turn named intros.

### Role slots as first-class memory objects (P3: 5/5)

Slots keyed `<Holder>/<Category>/<Role>` (e.g. `User/Employment/boss`). Slot value is a pointer (`@Marcus`). Slot has its own append-only log of holder-changes.

Boss changes: single append to `User/Employment/boss` log (`@Marcus` → `@Alice`). Marcus's own profile log is untouched; Alice's profile log is untouched. **Fanout of a role change = 1 append, not 3.**

Qualitative facts ("Marcus is a Scorpio") stay on Marcus's entity log, not the slot log. Marcus's profile survives intact across boss-changes. gpt-5-mini correctly separates slot_update from entity_fact first-try — no double-emission observed.

Opinion: not over-engineered. The deterministic "who is the current boss" query is a one-liner against the slot log; the alternative (scan all entity logs for current-role state) is messy.

### Salience-gated entity extraction (P4: 6/6 after 1 prompt iteration)

LAZY approach with multi-signal scoring:
- LLM emits per-candidate: `is_named`, `has_identifying_detail`, `has_state_change`, `grouping_key`
- Deterministic scorer: threshold = 2 signals → create entity; below → defer
- Grouping_key deduplicates across aliases ("grandmother's blue ceramic bowl" vs "the bowl")

Prompt iteration needed: explicitly state "possessive 'my' does NOT count as identifying", "passive use (knocking over) does NOT count as state change". Without these, the LLM defaults permissive.

Uses: specific tracked objects (the grandmother's bowl that gets broken later) get entities; common-noun distractors (mugs, tissues, pens) stay deferred.

### Integration test (13-turn composite): 1/1

One fused LLM call per turn handles all four mechanisms together. Scenario covered: 4 coref merges (2 cross-turn, 2 same-turn appositive), role transfer (`boss = @Marcus → @Alice`), 6 admitted entities, several deferred common nouns, sentimental item with later state-change. Conditional coref call (18 total across 13 turns). All four dimensions pass simultaneously.

### Round 7 caveats

- **Scenarios designed by the subagent, 5-9 per problem.** 100% scores reflect principled design, not generalization evidence. Real-world accuracy will be lower on surprise patterns.
- **Retroactive log rewrite applier is NOT implemented.** Coref merge signal carries the metadata, but moving entries between topic logs is deferred.
- **Same-name different-entity disambiguation untouched** ("Marcus my boss" vs "Marcus my brother-in-law").
- **Role-slot consolidation interaction** — slots are append-only history; they need a consolidator that knows not to drift.
- **Long-horizon salience re-scoring** — a deferred common noun that becomes tracked 200 turns later has no escalation mechanism.

### Updated pipeline (rounds 6 + 7 integrated)

```
event memory (raw text, stateless)
  ↓ per-turn OR 5-turn fused extraction LLM call, emits:
      - appends / append_refs per topic
      - multi-label routes (gated by new-entity / relationship-event)
      - slot_updates (on `<Holder>/<Category>/<Role>` slot logs)
      - entity_facts (with salience signals)
      - descriptor_mentions (pushed to coref buffer)
  ↓ coref resolver (conditional: only on named-entity intro + non-empty buffer)
      → CoreferenceMerge signals
  ↓ salience scorer (deterministic, threshold=2)
  ↓ topic router (R7 entity-first + embedding dedup)
  ↓ append-only per-entity logs + per-role slot logs
  ↓ background consolidation at ~10 entries/topic
  ↓ deferred: retroactive log-rewrite applier (unimplemented)
query interface
  ↑ C4 query-gated retrieval (top-K ∪ last-K)
  → reader LLM
```

---

## Artifacts

- `storage_representations/REPORT.md` + `scenarios.json` + `round{1,2}.py` + `results/`
- `update_commands/REPORT.md` + `scenarios.json` + `round{1,2,3}.py` + `results/`
- `update_commands/round4_fine_ops.py` + `results/round4_report.md`
- `update_commands/round5/round5_experiment.py` + `appliers.py` + `candidates.py` + `scenarios.json` + `results/round5_regraded_report.md`
- `round6/threads/` — R1-R7 routing + scenarios + cache + report
- `round6/compaction/` — C1-C7 compaction + 3 profiles × 8 Qs each + Pareto data
- `round6/pipeline/` — ARCHITECTURE.md + pipeline.py prototype + 4 scenarios + eval
- `round7/` — ARCHITECTURE.md + schemas.py + P1-P4 scenarios + integration test
- `round8_buffered/` — buffered_pipeline.py + E1-E6 scenarios + comparison vs immediate-write
- `prior_art/PRIOR_ART.md`

---

## Round 8: Buffered-commit window

Proposed alternative to retroactive log-rewrite: hold events in a buffer; commit to topic logs when they reach position `commit_age` (middle of `window_size`). Events stay mutable during the buffer period — coref, salience, multi-label gate all re-run until commit.

### Results

- **In-window coref/salience/correction**: buffered-commit wins cleanly over immediate-write. Turn-1 "my boss" fact gets retroactively re-routed to Marcus/… when Marcus is named at turn 10 (distance 9, within `commit_age=15`).
- **Beyond-window coref**: still fails (expected). Retroactive rewrite remains the fallback.
- **End-of-stream flush**: commits with reduced future context. Tolerable.
- **Query path**: must read buffer ∪ committed logs to see facts still pending. Confirmed viable.

### Two failure modes discovered

1. **Succession vs correction collision.** "Marcus left — Alice is my new boss" (succession: both in slot history) vs "Actually, Alice, not Marcus" (correction: only Alice) are structurally similar but require opposite handling. First-pass implementation dropped succession predecessors. Fixed heuristically by requiring explicit correction cues — works but fragile.

2. **Silent salience drops.** "Allergic to peanuts" got salience-deferred and never committed. Medical/user-core facts leak through the salience gate. Fix: category-based whitelist (`User/Medical`, `User/Biography` always admit) or tighter salience signals.

### Append-only tension

Buffered-commit as currently implemented can **drop** entries pre-commit on corrections — violates round-5's principle that retractions are appends, not deletions. **Purer variant**: pre-commit can re-route target topic, but cannot drop entries. Corrections commit the original AND an invalidate relation. One extra log entry per correction, but preserves audit and round-5 semantics. **Recommended over drop-pre-commit** — untested but principled.

### Updated pipeline

```
event arrives → buffered entry (no commit)
  ↓ each subsequent turn (while age < commit_age=15):
    - re-run coref against new named entities (may update target topic)
    - re-run salience (may flip defer ↔ admit)
    - re-run multi-label gate if state changes
    - on correction: append invalidate relation (pure variant)
  ↓ commit at age=15 → topic logs
  ↓ fallback: retroactive rewrite for coref detected >window later
query reads: committed logs ∪ in-memory buffer
```

### Round 8 open items

- Pure append-only variant (commit + invalidate on correction) untested
- Salience category-whitelist for medical/user-core facts
- Beyond-window coref still needs retroactive-rewrite fallback (rare but real)
- Succession/correction disambiguation beyond heuristic cues

### Recommendation

Ship buffered-commit with `commit_age=15` as the default. Keep retroactive rewrite as an explicit fallback path. Add category-based salience whitelist. Migrate from drop-pre-commit to commit+invalidate for cleaner semantics.

---

## Round 9: AEN-1 (single-log) vs partitioned at 100-turn scale

Tested three architectures (AEN-1 single log with @mentions, AEN-1 + LLM materialized views, round-7 entity-partitioned) at 110 turns / ~100-170 entries across 20 state-tracking questions.

### Results (after adjusting for grader word-matching artifacts)

- **AEN-1**: ~18/20, $0 (cache hit)
- **AEN-1 + views**: 11/20 — materialization drops nuance. Pareto-dominated.
- **Partitioned**: ~18/20, $0.45 — lost Priya's Google employment history (earlier entry superseded by later one in the entity log)

**Key finding**: AEN-1 ties partitioned on accuracy at modest scale with far less structural complexity. LLM-generated views are a net negative — same failure mode as round-6B's active consolidation.

---

## Round 10: AEN-1 + structural indexes at 200-2000 entries

Architecture addition: lossless, deterministic indexes maintained at write time on top of AEN-1.

```
AEN-1 log (authoritative)
  + mention_index:      @entity → list[uuid]
  + category_index:     category → list[uuid] (if present)
  + supersede_head:     (@entity, predicate) → current_entry_uuid
  + embedding vectors
```

No LLM rewriting. Pure data structures.

### Degradation curve (flat)

| Entries | Plain AEN-1 | Indexed AEN-1 |
|---|---|---|
| 200 | 100% (45/45) | 100% (45/45) |
| 500 | 96% (43/45) | **100%** (45/45) |
| 1000 | 96% (53/55) | **100%** (55/55) |
| 2000 | 100% (5/5) | 100% (5/5) |

**Scale-invariant on tested metrics.** 18-element supersede chain at 2000 entries: resolved perfectly.

### Why it holds

Chain coherence is a structural property, not a retrieval property. Once retrieval lands on *any* chain entry, transitive ref-walk gives the rest. `supersede_head` is O(entities × predicates_per_entity), not O(log_size). Current-state queries bypass embedding entirely.

### The one differentiator: untagged-pronoun scenarios

Plain AEN-1 failed 0/2 on scenarios where later entries in a supersede chain were tagged with pronouns ("she started a new job") instead of `@Priya`. Indexed AEN-1 won 2/2 because `supersede_head` is populated from the ref's `predicate` metadata at write time, independent of later entries' tagging.

Indexes compensate for writer slips. This is exactly their value at scale.

### Big caveat

Round 10 used pre-built canonical log entries, **not LLM-written ones.** Retrieval + reader were isolated from writer reliability. Round 9's real-LLM writer at 110 turns had ref-emission gaps; at 10K+ turns these gaps compound. **Writer reliability is the dominant scaling risk — not retrieval.**

### Extrapolation to 100M tokens

On the evidence: **yes, AEN-1 + structural indexes extrapolates** — IF the writer reliably emits @-tags and supersede refs.

The supersede_head map, mention index, and ref-graph walk are all scale-invariant on log size. Embedding only needs to surface one chain entry among many; even at 100K entries this is well within top-K=60 capability when mention-filtered.

### What's still untested at 100M

1. **Writer ref-emission reliability over 10K+ turns** — the load-bearing unknown.
2. **Reader context overflow** when `mention_index[@User]` holds thousands of UUIDs and the 60-entry cap forces lossy selection.
3. **Same-name disambiguation** ("Marcus the boss" vs "Marcus the cousin") — untested at scale.
4. **Multi-entity set queries** at large N (10+ household members, 20+ contacts).
5. **Pure recall@K** separated from reader-LLM tolerance — to disentangle retrieval quality from reader tolerance.

### Recommendation

**Ship AEN-1 + structural indexes as the default storage architecture.** The retrieval + indexing layer is scale-invariant up to ~100K entries on the evidence. Round 11 focus should be **writer-reliability under stress** (ref-emission rate over 1000+ LLM-written turns, drift in @-tag consistency, supersede-miss frequency) — not another retrieval experiment.

### Final architecture stack (rounds 6 + 7 + 8 + 9 + 10 integrated)

```
event memory (stateless, raw text)
  ↓ per-turn OR 5-turn fused extraction LLM call
  ↓ buffered-commit window (commit_age=15, corrections via invalidate relation)
      - coref resolves in-buffer against named-entity intros
      - salience re-scores as evidence accumulates (category-whitelist fix)
      - multi-label gate re-evaluates
  ↓ commit →
        AEN-1 single log (authoritative)
        + mention_index, category_index (if used)
        + supersede_head map
        + embedding vectors
  ↓ background consolidation (C6 relation-aware, with (root, sub-attribute) fix)
query interface:
  - Current-state: O(1) consult of supersede_head
  - History: transitive ref-walk from chain head
  - Entity profile: mention_index filter + top-K embedding
  - Multi-entity: intersect mention_index entries
  ↓ reader LLM
```

---

## Rounds 11-15: Writer-side and entity-layer evidence at scale

### Round 11: single ref type wins

Phase 1 (110 turns): simplified `revise/add/remove/noop` (one ref type) scored 15/20 vs four-typed-relations 14/20 at half the LLM cost.

Phase 3 (1000 turns): simplified ref_emission_rate **89%** vs typed **68%**; simplified Q/A **15/30** vs typed **13/30**. Typed writer drops more refs because relation-choice adds friction.

**Drop the four-relation schema.** Single `ref` type for all updates (invalidation expressed as more text, not a different relation).

### Round 12: entity registry beats baseline on disambiguation, loses on LRU eviction

`aen2_registry` (entity IDs + LRU=20 active cache, no persistence beyond LRU): scored +11pp on same-name (S1), +15pp on different-name (S2), +24pp on pronoun chains (S4). But lost **-23pp on LRU stress (S3)** because entities evicted from LRU were unreachable.

### Round 13/13b: persistent registry + lazy embedding pull recovers descriptors

`aen3_persistent` (same as aen2 + persistent registry + embedding-search-then-LLM-pick fallback for descriptors that miss the active cache): structural fix is correct but original descriptor-pick prompt was too strict and graders had a surface-normalization bug.

`aen3b` (with grader fix + soft descriptor prompt + per-turn description accumulation): **97% on S3 (descriptors 100%)**, **100% on S6 (134-turn long delay)** — matches baseline AND keeps the disambiguation wins from rounds 12-13.

Three load-bearing fixes:
1. Surface-normalization in grader (~20pp)
2. Soft descriptor-pick prompt (match on consistent role/features, not strict feature-alignment)
3. Per-turn snippet accumulation into description blob (no LLM cost — pure concat; gives embedding search rich text to match against)

### Round 14: writer ref-emission collapses at scale

Dense-chains scenario (743 turns, 86 transitions, 14 active predicates) — *real* chain stress, unlike round 11's 24 transitions in 1000 turns.

aen1_simple ref-emission per bucket:

| Bucket | Trans | Emit | Correct |
|---|---|---|---|
| 0-100 | 8 | 62% | 50% |
| 100-200 | 14 | 71% | 43% |
| 200-300 | 10 | 60% | 30% |
| 300-400 | 9 | 44% | 22% |
| 400-500 | 10 | 30% | 20% |
| 500-600 | 10 | 50% | 0% |
| 600-700 | 13 | 31% | 8% |
| 700-800 | 12 | **17%** | **8%** |

End-to-end Q/A: 18-19/32 (~56-59%).

**Mechanism**: writer's prompt window can't hold all chain heads as the log grows. Chains quiet for many turns fall out of view; writer can't ref what it can't see. **Round 11's "89% ref emission" was a sparse-transitions artifact.**

This invalidated the unconditional shipping recommendation. Writer reliability needed a fix.

### Round 15: at-write-time active-chain injection FIXES the collapse

`aen1_active`: before each writer batch, identify the entities mentioned and inject their active chain heads (looked up from the structural `supersede_head` index) into the writer's prompt as an "ACTIVE STATE" block. The writer can now ref any chain head regardless of distance — the structural index is O(1) and scale-invariant.

cap=100 (100-token max active-state block) result on the same dense-chains scenario:

| Bucket | R14 emit | R15 emit | Δ | R14 correct | R15 correct | Δ |
|---|---|---|---|---|---|---|
| 0-100 | 62% | 88% | +26 | 50% | 75% | +25 |
| 100-200 | 71% | 100% | +29 | 43% | 93% | +50 |
| 200-300 | 60% | 100% | +40 | 30% | 60% | +30 |
| 300-400 | 44% | 78% | +34 | 22% | 56% | +34 |
| 400-500 | 30% | 100% | +70 | 20% | 80% | +60 |
| 500-600 | 50% | 90% | +40 | 0% | 50% | +50 |
| 600-700 | 31% | 85% | +54 | 8% | 62% | +54 |
| **700-800** | **17%** | **83%** | **+66** | **8%** | **67%** | **+59** |

Mean ref-emission across all buckets: **47% → 90%**. Mean ref-correctness: **20% → 68%**.

**The drift is gone.** Writer ref-emission stays in the 78-100% band across all buckets — no degradation curve at scale.

QA didn't complete due to embed budget cap; ref-emission/correctness curves are the structural signal that downstream Q/A inherits.

## Final integrated architecture (rounds 6-15)

```
event memory (stateless, raw text)
  ↓ per-turn OR 5-turn fused extraction call
  ↓ buffered-commit window (commit_age=15)
      - in-window coref resolves against active LRU entity cache
      - on correction: append invalidate-relation entry (no drop)
      - salience re-scores
      - multi-label gate re-evaluates
  ↓ entity-resolution layer (aen3b)
      - active LRU cache (~20 entities, recently mentioned)
      - persistent registry: by_id, aliases_index, desc_embed_index
      - lazy embedding-pull from persistence on cache miss → LLM picks
      - per-turn snippet accumulation into entity descriptions
  ↓ writer with active-chain injection (aen1_active)
      - for each entity in batch, look up supersede_head heads
      - inject "ACTIVE STATE" block into writer prompt
      - writer emits text with @ent_id tags + single-type refs
  ↓ commit →
        AEN-1 single log (authoritative)
        + mention_index (lossless, deterministic)
        + supersede_head map (lossless, deterministic, O(1) current state)
        + embedding vectors per entry
        + per-entity description embeddings
  ↓ background consolidation
query interface:
  - Current-state: O(1) consult of supersede_head
  - History: transitive ref-walk from chain head
  - Entity profile: mention_index filter + top-K embedding
  - Multi-entity: intersect mention_index entries
  ↓ reader LLM
```

## Status of the eight known failure modes

| Failure mode | Mitigation | Status |
|---|---|---|
| Multi-label routing | Decision-tree gate (P1 round 7) | Tested 9/9 |
| Cross-turn coref (in-window) | Buffered-commit (round 8) | Tested 6/6 within window |
| Role updates with fanout | Role slots OR single-log w/ supersede | Tested 5/5 |
| Salience extraction | Multi-signal lazy filter (P4 round 7) | Tested 6/6, plus medical whitelist needed |
| Same-name disambiguation | Persistent registry + LLM-judged disambiguation (round 13b) | Tested S1, S6 100% |
| Different-name same-entity | Alias addition + LLM confirm (round 13b) | Tested S2 93% |
| Entity descriptor recovery (LRU eviction) | Embedding pull + LLM pick (round 13b) | S3 97%, S6 100% |
| **Writer ref-emission at scale** | **Active-chain injection at write-time (round 15)** | **78-100% across 800 turns** |

## What's still untested at production scale

- Round 15 cap=200 ablation didn't run; cap=100 chosen pragmatically
- End-to-end Q/A on round-15 dense-chains (budget cap hit during judging)
- Aen3b on S6 + dense-chains combined (entity layer + writer fix together at length)
- 5000+ turn lifetime simulation
- Multiple-user chat (all stress so far is single-user-with-mentioned-entities)
- Real-world conversational data (LoCoMo or actual chat traces)

The architecture is **shippable on the evidence at the scales tested**, with the eight previously-flagged failure modes each having a concrete, evidence-backed mitigation. Production validation remains the outstanding work.

## Rounds 16-21: writer schedule, variable binding, partitioned cognition

### Round 16: world scoping + sliding writer + ref correctness
- 16B-v2: 2-cat world (real / non_real) wins over 3-cat or 4-cat on novel and mixed scenarios. (Later corrected: world=non_real ONLY for fiction/thought-experiments; user's plans/conditionals stay in real-world with mental-state predicates. See feedback_world_scope.)
- 16A: sliding K=3 W=15 writer (target at END of window) on multi_batch_coref scored strict=1/8, soft=8/8, ref=7/8, QA=7/8.
- 16C v2: deterministic relinker that walks log chronologically, skips clarify-style entries when computing supersede_head. +28pp Q/A vs naive.

### Round 17: extra_memory parity test
- Adopted extra_memory architecture for parity check: MemoryEntry with cluster_id + MemoryResolution layer + per-cluster history.
- Identified extra_memory writer-prompt issues (regex hallucinations of @Big/@Pretty, predicate synonym gaps for next_job/manager/interest, ResolutionRow.metadata SQLAlchemy collision).
- User pruned cruft (question-kind heuristic, reranker integration, anchor_label genericization at ingest time).
- anchor_label kept for query-time only.

### Round 18: K-block centered window
- Centered window with target K-block at MIDDLE of window (W = w_past + K + w_future). Sliding by K. Cost = N/K LLM calls.
- Controlled A/B vs round 16a's last-window K=3 W=15:
    - centered_K3_w6_w6 (window=15): QA = 6/8 (vs last-window 7/8). Centering target alone, at the same window size, didn't help and slightly hurt because the writer over-emitted and shifted resolution location.
    - centered_K3_w7_w14 (window=24, 2x fudge on w_future): QA = 6/8. Larger window did not recover — bigger prompt = more drift.
- Conclusion: centering is orthogonal at this configuration; v2's last-window with active-state injection already captures cross-batch coref via active state at name-turn fire.

### Round 19: variable-binding writer (the major architectural shift)

User pushed framing A: anonymous descriptors are existential variables; later names are LABEL ASSIGNMENTS to those variables, not new facts about a name. Schema: LogEntry has cluster_id (chain identity); a separate Resolution event binds canonical_label to a cluster.

Two implementations tested:
- v1 (separate Entry + Resolution items in writer output): schema too complex for the model. Multi_batch_coref QA = 5/8 (w7+w14) and 2/8 (w7+w7). Writer over-emitted filler (273-384 entries for 126 turns) and frequently mis-attributed predicates.
- **v2 (single emit, canonical_label as field on entry)**: 37 entries / 126 turns. **Multi_batch_coref QA = 8/8** — best result so far. Wins via:
    1. Single-emit schema (post-processor derives Resolutions from any entry where canonical_label is set on an existing cluster).
    2. Hard filler skip (explicit DO-NOT-EMIT examples for body sensations, weather, transient feelings).
    3. Active-state injection filtered to durable predicates only — breaks the ephemeral-chain feedback loop where filler sensations were echoed back as "active chains".

V2 generalization on harder scenarios (K=3 w_past=7 w_future=7, fresh caches):
- multi_batch_coref: 8/8 (100%)
- dormant_chains[:200]: 9/10 (90%)
- dense_chains[:200]: 17/23 (74%)

Dense_chains is the weak point — writer skips hobby/car/gym/commute predicates as "routine". V3 broader prompt picks them up but creates retrieval noise (10/23, regression). The bottleneck is retrieval-side dedup of chain history at current-state queries, not write-side coverage.

### Round 20: write-time cognition pass (entry_type filter, flat-mixed storage)

Added a second LLM pass after the writer (the "cognizer") that emits cognition entries (predicates @User.expectation, @User.plan, @User.belief, @User.fear, @User.confirmation). Same store, type-filtered at retrieval.

| variant | obs / cog | QA judge |
|---|---|---|
| coref cog_off | 51 / 0 | 6/8 (prompt micro-regression vs v2) |
| coref cog_on | 66 / 98 | 7/8 — gained Quentin+Nadia, **lost Marcus (boss→Quentin via cognition contamination)** |
| hbr cog_off | 24 / 0 | 5/5 (the conditional ENTRY itself is retrievable) |
| hbr cog_on | 26 / 46 | 5/5 — cognition didn't add value |

Findings:
1. The Marcus→Quentin swap is the project_em_ingest_substrate failure mode: cognition entries flat-mixed with same (subject, predicate) chain key as observations corrupt the chain_head pointer.
2. Hypothetical-becomes-real works at 5/5 WITHOUT a cognition pass. The conditional ("If hired at @Notion, my boss will be @Sam") is itself an observation entry; normal kNN + mention-index retrieval surfaces it. This validates feedback_no_explicit_conditional_links — spreading activation = normal retrieval.
3. Cognition over-emits (98 in coref ≈ 1.5x observations). Doubles LLM cost.

### Round 21: partitioned storage (the architectural fix)

Split the memory store into two IndexedCollections (observations + cognition), each with its own indexes. Active state at write-time pulls from observations only. Retrieval is multi-collection with keyword-based gating.

| variant | obs / cog | QA judge | vs R20 |
|---|---|---|---|
| coref part_cog_off | 34 / 0 | 7/8 | +1 (recovered from prompt regression) |
| coref part_cog_on | 32 / 72 | 7/8 | **Marcus stays correct, no swap** |
| hbr part_cog_off | 9 / 0 | 4/5 | -1 (v2 prompt skips possessions) |
| hbr part_cog_on | 12 / 27 | 4/5 | cognition didn't recover Q05 (gating too strict) |

**Architectural win confirmed**: partition prevents cognition from corrupting observation chain heads. R20's contamination is gone.

Remaining gaps (post-architecture):
1. Writer prompt tradeoff: v2 skips possessions (HBR Q05 "Did User end up buying a bike?" fails).
2. Cognition retrieval gating too strict: keyword "expect/plan/think/feel" misses confirmation-of-plan questions like "Did User end up doing X?".
3. Cognizer over-emits (72/27 entries vs 32/12 observations) — should fire only on clear triggers (conditionals, contradictions, naming events).

## Locked architecture (rounds 6-21)

```
event memory (stateless, raw text)
  ↓ K=3 centered sliding window (target K turns at MIDDLE; w_past=7, w_future=7)
  ↓ writer: variable-binding schema (LogEntry with cluster_id, canonical_label, subject, predicate)
        - single emit type; canonical_label as field
        - hard filler skip with explicit categories
        - active-state injection filtered to durable predicates
  ↓ optional cognizer: emits cognition entries to a SEPARATE collection
        - predicates: @User.expectation/.plan/.belief/.fear/.confirmation
        - cluster_ids in shared namespace; cross-collection mentions link
  ↓ commit →
        observations IndexedCollection
            chain_head[(subject, predicate)] = cluster_id
            cluster_label[cluster_id] = canonical_label (latest resolution)
            mention_index[@entity] = [uuid]
            embed_by_uuid[uuid] = vector
        cognition IndexedCollection (same schema, separate indexes)
        + (future) images, audio, metacognition collections per modality
query interface:
  - factual: query observations only
  - cognitive (expect/plan/think/feel): query observations + cognition, merge ranked
  ↓ reader LLM (sees retrieved entries + cluster label table; substitutes labels)
```

## Status of failure modes (R6-R21 update)

| Failure mode | Mitigation | Status |
|---|---|---|
| Multi-label routing | Decision-tree gate (P1 round 7) | Tested 9/9 |
| Cross-turn coref (in-window) | Buffered-commit + active-state at name-turn (R15) | 5-7/8 strict, 7/8 ref, 8/8 QA on multi_batch_coref |
| Cross-batch coref | Variable binding via cluster_id (R19 v2) | 8/8 QA on multi_batch_coref |
| Hypothetical-becomes-real | Conditional entry as observation; normal retrieval (no special activation) | 5/5 QA at v2 baseline (R20) |
| Same-name disambiguation | Persistent registry + LLM-judged disambig (R13b) | Tested S1, S6 100% |
| Cognition pollution of facts | Partitioned storage (R21) | Marcus stays correct in cog_on |
| Writer ref-emission at scale | Active-chain injection at write-time (R15) | 78-100% across 800 turns |
| Cognition contamination of chain heads | Separate collection per modality (R21) | Validated on coref |
| Cognition surfacing too aggressive | Keyword retrieval gate + cognizer fires only on triggers | TBD round 22 |
| HBR possessions / "did User end up X" | Writer prompt + cognition retrieval gating | TBD round 22 |

## Architectural commitments

- LogEntry: uuid, ts, cluster_id, text, subject, predicate, canonical_label, mentions, refs, collection.
- No `world` field. Hierarchical cluster_ids encode separation (e.g., `dune.paul_atreides`). See feedback_hierarchical_cluster_ids.
- No explicit conditional/triggers/depends_on links. Conditional structure is implicit in mentions+text. See feedback_no_explicit_conditional_links.
- Mental-state predicates (`@User.expectation` etc.) live in `cognition` collection; observation-shape factual chains stay in `observations`.
- Memory partitioned per modality; each collection has own encoder/index/retrieval policy. See feedback_memory_partitions_per_modality.
- Memory IS cognition's substrate, not passive storage. Writer + cognizer + (future) read-time cognizer + metacognition all emit to the unified event stream. See feedback_unified_cognition_stream.

### Round 22: cognition substrate becomes additive (the working architecture)

Three deltas on R21's partitioned base:
1. **Writer prompt expansion** — added possessions, hobbies, recurring routines, confirmed plans as chain-worthy.
2. **Cognition retrieval gate broader** — includes "end up", "actually", "did user end", "would happen", "going to do" alongside expect/plan/think/feel.
3. **Cognizer trigger-gated** — strict CONDITIONAL / CONFIRMATION / CONTRADICTION / NAMED-HOPE-FEAR triggers. Default empty. Max ~1 per K-block.

| variant | obs / cog | QA judge | vs R21 |
|---|---|---|---|
| r22_coref_cog_off | 28 / 0  | 7/8 | = |
| r22_coref_cog_on  | 32 / 13 | 7/8 | cog 5x quieter, no contamination |
| r22_hbr_cog_off   | 12 / 0  | 4/5 | (different miss: Q05 recovered, Q04 lost without cognition) |
| **r22_hbr_cog_on** | 20 / 17 | **5/5** | **+1 — cognition recovered Q04 conditional plan** |

**The cognition substrate now works**: architecturally neutral on factual queries (no contamination, thanks to partition), additive on cognitive queries (Q04 "What did User intend to do if moving to Berlin?" recovered via CONDITIONAL-trigger cognition entry). This is the first round where the cognition pass gave a positive end-to-end QA delta.

**Required pieces (all needed):**
- R19 v2: variable-binding writer (single emit, canonical_label as field, hard filler skip).
- R21: partitioned storage (observations vs cognition in separate IndexedCollections).
- R22: writer prompt expansion + retrieval gate broadening + cognizer trigger gating.

Cost: ~30% more LLM calls with cognition on vs off. Cognition adds ~$0.10/scenario.


### Round 22 generalization across all scenarios

| scenario | cog_off | cog_on |
|---|---|---|
| multi_batch_coref (8 Qs)        | 7/8   | 7/8   |
| hypothetical_becomes_real (5 Qs)| 4/5   | **5/5** |
| dormant_chains[:200] (10 Qs)    | 10/10 | 10/10 |
| dense_chains[:200] (23 Qs)      | 14/23 | **20/23** |
| **aggregate**                   | 35/46 (76%) | **42/46 (91%)** |

The cognition pass adds **+15 percentage points aggregate** on top of the partitioned variable-binding base. The dense_chains delta (+6 questions) is the most surprising: cog_on causes the writer to emit fewer observation entries (100 vs 134) which reduces retrieval noise, while cognition's confirmation entries anchor current-state queries better than chain-history flooding.

**Cost:** cog_on ≈ +30% LLM calls vs cog_off. For +15pp absolute QA improvement, strong ROI.

**This validates the full architectural stack** at gpt-5-mini caliber across four scenario shapes:
- Variable binding (R19 v2): single-emit schema, hard filler skip, durable-predicate active state
- Partitioned storage (R21): observations and cognition in separate IndexedCollections
- Trigger-gated cognizer + broader retrieval gate (R22): cognition is neutral on factual, additive on cognitive


### Round 22 fair head-to-head: variable binding (R22) vs persistent registry (R13b-fair)

Built `aen3b_fair.py` to put R13b's persistent registry + per-turn coref pass on top of R22's general optimizations (K=3 centered window scheduler + filler-skip writer prompt + active-state filtered to durable predicates). This isolates the architectural delta:

- R13b-fair: persistent entity registry, per-turn coref pass, LRU active cache, lazy embedding pull on cache miss, LLM-judged disambiguation. Writer sees `@ent_<id>` tags after coref rewriting.
- R22 cog_on: variable binding (cluster_id + canonical_label-as-field), partitioned observations + cognition collections, trigger-gated cognizer.

Both use the same scheduler (K=3 centered, w_past=w_future=7), the same filler-skip writer prompt structure, and the same durable-predicate active-state filtering.

| scenario | R13b-fair | R22 cog_off | R22 cog_on |
|---|---|---|---|
| HBR (5)             | 4/5   | 4/5   | **5/5** |
| multi_batch_coref (8) | **3/8** | 7/8 | 7/8 |
| dorm[:200] (10)     | 10/10 | 10/10 | 10/10 |
| dense[:200] (23)    | **20/23** | 14/23 | 20/23 |
| **aggregate**       | **37/46 (80%)** | 35/46 (76%) | **42/46 (91%)** |
| **cost**            | **$2.74 (913 LLM)** | ~$0.40 | ~$0.55 |

**Findings:**

1. **Variable binding decisively wins descriptor→name binding** (multi_batch_coref 7/8 vs 3/8). R22's `chain_head[(@User, predicate)] = cluster_id` pointer holds slot identity steady from descriptor through name reveal. R13b's registry creates separate entity_ids per surface ("the boss" → ent_X, "Marcus" → ent_Y) and the writer must re-pair entities with predicate slots independently — and on this scenario, it mis-pairs Q05 (friend → "Theo" instead of Sana), Q06 (neighbor → "Sana" instead of Alice), Q08 (senior → "Quentin" instead of Nadia).

2. **R13b-fair surprisingly ties cognition on dense_chains** (20/23). Per-turn coref + persistent registry handles many-chain density well — the writer always has the canonical entity_id for each mention, so chain transitions stay clean. R22 cog_off without cognition fragments at 14/23; R22 cog_on recovers via cognition's confirmation entries.

3. **Both architectures equal on long-gap supersession** (dorm 10/10). Both registry persistence and chain_head pointer survive 50+ turn dormancy.

4. **R13b-fair costs ~5x more.** The per-turn coref pass is +1 LLM call per turn (200+ extra calls on dense alone). R22's writer sees active state directly without a separate coref step.

5. **Cognition pass is uniquely valuable for cognitive/conditional queries** (HBR 4/5 → 5/5 via Q04 conditional plan recovery), and adds bonus value on dense_chains (14 → 20/23) by reducing observation over-emission while anchoring current-state via @User.confirmation entries.

**Architectural verdict:**
- Variable binding is architecturally superior on descriptor binding AND cheaper overall.
- Persistent registry is competitive on chain density and long-gap supersession.
- Cognition pass uniquely covers cognitive/conditional queries.
- The right production system combines variable binding's chain_head + cluster_id with (a) registry-style per-cluster description accumulator (R13b's contribution) for richer entity matching at scale, (b) partitioned cognition collection.


### Round 23: prose-fact + DSU (symmetric mention/entity design)

User pushed for: (a) all facts are prose, no specialized fact subtypes; (b) symmetric design where every mentioned entity has an opaque mention_id, opaque entity_ids canonical via disjoint-set; (c) names/descriptions are facts (prose with mentions), not separate fields; (d) no mandatory subject/predicate/object structure.

Built `aen6_prose.py` with:
- Fact: prose text + mention_ids list, in a collection
- Mention: opaque per-occurrence ID + surface
- EntityRegistry: DSU with `merge(m1, m2)`, `split(m1)`, `get_canonical(m)`, BindingEvent audit log
- No predicate field, no chain_head (current state derived at retrieval time)
- Writer emits prose facts + per-mention `resolves_to` decisions (existing entity_id or "new")
- Retrieval: surface match → mention_ids → canonical entity_ids → entity_facts; plus kNN

| variant | coref (8) | HBR (5) | dorm (10) | dense (23) | aggregate |
|---|---|---|---|---|---|
| R23 v1 (initial)  | 2  | 5 | 10 | 11 | 28/46 (61%) |
| R23 v2 (refined)  | 6  | 5 | 9  | 14 | 34/46 (74%) |
| R23 v3 (+cog)     | 6  | 5 | 9  | 11 | 31/46 (67%) |

V1 over-emitted (124/319 facts) and under-merged (51/236 entities). V2's writer-prompt refinement (hard filler skip + "DEFAULT TO REUSING EXISTING ENTITIES" + explicit name-reveal example showing both descriptor and name resolving to the same existing entity_id) brought it to 34/46 — competitive with R22 cog_off (35/46).

V3 added the R22-style cognition pass on top of v2. **Cognition REGRESSED dense 14→11.** Why: R22's cognition makes the writer conservative on observations (134→100); R23's cognition AMPLIFIES observation count (166→181) plus 41 cognition facts. More retrieval flooding, not less.

**Final architectural ranking from this session:**

| architecture | aggregate | scenario strengths |
|---|---|---|
| R22 cog_on | 42/46 (91%) | wins overall; variable binding + partition + trigger-gated cognition |
| R13b-fair | 37/46 (80%) | registry competitive when general optimizations match |
| R22 cog_off | 35/46 (76%) | variable binding alone is strong |
| **R23 v2** | **34/46 (74%)** | prose-fact + DSU; cleanest schema; near-tied |
| R23 v3 | 31/46 (67%) | cognition amplifies prose over-emission |
| R23 v1 | 28/46 (61%) | over-emission baseline |

**The conceptually-cleanest architecture (R23 v2) is competitive but doesn't beat R22 on the current benchmark.** R23's claimed advantages (anonymous-subject chains, no same-name collision, property-of-property representation, contextualized fact support) aren't exercised by current test scenarios — multi_batch_coref/HBR/dorm/dense use unique names per chain and SPO-shaped questions.

**For R23 to surpass R22 we'd need:** (a) scenarios that exercise R23's strengths (multi-Alice disambig, contextualized facts like "sky color in Tokyo at sunset", anonymous-subject reasoning); (b) cognition integration that doesn't amplify observation count on prose substrate; (c) hybrid: prose-fact base with optional chain anchors when natural.

**For now: R22 cog_on (variable binding + partitioned cognition) is the production-validated architecture at 91% aggregate.** R23's prose-fact + DSU is the cleaner long-term design but needs (a)-(c) to demonstrate its theoretical advantages on benchmarks.

