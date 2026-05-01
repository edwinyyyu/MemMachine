# R25 — Working Memory Architecture (current best design)

Best-confirmed result on a 3-scenario slice of the R24 suite: **34/39**
(mbc 8/8, snd 6/8, dense 20/23) at all-medium reasoning, vs baseline 33/39.

Architecture is built on top of the R23 prose-fact + DSU substrate, with
R24's recursive-cognition reflector. R25 adds: working-memory block,
COLLIDING SURFACES disambiguation, pronoun antecedent rule, within-fire
`_anon` placeholder, named-default-new rule, and reader-side Entity
A/B/C sidebar — all motivated by cognitive analogs (entity-in-context vs
concrete-entity, short-term thread vs long-term store).

Diagnostic spot-check finding: writer at medium reasoning is the bottleneck
on snd's modifier-disambiguation. Writer=high lifts snd 6→7/8. Architecture
is sound; medium underutilizes the rules.

## LLM stages (writer + reflector + reader + dedup)

```
┌────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│                                  File                                  │                                  Role                                  │
├────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py                     │ WRITE_PROMPT (lines 555–839) + write_window (840–1035). Writer        │
│   :555 WRITE_PROMPT                                                    │ emits prose facts with mentions; resolves_to ∈ {<mention_id>, "user", │
│   :840 write_window                                                    │ "_anon_<label>", "new"}. Identity-vs-role rule, named-default-new      │
│                                                                        │ rule, pronoun antecedents, modifier disambiguation under collision.   │
├────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ round24_recursive_cognition/architectures/aen7_recursive.py            │ REFLECT_PROMPT (lines 360–478) + reflect_on_fact (582–639). Mental-   │
│   :360 REFLECT_PROMPT                                                  │ state-only verbs (expects/plans/hopes/confirms/contradicts/believes/  │
│   :582 reflect_on_fact                                                 │ felt). Five triggers: CONDITIONAL, CONFIRMATION, CONTRADICTION,        │
│                                                                        │ NAMED HOPE/FEAR, EMOTIONAL_STATE.                                     │
├────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py                     │ READ_PROMPT (1179–1224) + answer_question (1354–1372). Reader sees    │
│   :1179 READ_PROMPT (used by both v2 and v7)                           │ Entity A/B/C sidebar with discriminating excerpts + COLLIDING          │
│   :1354 answer_question                                                │ SURFACES warning. Disambiguation rule for question-side colliding      │
│ round24_recursive_cognition/architectures/aen7_recursive.py            │ surfaces. v7 answer_question (781–805) adds multi-collection retrieval │
│   :781 answer_question (multi-collection + expand_hops=1 + llm_dedup)  │ (observations + cognition), expand_hops=1, llm_dedup=True.            │
├────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ round24_recursive_cognition/architectures/aen7_recursive.py            │ Hybrid cosine+LLM dedup: cosine 0.80 prefilter, batched LLM judge      │
│   :81 DEDUP_PROMPT                                                     │ with transition-aware prompt, union-find merge of decided duplicates.  │
│   :120 _llm_dedup_filter                                               │ Critical: argument-flip-safe ("Alice paid Bob" ≠ "Bob paid Alice").   │
└────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

## Data model & registry

```
┌─────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
│                              File                               │                                 Role                                  │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ Mention(mention_id, surface, fact_uuid, ts) — per-occurrence handle.  │
│   :67 Mention, :75 Fact, :84 BindingEvent                       │ Fact(fact_uuid, ts, text, mention_ids, collection) — atomic prose.   │
│                                                                 │ BindingEvent — provenance for DSU merges.                             │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ EntityRegistry — DSU over mentions. mention_to_entity[m] = e_xxx.    │
│   :100 EntityRegistry                                           │ register/get_canonical/merge/split. Entity_ids stay system-internal;  │
│                                                                 │ never appear in any prompt rendered to the LLM.                       │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ IndexedCollection — per-collection structures: by_uuid, mentions_by_  │
│   :193 IndexedCollection, :205 MemoryStore                      │ id/fact/surface, facts_by_entity, embed_by_uuid. MemoryStore holds   │
│                                                                 │ collections + shared registry. build_collection (218–270).            │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round7/experiments/_common.py                                   │ Cache (sha-keyed by model + reasoning_effort + prompt). Budget        │
│   :Cache, :Budget, :llm, :embed_batch                           │ (max_llm/max_embed limits). llm() default reasoning_effort="low";     │
│                                                                 │ overridden to "medium" at all four call sites in v2/v7. embed_batch  │
│                                                                 │ batches text-embedding-3-small calls.                                 │
└─────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

## Working-memory rendering (R25 additions)

```
┌─────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
│                              File                               │                                 Role                                  │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ WORKING MEMORY block: recent N=7 facts in chronological order, each  │
│   :414 render_working_memory                                    │ tagged with named/(unnamed: <descriptor>) entities. Recency-driven    │
│                                                                 │ recall into working memory. Pronoun antecedent source.                │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ ACTIVE MENTIONS block: mentions grouped by current DSU class with    │
│   :272 render_active_mentions                                   │ render-local Group A/B/C aliases (no entity_ids exposed). Per-       │
│                                                                 │ mention disambiguator excerpts. Top "⚠ COLLIDING SURFACES" warning   │
│                                                                 │ lists each colliding surface with each group's most-recent fact      │
│                                                                 │ inline as discriminator. Association-driven recall into working      │
│                                                                 │ memory.                                                               │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ render_recent_facts (406–412) — small fallback block, kept for       │
│   :406 render_recent_facts                                      │ context. annotate_context_turn (501–553) — inline mention_id tags    │
│   :501 annotate_context_turn                                    │ on prior CONTEXT turns; only used when inline_anchors=True (off in   │
│                                                                 │ R25; kept for ablations).                                            │
└─────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

## Retrieval & reading

```
┌─────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
│                              File                               │                                 Role                                  │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ retrieve(question, ...) — surface-match (uncapped, entity-keyed) ∪    │
│   :1115 retrieve                                                │ kNN(top_k=14). Returns facts + resolution_map (entity → mentions).   │
│   :1090 extract_question_surfaces                               │ Surface tokenization with STOP-list (1037).                          │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round24_recursive_cognition/architectures/aen7_recursive.py     │ retrieve(...) extends v2: adds collections, llm_dedup (cosine 0.80   │
│   :189 retrieve (overrides v2.retrieve)                         │ + LLM judge), expand_hops (multi-hop entity expansion, depth=1),    │
│                                                                 │ per-collection caps. Default collections = ["observations",          │
│                                                                 │ "cognition"] when reflector is on.                                   │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ format_facts_for_read (1226–1258) — annotates each retrieved fact    │
│   :1226 format_facts_for_read                                   │ with [surface → Entity A/B/C] aliases. format_resolution_map (1260– │
│   :1260 format_resolution_map                                   │ 1338) — per-entity sidebar with surfaces + discriminating fact      │
│   :1340 _build_eid_alias                                        │ excerpts; reader-side ⚠ COLLIDING SURFACES warning. Render-local     │
│                                                                 │ aliases keep entity_ids invisible to the reader.                     │
└─────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

## Ingestion driver

```
┌─────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
│                              File                               │                                 Role                                  │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round24_recursive_cognition/architectures/aen7_recursive.py     │ ingest_turns(turns, ...) — K=3 centered sliding window               │
│   :640 ingest_turns                                             │ (w_past=7, w_future=7). Writer fires once per K-batch; reflector     │
│                                                                 │ runs after writer with reflection_budget=2, reflection_max=3. Index  │
│                                                                 │ rebuilt every 4 fires. Builds two collections: observations (writer  │
│                                                                 │ output) + cognition (reflector output).                              │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round23_prose_facts/architectures/aen6_prose_v2.py              │ ingest_turns(turns, ...) — basic version without reflector. Used as  │
│   :1374 ingest_turns                                            │ fallback / ablation. Pre-registers User as m_user_root → e_user.     │
└─────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

## Experiment harness

```
┌─────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
│                              File                               │                                 Role                                  │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round24_recursive_cognition/experiments/run_one_noinline.py     │ Single-scenario runner. Ingests scenario turns, grades QA via        │
│                                                                 │ deterministic + judge-LLM. Writes per-scenario JSON + cache. Used    │
│                                                                 │ for R25 noinline (mention_id, no inline anchors) experiments.        │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round24_recursive_cognition/experiments/run_one_anchored.py     │ Same as run_one_noinline but with inline_anchors=True. Used for      │
│                                                                 │ ablations against the noinline default.                              │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round24_recursive_cognition/scenarios/                          │ 34 scenario modules covering BEAM-style memory taxonomy: cross-fire  │
│   multi_batch_coref_wrap.py, same_name_disambig.py,             │ coref, dense state chains, same-name disambiguation, conjunction,    │
│   dense_chains_wrap.py, ... (34 total)                          │ counterfactual, theory-of-mind, etc. Each has generate() →           │
│                                                                 │ build_questions() with deterministic + judge expectations.           │
├─────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ round14_chain_density/experiments/run.py                        │ grade_deterministic + judge_with_llm — graders shared across all     │
│                                                                 │ rounds.                                                              │
└─────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

## Concept naming (writer/reader never see `entity_id`)

- **mention_id** = entity-in-context, per-occurrence handle (e.g.,
  `m0093_1`). Stable per occurrence. Leaks to the writer; appears in
  ACTIVE MENTIONS, WORKING MEMORY refs, `resolves_to` outputs, and the
  reader's facts annotation.
- **entity_id** = concrete-entity, DSU equivalence class label (e.g.,
  `e_m0093_1`). Volatile (changes with merges/splits). NEVER appears in
  any prompt — exposed only as render-local Group A/B/C / Entity A/B/C
  aliases that are regenerated each render.
- **`resolves_to`** values: `<mention_id>` (bind to that prior mention's
  current class via DSU) | `"user"` (bind to canonical User mention
  m_user_root → e_user) | `"_anon_<label>"` (within-fire shared
  placeholder for fresh anonymous entity) | `"new"` (fresh entity).

## Memory subsystems analog (cognitive grounding)

- **Long-term store** = full IndexedCollection (vector index + entity
  registry + fact/mention DB). All facts retained.
- **WORKING MEMORY block** = recency-driven recall into the prompt's
  working memory. The "short-term thread" of the conversation.
- **ACTIVE MENTIONS block** = association-driven recall (target-surface
  match + recency) into working memory. The "you've been thinking about
  these entities" view, with collision flags for ambiguous surfaces.
- **READER sidebar** = per-question working memory at retrieval time —
  Entity A/B/C aliases plus discriminating excerpts so the reader can
  bind question modifiers to entities at answer time.

## Open / known limitations

- **snd Q08 distinct-count** under-counts even at writer=high. Writer-
  side individuation of same-name entities under accumulated noise is
  hard for current LLMs.
- **dense Q06/Q22/Q23** baseline failures (yoga hobby, Chicago "ever",
  Anthropic "ever") — judge brittleness or scenario-expectation issues,
  not architecture.
- Spontaneous-association ("deja vu"), implicit-relational inference
  with world knowledge, and density-aware writes are future directions.
