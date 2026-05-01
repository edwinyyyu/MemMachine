# R25 — Working-Memory Architecture for Coreference + Disambiguation

Goal: an LLM-driven semantic-memory writer/reader that handles cross-fire
descriptor→name binding, same-surface disambiguation (multiple Alices),
role-vs-identity distinction (manager succession), and same-fire shared
anonymous entities — using cognitively-motivated context blocks rather than
inline anchors.

## Concepts (writer never sees `entity_id`)

- **mention_id** = entity-in-context. Per-occurrence handle (e.g. `m0093_1`).
  Stable, immutable. **Leaks to writer.**
- **entity** = concrete-entity. DSU equivalence class over mention_ids.
  Volatile (changes with merges/splits). **Never leaks** — rendered as
  per-prompt-render Group A/B/C aliases.
- **Working memory** = recency-driven recall (last N facts in chrono order).
- **Active mentions** = association-driven recall (entity groups matching
  TARGET surfaces).
- **Long-term store** = full IndexedCollection (all facts/mentions indexed).

The writer's `resolves_to` accepts:
  - `<mention_id>` — bind to that prior mention's class
  - `"user"` — canonical User self-reference
  - `"_anon_<label>"` — fresh entity shared across multiple facts in this fire
  - `"new"` — fresh isolated mention

## Score (3-scenario slice, all-medium reasoning)

| Variant                                    | mbc | snd | dense | total |
|--------------------------------------------|-----|-----|-------|-------|
| Baseline (entity_id, no inline, low)       | 6   | 7   | 20    | 33    |
| Anchor v1 (entity_id + inline, low)        | 8   | 5   | 17    | 30    |
| Noinline (mention_id, no inline, low)      | 6   | 6   | 19    | 31    |
| **R25 working-memory (all-medium)**        | **8** | 6 | 20  | **34** |
| R25 + writer=high (snd spot-check)         | —   | **7** | — | —     |

Diagnostic: snd's residual modifier-disambiguation failures are
**reasoning-bound at writer=medium**, not architecture-bound. snd jumps to
7/8 with writer=high.

## Architecture (file → role)

### Concepts / data model

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round23_prose_facts/architectures/aen6_prose_v2.py          │ Mention, Fact, BindingEvent, EntityRegistry (DSU), MemoryStore,    │
│                                                                      │ IndexedCollection.                                                 │
│                                                                      │ EntityRegistry.register / get_canonical / merge / split.           │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Writer prompt + working-memory context blocks

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round23_prose_facts/architectures/aen6_prose_v2.py          │ WRITE_PROMPT — full writer prompt with co-reference rules,         │
│                                                                      │ named-default-new rule, pronoun antecedents, disambiguation,       │
│                                                                      │ within-fire `_anon_<label>` placeholder rule, and 6 examples       │
│                                                                      │ using neutral names (puppy/Mochi, Pat, Riley, Casey, Jordan,       │
│                                                                      │ Avery) and unrelated domains (mandolin, book club, climbing).      │
│                                                                      │                                                                    │
│                                                                      │ render_working_memory(prior_facts, obs_idx, registry, last_n=7)    │
│                                                                      │   → recency-driven recall: last N facts in chrono order with       │
│                                                                      │   each referenced entity tagged as named or                        │
│                                                                      │   "(unnamed: <descriptor>)". Pronoun antecedent source.            │
│                                                                      │                                                                    │
│                                                                      │ render_active_mentions(obs_idx, registry, target_surfaces, ...)    │
│                                                                      │   → association-driven recall: prior mentions grouped by current   │
│                                                                      │   DSU class with render-local Group A/B/C aliases, per-mention     │
│                                                                      │   discriminating excerpts, and a self-contained                    │
│                                                                      │   ⚠ COLLIDING SURFACES warning that lists each colliding surface   │
│                                                                      │   alongside each group's most-recent fact text as a discriminator. │
│                                                                      │                                                                    │
│                                                                      │ render_recent_facts — chronological recent facts (cap=8) for       │
│                                                                      │ secondary context.                                                 │
│                                                                      │                                                                    │
│                                                                      │ extract_window_surfaces — capitalized noun surfaces in target      │
│                                                                      │ window (used to populate target_surfaces for active recall).       │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Writer execution + DSU bindings

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round23_prose_facts/architectures/aen6_prose_v2.py          │ write_window — main writer entry point per fire.                   │
│                                                                      │   Phase 1: assign fresh mention_ids; register all mentions.        │
│                                                                      │   Phase 2 resolution dispatch:                                     │
│                                                                      │     "new" / ""        → no merge                                   │
│                                                                      │     "user"            → merge with canonical m_user_root           │
│                                                                      │     <mention_id>      → DSU merge with that prior mention's class  │
│                                                                      │     <entity_id> (back-compat)  → merge with that class             │
│                                                                      │     "_anon_<label>"   → fire-scoped placeholder; first occurrence  │
│                                                                      │       anchors a fresh class, later occurrences merge into it       │
│                                                                      │       (lifts intra_resolve to per-fire scope so the writer can     │
│                                                                      │       share anonymous entities ACROSS facts in one emission).      │
│                                                                      │                                                                    │
│                                                                      │   Reminder block injected at TARGET TURNS boundary inside the      │
│                                                                      │   conversation window: plain-language nudge to scan ACTIVE         │
│                                                                      │   MENTIONS before defaulting any TARGET mention to "new".          │
│                                                                      │                                                                    │
│                                                                      │   Calls llm(reasoning_effort="medium").                            │
│                                                                      │                                                                    │
│                                                                      │ ingest_turns — K=3 sliding-window driver (w_past=7, w_future=7).   │
│                                                                      │   Rebuilds collection every 4 fires.                               │
│                                                                      │   `inline_anchors=False` (default; inline anchoring is implemented │
│                                                                      │   but disabled — see annotate_context_turn).                       │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Reflector (recursive cognition)

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round24_recursive_cognition/architectures/aen7_recursive.py │ REFLECT_PROMPT — mental-state-only verb-domain (expects/plans/    │
│                                                                      │ hopes/fears/confirms/contradicts/believes/felt). 5 triggers:      │
│                                                                      │   1. CONDITIONAL ("if X then Y")                                   │
│                                                                      │   2. CONFIRMATION (prior plan met)                                 │
│                                                                      │   3. CONTRADICTION (incl. silent)                                  │
│                                                                      │   4. NAMED HOPE/FEAR                                               │
│                                                                      │   5. EMOTIONAL_STATE (fires on tonal cues)                         │
│                                                                      │                                                                    │
│                                                                      │ reflect_on_fact — per-fact recursive expansion with budget         │
│                                                                      │ (reflection_budget=2, reflection_max=3).                           │
│                                                                      │                                                                    │
│                                                                      │ ingest_turns (overrides aen6) — adds cog facts and reflection      │
│                                                                      │ pass per fire. Calls llm(reasoning_effort="medium") for the        │
│                                                                      │ reflector.                                                         │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Retrieval + dedup

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round23_prose_facts/architectures/aen6_prose_v2.py          │ retrieve — surface-match (uncapped, all surfaces of question →    │
│                                                                      │ all facts whose entities have those surfaces) ∪ kNN top-K over     │
│                                                                      │ embeddings. Returns (facts, resolution_map).                       │
│                                                                      │                                                                    │
│                                                                      │ extract_question_surfaces — capitalized non-stopword tokens from   │
│                                                                      │ the question; STOP includes pronouns, articles, modals.            │
│                                                                      │                                                                    │
│ research/round24_recursive_cognition/architectures/aen7_recursive.py │ retrieve (overrides aen6) — adds:                                  │
│                                                                      │   - Multi-hop expansion (expand_hops=1, expand_top_k by frequency) │
│                                                                      │   - Per-collection caps for kNN-only (obs_cap=50, cog_cap=15)      │
│                                                                      │   - Defaults to ["observations","cognition"]                       │
│                                                                      │   - Optional LLM dedup pass (llm_dedup=True at QA time)            │
│                                                                      │                                                                    │
│                                                                      │ DEDUP_PROMPT + _llm_dedup_filter — hybrid cosine (0.8 prefilter)   │
│                                                                      │ + batched LLM judge with transition-aware prompt (NEVER mark       │
│                                                                      │ Marcus→Alice manager change as dup).                               │
│                                                                      │ Calls llm(reasoning_effort="medium") for dedup.                    │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Reader (answer_question)

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round23_prose_facts/architectures/aen6_prose_v2.py          │ READ_PROMPT — instructs reader to use chronological order,         │
│                                                                      │ infer/count/distinguish/attribute, and DISAMBIGUATE colliding      │
│                                                                      │ surfaces via question modifiers using only the matched entity.     │
│                                                                      │                                                                    │
│                                                                      │ _build_eid_alias — maps canonical entity_ids in the                │
│                                                                      │ resolution_map to render-local Entity A/B/C aliases (entity_ids    │
│                                                                      │ never leak to the reader either).                                  │
│                                                                      │                                                                    │
│                                                                      │ format_facts_for_read — chronological fact list with each          │
│                                                                      │ mention annotated as "[surface → Entity X]".                       │
│                                                                      │                                                                    │
│                                                                      │ format_resolution_map — per-entity sidebar with surfaces +         │
│                                                                      │ a few discriminating fact excerpts; ⚠ COLLIDING SURFACES           │
│                                                                      │ warning at top.                                                    │
│                                                                      │                                                                    │
│ research/round24_recursive_cognition/architectures/aen7_recursive.py │ answer_question — reader entry point. Computes eid_alias once      │
│                                                                      │ and threads it into both render functions for consistent labels.   │
│                                                                      │ Calls llm(reasoning_effort="medium") for the reader.               │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Shared infra

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round7/experiments/_common.py                               │ llm(prompt, cache, budget, reasoning_effort="low") — gpt-5-mini   │
│                                                                      │ wrapper. R25 callers explicitly pass "medium". Cache key includes  │
│                                                                      │ (model, reasoning_effort, prompt).                                 │
│                                                                      │                                                                    │
│                                                                      │ embed_batch — text-embedding-3-small wrapper, batched, cached.     │
│                                                                      │                                                                    │
│                                                                      │ Cache (json), Budget (call counters with stop thresholds).         │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Experiment harness + scenarios

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round24_recursive_cognition/experiments/run_one_noinline.py │ R25 single-scenario runner. Sets `inline_anchors=False`, writes    │
│                                                                      │ to `r24_noinline_<scenario>.json` cache + result. Used for the     │
│                                                                      │ canonical R25 architecture runs.                                   │
│                                                                      │                                                                    │
│ research/round24_recursive_cognition/experiments/run_one.py          │ Original R24 runner. Default `inline_anchors=False`; result name   │
│                                                                      │ `r24_one_<scenario>.json`. Used for baseline numbers (pre-R25      │
│                                                                      │ prompt, low reasoning).                                            │
│                                                                      │                                                                    │
│ research/round24_recursive_cognition/experiments/run_one_anchored.py │ Variant runner with `inline_anchors=True` (kept for back-compat;   │
│                                                                      │ R25 found inline anchors brittle and disabled them by default).    │
│                                                                      │                                                                    │
│ research/round14_chain_density/experiments/run.py                    │ grade_deterministic + judge_with_llm — shared QA grading.          │
│                                                                      │                                                                    │
│ research/round24_recursive_cognition/scenarios/                      │ 30+ scenario modules. Key R25 slice:                               │
│                                                                      │   multi_batch_coref_wrap.py (mbc — cross-fire descriptor→name)     │
│                                                                      │   same_name_disambig.py    (snd — three Alices)                    │
│                                                                      │   dense_chains_wrap.py     (dense — manager/city succession)       │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

### Results

```
┌──────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
│                                File                                  │                                Role                                │
├──────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
│ research/round24_recursive_cognition/results/r24_noinline_*.json     │ R25 architecture results per scenario.                             │
│ research/round24_recursive_cognition/results/r24_one_*.json          │ Baseline + earlier R24 results per scenario.                       │
│ research/round24_recursive_cognition/results/STATUS.md               │ R24 status doc; precursor to this design.                          │
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

## Wins vs the SotA event_memory.py and over baseline

vs `packages/server/src/memmachine_server/episodic_memory/event_memory/event_memory.py`
(no entity model, no co-reference, no supersession, only flat semantic kNN):
- Adds an entity model with mention_id/entity DSU.
- Cross-fire descriptor→name binding (mbc 8/8).
- Same-surface disambiguation via collision warning + per-group discriminating
  excerpts (snd 6/8 at medium, 7/8 at high).
- Role-vs-identity distinction for succession events (dense 20/23).
- Within-fire shared anonymous entities (`_anon_<label>` placeholder).
- Within-fact co-reference (multiple mentions sharing resolves_to).

vs the entity_id-baseline writer (R24 default):
- mention_id semantics keeps writer claims per-occurrence and revisable.
- Working memory + active mentions = explicit short-term context blocks.
- Entity_ids never leak (writer or reader) — all class labels are
  render-local Group A/B/C / Entity A/B/C aliases.

## Open / future directions

1. **Spontaneous association ("deja vu")** — concept-expansion retrieval +
   background association mining. High value, hard.
2. **World-knowledge bridging at retrieval/cognition time** — enables both
   implicit relational inference (Park Slope ⊂ Brooklyn) and implicit
   contradiction detection (Tokyo trip ⇒ international travel).
3. **Density-aware writes with temporal-middle eviction** within similarity
   radius (preserves history with "I used to think X, now Y" intact;
   bounds intermediate hedges).
4. **Cross-session continuity** — same family as deja vu / pattern recognition.
5. **Diagnostic finding (writer-high spot check)**: snd is reasoning-bound
   at writer=medium; better/future models should solve modifier
   disambiguation at medium-equivalent reasoning. Architecture is sound.

## Reasoning levels

All four LLM call sites pass `reasoning_effort="medium"` explicitly
(API default; our `_common.llm()` function default is `"low"` and is
overridden in callers). High reasoning is reserved for diagnostic spot
checks of single scenarios, not production runs.
