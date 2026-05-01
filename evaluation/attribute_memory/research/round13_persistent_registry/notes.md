# Round 13 — Persistent entity registry with LRU active cache

Goal: fix round 12's bug where the LRU dict WAS the registry. In round 13
the registry is durable storage; the LRU is just the in-context cache that
the coref LLM sees per turn.

## Architecture (`aen3_persistent`)

- `PersistentRegistry` — durable, indexed three ways:
    - `by_id`: ent_id -> Entity
    - `aliases_index`: lower(alias) -> list[ent_id]
    - `desc_embeds`: ent_id -> np.ndarray (description embedding)
- `lru_order` (~20) selects which entries the coref LLM sees per turn.
- Coref LLM emits one of 4 actions per mention:
    - `resolve` (cache hit, no LLM call)
    - `lookup` (search the full registry by alias OR description-embedding)
    - `create` (new entity)
    - `skip` (not an entity)
- `lookup` flow:
    1. Try exact alias-match in full registry. If 1 hit, take it. If
       multiple, fire `disambiguate_alias` LLM call.
    2. Otherwise (descriptor case), embed the rich `lookup_descriptor`
       phrase, retrieve top-K=5 candidates by cosine over description
       embeddings, and fire `descriptor_pick` LLM call to choose a match
       OR say create_new.
    3. If still no match, create a fresh entity.
- Embedding is candidate-filter only. The LLM picks the final answer.

## Costs

Every turn costs 1 coref LLM call. `lookup` actions on descriptors cost
+1 LLM (descriptor-pick) and +N embed calls (one per query + one per
new entity description). Most turns don't fire embed search.

## Key prompt differences from aen2

- `lookup` action replaces `resolve_by_alias`, with a richer
  `lookup_descriptor` field for descriptor mentions.
- Coref prompt explicitly tells the LLM that descriptor mentions for
  entities not in cache go via `lookup`, with paraphrased semantic
  description text (which is what we embed).
- Description maintenance: capped at 400 chars; description embedding
  recomputed when description changes.

## Running log
