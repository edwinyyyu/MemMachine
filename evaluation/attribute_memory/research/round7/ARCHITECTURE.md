# Round 7 Architecture — Integrated Entity-Layer Design

Addresses four problems identified at the end of round 6:

| # | Problem | Mechanism |
|---|---------|-----------|
| P1 | Multi-label routing (too eager) | **Decision-tree gate**: multi-label only when fact introduces a new entity OR is a bi-entity relationship-making event |
| P2 | Cross-turn anonymous→named coref | **Context buffer** of unresolved anonymous descriptors; triggered coref-resolve call only when a named entity is introduced and buffer is non-empty |
| P3 | Role-update fanout | **Role slots** as first-class memory objects (e.g. `User/Employment/boss`), separated from entity profiles |
| P4 | Salience (common-noun noise) | **Lazy admission** scored on signals (named, repeated, state-change, identifying-detail); threshold = 2 |

## Data model (schemas.py)

### Per-entity topic logs (from round 5/6)
Unchanged. Each entity gets an append-only log keyed `<Entity>/<Category>`.

### New: role slots
```
RoleSlot(slot_id="User/Employment/boss", history=[RoleSlotEntry...])
RoleSlotEntry(slot_id, ts, filler="@Marcus" | None, invalidated: bool,
              source_turn, source_fact)
```
A slot is filled by a *pointer* to an entity, not embedded data. When boss changes, one append lands on the slot's mini-log. The entities Marcus/Alice themselves are untouched — their profile logs continue to carry qualitative facts (Scorpio, coffee preferences). **Fanout of a role change is 1 append**, not 3.

### New: coref buffer
Ring buffer of unresolved anonymous mentions, bounded by turn-count (30) and count (20). Only active mentions are kept.
```
AnonymousMention(turn_idx, descriptor, topic, fact_text)
CoreferenceMerge(canonical_entity, anonymous_topic, anonymous_descriptor, matched_mention_turn_idx, rationale)
```

### New: salience candidates
```
SalienceCandidate(descriptor, first_seen_turn, mention_count,
                  has_name, has_state_change, has_identifying_detail)
salience_score(c) = 2*has_name + (mention_count>=2) + has_state_change + has_identifying_detail
admit iff score >= 2
```

## Prompt design

### P1 gate prompt
Single hard rule up front: "emit multiple topics ONLY in these two cases: (A) new entity introduced, (B) relationship-making event with 2+ parties undergoing state change." Four worked examples: positive (engagement, pet adoption), negative ("User is nurse and diabetic" → single).

### P2 coref prompt
Fired only when a named entity is introduced AND buffer is non-empty. Supplies the introducing text verbatim + full buffer. Asks for an **unambiguous** match (appositive, explicit resolution). Rejects distractor names (someone mentioned but unrelated).

### P3 role-slot prompt
Per-fact; emits `slot_updates` (role changes) and/or `entity_facts` (qualitative). Key rule: "Marcus is my boss" → ONE slot_update, NO entity_fact (don't duplicate). "Marcus likes coffee black" → entity_fact, NO slot_update.

### P4 salience prompt
Per-turn; flags noun phrases as candidates with signals. Two signal tightenings relative to a first draft:
- `has_identifying_detail`: possessive "my" alone does NOT count
- `has_state_change`: one-off passive use ("drank coffee from mug", "cat knocked over glass") does NOT count. Only persistent changes (moved, broke, got, lost).

## Integration — fused single-call design

In the shipping integration, per turn:
1. **One fused LLM call** (`FUSED_PROMPT`) that emits: facts with routing, slot_updates, anonymous_descriptors, named_entity_introduced, salience_candidates. This replaces four separate calls with one.
2. **Conditional coref-resolve call**: only when a named entity is introduced AND the anon buffer is non-empty.

So the per-turn cost is **1 LLM call baseline + 1 extra when a new name is introduced with pending anonymous descriptors**. Embedding calls are amortized against existing extraction/routing pipelines.

## Composability

- P1 introduces new entities → added to the `known_entities` set that P2 uses to detect "is this fact introducing a NEW named entity?"
- P2 merges anon → named → this is the input that P3 role slots need (the fact "my boss is X" becomes a slot_update once X is named; the merge trigger can reclassify previously-anonymous facts as referring to Marcus)
- P4 gates what's added to `known_entities`. Common nouns (a mug, a tissue) never enter the entity set, so they don't pollute P1's new-entity test.

## Test strategy

Per-problem isolation tests (`experiments/p{1,2,3,4}_*.py`) then an integration test (`experiments/integration.py`). Deterministic graders throughout; no LLM-as-judge needed.

## What this replaces / leaves unchanged

- Entity-first routing (R7) is unchanged as the *baseline* router — multi-label gate wraps it.
- Append-only logs per entity are unchanged.
- 5-turn batching + C4 query-gated retrieval — unchanged.
- Consolidation — unchanged (role slot logs are additional append-only logs that consolidate the same way).
