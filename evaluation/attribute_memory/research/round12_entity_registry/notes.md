# Round 12 — Entity registry architecture

Goal: decouple entity identity from name. Test whether an LRU entity cache
+ LLM-driven coref pass produces correct disambiguation for:

- S1 same-name different-entity (boss-Alice vs HS-Alice)
- S2 different-name same-entity (Marcus -> Mark alias add)
- S3 LRU stress (26 entities, late descriptor mentions)
- S4 pronoun chains (he/she/they within window)
- S5 silent context (default to LRU-most-recent)

Hard cap: 500 LLM + 100 embed, $5. Target $2-3.

## Architectural decisions

- Single-pass LLM coref. No embeddings for final decision (per instructions).
- Alias disambiguation only fires when registry has multiple candidates with
  the same alias. Otherwise resolve directly.
- Pre-seed `ent_user` for the speaker.
- LRU=20 starting point.
- Rewritten turn text uses `[@ent_NNNNN / surface]` so the writer LLM can
  pick up the canonical id without re-corefing.

## Running log
