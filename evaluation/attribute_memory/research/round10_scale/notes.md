# Round 10: Scale stress test — AEN-1 + structural indexes

## Goal

Stress-test AEN-1 at 200/500/1000 entries (and beyond) and argue whether the
architecture extrapolates to 100K entries (~100M-token lifetime).

## Approach

- **Deterministic scenario generation** — no LLM writer calls. Compose
  scenarios from entity + event templates. Pre-build the post-write log
  (uuid/ts/text/mentions/refs) the way an ideal AEN-1 writer would.
- This isolates the **retrieval + reader** path. Writer reliability was
  benchmarked in round 9; we test architecture under scale.
- Cost: only embedding + reader LLM calls.

## Architectures compared

- `aen1_plain` — round-9-style retrieval: embedding top-K + mention-filter
  (limited to 20) + ref-chain neighbors (single-step both directions).
- `aen1_indexed` — adds three structural indexes built deterministically at
  write time:
  - `mention_index: @entity -> [uuid]`
  - `superseded_by: uuid -> uuid` (the entry that supersedes/invalidates)
  - `supersede_head: (@entity, predicate) -> uuid` of the current head
  - `embed_by_uuid` for embedding lookup
  Retrieval branches on a regex-detected question-kind:
    - current/supersede/history/entity → look up `supersede_head` for any
      entity in the question, walk the FULL chain back to root.
    - all kinds → mention-filter (top-K by embedding within the filter)
    - history/default/entity → embedding top-K across whole log
    - all kinds → walk single-step refs of selected entries
    - cap final selection at 60 entries.

## Scenarios designed (12 deterministic generators × 200/500/1000 scales)

Round 1 (smoke):
- `dense` — User has 5-20 employer transitions plus 80%+ filler.
- `distractors` — 80 named entities, most with 1-2 mentions; 3 focal entities.
- `interleaved` — 7 tracked predicates with 2-4-step chains, scattered.
- `deep_chain` — single predicate (User.boss) with 10-15-step chain.

Round 2 (adversarial):
- `paraphrase` — each transition gets multiple paraphrase clarifications;
  semantic-distractor filler ("read about Stripe IPO").
- `cross_entity` — User AND Jamie AND Priya all have employer chains
  using identical templates.
- `rare_fact` — 5 facts, each stated ONCE, buried in 195/495/995 distractors.

Round 3 (writer noise):
- `decay` — gold fact is at position 5 of 1000, surrounded by entries that
  use the same template about other entities.
- `misleading` — 20 decoy entities all have employer chains identical-template
  to User's chain.
- `untagged` — Priya has 3 transitions but only the first is `@`-tagged;
  later ones use pronoun "she", no @-mentions.

Round 4 (extreme):
- `extreme_c15` / `extreme_c18` — 15- and 18-step employer chain at
  1000 and 2000 entries, with paraphrase clarifications interleaved.

## Bugs found and fixed

1. **`extract_question_entities` keeps `'s` in possessives** — "Jamie's"
   became `@Jamie's`, missing the `@Jamie` index. Fix: strip trailing `'s`.
   Restored 5 of 9 indexed failures on first sweep.
2. **UUID collision in `gen_misleading`** — fall-through after the User-slot
   block produced two entries with the same ts/uuid. Cosmetic generator bug,
   not a real architecture issue. Fix: `continue` after each branch.
3. **Indexed history queries didn't walk full chain** — the supersede-head
   lookup was gated to current/supersede only; for history queries the
   single-step neighbor walk missed deep chain heads. Fix: include
   supersede_head match for history/entity kinds and walk back transitively
   up to 50 steps.

## Final results

```
scale       plain          indexed
 200      45/45 (100%)    45/45 (100%)
 500      43/45 ( 96%)    45/45 (100%)
1000      53/55 ( 96%)    55/55 (100%)
2000       5/5  (100%)     5/5  (100%)
```

Indexed beats plain ONLY on `untagged` scenario (writer noise) at scale
≥500: indexed 2/2, plain 0/2. The supersede_head lookup correctly finds
the current Cursor entry even when Priya's chain is half-pronoun.

All other (12 scenarios × 3 scales) cells are tied at 100% for both archs.

## Total cost

LLM ≈ 230 (well under 500 cap), embed ≈ 90 (well under 300 cap).
Total ≈ $0.70.
