# Remaining failure diagnosis

Categorizing the ~16 failed questions across 18 scenarios into:
- **A** = architectural gap (real memory-system issue, fixable with code)
- **B** = reading-time variance (LLM picks wrong interpretation)
- **C** = harness brittleness (expected_contains requires specific words)

## Failures by category

### A — Architectural gaps

| Scenario | Q | What fails | Root cause |
|---|---|---|---|
| coref | Q06/Q08 | "Who is neighbor?" → wrong name; "Who is senior?" → not named | Writer disambiguation when multiple anonymous descriptors active simultaneously. Pronoun cues (he/she) don't always disambiguate correctly. The K=3 windowed writer sees several anon-descriptor entities; when name reveal arrives, model picks wrong binding. |
| ic | Q06 | "Who is User's boss's mentor's son?" → unknown | **Multi-hop retrieval not implemented.** Retrieval is single-hop entity-keyed (find Olivia → her facts). It doesn't chain: question entity (User) → boss (Marcus) → Marcus's mentor (Olivia) → Olivia's son (Theo). Surface-match for "Theo" isn't triggered by the question text. |
| pattern | Q02/Q07 | Race-condition / cache-invalidation count | **Pattern-instance counting** — each bug is its own fact, no shared pattern entity for aggregation. Cosine+LLM dedup helped but doesn't fully solve (the LLM has to look at ~20 pairs and decide). A pattern-entity layer at write time would be cleaner. |
| snd | Q07 | "Where does Alice the neighbor work?" — sometimes can't retrieve | Cross-fire entity binding: descriptor "new neighbor (a nurse at Mt Sinai)" emitted at fire 1; name reveal "Alice introduced herself" at fire 2. Writer in fire 2 doesn't bind Alice to the prior "neighbor" entity reliably. The two never DSU-merge → query for "Alice's workplace" can't reach the "Mt Sinai" fact. |

### B — Reading-time variance

| Scenario | Q | What fails | Root cause |
|---|---|---|---|
| dense | Q22 | "Did User ever live in Chicago?" → "No, planned but didn't" | Strict reading. User said "moving to Chicago" then later "moving to Denver instead". Did they live there? Strict: no, never moved. Lenient: yes, briefly intended. Architecture has both facts; reader picks strict. |
| dense | Q23 | "Did User ever work at Anthropic?" → similar | Same pattern — planned but not realized. |
| dense | Q06 | "What hobby is User into?" → "Yoga" but expected something else | Multiple hobby transitions; "current" hobby ambiguous. The latest fact may say something else; reader's "current" interpretation differs from question's intent. |
| dorm | Q05 | "Where does User work now?" → "Stripe" instead of "Stripe (will start Anthropic)" | Reader gives concise answer; question wanted both current + future. |
| em_recall | Q03 | "Was User anxious?" → "no" | "Internally screaming" was emitted by writer as fact, but reader classified as frustration not anxiety. Emotion category ambiguous. |
| evolving | Q01/Q08 | "What was originally called gamma?" → "auth" instead of "production auth" | Multiple valid names (gamma → v2 → production auth). Reader picks one; question wanted a specific one. |

### C — Harness brittleness

| Scenario | Q | What fails | Root cause |
|---|---|---|---|
| pattern | Q06/Q08 | Answer correct but missing required keyword ("race"/"concurrent") | expected_contains lists multiple required words; semantic match without literal word fails LLM judge. |
| em_recall | Q04/Q05/Q06 | "memory leak"/"Sara"/"Daisy" answered correctly; missing "bug"/"Tokyo"/"dog" | Synonym not in expected_contains. |
| ToM | Q03 | "thrilled" required, answer says "super happy" | Synonym mismatch. |

## Counts
- **A** failures: 5 question-pairs (architectural fixes possible)
- **B** failures: 6 questions (reading variance, partially addressable)
- **C** failures: ~5+ (harness, not architecture)

## Fix priority

Highest leverage:
1. **Multi-hop retrieval** (Category A): unlocks indirect_chain Q06 + likely helps other scenarios. Implement as a read-time expansion: Q's entity → find facts → those facts' entities → expand retrieval. Bounded depth=2.
2. **Pattern-instance entity layer** (Category A): writer or reflector creates a pattern entity (`PATTERN:race_condition`) that all matching events link to. Counting = entity class size.
3. **Cross-fire entity binding** (Category A): writer schema extension — `resolves_to: "new:n01"` for intra-fire entity introduction; persist these local IDs in a per-window map for the next fire to reference.

Lower leverage (B/C):
4. Reader prompt tuning for strict-vs-lenient interpretation (B)
5. Harness expected_contains relaxation (C — out of architecture scope)
