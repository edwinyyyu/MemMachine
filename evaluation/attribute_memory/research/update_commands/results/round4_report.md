# Round 4 — fine-grained update operations

Question: beyond the 4-op baseline (`revise[n]` whole-value replace,
`remove[n]`, `add`, `noop`), do ANY finer-grained verbs improve update
quality on gpt-5-mini?

Model: `gpt-5-mini` at `reasoning_effort=low`. Sample: 10 targeted
scenarios. Grading: a **deterministic applier + strict diff** against a
hand-authored `rows_after` for every scenario (not an LLM judge — Round
1's judge was too lenient on paraphrase).

Budget: 50 author calls, $0.12 against $1.00 cap. Cached in
`cache/round4_cache.json`.

## Candidates

| Key | Adds over baseline |
|-----|-------------------|
| `C1_baseline` | — (revise/remove/add/noop only) |
| `C2_member_ops` | `add_member[n] member`, `remove_member[n] member` (set rows only) |
| `C3_string_patch` | `patch[n] old_substring new_substring` |
| `C4_append_to` | `append_to[n] new_text` |
| `C5_conf_verbs` | `strengthen[n]`, `weaken[n]` (confidence-only) |

All share the Round-2/3 "copy editor + noop-discipline" framing. Only
the verb set and per-verb instructions differ. C2 additionally annotates
each fact-sheet line with `[cardinality=single|set]`.

## Scenarios (10)

| ID | Intent | Forces |
|----|--------|--------|
| R01 | strengthen_confidence | value body unchanged, tag hedged→confirmed |
| R02 | append_qualifier | single-value, add parenthetical qualifier |
| R03 | add_member | 2-member set → 3-member set |
| R04 | remove_member | 3-member set → 2-member set (a pet died) |
| R05 | correction | whole-value replace (Seattle→Portland) |
| R06 | multi_op | qualifier add + set member add in one turn |
| R07 | replace_trap | sounds incremental but whole replace is right |
| R08 | add_member | 10-member set; full-rewrite is high-token-cost and fragile |
| R09 | weaken_confidence | value body unchanged, tag confirmed→hedged |
| R10 | patch | one-word swap in a long value ("Duolingo"→"Anki") |

## Leaderboard (state-correct by deterministic applier)

| Candidate | Correct | Accuracy | Avg out chars | Op totals |
|-----------|---------|----------|---------------|-----------|
| `C2_member_ops` | **5/10** | 50% | **104** | `{"revise": 7, "add_member": 3, "remove_member": 1}` |
| `C5_conf_verbs` | 4/10 | 40% | 138 | `{"revise": 10, "add": 1, "weaken": 1}` |
| `C1_baseline` | 3/10 | 30% | 143 | `{"revise": 11, "add": 1}` |
| `C3_string_patch` | 3/10 | 30% | 156 | `{"revise": 11, "add": 1}` |
| `C4_append_to` | 2/10 | 20% | 149 | `{"revise": 11}` |

Zero parse failures anywhere. Zero `patch` paraphrase misses for C3 —
because the author never emitted `patch`, even on the perfect-fit
scenario R10.

## Correctness matrix

| Scenario | C1 base | C2 mem | C3 patch | C4 append | C5 conf |
|----------|:-------:|:------:|:--------:|:---------:|:-------:|
| R01 conf strengthen only | N | N | N | N | N |
| R02 qualifier add | N | N | N | N | N |
| R03 add set member | N | N | N | N | N |
| R04 remove set member | **Y** | **Y** | N | N | N |
| R05 whole replace | N | **Y** | **Y** | **Y** | **Y** |
| R06 multi small edits | N | N | N | N | N |
| R07 replace trap | **Y** | **Y** | **Y** | **Y** | **Y** |
| R08 long set + 1 add | N | **Y** | N | N | N |
| R09 weaken only | N | N | N | N | **Y** |
| R10 one-word swap | **Y** | **Y** | **Y** | N | **Y** |

## Op-distribution & bias

- **All five schemas are heavily revise-biased.** Across 50 turns:
  `revise` = 50 calls. `add_member` = 3, `remove_member` = 1, `weaken`
  = 1, `add` = 4. No `strengthen`, no `patch`, no `append_to`.
- **C3 `patch` was never used.** Even on R10 (literal substring swap
  "Duolingo"→"Anki" — the exact case the verb was designed for), the
  author still chose `revise`. Adding the verb without moving usage is
  pure schema weight with no upside. A stronger "prefer patch when
  possible" instruction might change this, but that itself becomes a
  schema-bias concern.
- **C4 `append_to` was never used.** Author preferred whole-value
  revise even for R02 (pure qualifier append) and R06's qualifier add.
  Same problem as C3: adding the verb didn't redirect behavior.
- **C5 `strengthen` was never used.** `weaken` fired once, on R09 — and
  that was the single scenario where only C5 got state-correct, because
  every revise-based schema paraphrased "runs 5km every morning" into
  "tries to run" / "aspires to run" / "intends to run", corrupting the
  stored body. `weaken` preserved the body verbatim. Real win on R09,
  zero activation elsewhere.
- **C2 member ops were actually used.** 3 `add_member` and 1
  `remove_member` fired, all on set-cardinality rows as instructed.
  No misuse on single-valued rows. The cardinality tag in the sheet
  rendering successfully routed behavior.

## Failure mode analysis

### Paraphrase drift on `revise` (dominant failure; affects R01, R02, R03, R06, R09)

Across these scenarios, authors paraphrased the value body during
whole-row revise. Observed corruptions:

- R01: `"vegan"` → `"vegan; 3 years"` / `"vegan, duration: 3 years"` /
  `"vegan (3 years)"` — author folded the turn's "3 years" detail into
  the value.
- R02: `"peanut allergy"` → `"anaphylactic"` (dropped the noun!) or
  `"anaphylactic peanut allergy"` (reordered) or
  `"severe peanut allergy"` (reordered).
- R09: `"runs 5km every morning"` → `"intends to run ..."` / `"tries
  to run ..."` / `"aspires to run ..."`.

This is the **core argument for finer-grained ops**: confidence-only
and append-only operations would preserve the body verbatim. But our
data shows the LLM *does not pick those verbs* when given the option.
Only `weaken` fired (once, R09). `strengthen`, `patch`, `append_to`
never fired — so the body-preservation upside is theoretical.

### Set corruption during full-rewrite (R08)

When the list contains a value with an internal comma
(`"Gödel, Escher, Bach"`), a revise that re-emits the full
comma-separated list breaks the member into three fragments at backend
parse time. All four revise-based schemas (C1, C3, C4, C5) hit this on
R08. Only **C2 member_ops** sidestepped it by emitting
`add_member[n]` — which bypasses list re-emission entirely. **R08 is
structural evidence that member ops beat revise for long sets with
complex members.**

### Spurious adjacent rows (R04, R05)

On R04, C3, C4, C5 either added a `deceased` row or embedded
`"Milo (cat, deceased)"` into the list rather than removing it. On R05
baseline split `"moved last month"` out into a second `move_date`
row — the turn carries two arguably-separate facts, and the author
couldn't resist capturing both. This mirrors the
"spurious-adjacent-fact" pattern flagged in Round 2/3 and is unrelated
to verb set.

### C3 paraphrase-miss count

**Zero**, because the author never emitted `patch`. The verb's
designed failure mode (LLM paraphrases `old_substring` so it doesn't
match the value) did not occur — but neither did any correct patch
emission. C3 failed the same scenarios as C1 for the same reasons.

## Token cost

| Candidate | Avg out chars / turn | Notes |
|-----------|---------------------:|-------|
| C2_member_ops | **104** | Drops hard on R03/R08 (set member ops emit short commands) |
| C5_conf_verbs | 138 | Drops on R09 (weaken op = 27 chars) |
| C1_baseline | 143 | — |
| C4_append_to | 149 | Verb adds length instructions but isn't used |
| C3_string_patch | 156 | Same; adds patch weight w/o usage |

On R08 alone: C2 emits 58 chars (`add_member`). Every other schema
emits 264-289 chars (full-list re-emission). For a production system
handling long sets, this is a meaningful cost differential.

## Limitations

- **N=10 scenarios, no replicates.** Differences of 1-2 scenarios are
  within noise.
- **Strict diff is harsh on semantically-equivalent paraphrases.** An
  LLM-judge would have scored several R01/R02/R06 turns as "correct in
  spirit". But for a durable memory system, the paraphrase corruption
  IS a real cost — "vegan; 3 years" downstream muddies retrieval and
  stacks up over many turns. Strict is the right grade for this
  question.
- **Cardinality annotations (C2) are a confound.** C2 got both a new
  verb set AND additional sheet metadata. A follow-up could isolate
  whether the win comes from the verbs or the cardinality hint alone.

## Recommendation

**Keep the 4-op baseline. Do not adopt C3 (`patch`), C4 (`append_to`),
or C5 (`strengthen`/`weaken`) as user-facing verbs. Consider adopting
C2 (`add_member` / `remove_member`) conditionally.**

Bullet rationale:

1. **`patch` (C3) and `append_to` (C4) are dead weight.** Zero
   activations across 20 exposures. Any intended upside is gated
   behind a behavioral change the model does not make. Adding them
   costs prompt tokens, parse complexity, and reviewer surface area
   without a single correctness win.

2. **`strengthen` / `weaken` (C5) are underused but have one real
   win.** `weaken` fired on R09 and was the only schema that preserved
   the value body. But the revise-paraphrase bug that R09 exposes also
   lands on R01 (where `strengthen` should have fired) — and `strengthen`
   did not fire. Until we can force the verb via prompt without
   reintroducing schema bias, the net value is ~0-1 scenarios out of
   10. A cheaper remedy is to append to the revise instructions: "When
   confidence changes and the value body is unchanged, copy the body
   verbatim from the fact sheet before appending the new tag."

3. **`add_member` / `remove_member` (C2) earn adoption IF we handle
   long/complex sets.** On R08 (10-member set, one add) C2 is the only
   schema that stays correct — and it does so with 58 output chars
   vs. 264-289 for revise. On R04 (remove member with complex value),
   C2's `remove_member` worked cleanly where C3/C4/C5 drifted into
   spurious additive rows. Adoption conditions:
   - Sheet must annotate set vs. single cardinality (the author needs
     to know which verb is legal).
   - Backend applier must accept fuzzy member match (exact first, then
     single-substring match; seen once in our testing, worked).
   - Keep revise available for set rewrites where wholesale reordering
     or renaming is intended — do NOT remove it.

4. **None of these verbs are the bottleneck.** The real failure across
   R01, R02, R03, R06 is that `revise` authors paraphrase the value
   body instead of copying it. That is a framing problem and should be
   fixed at the prompt level:
   - "When revising a row, copy the existing value verbatim and only
     change the parts called out by the new statement."
   - "Never silently reorder, abbreviate, or rephrase the value body."

   Fixing the framing would likely close more scenarios (5-6 of these
   10) than any new verb we tested.

### Final recommendation

Baseline + optional member ops (C1 + C2's two verbs with cardinality
tags), plus an explicit "preserve value body verbatim" rule added to
the framing. Do not add `patch`, `append_to`, `strengthen`, or
`weaken`.
