# Update-Command Schema Research

Research question: given an `attribute_memory` system that stores
`(topic, category, attribute, value)` rows and historically emits only
`add` / `delete` commands, design and test alternative update-command
schemas so that a `gpt-5-mini`-caliber author can emit the right command
type without bias, express all common modification intents, remain
semantically obvious, and be robust to paraphrase.

- Model: `gpt-5-mini`, `reasoning_effort=low`.
- Storage representation fed to the LLM author: markdown bullet lists
  grouped by `topic.category` headings (Candidates 1-4) or a numbered
  fact sheet with each row labeled `[n]` (Candidate 5, winner).
- LLM budget: 146 / 150 calls total, ~$0.36 against $1.50 cap.
- All LLM calls cached under `cache/` keyed on `(model, prompt)`.

## Files

| Path | Purpose |
|------|---------|
| `scenarios.json` | 12-scenario labeled sample set |
| `common.py` | Shared cache / budget / env bootstrap |
| `round1.py` | 5 schema x framing candidates x 12 scenarios |
| `round2.py` | 2 finalists with tightening x 6 stress scenarios |
| `round3.py` | Winner on 4 domain-distinct scenarios |
| `results/round{1,2,3}_{results.json,report.md}` | Per-round detail |
| `cache/round{1,2,3}_cache.json` | LLM-response caches |

## Sample set (12 scenarios)

| ID | Intent | What it tests |
|----|--------|---------------|
| S01 | new_fact | introduce fact, no prior collision |
| S02 | correction | correct a value AND introduce a related new fact |
| S03 | add_member | 1-pet -> 2-pets set growth |
| S04 | remove_member | 3 allergies -> remove peanuts |
| S05 | strengthen_confidence | "might hike" -> "did hike" |
| S06 | weaken_confidence | "vegan" -> "trying vegan for a month" |
| S07 | noop | weather/seasonal venting |
| S08 | noop | phatic "haha yeah, thanks" |
| S09 | multi_op | pet died + laid off in one turn |
| S10 | ambiguous | "that allergy is gone" with 3 stored |
| S11 | conflict | "married" -> "never married, engaged" |
| S12 | correction | paraphrased role change ("promoted to staff") |

Round 3 generalization set (domain-distinct): D01 travel update, D02
finance accounts, D03 work-venting noop, D04 multi-user family.

## Candidates

| Key | Schema | Framing |
|-----|--------|---------|
| `baseline_addel` | `add` / `delete` on `(cat, attr, val)` | direct memory system |
| `cud_triple` | `add` / `update` / `delete` + `noop` | dossier editor |
| `member_ops` | `add` / `update` / `delete` / `add_member` / `remove_member` + `noop` | librarian card catalog |
| `intent_ops` | 8 intent verbs (`introduce_fact`, `correct_fact`, `add_to_set`, `remove_from_set`, `retire_fact`, `strengthen_confidence`, `weaken_confidence`, `nothing_to_change`) | biographer draft |
| `indexed_patch` | `keep` / `revise` / `remove` / `add` / `noop` with numbered prior facts | copy editor markup |

## Round 1 results

| Candidate | Correct | Op totals |
|-----------|---------|-----------|
| `baseline_addel` | **8/12 (67%)** | `{"add": 15, "delete": 8}` |
| `member_ops` | **8/12 (67%)** | `{"add": 4, "update": 7, "remove_member": 2, "noop": 2}` |
| `indexed_patch` | **8/12 (67%)** | `{"add": 4, "revise": 9, "noop": 1}` |
| `cud_triple` | 7/12 (58%) | `{"add": 4, "update": 9, "noop": 2}` |
| `intent_ops` | 7/12 (58%) | `{"introduce_fact": 6, "correct_fact": 6, "remove_from_set": 2, "nothing_to_change": 2, "retire_fact": 1}` |

### Bias patterns

- **`baseline_addel`** is structurally `add`-biased (15 add vs 8 delete).
  Cannot express noop, update, or set membership. Ties the winners only
  because the LLM-judge accepts paraphrase matches — in practice
  `delete(old_paraphrase) + add(new)` usually leaves the old row
  undeleted (exact string match fails) and leaks a duplicate.
- **`cud_triple`** over-uses `update` against attributes that don't yet
  exist (S03: `update` on a set that should have been a member-add).
- **`member_ops`** has balanced verb spread but still prefers `update`
  over `add_member` for ambiguous cases.
- **`intent_ops`** with 8 verbs doesn't reduce bias; the author often
  picks `correct_fact` where `*_confidence` was called for. More verbs
  create new biases rather than eliminating old ones.
- **`indexed_patch`** shows the cleanest distribution relative to
  schema size. Index references structurally resolve paraphrase.

### Universal failure across Round 1: noop discipline

5/5 schemas treated S07 ("weather is gloomy, I hate November") as a new
user preference. This is a framing problem, not a schema problem.

## Round 2 (finalist stress test)

Finalists `member_ops` and `indexed_patch`, tightened with:
1. Explicit "DO NOT write" block (weather/mood/filler/repetition).
2. Fixed-vocabulary confidence tags: `(confirmed)`, `(hedged)`, `(intended)`.

Scenarios: S02, S05, S06, S07, S09, S10.

| Candidate | Correct | Op totals |
|-----------|---------|-----------|
| `indexed_patch_v2` | **5/6 (83%)** | `{"revise": 5, "add": 2, "noop": 2}` |
| `member_ops_v2` | 4/6 (67%) | `{"update": 4, "add": 1, "noop": 2, "remove_member": 1}` |

Both fixed S07 (weather noop) and S05/S06 (confidence). `indexed_patch_v2`
additionally nailed S02 multi-change (`revise [1]` + `add` new row) where
every Round-1 candidate missed one of two edits. Remaining winner failure
(S09): emitted an extra `deceased_pets` row alongside a correct pets-list
revise — a spurious-adjacent-fact pattern, not a structural fault.

## Round 3 (domain generalization)

Winner `indexed_patch_v2` unchanged on 4 domain-distinct scenarios:

| Scenario | Correct | Notes |
|----------|---------|-------|
| D01 travel update | yes | clean `revise` of flight time |
| D02 finance accounts | no | accounts revised correctly; judge flagged extra `user.investments` row as spurious (arguable) |
| D03 work venting noop | yes | clean `noop` |
| D04 family multi-member | yes | children-in-school + hobbies both updated |

3/4. The one failure is a judge-boundary call. Winner generalizes across
travel, finance, work, and family domains without prompt modification.

## Recommendation

### a) Full command schema

Author sees prior state as a numbered fact sheet:
```
[n] topic.category | attribute: value
```
- Set-valued attributes render as a comma-separated list within the
  value string.
- Confidence, if present, renders as a trailing parenthetical tag from
  a fixed vocabulary: `(confirmed)`, `(hedged)`, `(intended)`.

Commands (TypeScript-style):
```ts
type UpdateCommand =
  | { op: "keep";   index: number }
  | { op: "revise"; index: number; new_text: string }
  | { op: "remove"; index: number }
  | { op: "add";    new_text: string }
  | { op: "noop" };
```
`new_text` MUST be a fact-sheet line in the canonical form
`"topic.category | attribute: value"`. The author response is a JSON
array; an empty array is equivalent to a single `{"op":"noop"}`.

### b) Author-time system prompt (full text)

The exact prompt from `indexed_patch_v2`:

```
You are a copy editor marking up a numbered fact sheet. Each numbered line is one
fact, in the form:
  [n] topic.category | attribute: value

Before emitting any edit, decide: does this statement contain something that
BELONGS on a permanent fact sheet about the person? If not, emit noop.

DO NOT write to the sheet (emit noop instead):
- Weather comments, seasonal gripes, chitchat ("the weather is gloomy", "I hate November").
- Transient moods or fleeting reactions ("I'm tired", "ugh", "cool", "thanks").
- Generic filler / acknowledgements ("haha yeah", "totally", "sure thing").
- Repetitions of facts already on the sheet that add no new detail.

DO write to the sheet:
- Durable attributes (where they live, what they do, names, relationships).
- Durable preferences/traits (diet, allergies, hobbies, values).
- Plans/events the person commits to or reports completing.
- Any correction, addition, or removal to an existing fact.

Emit a JSON array of edits. Each edit is one of:

  {"op": "keep",   "index": n}
      // fact is still exactly right; keep verbatim (optional, rarely needed)
  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n]
  {"op": "remove", "index": n}
      // strike fact [n] from the sheet
  {"op": "add",    "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line at the end
  {"op": "noop"}
      // the statement does not require any change

Schema rules:
- For set-valued attributes (the value looks like a comma-separated list),
`revise` with the new full comma-separated list.
- Confidence markers — when confidence changes, `revise` the line and append
EXACTLY one of: (confirmed), (hedged), (intended) at the end of the value.
E.g. "user.activities | hiking_plan: summited Mt Rainier (confirmed)".
- When one turn carries multiple distinct changes, emit multiple edits
(one per logical change).
- Match facts by MEANING: if the statement refers to fact [n] in paraphrase,
that is still fact [n].
- An ambiguous referent (multiple facts could match) -> prefer noop over a
blind edit.

CURRENT FACT SHEET:
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
```

### c) Set semantics

Set-valued attributes render as a single comma-separated value on one
fact-sheet line. Authors edit sets via `revise [n]` with the new full
list. No `add_member` / `remove_member` verbs on the wire.

Rationale: `member_ops` (which had those verbs) confused single-valued
vs. set-valued attributes in S03. `indexed_patch` with only `revise`
handled every set scenario (S03, S04, D04) correctly. The backing store
can still decompose comma-separated values into normalized set
membership — that's a backend concern.

### d) No-op handling

Two required mechanisms:
1. **Schema-level:** first-class `{"op": "noop"}` command (authors use
   it; judges verify it; `[]` is also accepted).
2. **Framing-level:** the "DO NOT write" block enumerating weather,
   mood, filler, repetition. This was the single most impactful
   addition in Round 2.

Noop discipline is framing-load-bearing, not schema-load-bearing. Every
Round-1 schema with a noop verb still failed S07 (weather); every Round-2
schema with noop verb AND the framing block passed S07.

### e) Confidence representation

Fixed parenthetical tag at the end of the value: `(confirmed)`,
`(hedged)`, `(intended)`. No other parenthetical allowed. Revising a
value replaces any earlier tag.

Not a separate verb. Round 1's `intent_ops` had
`strengthen_confidence` / `weaken_confidence` verbs and the author
still chose `correct_fact` for those cases. Round 2's fixed-vocabulary
tag inside `revise` immediately fixed both scenarios for both
finalists.

### f) Canonicalization / paraphrase-matching

Paraphrase resolution is delegated to the author via stable indices.
The author sees `[3]`, decides whether the new statement refers to
fact `[3]`, emits `revise 3`. The system never needs to string-match
the old value. This fixes the baseline's exact-match delete problem.

For `add`, the author must produce `topic.category | attribute: value`
in a form consistent with existing rows. Misrouting an `add` that
should have been a `revise` is a noop-discipline/reading failure, not
a paraphrase failure. Value-surface canonicalization ("peanuts" vs
"peanut allergy") stays a backend concern — store a normalized key
alongside the surface form.

## Open questions and honest negative results

- **Is indexed_patch really better than baseline?** At Round 1 scoring
  they tied 8/12. Baseline ties only because the LLM-judge
  paraphrase-matches `delete(old) + add(new)` as a valid replacement.
  On actual storage identity (did the old row get deleted? did a
  duplicate get created?), baseline collapses. The judge cannot
  distinguish this.
- **Judge noise on confidence.** The judge marked `member_ops` correct
  on S05 and wrong on S06 despite near-identical value formats.
  Round 2's fixed vocabulary reduced but did not eliminate this. A
  deterministic applier + text-diff grader would be more reliable.
- **Spurious-adjacent-fact pattern** (S09, D02). Winner adds a
  defensible-but-extra fact alongside a correct revise. Judge
  penalizes. Unclear whether over-capture is acceptable if it's
  always extra-true vs. wrong. Needs a follow-up study.
- **Only one author model.** All results at gpt-5-mini / low. Ranking
  may reorder at higher effort or smaller models. Likely the framing
  parts matter more for smaller models; schema choice matters more
  for larger ones.
- **Ambiguous referent** (S10). All Round-1 candidates except
  `indexed_patch` correctly emitted noop. `indexed_patch_v2`'s
  "prefer noop over blind edit" rule fixes this. In production, the
  memory system may want to route ambiguous turns to a clarification
  loop rather than silently drop them.
- **`keep` verb utility.** `keep` was not used by the author in any
  scenario. Keep-or-drop in a v3 is an open question; it adds
  parse/prompt weight with no demonstrated benefit.
- **Multi-user / family** (D04 only). More scenarios needed. A
  "promote-member" op (Maya grade 8 -> Maya high school) might
  reduce set-member churn.

## Summary of verdict

The winner is the **indexed-patch schema + copy-editor framing** with
a noop-discipline block and fixed-vocabulary confidence tags. Load-
bearing engineering choices, in order of impact:

1. **Numbered indices for paraphrase resolution** — eliminates exact
   string-match from the delete/update path.
2. **Explicit noop-discipline block in the framing** — fixes the
   universal "weather chitchat becomes a preference" failure.
3. **Fixed-vocabulary confidence tags** — removes confidence drift.
4. **Single `revise` verb for set edits** — simpler than member-op
   verbs and strictly-equally capable.
