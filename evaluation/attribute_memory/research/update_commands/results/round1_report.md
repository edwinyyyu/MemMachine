# Round 1 — update-command schema comparison

Model: `gpt-5-mini`. Scenarios: 12. LLM calls (new): 116 (~$0.29).

## Leaderboard (correct by LLM judge)

| Candidate | Correct | Accuracy | Parse Failures | Op totals |
|-----------|---------|----------|----------------|-----------|
| `baseline_addel` | 8/12 | 67% | 0 | {"add": 15, "delete": 8} |
| `member_ops` | 8/12 | 67% | 0 | {"add": 4, "update": 7, "remove_member": 2, "noop": 2} |
| `indexed_patch` | 8/12 | 67% | 0 | {"add": 4, "revise": 9, "noop": 1} |
| `cud_triple` | 7/12 | 58% | 0 | {"add": 4, "update": 9, "noop": 2} |
| `intent_ops` | 7/12 | 58% | 0 | {"introduce_fact": 6, "correct_fact": 6, "remove_from_set": 2, "nothing_to_change": 2, "retire_fact": 1} |


## `baseline_addel` — Baseline add/delete

Accuracy: **8/12 (67%)**. Op totals: `{"add": 15, "delete": 8}`. Parse failures: 0.

| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|--------|---------|-----------|----------|----------|-------|
| S01_new_fact | new_fact | True | 2/2 | 0 | `{"add": 3}` |  |
| S02_correction_paraphrased | correction | False | 1/2 | 0 | `{"delete": 1, "add": 1}` | Hometown updated correctly, but did not add college_location=Portland in educati |
| S03_add_set_member | add_member | True | 1/1 | 0 | `{"add": 1}` |  |
| S04_remove_set_member | remove_member | True | 1/1 | 0 | `{"delete": 1, "add": 1}` |  |
| S05_strengthen_confidence | strengthen_confidence | False | 0/1 | 1 | `{"delete": 1, "add": 1}` | Removed original but added a different attribute and didn't mark confidence as c |
| S06_weaken_confidence | weaken_confidence | False | 0/1 | 0 | `{"delete": 1, "add": 1}` | Value updated but confidence not weakened/hedged (missing confidence change). |
| S07_noop_weather | noop | False | 0/0 | 1 | `{"add": 1}` | Added a new dislike from casual small talk; should have been no-op for irrelevan |
| S08_noop_phatic | noop | True | 0/0 | 0 | `{}` |  |
| S09_multi_op | multi_op | True | 2/2 | 0 | `{"delete": 2, "add": 4}` | Effects match expected (Oliver removed; employment updated), but used delete/add |
| S10_ambiguous_referent | ambiguous | True | 1/1 | 0 | `{}` |  |
| S11_conflict | conflict | True | 1/1 | 0 | `{"delete": 1, "add": 1}` |  |
| S12_paraphrase_drift | correction | True | 1/1 | 0 | `{"delete": 1, "add": 1}` |  |


## `member_ops` — add/update/delete + member ops + noop (librarian)

Accuracy: **8/12 (67%)**. Op totals: `{"add": 4, "update": 7, "remove_member": 2, "noop": 2}`. Parse failures: 0.

| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|--------|---------|-----------|----------|----------|-------|
| S01_new_fact | new_fact | True | 2/2 | 0 | `{"add": 2}` |  |
| S02_correction_paraphrased | correction | False | 1/2 | 1 | `{"update": 1, "add": 1}` | Hometown updated correctly; college info added but under user.location (college_ |
| S03_add_set_member | add_member | True | 1/1 | 0 | `{"update": 1}` | Updated value merged entries instead of adding a list member; content present. |
| S04_remove_set_member | remove_member | True | 1/1 | 0 | `{"remove_member": 1}` |  |
| S05_strengthen_confidence | strengthen_confidence | True | 1/1 | 0 | `{"update": 1}` |  |
| S06_weaken_confidence | weaken_confidence | False | 0/1 | 0 | `{"update": 1}` | Value updated but confidence field not set to 'hedged' as expected. |
| S07_noop_weather | noop | False | 0/0 | 1 | `{"add": 1}` | Added an unnecessary preference entry for a casual complaint instead of performi |
| S08_noop_phatic | noop | True | 0/0 | 0 | `{"noop": 1}` |  |
| S09_multi_op | multi_op | False | 1/2 | 1 | `{"remove_member": 1, "update": 1}` | Removed member uses 'Oliver' not 'a cat named Oliver', so pet removal would fail |
| S10_ambiguous_referent | ambiguous | True | 1/1 | 0 | `{"noop": 1}` |  |
| S11_conflict | conflict | True | 1/1 | 0 | `{"update": 1}` |  |
| S12_paraphrase_drift | correction | True | 1/1 | 0 | `{"update": 1}` |  |


## `indexed_patch` — numbered-sheet keep/revise/remove/add (copy editor)

Accuracy: **8/12 (67%)**. Op totals: `{"add": 4, "revise": 9, "noop": 1}`. Parse failures: 0.

| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|--------|---------|-----------|----------|----------|-------|
| S01_new_fact | new_fact | True | 2/2 | 0 | `{"add": 1}` | Command uses a single 'add' text instead of two explicit 'introduce' facts; effe |
| S02_correction_paraphrased | correction | False | 1/2 | 0 | `{"revise": 1}` | Hometown correctly changed to Vancouver, but did not add college_location: Portl |
| S03_add_set_member | add_member | True | 1/1 | 0 | `{"add": 1}` |  |
| S04_remove_set_member | remove_member | True | 1/1 | 0 | `{"revise": 1}` | Used a generic revise instead of a remove_member command. |
| S05_strengthen_confidence | strengthen_confidence | True | 1/1 | 0 | `{"revise": 1}` |  |
| S06_weaken_confidence | weaken_confidence | True | 1/1 | 0 | `{"revise": 1}` |  |
| S07_noop_weather | noop | False | 0/0 | 1 | `{"add": 1}` | Added irrelevant sentiment memory for a throwaway complaint; should have been no |
| S08_noop_phatic | noop | True | 0/0 | 0 | `{"noop": 1}` |  |
| S09_multi_op | multi_op | False | 2/2 | 1 | `{"revise": 2, "add": 1}` | Added an extra 'deceased' fact for Oliver which was not in expected changes. |
| S10_ambiguous_referent | ambiguous | False | 1/1 | 1 | `{"revise": 1}` | Removed shellfish (acceptable) but also marked tree nuts 'confirmed'—an unjustif |
| S11_conflict | conflict | True | 1/1 | 0 | `{"revise": 1}` |  |
| S12_paraphrase_drift | correction | True | 1/1 | 0 | `{"revise": 1}` |  |


## `cud_triple` — add/update/delete + noop (dossier editor)

Accuracy: **7/12 (58%)**. Op totals: `{"add": 4, "update": 9, "noop": 2}`. Parse failures: 0.

| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|--------|---------|-----------|----------|----------|-------|
| S01_new_fact | new_fact | False | 2/2 | 0 | `{"add": 2}` | Second command uses attribute 'status' instead of 'job_status' and different op  |
| S02_correction_paraphrased | correction | False | 1/2 | 1 | `{"update": 1, "add": 1}` | Hometown updated correctly; college info added but under user.location (college_ |
| S03_add_set_member | add_member | False | 1/1 | 1 | `{"update": 1}` | Updated existing pet string format (a dog named Luna -> dog: Luna), an unnecessa |
| S04_remove_set_member | remove_member | True | 1/1 | 0 | `{"update": 1}` |  |
| S05_strengthen_confidence | strengthen_confidence | False | 0/1 | 1 | `{"update": 1}` | Did not replace 'hiking_plan' hedged entry; created a new differently named attr |
| S06_weaken_confidence | weaken_confidence | True | 1/1 | 0 | `{"update": 1}` |  |
| S07_noop_weather | noop | False | 0/0 | 1 | `{"add": 1}` | Irrelevant small-talk should not create memory; adding a disliked_months entry i |
| S08_noop_phatic | noop | True | 0/0 | 0 | `{"noop": 1}` |  |
| S09_multi_op | multi_op | True | 2/2 | 0 | `{"update": 2}` |  |
| S10_ambiguous_referent | ambiguous | True | 1/1 | 0 | `{"noop": 1}` |  |
| S11_conflict | conflict | True | 1/1 | 0 | `{"update": 1}` |  |
| S12_paraphrase_drift | correction | True | 1/1 | 0 | `{"update": 1}` |  |


## `intent_ops` — intent-level verbs (biographer)

Accuracy: **7/12 (58%)**. Op totals: `{"introduce_fact": 6, "correct_fact": 6, "remove_from_set": 2, "nothing_to_change": 2, "retire_fact": 1}`. Parse failures: 0.

| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|--------|---------|-----------|----------|----------|-------|
| S01_new_fact | new_fact | True | 2/2 | 0 | `{"introduce_fact": 3}` |  |
| S02_correction_paraphrased | correction | False | 1/2 | 0 | `{"correct_fact": 1}` | Hometown corrected, but did not add college_location = Portland to education cat |
| S03_add_set_member | add_member | True | 1/1 | 0 | `{"correct_fact": 1}` |  |
| S04_remove_set_member | remove_member | True | 1/1 | 0 | `{"remove_from_set": 1}` |  |
| S05_strengthen_confidence | strengthen_confidence | False | 0/1 | 0 | `{"correct_fact": 1}` | Value updated correctly but confidence not strengthened to confirmed. |
| S06_weaken_confidence | weaken_confidence | False | 1/1 | 0 | `{"correct_fact": 1}` | Value updated but command did not weaken confidence to 'hedged' as expected. |
| S07_noop_weather | noop | False | 0/0 | 1 | `{"introduce_fact": 1}` | Should have been no-op; adding a persistent dislike from small-talk weather comm |
| S08_noop_phatic | noop | True | 0/0 | 0 | `{"nothing_to_change": 1}` |  |
| S09_multi_op | multi_op | False | 2/2 | 1 | `{"remove_from_set": 1, "introduce_fact": 1, "correct_fact": 1}` | Introduced an extra deceased_pets fact not requested. |
| S10_ambiguous_referent | ambiguous | True | 1/1 | 0 | `{"nothing_to_change": 1}` |  |
| S11_conflict | conflict | True | 1/1 | 0 | `{"retire_fact": 1, "introduce_fact": 1}` | Removed 'spouse' fact and added 'engaged_to'—captures correct state but uses a d |
| S12_paraphrase_drift | correction | True | 1/1 | 0 | `{"correct_fact": 1}` |  |

## Bias analysis

Per-schema command-type distribution on the 12-scenario set. The `add`-only baseline cannot emit update/member/noop verbs at all, so its bias is structural. For the other schemas, skew toward any single op indicates schema-induced bias.

- **`baseline_addel`** op totals: `{"add": 15, "delete": 8}`
- **`member_ops`** op totals: `{"add": 4, "update": 7, "remove_member": 2, "noop": 2}`
- **`indexed_patch`** op totals: `{"add": 4, "revise": 9, "noop": 1}`
- **`cud_triple`** op totals: `{"add": 4, "update": 9, "noop": 2}`
- **`intent_ops`** op totals: `{"introduce_fact": 6, "correct_fact": 6, "remove_from_set": 2, "nothing_to_change": 2, "retire_fact": 1}`
