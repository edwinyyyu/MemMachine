# Round 5 -- Command Schema Design

Tests S1 (round-4 winner) vs. S2/S3 (upsert) vs. S4/S5 (append-only), with framing ablations on S1.

Model: `gpt-5-mini`. Scenarios: 14 (10 single-turn + 4 multi-turn chains).

## Leaderboard

| Candidate | Family | Correct | Accuracy | Op totals |
|-----------|--------|---------|----------|-----------|
| `S4_append_ref` | log | 10/14 | 71% | `{"append": 11, "append_ref": 20}` |
| `S5_append_plain` | log | 10/14 | 71% | `{"append": 30}` |
| `S2_upsert_replace` | row | 9/14 | 64% | `{"upsert": 22, "remove": 4, "add_member": 2, "remove_member": 1, "noop": 4}` |
| `S1_baseline_editor` | row | 8/14 | 57% | `{"add": 10, "revise": 13, "add_member": 1, "remove_member": 2, "noop": 4}` |
| `S3_upsert_only` | row | 8/14 | 57% | `{"upsert": 27, "noop": 4}` |
| `S1_diff_framing` | row | 6/14 | 43% | `{"revise": 4, "add_member": 3, "remove_member": 3, "noop": 3, "add": 3}` |
| `S1_archivist_framing` | row | 3/14 | 21% | `{"revise": 8, "add_member": 1, "add": 4, "remove_member": 1}` |

## Per-scenario correctness

| Scenario | S4_append_ref | S5_append_plain | S2_upsert_replace | S1_baseline_editor | S3_upsert_only | S1_diff_framing | S1_archivist_framing |
|----------|----|----|----|----|----|----|----|
| T01_new_fact | Y | Y | N | N | N | - | - |
| T02_correction_value | Y | Y | Y | Y | Y | Y | Y |
| T03_retraction | Y | Y | Y | N | N | N | N |
| T04_set_add | Y | Y | N | N | N | N | N |
| T05_set_remove | Y | Y | Y | Y | Y | Y | N |
| T06_confidence_weaken | N | N | N | N | N | Y | Y |
| T07_noop_weather | Y | Y | Y | Y | Y | Y | N |
| T08_noop_joke | Y | Y | Y | Y | Y | Y | N |
| T09_ambiguous_referent | Y | Y | Y | Y | Y | - | - |
| T10_multi_change | N | N | N | N | N | N | N |
| T11_paraphrased_correction_after_chain | N | N | Y | Y | Y | Y | Y |
| T12_chain_with_retraction | N | N | N | N | N | - | - |
| T13_retrieval_probe | Y | Y | Y | Y | Y | - | - |
| T14_long_chain_preference_evolution | Y | Y | Y | Y | Y | - | - |


## `S4_append_ref` -- S4 append + append_ref / journal framing

- Schema family: log

- Correct: **10/14** (71%)

- Op totals: `{"append": 11, "append_ref": 20}`

- Total output chars: 4606


| Scenario | Correct | Tally | Notes |
|----------|---------|-------|-------|
| T01_new_fact | True | `{"append": 1}` |  |
| T02_correction_value | True | `{"append_ref": 1}` |  |
| T03_retraction | True | `{"append_ref": 1}` |  |
| T04_set_add | True | `{"append_ref": 1}` |  |
| T05_set_remove | True | `{"append_ref": 1}` |  |
| T06_confidence_weaken | False | `{"append_ref": 1}` |  |
| T07_noop_weather | True | `{"append": 1}` |  |
| T08_noop_joke | True | `{"append": 1}` |  |
| T09_ambiguous_referent | True | `{"append_ref": 1}` |  |
| T10_multi_change | False | `{"append_ref": 2}` |  |
| T11_paraphrased_correction_after_chain | False | `{"append": 2, "append_ref": 3}` |  failed_checks=['leg_day is NOT Monday'] |
| T12_chain_with_retraction | False | `{"append": 2, "append_ref": 5}` |  failed_checks=['peanut allergy is retracted'] |
| T13_retrieval_probe | True | `{"append_ref": 1}` |  |
| T14_long_chain_preference_evolution | True | `{"append": 4, "append_ref": 3}` |  |


## `S5_append_plain` -- S5 append only / journal framing

- Schema family: log

- Correct: **10/14** (71%)

- Op totals: `{"append": 30}`

- Total output chars: 3927


| Scenario | Correct | Tally | Notes |
|----------|---------|-------|-------|
| T01_new_fact | True | `{"append": 1}` |  |
| T02_correction_value | True | `{"append": 1}` |  |
| T03_retraction | True | `{"append": 1}` |  |
| T04_set_add | True | `{"append": 1}` |  |
| T05_set_remove | True | `{"append": 1}` |  |
| T06_confidence_weaken | False | `{"append": 1}` |  |
| T07_noop_weather | True | `{"append": 1}` |  |
| T08_noop_joke | True | `{"append": 1}` |  |
| T09_ambiguous_referent | True | `{"append": 1}` |  |
| T10_multi_change | False | `{"append": 2}` |  |
| T11_paraphrased_correction_after_chain | False | `{"append": 5}` |  failed_checks=['leg_day is NOT Monday'] |
| T12_chain_with_retraction | False | `{"append": 6}` |  failed_checks=['peanut allergy is retracted'] |
| T13_retrieval_probe | True | `{"append": 1}` |  |
| T14_long_chain_preference_evolution | True | `{"append": 7}` |  |


## `S2_upsert_replace` -- S2 upsert + member ops / editor framing

- Schema family: row

- Correct: **9/14** (64%)

- Op totals: `{"upsert": 22, "remove": 4, "add_member": 2, "remove_member": 1, "noop": 4}`

- Total output chars: 4634


| Scenario | Correct | Tally | Notes |
|----------|---------|-------|-------|
| T01_new_fact | False | `{"upsert": 1}` |  diff:missing=1,extra=1,wrong=0 |
| T02_correction_value | True | `{"upsert": 1}` |  |
| T03_retraction | True | `{"remove": 1}` |  |
| T04_set_add | False | `{"add_member": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T05_set_remove | True | `{"remove_member": 1}` |  |
| T06_confidence_weaken | False | `{"noop": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T07_noop_weather | True | `{"noop": 1}` |  |
| T08_noop_joke | True | `{"noop": 1}` |  |
| T09_ambiguous_referent | True | `{"noop": 1}` |  |
| T10_multi_change | False | `{"upsert": 1, "remove": 1}` |  diff:missing=1,extra=0,wrong=1 |
| T11_paraphrased_correction_after_chain | True | `{"upsert": 5}` |  |
| T12_chain_with_retraction | False | `{"upsert": 6, "remove": 2}` |  failed_checks=['peanut allergy is retracted'] |
| T13_retrieval_probe | True | `{"upsert": 1}` |  |
| T14_long_chain_preference_evolution | True | `{"upsert": 7, "add_member": 1}` |  |


## `S1_baseline_editor` -- S1 baseline / editor framing

- Schema family: row

- Correct: **8/14** (57%)

- Op totals: `{"add": 10, "revise": 13, "add_member": 1, "remove_member": 2, "noop": 4}`

- Total output chars: 3392


| Scenario | Correct | Tally | Notes |
|----------|---------|-------|-------|
| T01_new_fact | False | `{"add": 1}` |  diff:missing=1,extra=1,wrong=0 |
| T02_correction_value | True | `{"revise": 1}` |  |
| T03_retraction | False | `{"revise": 1}` |  diff:missing=0,extra=1,wrong=0 |
| T04_set_add | False | `{"add_member": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T05_set_remove | True | `{"remove_member": 1}` |  |
| T06_confidence_weaken | False | `{"noop": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T07_noop_weather | True | `{"noop": 1}` |  |
| T08_noop_joke | True | `{"noop": 1}` |  |
| T09_ambiguous_referent | True | `{"noop": 1}` |  |
| T10_multi_change | False | `{"remove_member": 1, "revise": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T11_paraphrased_correction_after_chain | True | `{"add": 4, "revise": 1}` |  |
| T12_chain_with_retraction | False | `{"add": 4, "revise": 2}` |  failed_checks=['peanut allergy is retracted'] |
| T13_retrieval_probe | True | `{"revise": 1}` |  |
| T14_long_chain_preference_evolution | True | `{"add": 1, "revise": 6}` |  |


## `S3_upsert_only` -- S3 upsert only / editor framing

- Schema family: row

- Correct: **8/14** (57%)

- Op totals: `{"upsert": 27, "noop": 4}`

- Total output chars: 5532


| Scenario | Correct | Tally | Notes |
|----------|---------|-------|-------|
| T01_new_fact | False | `{"upsert": 1}` |  diff:missing=1,extra=1,wrong=0 |
| T02_correction_value | True | `{"upsert": 1}` |  |
| T03_retraction | False | `{"upsert": 1}` |  diff:missing=0,extra=1,wrong=0 |
| T04_set_add | False | `{"upsert": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T05_set_remove | True | `{"upsert": 1}` |  |
| T06_confidence_weaken | False | `{"noop": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T07_noop_weather | True | `{"noop": 1}` |  |
| T08_noop_joke | True | `{"noop": 1}` |  |
| T09_ambiguous_referent | True | `{"noop": 1}` |  |
| T10_multi_change | False | `{"upsert": 2}` |  diff:missing=0,extra=0,wrong=2 |
| T11_paraphrased_correction_after_chain | True | `{"upsert": 6}` |  |
| T12_chain_with_retraction | False | `{"upsert": 6}` |  failed_checks=['peanut allergy is retracted'] |
| T13_retrieval_probe | True | `{"upsert": 1}` |  |
| T14_long_chain_preference_evolution | True | `{"upsert": 7}` |  |


## `S1_diff_framing` -- S1 baseline / minimal-diff framing (Q1)

- Schema family: row

- Correct: **6/14** (43%)

- Op totals: `{"revise": 4, "add_member": 3, "remove_member": 3, "noop": 3, "add": 3}`

- Total output chars: 1188


| Scenario | Correct | Tally | Notes |
|----------|---------|-------|-------|
| T02_correction_value | True | `{"revise": 1}` |  |
| T03_retraction | False | `{"revise": 1}` |  diff:missing=0,extra=1,wrong=0 |
| T04_set_add | False | `{"add_member": 2}` |  diff:missing=0,extra=0,wrong=1 |
| T05_set_remove | True | `{"remove_member": 1}` |  |
| T06_confidence_weaken | True | `{"revise": 1}` |  |
| T07_noop_weather | True | `{"noop": 1}` |  |
| T08_noop_joke | True | `{"noop": 1}` |  |
| T10_multi_change | False | `{"remove_member": 1, "revise": 1}` | errors=1 diff:missing=0,extra=0,wrong=2 |
| T11_paraphrased_correction_after_chain | True | `{"add": 3, "noop": 1, "remove_member": 1, "add_member": 1}` |  |


## `S1_archivist_framing` -- S1 baseline / archivist framing (Q1)

- Schema family: row

- Correct: **3/14** (21%)

- Op totals: `{"revise": 8, "add_member": 1, "add": 4, "remove_member": 1}`

- Total output chars: 1639


| Scenario | Correct | Tally | Notes |
|----------|---------|-------|-------|
| T02_correction_value | True | `{"revise": 1}` |  |
| T03_retraction | False | `{"revise": 1}` |  diff:missing=0,extra=1,wrong=0 |
| T04_set_add | False | `{"add_member": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T05_set_remove | False | `{"revise": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T06_confidence_weaken | True | `{"revise": 1}` |  |
| T07_noop_weather | False | `{"add": 1}` |  diff:missing=0,extra=1,wrong=0 |
| T08_noop_joke | False | `{"add": 1}` |  diff:missing=0,extra=1,wrong=0 |
| T10_multi_change | False | `{"remove_member": 1, "revise": 1}` |  diff:missing=0,extra=0,wrong=1 |
| T11_paraphrased_correction_after_chain | True | `{"add": 2, "revise": 3}` |  |
