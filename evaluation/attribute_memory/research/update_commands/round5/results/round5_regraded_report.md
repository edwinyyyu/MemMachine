# Round 5 -- Regraded Results

Regrading addresses three grader bugs in the original run:

1. Log-based retraction check was failing on any entry mentioning the retracted topic, including the retraction entry itself. Fixed to check the LATEST live claim's semantics.
2. T11 leg-day check was too coarse (substring of 'monday' in the gym_days entry). Fixed to look at the latest claim specifically about leg day.
3. Row-family retraction-via-confidence=negated was counted as 'still present'. Fixed to treat negated rows as retracted.

Single-turn scenarios keep their original grading from the deterministic applier.

## Leaderboard (regraded)

| Candidate | Family | Correct | Accuracy |
|-----------|--------|---------|----------|
| `S4_append_ref` | log | 12/14 | 86% |
| `S5_append_plain` | log | 12/14 | 86% |
| `S1_baseline_editor` | row | 8/14 | 57% |
| `S2_upsert_replace` | row | 8/14 | 57% |
| `S3_upsert_only` | row | 6/14 | 43% |
| `S1_diff_framing` | row | 5/9 | 56% |
| `S1_archivist_framing` | row | 3/9 | 33% |

## Per-scenario correctness (regraded)

| Scenario | S4_append_ref | S5_append_plain | S1_baseline_editor | S2_upsert_replace | S3_upsert_only | S1_diff_framing | S1_archivist_framing |
|----------|----|----|----|----|----|----|----|
| T01_new_fact | Y | Y | N | N | N | - | - |
| T02_correction_value | Y | Y | Y | Y | Y | Y | Y |
| T03_retraction | Y | Y | N | Y | N | N | N |
| T04_set_add | Y | Y | N | N | N | N | N |
| T05_set_remove | Y | Y | Y | Y | Y | Y | N |
| T06_confidence_weaken | N | N | N | N | N | Y | Y |
| T07_noop_weather | Y | Y | Y | Y | Y | Y | N |
| T08_noop_joke | Y | Y | Y | Y | Y | Y | N |
| T09_ambiguous_referent | Y | Y | Y | Y | Y | - | - |
| T10_multi_change | N | N | N | N | N | N | N |
| T11_paraphrased_correction_after_chain | Y | Y | Y | Y | N | N | Y |
| T12_chain_with_retraction | Y | Y | Y | N | N | - | - |
| T13_retrieval_probe | Y | Y | N | N | N | - | - |
| T14_long_chain_preference_evolution | Y | Y | Y | Y | Y | - | - |

## Op distribution

| Candidate | Op totals |
|-----------|-----------|
| `S4_append_ref` | `{"append": 11, "append_ref": 20}` |
| `S5_append_plain` | `{"append": 30}` |
| `S1_baseline_editor` | `{"add": 10, "revise": 13, "add_member": 1, "remove_member": 2, "noop": 4}` |
| `S2_upsert_replace` | `{"upsert": 22, "remove": 4, "add_member": 2, "remove_member": 1, "noop": 4}` |
| `S3_upsert_only` | `{"upsert": 27, "noop": 4}` |
| `S1_diff_framing` | `{"revise": 4, "add_member": 3, "remove_member": 3, "noop": 3, "add": 3}` |
| `S1_archivist_framing` | `{"revise": 8, "add_member": 1, "add": 4, "remove_member": 1}` |
