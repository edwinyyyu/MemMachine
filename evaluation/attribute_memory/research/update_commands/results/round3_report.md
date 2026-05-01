# Round 3 — domain-neutral generalization check (indexed_patch_v2)

Winner from Round 2 (`indexed_patch_v2`) tested on 4 scenarios with distinctly different domain flavor (travel, finance, work-venting, multi-user family). The prompt itself was NOT altered.

Accuracy: **3/4 = 75%**. Op totals: `{"revise": 4, "add": 1, "noop": 1}`.

| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|--------|---------|-----------|----------|----------|-------|
| D01_travel_update | correction | True | 1/1 | 0 | `{"revise": 1}` |  |
| D02_finance_setmember | add_member | False | 1/1 | 1 | `{"revise": 1, "add": 1}` | Accounts updated correctly but an extra 'user.investments' fact was added (spuri |
| D03_work_noop_venting | noop | True | 0/0 | 0 | `{"noop": 1}` |  |
| D04_family_multi_member | multi_op | True | 3/3 | 0 | `{"revise": 2}` |  |


## Per-scenario outputs

### D01_travel_update — Travel booking update: flight time changed.

Turn: `User: Airline just rescheduled me — new departure is 11:30am, still Mar 15.`

Prior: `{"user.travel": {"upcoming_flight": "SFO -> NRT, Mar 15 9am, seat 12A"}}`

```
[{"op":"revise","index":1,"new_text":"user.travel | upcoming_flight: SFO -> NRT, Mar 15 11:30am, seat 12A"}]
```
Judge: `{"correct": true, "expected_hits": 1, "expected_total": 1, "spurious_changes": 0, "op_choice_quality": "good", "notes": ""}`

### D02_finance_setmember — Add a new account to a list of financial accounts.

Turn: `User: I opened a Fidelity brokerage account this morning to start dollar-cost averaging into index funds.`

Prior: `{"user.finance": {"accounts": ["Chase checking", "Vanguard 401k"]}}`

```
[
  {"op": "revise", "index": 1, "new_text": "user.finance | accounts: Chase checking, Vanguard 401k, Fidelity brokerage"},
  {"op": "add", "new_text": "user.investments | strategy: dollar-cost averaging into index funds (intended)"}
]
```
Judge: `{"correct": false, "expected_hits": 1, "expected_total": 1, "spurious_changes": 1, "op_choice_quality": "right_effect_wrong_label", "notes": "Accounts updated correctly but an extra 'user.investments' fact was added (spurious)."}`

### D03_work_noop_venting — User vents about a meeting — no durable fact.

Turn: `User: this meeting could've been an email, I'm losing my mind.`

Prior: `{"user.work": {"role": "product manager at Figma", "team": "design systems"}}`

```
[{"op":"noop"}]
```
Judge: `{"correct": true, "expected_hits": 0, "expected_total": 0, "spurious_changes": 0, "op_choice_quality": "good", "notes": ""}`

### D04_family_multi_member — Multi-user/family: remove one child from school list when they graduate; add a new hobby.

Turn: `User: Maya just graduated 8th grade so she's off to high school now, and I picked up woodworking last month.`

Prior: `{"user.family": {"children_in_school": ["Maya (grade 8)", "Noah (grade 5)"]}, "user.hobbies": {"hobbies": ["birding", "pickleball"]}}`

```
[
  {"op": "revise", "index": 1, "new_text": "user.family | children_in_school: Maya (high school), Noah (grade 5)"},
  {"op": "revise", "index": 2, "new_text": "user.hobbies | hobbies: birding, pickleball, woodworking"}
]
```
Judge: `{"correct": true, "expected_hits": 3, "expected_total": 3, "spurious_changes": 0, "op_choice_quality": "good", "notes": ""}`
