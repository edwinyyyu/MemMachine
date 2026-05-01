# Round 2 — finalist stress test with noop-discipline + confidence markers

Scenarios: ['S02_correction_paraphrased', 'S05_strengthen_confidence', 'S06_weaken_confidence', 'S07_noop_weather', 'S09_multi_op', 'S10_ambiguous_referent']

Model: `gpt-5-mini`. Budget used: ~138/150 (~$0.35).

## Leaderboard

| Candidate | Correct | Accuracy | Op totals |
|-----------|---------|----------|-----------|
| `indexed_patch_v2` | 5/6 | 83% | {"revise": 5, "add": 2, "noop": 2} |
| `member_ops_v2` | 4/6 | 67% | {"update": 4, "add": 1, "noop": 2, "remove_member": 1} |


## `indexed_patch_v2`

Accuracy: **5/6**. Op totals: `{"revise": 5, "add": 2, "noop": 2}`.

| Scenario | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|---------|-----------|----------|----------|-------|
| S02_correction_paraphrased | True | 2/2 | 0 | `{"revise": 1, "add": 1}` |  |
| S05_strengthen_confidence | True | 1/1 | 0 | `{"revise": 1}` |  |
| S06_weaken_confidence | True | 1/1 | 0 | `{"revise": 1}` |  |
| S07_noop_weather | True | 0/0 | 0 | `{"noop": 1}` |  |
| S09_multi_op | False | 2/2 | 1 | `{"revise": 2, "add": 1}` | Added an unnecessary deceased_pets entry; employer updated but company name omit |
| S10_ambiguous_referent | True | 1/1 | 0 | `{"noop": 1}` |  |


## `member_ops_v2`

Accuracy: **4/6**. Op totals: `{"update": 4, "add": 1, "noop": 2, "remove_member": 1}`.

| Scenario | Correct | Hits/Total | Spurious | Op tally | Notes |
|----------|---------|-----------|----------|----------|-------|
| S02_correction_paraphrased | False | 1/2 | 1 | `{"update": 1, "add": 1}` | Hometown updated correctly; college location added but under user.location as co |
| S05_strengthen_confidence | True | 1/1 | 0 | `{"update": 1}` |  |
| S06_weaken_confidence | True | 1/1 | 0 | `{"update": 1}` |  |
| S07_noop_weather | True | 0/0 | 0 | `{"noop": 1}` |  |
| S09_multi_op | False | 1/2 | 0 | `{"remove_member": 1, "update": 1}` | First removal acceptable; employer update omits Acme Corp/laid-off-from detail,  |
| S10_ambiguous_referent | True | 1/1 | 0 | `{"noop": 1}` |  |


## Side-by-side outputs

### S02_correction_paraphrased — User corrects a prior value; new value is a paraphrase-linked replacement (hometown).

Turn: `User: Actually I grew up in Vancouver, not Portland — I only went to college in Portland.`

Prior (json): `{"user.location": {"hometown": "Portland", "current_city": "Seattle"}}`

**`member_ops_v2`** — correct=False  hits=1/2  notes: Hometown updated correctly; college location added but under user.location as college_city instead of introducing educat

```
[
  {"op": "update", "category": "user.location", "attribute": "hometown", "value": "Vancouver (confirmed)"},
  {"op": "add", "category": "user.location", "attribute": "college_city", "value": "Portland (confirmed)"}
]
```

**`indexed_patch_v2`** — correct=True  hits=2/2  notes: 

```
[
  {"op": "revise", "index": 1, "new_text": "user.location | hometown: Vancouver"},
  {"op": "add", "new_text": "user.education | college_city: Portland"}
]
```


### S05_strengthen_confidence — A hedged plan becomes a completed action.

Turn: `User: I did it — summited Mt Rainier yesterday, it was amazing.`

Prior (json): `{"user.activities": {"hiking_plan": {"value": "might hike Mt Rainier this weekend", "confidence": "hedged"}}}`

**`member_ops_v2`** — correct=True  hits=1/1  notes: 

```
[{"op":"update","category":"user.activities","attribute":"hiking_plan","value":"summited Mt Rainier yesterday (confirmed)"}]
```

**`indexed_patch_v2`** — correct=True  hits=1/1  notes: 

```
[{"op":"revise","index":1,"new_text":"user.activities | hiking_plan: summited Mt Rainier yesterday (confirmed)"}]
```


### S06_weaken_confidence — A firm identity claim becomes a trial.

Turn: `User: Well, I'm not exactly vegan anymore — I'm trying it for a month to see if I feel better.`

Prior (json): `{"user.diet": {"diet": {"value": "vegan", "confidence": "confirmed"}}}`

**`member_ops_v2`** — correct=True  hits=1/1  notes: 

```
[
  {
    "op": "update",
    "category": "user.diet",
    "attribute": "diet",
    "value": "trying vegan for a month (hedged)"
  }
]
```

**`indexed_patch_v2`** — correct=True  hits=1/1  notes: 

```
[{"op":"revise","index":1,"new_text":"user.diet | diet: trying vegan for one month (intended)"}]
```


### S07_noop_weather — Irrelevant small talk turn — no memory-worthy content.

Turn: `User: ugh the weather is so gloomy today, I hate November in this city.`

Prior (json): `{"user.basics": {"name": "Alex", "city": "Seattle"}}`

**`member_ops_v2`** — correct=True  hits=0/0  notes: 

```
[{"op":"noop"}]
```

**`indexed_patch_v2`** — correct=True  hits=0/0  notes: 

```
[{"op":"noop"}]
```


### S09_multi_op — Single turn carries 2 independent changes: remove a set member AND introduce a new fact.

Turn: `User: Rough week — Oliver passed away on Tuesday, and I also got laid off Friday from Acme.`

Prior (json): `{"user.pets": {"pets": ["a dog named Luna", "a cat named Oliver"]}, "user.work": {"employer": "Acme Corp"}}`

**`member_ops_v2`** — correct=False  hits=1/2  notes: First removal acceptable; employer update omits Acme Corp/laid-off-from detail, so replacement intent not fully achieved

```
[
  {"op": "remove_member", "category": "user.pets", "attribute": "pets", "member": "Oliver"},
  {"op": "update", "category": "user.work", "attribute": "employer", "value": "Unemployed (confirmed)"}
]
```

**`indexed_patch_v2`** — correct=False  hits=2/2  notes: Added an unnecessary deceased_pets entry; employer updated but company name omitted.

```
[
  {"op": "revise", "index": 1, "new_text": "user.pets | pets: a dog named Luna"},
  {"op": "revise", "index": 2, "new_text": "user.work | employer: none (laid off Friday)"},
  {"op": "add", "new_text": "user.pets | deceased_pets: Oliver (died Tuesday)"}
]
```


### S10_ambiguous_referent — User has multiple allergies; says 'that allergy is gone' without specifying — LLM should either defer or pick most-recent.

Turn: `User: Great news, turns out that allergy is actually gone according to my new test!`

Prior (json): `{"user.health": {"allergies": ["peanuts (diagnosed 2015)", "tree nuts (diagnosed 2022)", "shellfish (diagnosed 2023)"]}}`

**`member_ops_v2`** — correct=True  hits=1/1  notes: 

```
[{"op":"noop"}]
```

**`indexed_patch_v2`** — correct=True  hits=1/1  notes: 

```
[{"op":"noop"}]
```

