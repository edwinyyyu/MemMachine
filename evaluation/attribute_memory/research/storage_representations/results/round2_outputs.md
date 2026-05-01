# Round 2 — readback, drift, retrieval, judge

Finalists: C1_flat_triples, C2_typed_hierarchy, C3_dossier_markdown.

Model: gpt-5-mini.


## Judge averages (1-5)

| Representation | Fidelity | Confidence | Cardinality | Negation | n |

|---|---|---|---|---|---|

| C1_flat_triples | 4.583 | 4.25 | 4.583 | 4.667 | 12 |

| C2_typed_hierarchy | 4.583 | 4.583 | 4.583 | 4.75 | 12 |

| C3_dossier_markdown | 4.917 | 4.75 | 4.417 | 5.0 | 12 |


Winner tally: `{'C1_flat_triples': 1, 'C2_typed_hierarchy': 3, 'C3_dossier_markdown': 2, 'tie': 6}`, judge parse failures: 0


## Per-scenario readbacks, drift, retrieval, judge


### simple_first_person

**Source:** `I live in Portland with my partner Jamie. We moved here from Austin about three years ago.`

**Truth:**
```
  - subject=user | attribute=city | value=Portland | confidence=stated
  - subject=user | attribute=partner_name | value=Jamie | confidence=stated
  - subject=user | attribute=previous_city | value=Austin | confidence=stated
  - subject=user | attribute=years_at_city | value=~3 | confidence=approximate
  - REL: user -[partner]-> Jamie
```


#### C1_flat_triples

**Author-time rep:**
```
(speaker | biography | current_city | Portland)
(speaker | relationships | partner | Jamie)
(speaker and Jamie | biography | previous_city | Austin)
(speaker | biography | time_since_move | about three years ago)
```

**Readback:**
```
- current_city: Portland [stated]
- partner: Jamie (cardinality=1) [stated]
- previous_city (speaker and Jamie): Austin [stated]
- time_since_move: about three years ago [stated]
```

**Drift paraphrase source:**

> Jamie and I have been in Portland for about three years now — we came up from Austin together.

**Drift rep:**
```
(Jamie and I | biography | current_city | Portland)
(Jamie and I | biography | duration_in_current_city | approximately three years)
(Jamie and I | biography | previous_city | Austin)
(Jamie and I | relationships | moved_to_current_city_together | came from Austin together)
```

**Drift readback:**
```
- subject: Jamie and I [stated]
- category: biography, relationships (cardinality=2) [stated]
- current_city: Portland [stated]
- duration_in_current_city: approximately three years [approximate]
- previous_city: Austin [stated]
- moved_to_current_city_together: came from Austin together [stated]
```

**Retrieval new statement:**

> We're thinking of moving to Seattle next summer.

**Retrieval output:**
```
RELEVANT: (speaker | biography | current_city | Portland) | The new statement says they are thinking of moving to Seattle, which would change or relate to their current city being Portland.

RELEVANT: (speaker | relationships | partner | Jamie) | Plans to move often involve a partner; this fact is relevant to whether Jamie is part of the move and how the move affects relationships.

RELEVANT: (speaker and Jamie | biography | previous_city | Austin) | Their prior move history (from Austin) provides context for relocation patterns and may be relevant when considering another move.

RELEVANT: (speaker | biography | time_since_move | about three years ago) | Indicates how long they've been in their current city (Portland); relevant for planning/timing of another move next summer.
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "personal": {
    "location": {
      "current_city": {
        "kind": "partial_functional",
        "value": "Portland",
        "confidence": "stated"
      },
      "household_members": {
        "kind": "set",
        "value": [
          "partner: Jamie"
        ],
        "confidence": "stated",
        "cardinality": 1,
        "note": "Subject lives with a partner named Jamie"
      }
    },
    "relationship": {
      "partner_name": {
        "kind": "partial_functional",
        "value": "Jamie",
        "confidence": "stated"
      }
    },
    "migration": {
      "previous_city": {
        "kind": "functional",
        "value": "Austin",
        "confidence": "stated"
      },
      "move_time": {
        "kind": "functional",
        "value": "about three years ago",
        "confidence": "approximate",
        "note": "Relative time expressed from moment of statement"
      }
    }
  }
}
```

**Readback:**
```
- current_city: Portland [stated]
- household_members: partner: Jamie (cardinality=1) [stated]
- partner_name: Jamie [stated]
- previous_city: Austin [stated]
- move_time: about three years ago [approximate]
```

**Drift paraphrase source:**

> Jamie and I have been in Portland for about three years now — we came up from Austin together.

**Drift rep:**
```
{
  "location": {
    "residence": {
      "current_city": {
        "kind": "functional",
        "value": "Portland",
        "confidence": "stated",
        "note": "Speaker says 'have been in Portland for about three years now.'"
      },
      "duration_in_current_city": {
        "kind": "functional",
        "value": "about three years",
        "confidence": "stated",
        "note": "Exact start date not provided; phrased as 'about three years now.'"
      }
    }
  },
  "migration": {
    "origin": {
      "origin_city": {
        "kind": "set",
        "value": [
          "Austin"
        ],
        "cardinality": 1,
        "confidence": "stated",
        "note": "Text: 'we came up from Austin together.'"
      },
      "moved_together_with": {
        "kind": "functional",
        "value": "Jamie",
        "confidence": "stated",
        "note": "Indicates the speaker and Jamie moved from Austin to Portland together."
      }
    }
  },
  "relationships": {
    "peer_relation": {
      "mentioned_person": {
        "kind": "set",
        "value": [
          "Jamie"
        ],
        "cardinality": 1,
        "confidence": "stated",
        "note": "Jamie is mentioned as accompanying the speaker."
      },
      "shared_history_with_Jamie": {
        "kind": "functional",
        "value": "have lived in Portland together for about three years; came from Austin together",
        "confidence": "stated",
        "note": "Combined facts about co-migration and co-residency stated in the text."
      }
    }
  }
}
```

**Drift readback:**
```
- current_city: Portland [stated]
- duration_in_current_city: about three years [stated]
- origin_city: Austin (cardinality=1) [stated]
- moved_together_with: Jamie [stated]
- mentioned_person: Jamie (cardinality=1) [stated]
- shared_history_with_Jamie: have lived in Portland together for about three years; came from Austin together [stated]
```

**Retrieval new statement:**

> We're thinking of moving to Seattle next summer.

**Retrieval output:**
```
RELEVANT: "current_city": { "kind": "partial_functional", "value": "Portland", "confidence": "stated" } | Relevant because the new statement indicates a planned move from the current city (Portland) to Seattle, so current location may change.
RELEVANT: "household_members": { "kind": "set", "value": [ "partner: Jamie" ], "confidence": "stated", "cardinality": 1, "note": "Subject lives with a partner named Jamie" } | Relevant because a planned move may involve household members (partner Jamie) and their status or plans could be affected.
RELEVANT: "partner_name": { "kind": "partial_functional", "value": "Jamie", "confidence": "stated" } | Relevant because the partner may be part of the planned move to Seattle.
RELEVANT: "previous_city": { "kind": "functional", "value": "Austin", "confidence": "stated" } | Relevant as background migration history (moved from Austin) which may contextualize the new planned move.
RELEVANT: "move_time": { "kind": "functional", "value": "about three years ago", "confidence": "approximate", "note": "Relative time expressed from moment of statement" } | Relevant because it records past move timing and may be useful when considering timeline/context for the new planned move.
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Subject

### Biography
- Lives in Portland with partner Jamie [stated].

### Relationships
- Partner: Jamie [stated].
- Lives with partner Jamie [stated].

### Location
- Current residence: Portland [stated].
- Previous residence: Austin [stated].

### Timeline
- Moved from Austin to Portland about three years ago [approximate].
- Has lived in Portland for about three years [approximate].
```

**Readback:**
```
Lives in Portland with partner: Jamie [stated]
Lives with partner: Jamie [stated]
Partner: Jamie [stated]
Current residence: Portland [stated]
Previous residence: Austin [stated]
Moved from Austin to Portland: about three years ago [approximate]
Duration in Portland: about three years [approximate]
Biography - Lives in Portland with partner: Jamie [stated]
```

**Drift paraphrase source:**

> Jamie and I have been in Portland for about three years now — we came up from Austin together.

**Drift rep:**
```
## Jamie and the speaker

### Biography
- People involved: Jamie and the speaker (2 total) [stated]

### Location
- Current residence: Portland [stated]
- Duration in Portland: about three years [approximate]
- Previous residence: Austin [stated]
- Moved together from Austin to Portland [stated]
```

**Drift readback:**
```
- People involved: Jamie, the speaker (cardinality=2) [stated]
- Current residence: Portland [stated]
- Duration in Portland: about three years [approximate]
- Previous residence: Austin [stated]
- Moved together: from Austin to Portland [stated]
```

**Retrieval new statement:**

> We're thinking of moving to Seattle next summer.

**Retrieval output:**
```
RELEVANT: Lives in Portland with partner Jamie [stated]. | Relevant because planning a move from Portland to Seattle would change current residence and may affect who they live with.

RELEVANT: Partner: Jamie [stated]. | Relevant because the move statement uses "we" and implies Jamie is involved in the decision to move.

RELEVANT: Lives with partner Jamie [stated]. | Relevant because moving could alter current cohabitation arrangements.

RELEVANT: Current residence: Portland [stated]. | Relevant because the plan to move to Seattle would change the user's current city of residence.

RELEVANT: Moved from Austin to Portland about three years ago [approximate]. | Relevant because it provides recent relocation history that may inform timing/frequency of moves and context for another move.

RELEVANT: Has lived in Portland for about three years [approximate]. | Relevant because it indicates how long they've been in Portland and may be updated if they move next summer.
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 3, "cardinality": 5, "negation": 5, "note": "Captures all factual content (current city, partner, previous city, move time) and cardinality explicitly, but mislabels the hedge on move time as stated rather than approximate."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Preserves all facts, correctly encodes the approximate move time, and explicitly represents partner/cardinality in a typed hierarchy."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 4, "negation": 5, "note": "All facts and the approximate timing are present (though repeated), but cardinality is implied by repetition rather than an explicit count."},
  "overall_winner": "B",
  "overall_justification": "Representation B best preserves facts, hedges, and explicit cardinality in a structured way; A mislabels confidence and C is redundant and less explicit about cardinality." 
}
```


### hedged_nuanced

**Source:** `I think I might be lactose intolerant? Cheese sometimes makes me feel bad but yogurt is fine. Probably going to try cutting back.`

**Truth:**
```
  - subject=user | attribute=possible_condition | value=lactose intolerance | confidence=suspected
  - subject=user | attribute=trigger_food | value=cheese | confidence=partial (sometimes)
  - subject=user | attribute=tolerated_food | value=yogurt | confidence=stated
  - subject=user | attribute=planned_change | value=reduce lactose | confidence=intention
```


#### C1_flat_triples

**Author-time rep:**
```
(speaker | medical | lactose intolerance | suspected)
(speaker | symptoms | reaction to cheese | sometimes makes me feel bad)
(speaker | diet | reaction to yogurt | yogurt is fine)
(speaker | plans | dietary change | probably going to try cutting back)
```

**Readback:**
```
- speaker: medical: lactose intolerance [suspected]
- speaker: symptoms: reaction to cheese (sometimes makes me feel bad) (cardinality=1) [stated]
- speaker: diet: reaction to yogurt: yogurt is fine [stated]
- speaker: plans: dietary change: probably going to try cutting back [probable]
```

**Drift paraphrase source:**

> Wondering if dairy is an issue for me — cheese can bother me at times but yogurt seems OK. Thinking about easing off it.

**Drift rep:**
```
(speaker | medical | dairy sensitivity | uncertain / wondering if dairy is an issue)
(speaker | medical | reaction to cheese | can bother at times)
(speaker | medical | reaction to yogurt | seems OK)
(speaker | intentions | dietary change | thinking about easing off dairy)
```

**Drift readback:**
```
- subject: speaker [stated]
- categories: medical, intentions (cardinality=2) [stated]
- dairy sensitivity: uncertain / wondering if dairy is an issue [approximate]
- reaction to cheese: can bother at times [approximate]
- reaction to yogurt: seems OK [approximate]
- dietary change: thinking about easing off dairy [approximate]
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "health": {
    "diagnosis": {
      "lactose_intolerance": {
        "kind": "functional",
        "value": "lactose_intolerance",
        "confidence": "suspected",
        "note": "Self-suspected by speaker, not presented as a medical diagnosis"
      }
    },
    "symptoms": {
      "food_responses": {
        "kind": "set",
        "value": [
          "cheese: sometimes makes me feel bad",
          "yogurt: fine"
        ],
        "confidence": "stated",
        "cardinality": 2,
        "note": "Speaker reports subjective reactions to specific dairy foods"
      }
    },
    "intentions": {
      "planned_actions": {
        "kind": "ordered_list",
        "value": [
          "try cutting back (on dairy)"
        ],
        "confidence": "stated",
        "note": "Speaker says 'Probably going to try cutting back.'"
      }
    }
  }
}
```

**Readback:**
```
- health.diagnosis.lactose_intolerance.kind: functional [stated]
- health.diagnosis.lactose_intolerance.value: lactose_intolerance [stated]
- health.diagnosis.lactose_intolerance.confidence: suspected [suspected]
- health.diagnosis.lactose_intolerance.note: Self-suspected by speaker, not presented as a medical diagnosis [stated]
- health.symptoms.food_responses.kind: set [stated]
- health.symptoms.food_responses.value: cheese: sometimes makes me feel bad, yogurt: fine (cardinality=2) [stated]
- health.symptoms.food_responses.confidence: stated [stated]
- health.symptoms.food_responses.note: Speaker reports subjective reactions to specific dairy foods [stated]
- health.intentions.planned_actions.kind: ordered_list [stated]
- health.intentions.planned_actions.value: try cutting back (on dairy) (cardinality=1) [stated]
- health.intentions.planned_actions.confidence: stated [stated]
- health.intentions.planned_actions.note: Speaker says "Probably going to try cutting back." [stated]
```

**Drift paraphrase source:**

> Wondering if dairy is an issue for me — cheese can bother me at times but yogurt seems OK. Thinking about easing off it.

**Drift rep:**
```
{
  "food_tolerance": {
    "dairy": {
      "possible_issue": {
        "kind": "partial_functional",
        "value": "wondering_if_dairy_is_an_issue",
        "confidence": "stated",
        "note": "Subject expresses uncertainty about dairy causing problems"
      },
      "cheese_reaction": {
        "kind": "functional",
        "value": "cheese_can_bother_me_at_times",
        "confidence": "stated",
        "note": "Intermittent adverse reaction to cheese is reported"
      },
      "yogurt_reaction": {
        "kind": "functional",
        "value": "yogurt_seems_ok",
        "confidence": "stated",
        "note": "Subject reports yogurt does not seem to cause issues"
      },
      "behavior_intention": {
        "kind": "partial_functional",
        "value": "thinking_about_easing_off_dairy",
        "confidence": "stated",
        "note": "Subject is considering reducing dairy intake"
      }
    }
  }
}
```

**Drift readback:**
```
- food_tolerance.dairy.possible_issue.kind: partial_functional [stated]
- food_tolerance.dairy.possible_issue.value: wondering_if_dairy_is_an_issue [stated]
- food_tolerance.dairy.possible_issue.note: Subject expresses uncertainty about dairy causing problems [stated]
- food_tolerance.dairy.possible_issue.confidence: stated [stated]
- food_tolerance.dairy.cheese_reaction.kind: functional [stated]
- food_tolerance.dairy.cheese_reaction.value: cheese_can_bother_me_at_times [stated]
- food_tolerance.dairy.cheese_reaction.note: Intermittent adverse reaction to cheese is reported [stated]
- food_tolerance.dairy.cheese_reaction.confidence: stated [stated]
- food_tolerance.dairy.yogurt_reaction.kind: functional [stated]
- food_tolerance.dairy.yogurt_reaction.value: yogurt_seems_ok [stated]
- food_tolerance.dairy.yogurt_reaction.note: Subject reports yogurt does not seem to cause issues [stated]
- food_tolerance.dairy.yogurt_reaction.confidence: stated [stated]
- food_tolerance.dairy.behavior_intention.kind: partial_functional [stated]
- food_tolerance.dairy.behavior_intention.value: thinking_about_easing_off_dairy [stated]
- food_tolerance.dairy.behavior_intention.note: Subject is considering reducing dairy intake [stated]
- food_tolerance.dairy.behavior_intention.confidence: stated [stated]
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Possible lactose intolerance (speaker)

### Medical
- The speaker thinks they might be lactose intolerant. [stated]
- Cheese sometimes makes the speaker feel bad. [stated]
- Yogurt does not make the speaker feel bad. [stated]

### Actions / Intent
- The speaker is probably going to try cutting back. [stated]
```

**Readback:**
```
- Lactose intolerance belief: speaker thinks they might be lactose intolerant [stated]
- Cheese effect: sometimes makes the speaker feel bad [stated]
- Yogurt effect: NOT makes the speaker feel bad [negated]
- Intended action: speaker is probably going to try cutting back [stated]
```

**Drift paraphrase source:**

> Wondering if dairy is an issue for me — cheese can bother me at times but yogurt seems OK. Thinking about easing off it.

**Drift rep:**
```
## Subject

### Medical
- Wonders whether dairy is an issue for them [stated].
- Cheese can bother them at times [stated].
- Yogurt seems okay for them [stated].

### Preferences / Diet
- Dairy products: cheese, yogurt (2 total) [stated].
- Thinking about easing off dairy intake [stated].
```

**Drift readback:**
```
Wonders whether dairy is an issue for them: yes [stated]
Cheese can bother them at times: yes [stated]
Yogurt seems okay for them: yes [stated]
Dairy products: cheese, yogurt (cardinality=2) [stated]
Thinking about easing off dairy intake: yes [stated]
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 4, "cardinality": 3, "negation": 5, "note": "Captures all factual items and the hedged diagnosis and plan, but mildly mislabels the cheese confidence as 'stated' rather than partial and mixes item cardinality across entries."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 4, "cardinality": 5, "negation": 5, "note": "Includes all facts, preserves diagnosis hedge and item-level details including cardinalities, but labels the planned action confidence as 'stated' instead of reflecting the speaker's 'probably' hedge."},
  "C_C3_dossier_markdown": {"fidelity": 4, "confidence": 4, "cardinality": 2, "negation": 5, "note": "Generally accurate and explicitly negates yogurt effects, but downplays the suspected hedge on diagnosis labeling and fails to represent set cardinality of reported foods."},
  "overall_winner": "B",
  "overall_justification": "Representation B most completely and precisely preserves facts and cardinality while largely retaining hedges; A is close but muddles confidence/cardinality slightly and C omits cardinality and weakens the diagnosis hedge."
}
```


### set_valued_pets

**Source:** `My house is a zoo right now. We've got two cats, Luna and Milo, a pitbull named Rex, and my daughter's hamster. The fish died last month so we're down from six to five.`

**Truth:**
```
  - subject=user | attribute=pets_count | value=5 currently (was 6) | confidence=stated
  - subject=user | attribute=pets | value=Luna (cat), Milo (cat), Rex (pitbull dog), daughter's hamster (unnamed) (cardinality=set) | confidence=stated
  - subject=user | attribute=former_pet | value=fish (died last month) | confidence=stated
  - REL: user -[has_child]-> daughter
  - REL: daughter -[owns_pet]-> hamster
```


#### C1_flat_triples

**Author-time rep:**
```
(household | description | state | "a zoo right now")
(household | pets count | current total | 5)
(household | pets count | previous total | 6)
(household | cats | count | 2)
(household | cats | names | Luna; Milo)
(household | dog | breed | pitbull)
(household | dog | name | Rex)
(household | hamster | owner | my daughter)
(household | hamster | existence | present)
(household | fish | status | died last month)
```

**Readback:**
```
- household description: a zoo right now [stated]
- household pets count: 5 [stated]
- household pets count previous total: 6 [stated]
- household pets count: 5; retracted 6 [corrected]
- household cats count: 2 [stated]
- household cats names: Luna, Milo (cardinality=2) [stated]
- household dog breed: pitbull [stated]
- household dog name: Rex [stated]
- household hamster owner: my daughter [stated]
- household hamster existence: present [stated]
- household fish status: died last month [stated]
```

**Drift paraphrase source:**

> Right now we have five pets — the two cats (Luna, Milo), Rex the pitbull, and my daughter's hamster. We used to have six before our fish died last month.

**Drift rep:**
```
(household | pets | current_count | 5)
(household | pets | cats_count | 2)
(household | pets | cat_names | Luna, Milo)
(household | pets | dog | Rex (pitbull))
(household | pets | daughter's_hamster | present (name not given))
(household | pets | former_count_before_death | 6)
(household | pets | fish | died last month)
```

**Drift readback:**
```
- current_count: 5 [stated]
- cats_count: 2 [stated]
- cat_names: Luna, Milo (cardinality=2) [stated]
- dog: Rex (pitbull) (cardinality=1) [stated]
- daughter's_hamster: present (name not given) (cardinality=1) [stated]
- former_count_before_death: 6 [stated]
- fish: died last month [stated]
```

**Retrieval new statement:**

> I just got a hamster for my son's birthday.

**Retrieval output:**
```
RELEVANT: (household | hamster | owner | my daughter) | New statement says the hamster was gotten for the user's son, which conflicts with the stored owner being the user's daughter (owner info may need update or clarification).

RELEVANT: (household | hamster | existence | present) | Stored memory already indicates a hamster is present; new statement confirms acquisition of a hamster and may refer to the same or a different hamster (existence status may need consolidation).

RELEVANT: (household | pets count | current total | 5) | Acquiring a new hamster could change the household pet count (this stored total may need to be updated).
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "household": {
    "home": {
      "description": {
        "kind": "functional",
        "value": "My house is a zoo right now.",
        "confidence": "stated",
        "note": "figurative statement made by speaker"
      }
    },
    "pets": {
      "pet_count": {
        "kind": "functional",
        "value": "5",
        "confidence": "stated",
        "note": "Speaker: 'The fish died last month so we're down from six to five.'"
      },
      "previous_pet_count": {
        "kind": "functional",
        "value": "6",
        "confidence": "stated",
        "note": "Count before the fish died"
      },
      "current_pets": {
        "kind": "set",
        "value": [
          "cat:Luna",
          "cat:Milo",
          "dog:pitbull:Rex",
          "hamster:daughter's_hamster"
        ],
        "confidence": "stated",
        "cardinality": 5,
        "note": "Four named/identified pets are listed; speaker states total is five, so one current pet is unnamed/unlisted"
      },
      "deceased_pets": {
        "kind": "set",
        "value": [
          "fish"
        ],
        "confidence": "stated",
        "cardinality": 1,
        "note": "Speaker: 'The fish died last month'"
      }
    }
  }
}
```

**Readback:**
```
- home.description: "My house is a zoo right now." [stated]
- home.description.note: figurative statement made by speaker [stated]
- pets.pet_count: 5 (was=6) [stated]
- pets.previous_pet_count: 6 [stated]
- pets.current_pets: cat:Luna, cat:Milo, dog:pitbull:Rex, hamster:daughter's_hamster (cardinality=5, one current pet unnamed/unlisted) [stated]
- pets.current_pets_listed_count: 4 (cardinality of listed/identified pets) [stated]
- pets.unnamed_current_pets: 1 (inferred from total count minus listed pets) [stated]
- pets.deceased_pets: fish (cardinality=1) [stated]
- pets.deceased_pets.timing: fish died last month [stated]
```

**Drift paraphrase source:**

> Right now we have five pets — the two cats (Luna, Milo), Rex the pitbull, and my daughter's hamster. We used to have six before our fish died last month.

**Drift rep:**
```
{
  "pets": {
    "counts": {
      "number_current_pets": {
        "kind": "functional",
        "value": "5",
        "confidence": "stated",
        "note": "Text explicitly states 'Right now we have five pets.'"
      },
      "number_previous_pets_before_death": {
        "kind": "functional",
        "value": "6",
        "confidence": "stated",
        "note": "Text states 'We used to have six before our fish died last month.'"
      }
    },
    "current": {
      "names": {
        "kind": "set",
        "value": [
          "Luna",
          "Milo",
          "Rex",
          "daughter's hamster",
          "one other (unspecified)"
        ],
        "cardinality": 5,
        "confidence": "stated",
        "note": "Text names Luna, Milo, Rex, and 'my daughter's hamster' and also states a total of five pets, implying one current pet is not named in the text."
      },
      "species": {
        "kind": "set",
        "value": [
          "cat",
          "cat",
          "pitbull (dog)",
          "hamster",
          "unspecified"
        ],
        "cardinality": 5,
        "confidence": "stated",
        "note": "Text specifies two cats, a pitbull (Rex), and a daughter's hamster; total count implies one additional unspecified pet."
      },
      "ownership_relations": {
        "kind": "set",
        "value": [
          "daughter_owns_a_hamster"
        ],
        "confidence": "stated",
        "note": "Text says 'my daughter's hamster.'"
      }
    },
    "former": {
      "pets_lost": {
        "kind": "set",
        "value": [
          "fish (died last month)"
        ],
        "cardinality": 1,
        "confidence": "stated",
        "note": "Text: 'We used to have six before our fish died last month.'"
      },
      "time_of_death": {
        "kind": "functional",
        "value": "last month",
        "confidence": "stated",
        "note": "Refers specifically to the fish's death."
      }
    }
  }
}
```

**Drift readback:**
```
- number_current_pets: 5 [stated]
- number_previous_pets_before_death: 6 [stated]
- current_names: Luna, Milo, Rex, daughter's hamster, one other (unspecified) (cardinality=5) [stated]
- current_species: cat, cat, pitbull (dog), hamster, unspecified (cardinality=5) [stated]
- ownership_relations: daughter_owns_a_hamster (cardinality=1) [stated]
- pets_lost: fish (died last month) (cardinality=1) [stated]
- time_of_death: last month [stated]
```

**Retrieval new statement:**

> I just got a hamster for my son's birthday.

**Retrieval output:**
```
RELEVANT: "pet_count": { "kind": "functional", "value": "5", "confidence": "stated", "note": "Speaker: 'The fish died last month so we're down from six to five.'" } | Relevant because getting a new hamster likely changes the household pet count (needs updating from 5 to reflect the new pet).

RELEVANT: "current_pets": { "kind": "set", "value": [ "cat:Luna", "cat:Milo", "dog:pitbull:Rex", "hamster:daughter's_hamster" ], "confidence": "stated", "cardinality": 5, "note": "Four named/identified pets are listed; speaker states total is five, so one current pet is unnamed/unlisted" } | Relevant because a newly acquired hamster for the son may need to be added to this set (or replace/duplicate the existing hamster entry); the cardinality and note about an unnamed pet may need updating.

RELEVANT: "pet_count": { "kind": "functional", "value": "5", "confidence": "stated", "note": "Speaker: 'The fish died last month so we're down from six to five.'" } | Relevant because the new hamster could affect the previous vs current count relationship noted here (may change from five back to six).
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Household

### Summary
- House described as "a zoo" [stated].

### Pets
- Cats: Luna, Milo (2 total) [stated].
- Dog: Rex; breed pitbull [stated].
- Hamster: my daughter's hamster [stated].
- Fish: died last month [stated].
- Number of pets: currently five; previously six [corrected].
```

**Readback:**
```
- Household description: "a zoo" [stated]
- Cats: Luna, Milo (cardinality=2) [stated]
- Dog: Rex [stated]
- Dog breed: pitbull [stated]
- Hamster: my daughter's hamster [stated]
- Fish: died last month [stated]
- Number of pets: five; retracted six [corrected]
```

**Drift paraphrase source:**

> Right now we have five pets — the two cats (Luna, Milo), Rex the pitbull, and my daughter's hamster. We used to have six before our fish died last month.

**Drift rep:**
```
## Pets

### Inventory
- Number of current pets: five [stated]
- Current pets: Luna, Milo, Rex (pitbull), my daughter's hamster (4 total); previously claimed five [corrected]
- Cat names: Luna, Milo (2 total) [stated]
- Dog: Rex the pitbull [stated]
- Hamster: the hamster belongs to the speaker's daughter [stated]

### History
- Previously had a fish among the pets [stated]
- Fish status: died last month [stated]
- Number of pets before the fish died: six [stated]

### Relationships
- Speaker has a daughter who owns the hamster [stated]
```

**Drift readback:**
```
- Number of current pets: five [stated]
- Current pets: Luna, Milo, Rex (pitbull), my daughter's hamster (cardinality=4, was=5) [corrected]
- Cat names: Luna, Milo (cardinality=2) [stated]
- Dog: Rex the pitbull [stated]
- Hamster: the hamster belongs to the speaker's daughter [stated]
- Previously had a fish among the pets: yes [stated]
- Fish status: died last month [stated]
- Number of pets before the fish died: six [stated]
- Speaker has a daughter who owns the hamster: yes [stated]
```

**Retrieval new statement:**

> I just got a hamster for my son's birthday.

**Retrieval output:**
```
RELEVANT: Hamster: my daughter's hamster [stated]. | The new statement says the user got a hamster for their son, which may contradict, update, or add to the existing fact that the household has a hamster belonging to the daughter.

RELEVANT: Number of pets: currently five; previously six [corrected]. | Acquiring a new hamster could change the current total number of pets, so this stored count may need to be updated.
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Representation A captures all facts (names, species, counts, deceased fish) and marks the corrected count and stated status clearly."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Representation B preserves all facts, adds a useful figurative note and correctly encodes current vs previous counts including the inferred unnamed pet."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Representation C includes the same core facts (names, species, dog breed, hamster ownership, fish died) and records the corrected pet count."},
  "overall_winner": "tie",
  "overall_justification": "All three representations faithfully preserve the source facts, confidence/stance, cardinalities (including the count change), and the correction; none shows a clear advantage over the others."
}
```


### set_valued_allergies

**Source:** `I'm allergic to peanuts and shellfish - the shellfish is the bad one, EpiPen territory. I used to be allergic to eggs as a kid but grew out of it.`

**Truth:**
```
  - subject=user | attribute=active_allergies | value=peanuts, shellfish (cardinality=set) | confidence=stated
  - subject=user | attribute=allergy_severity:shellfish | value=anaphylaxis-risk (EpiPen) | confidence=stated
  - subject=user | attribute=former_allergy | value=eggs (outgrown) | confidence=stated
```


#### C1_flat_triples

**Author-time rep:**
```
(speaker | medical | allergy | peanuts)
(speaker | medical | allergy | shellfish — severe; "EpiPen territory")
(speaker | medical | past allergy | eggs — allergic as a kid; grew out of it)
```

**Readback:**
```
- allergies: peanuts, shellfish (severe, "EpiPen territory") (cardinality=2, was=3) [stated]
- allergy: peanuts [stated]
- allergy: shellfish — severe; "EpiPen territory" [stated]
- past allergy: eggs — allergic as a kid; grew out of it [stated]
- allergy to eggs: NOT allergic; retracted allergic as a kid [corrected]
```

**Retrieval new statement:**

> Found out I'm also allergic to tree nuts now — not as bad as shellfish though.

**Retrieval output:**
```
RELEVANT: (speaker | medical | allergy | peanuts) | Relevant because the new statement adds another food allergy (tree nuts), which is related to other food allergies (peanuts) in the user's allergy profile.

RELEVANT: (speaker | medical | allergy | shellfish — severe; "EpiPen territory") | Relevant because the user compares severity of the new tree nut allergy to their existing severe shellfish allergy, indicating the tree nut allergy is less severe than the stored shellfish allergy.

RELEVANT: (speaker | medical | past allergy | eggs — allergic as a kid; grew out of it) | Relevant as part of the user's broader history of food allergies; the new tree nut allergy updates the pattern of current versus past allergies.
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "health": {
    "allergies": {
      "current_allergies": {
        "kind": "set",
        "value": [
          "peanuts",
          "shellfish"
        ],
        "confidence": "stated",
        "cardinality": 2
      },
      "severe_allergies": {
        "kind": "set",
        "value": [
          "shellfish"
        ],
        "confidence": "stated",
        "cardinality": 1,
        "note": "Described as 'EpiPen territory' indicating high severity."
      },
      "egg_allergy_status": {
        "kind": "functional",
        "value": "no",
        "confidence": "corrected",
        "note": "Previously allergic to eggs as a child but grew out of it."
      },
      "past_allergies": {
        "kind": "set",
        "value": [
          "eggs"
        ],
        "confidence": "stated",
        "cardinality": 1,
        "note": "Allergy occurred in childhood and was outgrown."
      }
    }
  }
}
```

**Readback:**
```
- health.allergies.current_allergies: peanuts, shellfish (cardinality=2) [stated]
- health.allergies.severe_allergies: shellfish (cardinality=1) [stated]
- health.allergies.severe_allergies.note: Described as 'EpiPen territory' indicating high severity [stated]
- health.allergies.egg_allergy_status: no; retracted allergic (previous childhood allergy) [corrected]
- health.allergies.past_allergies: eggs (cardinality=1) [stated]
- health.allergies.past_allergies.note: Allergy occurred in childhood and was outgrown [stated]
```

**Retrieval new statement:**

> Found out I'm also allergic to tree nuts now — not as bad as shellfish though.

**Retrieval output:**
```
RELEVANT: "current_allergies": { "kind": "set", "value": [ "peanuts", "shellfish" ], "confidence": "stated", "cardinality": 2 } | The user newly reports an additional allergy (tree nuts), so the current_allergies set, its value list, confidence, and cardinality would need updating to include tree nuts and adjust cardinality.

RELEVANT: "severe_allergies": { "kind": "set", "value": [ "shellfish" ], "confidence": "stated", "cardinality": 1, "note": "Described as 'EpiPen territory' indicating high severity." } | The user states tree nut allergy is "not as bad as shellfish," which bears on severity classification; this fact confirms shellfish remains severe and suggests tree nuts should not be added to severe_allergies.
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Allergies

### Medical
- Allergic to: peanuts, shellfish (2 total) [stated]
- Shellfish allergy severity: severe and potentially requires epinephrine ("EpiPen territory") [stated]
- Previously allergic to: eggs (1 total) [stated]
- Egg allergy status: resolved; grew out of it during childhood [stated]
```

**Readback:**
```
- Allergic to: peanuts, shellfish (cardinality=2) [stated]
- Shellfish allergy severity: severe; potentially requires epinephrine ("EpiPen territory") (cardinality=1) [stated]
- Previously allergic to: eggs (cardinality=1) [stated]
- Egg allergy status: resolved; grew out of it during childhood [stated]
- Egg allergy status: resolved; retracted allergic to eggs [corrected]
- Allergic to eggs: NOT allergic to eggs [negated]
```

**Retrieval new statement:**

> Found out I'm also allergic to tree nuts now — not as bad as shellfish though.

**Retrieval output:**
```
RELEVANT: - Allergic to: peanuts, shellfish (2 total) [stated] | The new statement adds a tree nut allergy, which affects the user's list of allergies and changes the count; this stored fact about current allergies will need to be updated to include tree nuts.
RELEVANT: - Shellfish allergy severity: severe and potentially requires epinephrine ("EpiPen territory") [stated] | The new statement compares the tree nut allergy severity to shellfish ("not as bad as shellfish"), so the stored severity of the shellfish allergy is relevant context for understanding the relative seriousness.
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Accurately records current allergies (peanuts, shellfish), severity for shellfish, past egg allergy and explicitly marks the egg allergy as retracted while preserving prior count."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Structured hierarchy cleanly captures current vs past allergies, marks shellfish as severe, and explicitly records that the egg allergy was retracted."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Clearly lists current allergies and shellfish severity, notes the past egg allergy and its resolution, and includes explicit negation that the user is no longer allergic to eggs."},
  "overall_winner": "tie",
  "overall_justification": "All three representations preserve the same set of facts, confidence/stance (stated vs retracted), cardinalities, and explicit negation of the egg allergy, so no representation is superior."
}
```


### correction_retraction

**Source:** `User: I work at Google. [next turn] Actually wait, sorry - I used to work at Google, I've been at Anthropic for the last year and a half.`

**Truth:**
```
  - subject=user | attribute=current_employer | value=Anthropic | confidence=corrected/final
  - subject=user | attribute=tenure_at_current_employer | value=~1.5 years | confidence=stated
  - subject=user | attribute=former_employer | value=Google | confidence=stated
  - RETRACTED: attribute=current_employer previously_stated=Google
```


#### C1_flat_triples

**Author-time rep:**
```
(User | employment | former employer | Google)
(User | employment | current employer | Anthropic)
(User | employment | tenure at current employer | for the last year and a half)
```

**Readback:**
```
- former employer: Google [stated]
- current employer: Anthropic [stated]
- tenure at current employer: for the last year and a half (approx. 1.5 years) [stated]
```

**Retrieval new statement:**

> Actually I took a new role at OpenAI last week.

**Retrieval output:**
```
RELEVANT: (User | employment | current employer | Anthropic) | Contradicted/changed by the new statement that the user took a new role at OpenAI (implies current employer may now be OpenAI rather than Anthropic).

RELEVANT: (User | employment | tenure at current employer | for the last year and a half) | Potentially outdated because if the user started a new role at OpenAI last week, their tenure at Anthropic (the stored current employer) may have ended; the duration claim may no longer apply.
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "user": {
    "employment": {
      "current_employer": {
        "kind": "partial_functional",
        "value": "Anthropic",
        "confidence": "stated",
        "note": "User: \"I've been at Anthropic for the last year and a half.\""
      },
      "current_employer_duration": {
        "kind": "functional",
        "value": "a year and a half",
        "confidence": "stated",
        "note": "Phrase: \"for the last year and a half.\""
      },
      "previous_employers": {
        "kind": "set",
        "value": [
          "Google"
        ],
        "cardinality": 1,
        "confidence": "stated",
        "note": "User: \"I used to work at Google.\""
      },
      "retracted_initial_claim": {
        "kind": "functional",
        "value": "Anthropic",
        "confidence": "corrected",
        "note": "Initial statement was \"I work at Google.\" which was then corrected to indicate the user used to work at Google and is currently at Anthropic."
      }
    }
  }
}
```

**Readback:**
```
- current_employer: Anthropic [stated]
- current_employer_duration: a year and a half [stated]
- previous_employers: Google (cardinality=1) [stated]
- retracted_initial_claim: Anthropic; retracted "I work at Google." [corrected]
- current_employer: NOT Google [negated]
```

**Retrieval new statement:**

> Actually I took a new role at OpenAI last week.

**Retrieval output:**
```
RELEVANT: "current_employer": { "kind": "partial_functional", "value": "Anthropic", "confidence": "stated", "note": "User: \"I've been at Anthropic for the last year and a half.\"" } | Relevant because the new statement that the user took a new role at OpenAI last week may supersede or change the user's current employer from Anthropic.

RELEVANT: "current_employer_duration": { "kind": "functional", "value": "a year and a half", "confidence": "stated", "note": "Phrase: \"for the last year and a half.\"" } | Relevant because if the user recently moved to OpenAI, the recorded duration at Anthropic may no longer be current or may need to be finalized.

RELEVANT: "previous_employers": { "kind": "set", "value": [ "Google" ], "cardinality": 1, "confidence": "stated", "note": "User: \"I used to work at Google.\"" } | Relevant because the user's employment history may need to be updated to include Anthropic as a past employer if they have left it, while Google remains a previous employer.

RELEVANT: "retracted_initial_claim": { "kind": "functional", "value": "Anthropic", "confidence": "corrected", "note": "Initial statement was \"I work at Google.\" which was then corrected to indicate the user is currently at Anthropic." } | Relevant because the record of corrections about the user's employer history may need updating in light of the new statement that the user moved to OpenAI.
```


#### C3_dossier_markdown

**Author-time rep:**
```
## User

### Employment
- Current employer: Anthropic; previously claimed Google [corrected]
- Duration at Anthropic: the last year and a half [stated]
- Former employer: Google [stated]
- Initially stated current employer: Google [stated]
```

**Readback:**
```
- Current employer: Anthropic; retracted Google [corrected]
- Duration at Anthropic: the last year and a half [stated]
- Former employer: Google [stated]
- Initially stated current employer: Google [stated]
```

**Retrieval new statement:**

> Actually I took a new role at OpenAI last week.

**Retrieval output:**
```
RELEVANT: Current employer: Anthropic; previously claimed Google [corrected] | Contradicted/needs update because the user says they took a new role at OpenAI last week, implying their current employer may no longer be Anthropic and should be updated to OpenAI (or reflect a job change).
RELEVANT: Duration at Anthropic: the last year and a half [stated] | Relevant because if the user changed employers last week, the duration at Anthropic may need to be finalized/adjusted to reflect their end date.
RELEVANT: Former employer: Google [stated] | Relevant as background employment info; remains true but may need to be preserved alongside the new OpenAI role.
RELEVANT: Initially stated current employer: Google [stated] | Relevant because it records a previous claim about current employment and may be superseded by the new OpenAI role.
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 4, "confidence": 4, "cardinality": 4, "negation": 2, "note": "Captures current employer, tenure, and former employer clearly, preserves 'stated'/approx labels, but fails to record the explicit retraction/negation of the initial claim."},
  "B_C2_typed_hierarchy": {"fidelity": 3, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Preserves confidence/negation and explicitly records cardinality, but introduces a factual confusion by labeling the retracted_initial_claim as 'Anthropic' (incorrect) which hurts fidelity."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 4, "negation": 5, "note": "Faithfully records the final employer, tenure, former employer, and the initial (retracted) claim with correct corrective labeling; it omits an explicit cardinality field but implies the single former employer."},
  "overall_winner": "C",
  "overall_justification": "Representation C most accurately and clearly preserves the facts including the retraction and confidence labels; B is strong on structure and negation but contains a factual labeling error, while A omits the explicit retraction."
}
```


### temporal_evolution

**Source:** `When I started this job I was remote from Denver. Got relocated to Chicago last April for an office role. Now as of this month I'm hybrid - three days in, two remote.`

**Truth:**
```
  - subject=user | attribute=current_city | value=Chicago | confidence=stated
  - subject=user | attribute=current_work_mode | value=hybrid 3-in-office, 2-remote | confidence=stated (as of this month)
  - subject=user | attribute=previous_city_at_job | value=Denver | confidence=stated
  - subject=user | attribute=work_mode_history | value=remote (initial) -> in-office (last April) -> hybrid (this month) | confidence=stated
```


#### C1_flat_triples

**Author-time rep:**
```
(speaker | biography | prior residence | Denver)
(speaker | employment | work arrangement at job start | remote from Denver)
(speaker | employment | relocated to | Chicago)
(speaker | employment | reason for relocation | for an office role)
(speaker | employment | relocation date | last April)
(speaker | employment | current work arrangement | hybrid)
(speaker | employment | hybrid schedule | three days in, two remote)
(speaker | employment | hybrid start date | as of this month)
```

**Readback:**
```
- prior residence: Denver (cardinality=1) [stated]
- residence: Denver, Chicago (cardinality=2, was=1) [stated]
- employment work arrangement at job start: remote from Denver (cardinality=1) [stated]
- employment relocated to: Chicago (cardinality=1) [stated]
- reason for relocation: for an office role (cardinality=1) [stated]
- relocation date: last April (cardinality=1) [stated]
- current work arrangement: hybrid (cardinality=1) [stated]
- hybrid schedule: three days in, two remote (cardinality=1) [stated]
- hybrid start date: as of this month (cardinality=1) [stated]
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "work_location_history": {
    "status": {
      "initial_location": {
        "kind": "functional",
        "value": "remote from Denver",
        "confidence": "stated",
        "note": "Describes the speaker's location when they started the job"
      },
      "relocation_destination": {
        "kind": "functional",
        "value": "Chicago",
        "confidence": "stated",
        "note": "Relocated to Chicago last April"
      },
      "relocation_date": {
        "kind": "functional",
        "value": "last April",
        "confidence": "stated",
        "note": "Relative time expression as stated in the text"
      },
      "relocation_reason": {
        "kind": "functional",
        "value": "for an office role",
        "confidence": "stated"
      },
      "current_arrangement": {
        "kind": "partial_functional",
        "value": "hybrid",
        "confidence": "stated",
        "note": "Described as 'now as of this month I'm hybrid'"
      },
      "current_arrangement_effective": {
        "kind": "functional",
        "value": "this month",
        "confidence": "stated",
        "note": "Relative time expression indicating when the hybrid arrangement began"
      },
      "hybrid_schedule": {
        "kind": "ordered_list",
        "value": [
          "three days in (office)",
          "two days remote"
        ],
        "confidence": "stated",
        "note": "Order reflects the stated schedule"
      }
    }
  }
}
```

**Readback:**
```
- initial_location: remote from Denver [stated]
- initial_location_kind: functional [stated]
- initial_location_note: Describes the speaker's location when they started the job [stated]
- relocation_destination: Chicago [stated]
- relocation_destination_kind: functional [stated]
- relocation_destination_note: Relocated to Chicago last April [stated]
- relocation_date: last April [stated]
- relocation_date_kind: functional [stated]
- relocation_date_note: Relative time expression as stated in the text [stated]
- relocation_reason: for an office role [stated]
- relocation_reason_kind: functional [stated]
- current_arrangement: hybrid [stated]
- current_arrangement_kind: partial_functional [stated]
- current_arrangement_note: Described as 'now as of this month I'm hybrid' [stated]
- current_arrangement_effective: this month [stated]
- current_arrangement_effective_kind: functional [stated]
- current_arrangement_effective_note: Relative time expression indicating when the hybrid arrangement began [stated]
- hybrid_schedule: three days in (office), two days remote (cardinality=2) [stated]
- hybrid_schedule_kind: ordered_list [stated]
- hybrid_schedule_note: Order reflects the stated schedule [stated]
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Subject

### Biography
- Started current job while remote from Denver [stated].

### Residence / Location
- Initial work location: Denver (remote) [stated].
- Relocated to Chicago last April for an office role [stated].

### Employment / Work arrangement
- Role type after relocation: office role [stated].
- Current work arrangement as of this month: hybrid, three days in-office and two days remote [stated].
```

**Readback:**
```
- Biography: Started current job while remote from Denver [stated]
- Initial work location: Denver (remote) [stated]
- Relocated to: Chicago (timeframe: last April) [stated]
- Residence/Location: Denver, Chicago (cardinality=2, was=1) [stated]
- Role type after relocation: office role [stated]
- Current work arrangement: hybrid, three days in-office and two days remote (as of this month) [stated]
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "C1 explicitly records the full location and work-mode timeline, dates, hybrid schedule, and marks items as stated, preserving counts and lack of negation."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 5, "cardinality": 4, "negation": 5, "note": "C2 preserves all facts and their stated status with rich typing and notes, but does not explicitly combine residence cardinality into a single field with an explicit 'was=1' marker (cardinality must be inferred)."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "C3 concisely captures the initial remote-from-Denver state, relocation to Chicago last April, and current hybrid schedule with an explicit cardinality note and stated markers."},
  "overall_winner": "tie",
  "overall_justification": "All three representations faithfully preserve the factual content and stated confidence; B is slightly weaker on explicit cardinality encoding, so A and C tie as the best for directly communicating counts and timeline." 
}
```


### referential_pronoun

**Source:** `(Earlier: user mentioned their sister Maya who lives in Barcelona and teaches Spanish.) User: She's coming to visit next month - I haven't seen her since COVID.`

**Truth:**
```
  - subject=user | attribute=upcoming_event | value=sister Maya visiting next month | confidence=stated
  - subject=user | attribute=last_saw_sister | value=before/during COVID (long gap) | confidence=stated
  - REL: user -[sister]-> Maya
```


#### C1_flat_triples

**Author-time rep:**
```
(Maya | relationships | relation to user | sister)
(Maya | biography | residence | Barcelona)
(Maya | biography | occupation | teaches Spanish)
(Maya | biography | gender | female)
(Maya | events | upcoming visit | next month)
(Maya | interactions | last time seen by user | since COVID)
```

**Readback:**
```
- relation to user: sister [stated]
- residence: Barcelona [stated]
- occupation: teaches Spanish [stated]
- gender: female [stated]
- upcoming visit: next month (cardinality=1) [stated]
- last time seen by user: since COVID [stated]
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "identity": {
    "name": {
      "kind": "functional",
      "value": "Maya",
      "confidence": "stated"
    },
    "gender": {
      "kind": "functional",
      "value": "female",
      "confidence": "stated",
      "note": "Referred to with pronoun 'she'"
    }
  },
  "relationship": {
    "relation_to_user": {
      "kind": "functional",
      "value": "sister",
      "confidence": "stated"
    }
  },
  "residence": {
    "city": {
      "kind": "functional",
      "value": "Barcelona",
      "confidence": "stated"
    }
  },
  "occupation": {
    "job_title": {
      "kind": "functional",
      "value": "Spanish teacher",
      "confidence": "stated",
      "note": "Described as 'teaches Spanish'"
    }
  },
  "visits": {
    "upcoming_visit": {
      "kind": "partial_functional",
      "value": "next month",
      "confidence": "stated",
      "note": "User: 'She's coming to visit next month'"
    },
    "last_time_seen": {
      "kind": "functional",
      "value": "since COVID",
      "confidence": "stated",
      "note": "User: 'I haven't seen her since COVID' (meaning they have not seen her since the COVID period)"
    }
  }
}
```

**Readback:**
```
- name: Maya [stated]
- gender: female [stated]
- gender_pronoun_reference: referred with pronoun 'she' [stated]
- relation_to_user: sister [stated]
- city: Barcelona [stated]
- job_title: Spanish teacher [stated]
- job_description: teaches Spanish [stated]
- upcoming_visit: next month [stated]
- upcoming_visit_quoted_by_user: "She's coming to visit next month" [stated]
- last_time_seen: since COVID [stated]
- seen_by_user_since_COVID: NOT yes (i.e., user has not seen her since the COVID period) [stated]
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Maya (sister)

### Biography
- Maya is the user's sister [stated].
- Maya lives in Barcelona [stated].
- Maya teaches Spanish [stated].

### Visits / Timing
- Maya is coming to visit next month [stated].
- The user has not seen Maya since COVID [stated].

### Relationship
- Relationship to subject: sibling (sister) [stated]
```

**Readback:**
```
- Name: Maya [stated]
- Relationship to user: sibling (sister) [stated]
- Residence: Barcelona [stated]
- Occupation: teaches Spanish [stated]
- Visit timing: coming to visit next month [stated]
- User has seen Maya since COVID: NOT true [stated, negated]
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 4, "confidence": 4, "cardinality": 5, "negation": 4, "note": "Captures the visit, relation, residence, job and last-seen timing but omits the sister's name explicitly and uses implied rather than explicit negation language."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 5, "cardinality": 4, "negation": 5, "note": "Includes the name, relation, visit timing, quoted source, and an explicit non-seen negation and clearly marks statements and pronoun reference, though it doesn't state visit cardinality."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 4, "cardinality": 4, "negation": 5, "note": "Contains the name, relation, location, job, visit timing and an explicit negation about seeing since COVID, but lacks an explicit confidence/quotation marker and cardinality detail."},
  "overall_winner": "B",
  "overall_justification": "Representation B most fully preserves facts, explicit confidence, and negation (including the original quote), making it the most faithful and interpretable; A is fine on cardinality but misses the name, and C is strong but less explicit about source/confidence."
}
```


### negation

**Source:** `I don't drink coffee - never have. Tea person through and through. And no, I'm not one of those people who won't touch caffeine at all, just don't like coffee specifically.`

**Truth:**
```
  - subject=user | attribute=drinks_coffee | value=no (never) | confidence=stated, emphatic
  - subject=user | attribute=preferred_beverage | value=tea | confidence=stated
  - subject=user | attribute=avoids_caffeine_generally | value=no | confidence=stated (explicit non-negation)
```


#### C1_flat_triples

**Author-time rep:**
```
(speaker | preferences | coffee consumption | never drinks coffee)
(speaker | preferences | beverage preference | prefers tea)
(speaker | preferences | attitude toward caffeine | not opposed to caffeine in general)
(speaker | preferences | opinion of coffee | specifically does not like coffee)
```

**Readback:**
```
- coffee consumption: never drinks coffee [stated]
- beverage preference: prefers tea [stated]
- attitude toward caffeine: not opposed to caffeine in general [stated]
- opinion of coffee: specifically does not like coffee [stated]
- coffee consumption: NOT drinks coffee [negated]
```

**Drift paraphrase source:**

> Coffee's not my thing — never touched it. I'm strictly tea. That said I do consume caffeine elsewhere; it's specifically coffee I dislike.

**Drift rep:**
```
(speaker | preferences | beverage preference | strictly tea)
(speaker | preferences | coffee attitude | dislikes coffee)
(speaker | consumption | coffee consumption history | never touched coffee)
(speaker | consumption | caffeine consumption | consumes caffeine from non-coffee sources)
```

**Drift readback:**
```
- beverage preference: strictly tea [stated]
- coffee attitude: dislikes coffee [stated]
- coffee attitude: NOT likes coffee [negated]
- coffee consumption history: NOT has ever touched coffee [negated]
- caffeine consumption: consumes caffeine from non-coffee sources [stated]
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "beverages": {
    "coffee": {
      "drinks_coffee": {
        "kind": "functional",
        "value": "no",
        "confidence": "negated",
        "note": "Speaker states 'I don't drink coffee - never have.'"
      },
      "history": {
        "kind": "functional",
        "value": "never_have",
        "confidence": "stated",
        "note": "Explicit phrasing 'never have.'"
      },
      "liking": {
        "kind": "functional",
        "value": "dislike",
        "confidence": "stated",
        "note": "Speaker says 'just don't like coffee specifically.'"
      }
    },
    "tea": {
      "preferred_beverage": {
        "kind": "functional",
        "value": "tea",
        "confidence": "stated",
        "note": "Speaker: 'Tea person through and through.'"
      }
    },
    "caffeine": {
      "avoids_all_caffeine": {
        "kind": "functional",
        "value": "no",
        "confidence": "negated",
        "note": "Speaker: 'And no, I'm not one of those people who won't touch caffeine at all.'"
      },
      "consumes_caffeine_in_non_coffee_forms": {
        "kind": "functional",
        "value": "yes",
        "confidence": "stated",
        "note": "Implied by rejecting total caffeine avoidance while not drinking coffee."
      }
    }
  }
}
```

**Readback:**
```
- coffee.drinks_coffee: NOT no [negated]
- coffee.history: never_have [stated]
- coffee.liking: dislike [stated]
- tea.preferred_beverage: tea [stated]
- caffeine.avoids_all_caffeine: NOT no [negated]
- caffeine.consumes_caffeine_in_non_coffee_forms: yes [stated]
```

**Drift paraphrase source:**

> Coffee's not my thing — never touched it. I'm strictly tea. That said I do consume caffeine elsewhere; it's specifically coffee I dislike.

**Drift rep:**
```
{
  "beverages": {
    "preferences": {
      "coffee_preference": {
        "kind": "functional",
        "value": "dislike",
        "confidence": "stated",
        "note": "Speaker: 'specifically coffee I dislike.'"
      },
      "tea_preference": {
        "kind": "functional",
        "value": "strictly tea",
        "confidence": "stated",
        "note": "Phrase: 'I'm strictly tea.'"
      }
    },
    "consumption": {
      "caffeine_consumption": {
        "kind": "partial_functional",
        "value": "consumes caffeine from sources other than coffee",
        "confidence": "stated",
        "note": "Phrase: 'I do consume caffeine elsewhere.'"
      },
      "coffee_ever_tasted": {
        "kind": "functional",
        "value": "no",
        "confidence": "stated",
        "note": "Phrase: 'never touched it.' (refers to coffee)"
      }
    }
  }
}
```

**Drift readback:**
```
- beverages.preferences.coffee_preference.kind: functional [stated]
- beverages.preferences.coffee_preference.value: dislike [stated]
- beverages.preferences.coffee_preference.note: "Speaker: 'specifically coffee I dislike.'" [stated]
- beverages.preferences.coffee_preference.confidence: stated [stated]
- beverages.preferences.tea_preference.kind: functional [stated]
- beverages.preferences.tea_preference.value: strictly tea [stated]
- beverages.preferences.tea_preference.note: "Phrase: 'I'm strictly tea.'" [stated]
- beverages.preferences.tea_preference.confidence: stated [stated]
- beverages.consumption.caffeine_consumption.kind: partial_functional [stated]
- beverages.consumption.caffeine_consumption.value: consumes caffeine from sources other than coffee [stated]
- beverages.consumption.caffeine_consumption.note: "Phrase: 'I do consume caffeine elsewhere.'" [stated]
- beverages.consumption.caffeine_consumption.confidence: stated [stated]
- beverages.consumption.coffee_ever_tasted.kind: functional [stated]
- beverages.consumption.coffee_ever_tasted: NOT yes [negated]
- beverages.consumption.coffee_ever_tasted.note: "Phrase: 'never touched it.' (refers to coffee)" [stated]
- beverages.consumption.coffee_ever_tasted.confidence: stated [stated]
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Speaker

### Preferences
- Does NOT drink coffee: never have [negated]
- Is a tea person through and through [stated]
- Does NOT avoid caffeine entirely: not one of those people who won't touch caffeine at all [stated]
- Dislikes coffee specifically [stated]
```

**Readback:**
```
- Drinks coffee: NOT ever (has never drunk coffee) [negated]
- Primary beverage preference: tea (tea person through and through) [stated]
- Avoids caffeine entirely: NOT true (does not avoid caffeine entirely) [stated]
- Attitude toward coffee: dislikes coffee specifically [stated]
```

**Drift paraphrase source:**

> Coffee's not my thing — never touched it. I'm strictly tea. That said I do consume caffeine elsewhere; it's specifically coffee I dislike.

**Drift rep:**
```
## Subject

### Preferences
- Does NOT drink coffee: never touched coffee [stated]
- Prefers tea exclusively: "strictly tea" [stated]
- Consumes caffeine in other forms besides coffee [stated]
- Specifically dislikes coffee (distinct from other caffeine sources) [stated]
```

**Drift readback:**
```
drinks coffee: NOT ever; "never touched coffee" [negated]
beverage preference: tea exclusively ("strictly tea") [stated]
consumes caffeine in other forms besides coffee: yes [stated]
dislikes coffee: specifically dislikes coffee (distinct from other caffeine sources) [stated]
dislike_scope: applies to coffee and not to other caffeine sources [stated]
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Accurately records that the user never drinks coffee, prefers tea, and does not avoid caffeine generally, with clear stated labels (though it redundantly repeats the coffee negation)."},
  "B_C2_typed_hierarchy": {"fidelity": 3, "confidence": 2, "cardinality": 4, "negation": 2, "note": "Contains the relevant facts but uses confusing double-negation tokens (e.g., 'NOT no') that make the coffee and caffeine negations ambiguous and reduces confidence in the values."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Clean, unambiguous restatement: never drinks coffee, tea preferred, and does not avoid caffeine entirely, with correct negation and stated confidence markers."},
  "overall_winner": "tie",
  "overall_justification": "A and C both preserve all facts, confidence, cardinality, and negation clearly; B is ambiguous due to inconsistent negation encoding, so A and C tie for best." 
}
```


### uncertain_inference

**Source:** `Looked at a house in Cambridge yesterday. Don't know, it was nice but the kitchen's tiny and the commute would be brutal. My wife loved it though.`

**Truth:**
```
  - subject=user | attribute=viewed_property | value=house in Cambridge (yesterday) | confidence=stated
  - subject=user | attribute=opinion_on_house | value=ambivalent (nice but concerns) | confidence=stated
  - subject=user | attribute=concerns_about_house | value=small kitchen, long commute (cardinality=set) | confidence=stated
  - subject=wife | attribute=opinion_on_house | value=positive | confidence=stated
  - REL: user -[married_to]-> wife
```


#### C1_flat_triples

**Author-time rep:**
```
(speaker | activity | viewed_property | house in Cambridge)
(house | location | city | Cambridge)
(house | feature | kitchen_size | tiny)
(house | commute | quality | would be brutal)
(speaker | relationship | spouse | wife)
(house | condition | overall_impression | nice)
(speaker | temporal_activity | viewing_date | yesterday)
```

**Readback:**
```
- speaker activity: viewed property (house in Cambridge) [stated]
- house location city: Cambridge [stated]
- house feature kitchen_size: tiny [stated]
- house commute quality: would be brutal [stated]
- speaker relationship spouse: wife [stated]
- house condition overall_impression: nice [stated]
- speaker temporal_activity viewing_date: yesterday [stated]
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "viewing": {
    "visit": {
      "date": {
        "kind": "functional",
        "value": "yesterday",
        "confidence": "stated",
        "note": "Relative to the time of the utterance"
      },
      "location": {
        "kind": "functional",
        "value": "Cambridge",
        "confidence": "stated"
      }
    }
  },
  "property": {
    "description": {
      "condition": {
        "kind": "functional",
        "value": "nice",
        "confidence": "stated",
        "note": "Speaker said 'it was nice'"
      },
      "kitchen_size": {
        "kind": "functional",
        "value": "tiny",
        "confidence": "stated",
        "note": "Speaker said 'the kitchen's tiny'"
      },
      "commute_assessment": {
        "kind": "functional",
        "value": "would be brutal",
        "confidence": "stated",
        "note": "Speaker's assessment of the commute if they were to live there"
      }
    }
  },
  "people": {
    "spouse": {
      "reaction": {
        "kind": "functional",
        "value": "loved it",
        "confidence": "stated",
        "note": "Speaker: 'My wife loved it though.'"
      }
    }
  },
  "speaker": {
    "sentiment_and_uncertainty": {
      "sentiment": {
        "kind": "functional",
        "value": "unsure",
        "confidence": "stated",
        "note": "Expressed uncertainty with 'Don't know'"
      }
    }
  }
}
```

**Readback:**
```
- viewing.visit.date: yesterday [stated]
- viewing.visit.date.kind: functional [stated]
- viewing.visit.date.note: Relative to the time of the utterance [stated]
- viewing.visit.location: Cambridge [stated]
- viewing.visit.location.kind: functional [stated]
- property.description.condition: nice [stated]
- property.description.condition.kind: functional [stated]
- property.description.condition.note: Speaker said 'it was nice' [stated]
- property.description.kitchen_size: tiny [stated]
- property.description.kitchen_size.kind: functional [stated]
- property.description.kitchen_size.note: Speaker said 'the kitchen's tiny' [stated]
- property.description.commute_assessment: would be brutal [stated]
- property.description.commute_assessment.kind: functional [stated]
- property.description.commute_assessment.note: Speaker's assessment of the commute if they were to live there [stated]
- people.spouse.reaction: loved it [stated]
- people.spouse.reaction.kind: functional [stated]
- people.spouse.reaction.note: Speaker: 'My wife loved it though.' [stated]
- speaker.sentiment_and_uncertainty.sentiment: unsure [stated]
- speaker.sentiment_and_uncertainty.sentiment.kind: functional [stated]
- speaker.sentiment_and_uncertainty.sentiment.note: Expressed uncertainty with 'Don't know' [stated]
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Subject

### Real estate
- Looked at a house in Cambridge yesterday. [stated]
- The house's kitchen is tiny. [stated]
- The commute for that house would be brutal. [stated]

### Opinions / Decision
- Undecided about the house. [stated]

### Relationships
- The subject has a wife. [stated]
- The wife loved the house. [stated]
```

**Readback:**
```
- Action: Looked at a house in Cambridge yesterday [stated]
- House feature: kitchen is tiny [stated]
- Commute assessment: commute for that house would be brutal [stated]
- Decision status: Undecided about the house [stated]
- Relationship: has a wife (cardinality=1) [stated]
- Wife's opinion: wife loved the house [stated]
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 3, "confidence": 1, "cardinality": 5, "negation": 5, "note": "Captures most explicit facts (viewing, location, tiny kitchen, brutal commute, 'nice', spouse relation and date) but omits the wife's positive reaction and does not record the speaker's ambivalence as a combined judgment."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Preserves all facts including wife's positive reaction and explicitly encodes speaker uncertainty and contextual notes, so meaning and qualifiers are fully recoverable."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Clearly records the viewing, location, both concerns, wife's positive opinion, and the speaker's undecided/uncertain stance, so the original facts are intact and recoverable."},
  "overall_winner": "tie",
  "overall_justification": "B and C both preserve the factual content and the speaker's uncertainty and cardinalities; A omits the wife's opinion and the expressed ambivalence, so it scores lower."
}
```


### novel_chunk

**Source:** `Eleanor Vance had never been outside of Yorkshire until the age of thirty-two. She was a bookseller, thin as a rake, with the sort of watchful gray eyes that made children quiet down without being told. The shop - her father's before it was hers - sat on the corner of Gillygate and had not, in living memory, turned a profit in any month that did not contain the word 'December'. Eleanor loved it anyway; or rather, she was bound to it the way one is bound to a difficult relative.`

**Truth:**
```
  - subject=Eleanor Vance | attribute=region | value=Yorkshire | confidence=stated
  - subject=Eleanor Vance | attribute=age_at_first_travel | value=32 | confidence=stated (she hadn't left before 32)
  - subject=Eleanor Vance | attribute=occupation | value=bookseller | confidence=stated
  - subject=Eleanor Vance | attribute=appearance | value=thin, watchful gray eyes | confidence=stated
  - subject=Eleanor Vance | attribute=shop_location | value=corner of Gillygate | confidence=stated
  - subject=Eleanor Vance | attribute=shop_inheritance | value=inherited from father | confidence=stated
  - subject=Eleanor Vance | attribute=shop_profitability | value=only profitable in December | confidence=stated (figurative)
  - subject=Eleanor Vance | attribute=attachment_to_shop | value=ambivalent bond (love + obligation) | confidence=stated
```


#### C1_flat_triples

**Author-time rep:**
```
(Eleanor Vance | biography | birthplace travel history | had never been outside of Yorkshire until the age of thirty-two)
(Eleanor Vance | biography | occupation | bookseller)
(Eleanor Vance | physical | build | thin as a rake)
(Eleanor Vance | physical | eyes | watchful gray)
(Eleanor Vance | behavior | effect on children | made children quiet down without being told)
(Eleanor Vance | relationships | attachment to business | loved the shop anyway / bound to it like a difficult relative)
(Eleanor Vance's shop | ownership | previous owner | her father)
(Eleanor Vance's shop | ownership | current owner | Eleanor Vance)
(Eleanor Vance's shop | location | street corner | corner of Gillygate)
(Eleanor Vance's shop | finance | profitability history | had not, in living memory, turned a profit in any month that did not contain the word "December")
```

**Readback:**
```
- birthplace/travel history: had never been outside of Yorkshire until the age of thirty-two [stated]
- occupation: bookseller [stated]
- physical build: thin as a rake [stated]
- eye color/appearance: watchful gray [stated]
- effect on children: made children quiet down without being told [stated]
- attachment to business: loved the shop anyway, bound to it like a difficult relative (cardinality=2) [stated]
- shop previous owner: her father (cardinality=1) [stated]
- shop current owner: Eleanor Vance (cardinality=1) [stated]
- shop location: corner of Gillygate [stated]
- shop profitability history: had not, in living memory, turned a profit in any month that did not contain the word "December" [stated]
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "personal": {
    "identity": {
      "name": {
        "kind": "functional",
        "value": "Eleanor Vance",
        "confidence": "stated"
      }
    },
    "age_events": {
      "age_left_yorkshire": {
        "kind": "functional",
        "value": "thirty-two",
        "confidence": "stated",
        "note": "Text states she had never been outside of Yorkshire until the age of thirty-two"
      }
    }
  },
  "location_history": {
    "region_before_age_32": {
      "kind": "functional",
      "value": "Yorkshire",
      "confidence": "stated",
      "note": "She had never been outside of Yorkshire until age thirty-two, implying residence in Yorkshire until that age"
    }
  },
  "occupation": {
    "profession": {
      "kind": "functional",
      "value": "bookseller",
      "confidence": "stated"
    }
  },
  "shop": {
    "ownership_history": {
      "kind": "ordered_list",
      "value": [
        "father",
        "Eleanor Vance"
      ],
      "confidence": "stated",
      "note": "Text: 'The shop - her father's before it was hers'"
    },
    "location": {
      "kind": "functional",
      "value": "corner of Gillygate",
      "confidence": "stated"
    },
    "profitability_pattern": {
      "kind": "functional",
      "value": "had not, in living memory, turned a profit in any month that did not contain the word 'December'",
      "confidence": "stated",
      "note": "Quoted phrasing preserved from source"
    },
    "emotional_relation": {
      "kind": "partial_functional",
      "value": "loved it anyway; bound to it the way one is bound to a difficult relative",
      "confidence": "stated",
      "note": "Describes Eleanor's feelings toward the shop"
    }
  },
  "physical_description": {
    "build": {
      "kind": "functional",
      "value": "thin as a rake",
      "confidence": "stated"
    },
    "eyes": {
      "kind": "functional",
      "value": "watchful gray",
      "confidence": "stated"
    },
    "effect_on_children": {
      "kind": "functional",
      "value": "made children quiet down without being told",
      "confidence": "stated"
    }
  }
}
```

**Readback:**
```
- name: Eleanor Vance [stated]
- age_left_yorkshire: thirty-two [stated]
- region_before_age_32: Yorkshire [stated]
- profession: bookseller [stated]
- shop ownership_history: father, Eleanor Vance (cardinality=2) [stated]
- shop location: corner of Gillygate [stated]
- shop profitability_pattern: had not, in living memory, turned a profit in any month that did not contain the word 'December' [stated]
- shop emotional_relation: loved it anyway; bound to it the way one is bound to a difficult relative [stated]
- physical build: thin as a rake [stated]
- eyes: watchful gray [stated]
- effect_on_children: made children quiet down without being told [stated]
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Eleanor Vance

### Biography
- She had never been outside of Yorkshire until the age of thirty-two. [stated]

### Occupation
- She was a bookseller. [stated]
- The shop was her father's before it was hers. [stated]
- The shop sat on the corner of Gillygate. [stated]
- The shop had not, in living memory, turned a profit in any month that did not contain the word "December". [stated]

### Physical description
- She was thin as a rake. [stated]
- She had watchful gray eyes. [stated]
- Her eyes made children quiet down without being told. [stated]

### Personal attitudes
- She loved the shop anyway. [stated]
- She felt bound to the shop the way one is bound to a difficult relative. [stated]
```

**Readback:**
```
- birthplace_travel_history: had not been outside of Yorkshire until age 32 [stated]
- occupation: bookseller [stated]
- shop_ownership_history: the shop was her father's before it was hers [stated]
- shop_location: corner of Gillygate [stated]
- shop_profit_pattern: had not, in living memory, turned a profit in any month that did not contain the word "December" [stated]
- physical_build: thin as a rake [stated]
- eye_color: watchful gray [stated]
- eye_effect_on_children: her eyes made children quiet down without being told [stated]
- attitude_toward_shop: loved the shop anyway [stated]
- felt_bound_to_shop: felt bound to the shop the way one is bound to a difficult relative [stated]
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Captures all facts (travel, occupation, appearance, shop history, effect on children, attachment) and preserves hedging, counts, and explicit negations exactly."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "All source facts are present and correctly typed, with stated confidence, correct ownership cardinality, and preserved negated profitability/travel claims."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "Dossier lists the same facts and nuances (including negations and emotional ambivalence) with no loss of factual content or confidence."},
  "overall_winner": "tie",
  "overall_justification": "All three representations accurately preserve the source facts, their stated confidence, cardinalities, and explicit negations, so no single representation outperforms the others."
}
```


### multi_user_chat

**Source:** `Amit: Just got back from Kyoto - best trip of my life. Priya, you should go, you'd love the temples. Priya: I've actually been twice! Ryan hated it though, too crowded for him. He only likes off-the-beaten-path stuff.`

**Truth:**
```
  - subject=Amit | attribute=recent_trip | value=Kyoto | confidence=stated
  - subject=Amit | attribute=opinion_of_kyoto | value=best trip of my life | confidence=stated
  - subject=Priya | attribute=kyoto_visits_count | value=2 | confidence=stated
  - subject=Ryan | attribute=opinion_of_kyoto | value=disliked - too crowded | confidence=stated (by Priya)
  - subject=Ryan | attribute=travel_preference | value=off-the-beaten-path | confidence=stated (by Priya)
```


#### C1_flat_triples

**Author-time rep:**
```
Amit | travel | recent trip destination | Kyoto
Amit | opinion | trip quality | best trip of my life
Amit | recommendation | suggested person to visit Kyoto | Priya
Priya | travel | times visited Kyoto | 2
Ryan | opinion | reaction to Kyoto | hated it
Ryan | preferences | dislikes | crowded places
Ryan | preferences | likes | off-the-beaten-path stuff
```

**Readback:**
```
- Amit travel recent trip destination: Kyoto [stated]
- Amit opinion trip quality: best trip of my life [stated]
- Amit recommendation suggested person to visit Kyoto: Priya [stated]
- Priya travel times visited Kyoto: 2 (cardinality=2) [stated]
- Ryan opinion reaction to Kyoto: hated it [stated]
- Ryan preferences: crowded places (dislikes), off-the-beaten-path stuff (likes) (cardinality=2) [stated]
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "amit": {
    "travel": {
      "recent_trip": {
        "kind": "functional",
        "value": "Kyoto",
        "confidence": "stated",
        "note": "Says 'Just got back from Kyoto'"
      },
      "trip_sentiment": {
        "kind": "functional",
        "value": "best trip of my life",
        "confidence": "stated",
        "note": "Direct quote: 'best trip of my life.'"
      }
    },
    "communication": {
      "recommendation_to_priya": {
        "kind": "functional",
        "value": "you should go",
        "confidence": "stated",
        "note": "Amit recommends Priya visit Kyoto"
      }
    }
  },
  "priya": {
    "travel": {
      "visits_to_kyoto": {
        "kind": "functional",
        "value": "2",
        "confidence": "stated",
        "note": "Says 'I've actually been twice!' referring to Kyoto"
      },
      "visited_locations": {
        "kind": "set",
        "value": [
          "Kyoto"
        ],
        "confidence": "stated"
      }
    }
  },
  "ryan": {
    "reaction": {
      "reaction_to_kyoto": {
        "kind": "functional",
        "value": "hated it",
        "confidence": "stated",
        "note": "Priya: 'Ryan hated it though, too crowded for him.'"
      }
    },
    "preferences": {
      "crowded_places": {
        "kind": "functional",
        "value": "dislikes crowded places",
        "confidence": "stated",
        "note": "Text: 'too crowded for him.'"
      },
      "travelling_style": {
        "kind": "functional",
        "value": "likes off-the-beaten-path stuff",
        "confidence": "stated",
        "note": "Text: 'He only likes off-the-beaten-path stuff.'"
      }
    }
  }
}
```

**Readback:**
```
- amit.travel.recent_trip: Kyoto [stated]
- amit.travel.trip_sentiment: best trip of my life [stated]
- amit.communication.recommendation_to_priya: you should go [stated]
- priya.travel.visits_to_kyoto: 2 [stated]
- priya.travel.visited_locations: Kyoto (cardinality=1) [stated]
- ryan.reaction.reaction_to_kyoto: hated it [stated]
- ryan.preferences.crowded_places: dislikes crowded places [stated]
- ryan.preferences.travelling_style: likes off-the-beaten-path stuff [stated]
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Kyoto trip

### Participants
- Amit is a speaker in the conversation [confirmed].
- Priya is a speaker in the conversation [confirmed].
- Ryan is mentioned in the conversation [confirmed].

### Trips
- Amit just got back from Kyoto. [stated]
- Amit described the Kyoto trip as "best trip of my life." [stated]
- Priya has been to Kyoto twice (2 total). [stated]

### Recommendations / Statements
- Amit told Priya she should go to Kyoto. [stated]
- Amit said Priya would love the temples in Kyoto. [stated]

### Reactions / Preferences
- Ryan hated Kyoto. [stated]
- Ryan found Kyoto "too crowded." [stated]
- Ryan only likes off-the-beaten-path stuff. [stated]
```

**Readback:**
```
- Participants: Amit, Priya, Ryan (cardinality=3) [confirmed]
- Amit is a speaker in the conversation: Amit [confirmed]
- Priya is a speaker in the conversation: Priya [confirmed]
- Ryan is mentioned in the conversation: Ryan [confirmed]
- Amit just got back from Kyoto: yes [stated]
- Amit described the Kyoto trip as "best trip of my life.": "best trip of my life." [stated]
- Priya has been to Kyoto twice: 2 (cardinality=2) [stated]
- Amit told Priya she should go to Kyoto: yes [stated]
- Amit said Priya would love the temples in Kyoto: yes [stated]
- Ryan hated Kyoto: yes [stated]
- Ryan found Kyoto "too crowded.": "too crowded." [stated]
- Ryan only likes off-the-beaten-path stuff: yes [stated]
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 4, "confidence": 5, "cardinality": 3, "negation": 5, "note": "Captures the main facts (Amit trip and opinion, Priya visits, Ryan reaction and preference) but omits Amit's comment about temples and includes an odd cardinality for Ryan's preferences."},
  "B_C2_typed_hierarchy": {"fidelity": 4, "confidence": 5, "cardinality": 3, "negation": 5, "note": "Structured and accurate on core facts (Amit trip/opinion, Priya visits, Ryan reaction/preferences) but does not record the temples mention and misrepresents visited_locations cardinality."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 5, "cardinality": 4, "negation": 5, "note": "Most complete: includes temples recommendation and correct Priya visit count and participant list, though it focuses on prose rather than explicit set-cardinality metadata."},
  "overall_winner": "C",
  "overall_justification": "Representation C preserves the full set of factual claims (including the temples recommendation) and visit counts most faithfully; A and B are close but each omits the temples detail and show minor cardinality issues."
}
```


### nuanced_preference

**Source:** `I like jazz, mostly bebop and hard bop. Not a huge fan of free jazz - it's fine in small doses but I wouldn't put it on at home.`

**Truth:**
```
  - subject=user | attribute=music_genre_liked | value=jazz | confidence=stated
  - subject=user | attribute=jazz_subgenres_preferred | value=bebop, hard bop (cardinality=set) | confidence=stated
  - subject=user | attribute=jazz_subgenres_disliked | value=free jazz (tolerated in small doses) | confidence=stated, qualified
```


#### C1_flat_triples

**Author-time rep:**
```
(speaker | preferences | music_genre_preference | jazz)
(speaker | preferences | preferred_jazz_subgenres | mostly bebop and hard bop)
(speaker | preferences | attitude_toward_free_jazz | not a huge fan)
(speaker | preferences | free_jazz_listening_conditions | fine in small doses; would not put it on at home)
```

**Readback:**
```
- music_genre_preference: jazz [stated]
- preferred_jazz_subgenres: mostly bebop, hard bop (cardinality=2) [stated]
- attitude_toward_free_jazz: NOT a huge fan [negated]
- free_jazz_listening_conditions: fine in small doses, would not put it on at home (cardinality=2) [stated]
- free_jazz_listening_at_home: NOT put it on at home [negated]
```

**Drift paraphrase source:**

> Big jazz fan, especially bebop and hard bop — free jazz I can handle a little but wouldn't put on at home.

**Drift rep:**
```
(subject | preferences | music_genre_fandom | big jazz fan)
(subject | preferences | preferred_jazz_styles | especially bebop and hard bop)
(subject | preferences | attitude_toward_free_jazz | can handle a little but wouldn't put on at home)
```

**Drift readback:**
```
- music_genre_fandom: big jazz fan [stated]
- preferred_jazz_styles: bebop, hard bop (cardinality=2) [stated]
- attitude_toward_free_jazz: can handle a little [stated]
- attitude_toward_free_jazz: NOT put on at home [stated, negated]
```

**Retrieval new statement:**

> Been getting into cool jazz lately, especially Chet Baker.

**Retrieval output:**
```
RELEVANT: (speaker | preferences | music_genre_preference | jazz) | Relevant because the new statement specifies a particular jazz substyle (cool jazz) and confirms continued interest in jazz music, so this broader genre preference should be considered when updating subgenre details.

RELEVANT: (speaker | preferences | preferred_jazz_subgenres | mostly bebop and hard bop) | Relevant because the new statement indicates interest in cool jazz (Chet Baker), which may require updating or adding to the user's preferred jazz subgenres.
```


#### C2_typed_hierarchy

**Author-time rep:**
```
{
  "music_preferences": {
    "genre_preferences": {
      "likes_jazz": {
        "kind": "functional",
        "value": "yes",
        "confidence": "stated",
        "note": "Explicit statement: 'I like jazz.'"
      },
      "preferred_subgenres": {
        "kind": "set",
        "value": [
          "bebop",
          "hard bop"
        ],
        "confidence": "stated",
        "cardinality": 2,
        "note": "User said 'mostly bebop and hard bop.'"
      },
      "attitude_toward_free_jazz": {
        "kind": "functional",
        "value": "not a huge fan",
        "confidence": "stated",
        "note": "Exact phrasing: 'Not a huge fan of free jazz.'"
      },
      "free_jazz_tolerated_in_small_doses": {
        "kind": "functional",
        "value": "yes",
        "confidence": "stated",
        "note": "User stated: 'it's fine in small doses.'"
      },
      "would_play_free_jazz_at_home": {
        "kind": "functional",
        "value": "no",
        "confidence": "stated",
        "note": "User stated: 'I wouldn't put it on at home.'"
      }
    }
  }
}
```

**Readback:**
```
- likes_jazz: yes [stated]
- preferred_subgenres: bebop, hard bop (cardinality=2) [stated]
- attitude_toward_free_jazz: not a huge fan [stated]
- free_jazz_tolerated_in_small_doses: yes [stated]
- would_play_free_jazz_at_home: NOT yes [negated]
```

**Drift paraphrase source:**

> Big jazz fan, especially bebop and hard bop — free jazz I can handle a little but wouldn't put on at home.

**Drift rep:**
```
{
  "music_preferences": {
    "jazz": {
      "is_jazz_fan": {
        "kind": "functional",
        "value": "yes",
        "confidence": "stated",
        "note": "Text: 'Big jazz fan'"
      },
      "favorite_subgenres": {
        "kind": "set",
        "value": [
          "bebop",
          "hard bop"
        ],
        "confidence": "stated",
        "cardinality": 2,
        "note": "Text: 'especially bebop and hard bop'"
      },
      "attitude_toward_free_jazz": {
        "kind": "functional",
        "value": "can_handle_a_little_but_wouldn't_put_on_at_home",
        "confidence": "stated",
        "note": "Text: 'free jazz I can handle a little but wouldn't put on at home'"
      }
    }
  }
}
```

**Drift readback:**
```
- music_preferences.jazz.is_jazz_fan: yes [stated]
- music_preferences.jazz.favorite_subgenres: bebop, hard bop (cardinality=2) [stated]
- music_preferences.jazz.attitude_toward_free_jazz: can_handle_a_little_but_wouldn't_put_on_at_home [stated]
```

**Retrieval new statement:**

> Been getting into cool jazz lately, especially Chet Baker.

**Retrieval output:**
```
RELEVANT: "likes_jazz": { "kind": "functional", "value": "yes", "confidence": "stated", "note": "Explicit statement: 'I like jazz.'" } | Relevant because the new statement affirms jazz interest and may refine what kinds of jazz the user likes (cool jazz / Chet Baker).

RELEVANT: "preferred_subgenres": { "kind": "set", "value": [ "bebop", "hard bop" ], "confidence": "stated", "cardinality": 2, "note": "User said 'mostly bebop and hard bop.'" } | Relevant because the user now mentions liking cool jazz (especially Chet Baker), which should be added to or update their preferred subgenres.

RELEVANT: "attitude_toward_free_jazz": { "kind": "functional", "value": "not a huge fan", "confidence": "stated", "note": "Exact phrasing: 'Not a huge fan of free jazz.'" } | Relevant because it describes preferences about jazz subgenres; the new statement about cool jazz is consistent with liking melodic/modal styles rather than free jazz.

RELEVANT: "free_jazz_tolerated_in_small_doses": { "kind": "functional", "value": "yes", "confidence": "stated", "note": "User stated: 'it's fine in small doses.'" } | Relevant as related context about the user's boundaries in jazz subgenre preferences, which remain pertinent given the new specific interest in cool jazz.
```


#### C3_dossier_markdown

**Author-time rep:**
```
## Subject

### Preferences
- Likes jazz [stated]
- Jazz subgenres: bebop, hard bop (2 total) [stated]
- Does NOT generally enjoy free jazz: not a huge fan; fine in small doses but would not put it on at home [negated]
```

**Readback:**
```
- Likes: jazz [stated]
- Jazz subgenres: bebop, hard bop (cardinality=2) [stated]
- Enjoyment of free jazz: NOT generally enjoy free jazz; fine in small doses but would not put it on at home [negated]
```

**Drift paraphrase source:**

> Big jazz fan, especially bebop and hard bop — free jazz I can handle a little but wouldn't put on at home.

**Drift rep:**
```
## Subject

### Preferences
- Is a big jazz fan [stated]
- Preferred jazz subgenres: bebop, hard bop (2 total) [stated]
- Can handle free jazz a little [stated]
- Would not put free jazz on at home [stated]
```

**Drift readback:**
```
- Is a big jazz fan: yes [stated]
- Preferred jazz subgenres: bebop, hard bop (cardinality=2) [stated]
- Can handle free jazz a little: yes [stated]
- Would not put free jazz on at home: NOT put free jazz on at home [stated]
```

**Retrieval new statement:**

> Been getting into cool jazz lately, especially Chet Baker.

**Retrieval output:**
```
RELEVANT: Likes jazz [stated] | The new statement reinforces and is consistent with the user's expressed interest in jazz.

RELEVANT: Jazz subgenres: bebop, hard bop (2 total) [stated] | The user now indicates interest in cool jazz (Chet Baker), so the stored list of jazz subgenres would need to be updated/added-to.

RELEVANT: Does NOT generally enjoy free jazz: not a huge fan; fine in small doses but would not put it on at home [negated] | The new statement about liking cool jazz does not contradict this dislike of free jazz but is relevant to maintain as a distinct preference boundary.
```


**Judge verdict:**
```
{
  "A_C1_flat_triples": {"fidelity": 5, "confidence": 5, "cardinality": 5, "negation": 5, "note": "All facts are present (likes jazz, prefers bebop/hard bop, dislikes free jazz but tolerates small doses) and hedges/negations and cardinalities are explicitly preserved."},
  "B_C2_typed_hierarchy": {"fidelity": 5, "confidence": 4, "cardinality": 5, "negation": 5, "note": "Covers all facts and negations clearly, but drops the subtle 'mostly' hedge on subgenre preference."},
  "C_C3_dossier_markdown": {"fidelity": 5, "confidence": 4, "cardinality": 5, "negation": 5, "note": "Captures the same facts and negations including tolerated small doses, but omits the 'mostly' qualifier on subgenre preference."},
  "overall_winner": "A",
  "overall_justification": "Representation A most completely preserves the nuance (including the explicit 'mostly' qualifier) while B and C are accurate but lose that subtle hedge on subgenre preference."
}
```
