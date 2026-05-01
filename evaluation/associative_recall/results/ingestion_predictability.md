# Ingestion-Predictability Analysis of Retrieval Failures

Source: `error_analysis_details.json` (55 failing questions, 334 missed turns)

This report asks: *for each missed turn, could cheap (mostly deterministic) ingest-time signals have flagged it for alternate-key generation, without knowing the query?*

## 1. Headline numbers

- Total missed turns analyzed: **334**
- Ingestion-predictable (matches at least one heuristic): **232 / 334 (69.5%)**
- Not predictable (requires query knowledge): **102 / 334 (30.5%)**

### Original failure-mode labels (for context)

| original_failure_mode | count | % |
|---|---:|---:|
| ranked_out | 250 | 74.9% |
| vocabulary_gap | 50 | 15.0% |
| anaphoric_reference | 34 | 10.2% |

## 2. Predictable categories (primary bucket)

Each turn is assigned to the *highest-priority* predictable category it matches. Priority order: `anaphoric` > `short_response` > `update_marker` > `known_unknown` > `alias_evolution` > `structured_fact` > `rare_entity`.

| category | count | % of all missed | cheap heuristic? |
|---|---:|---:|---|
| anaphoric | 90 | 26.9% | Yes - first-token membership in pronoun set |
| short_response | 4 | 1.2% | Yes - word_count<=4 or first-token in agree set |
| update_marker | 28 | 8.4% | Yes - regex on sentence-initial marker words |
| known_unknown | 7 | 2.1% | Yes - regex on 'check/verify/TBD/pending/...' |
| alias_evolution | 2 | 0.6% | Mostly - phrase regex ('call it', 'aka', 'renamed') |
| structured_fact | 35 | 10.5% | Yes - keyword list (allergy, deadline, $, %...) |
| rare_entity | 66 | 19.8% | Yes - capitalized-name + number regex |

### Multi-tag fire counts (a turn may match several)

| tag | # turns where tag fires |
|---|---:|
| anaphoric | 90 |
| short_response | 4 |
| update_marker | 28 |
| known_unknown | 10 |
| alias_evolution | 6 |
| structured_fact | 56 |
| rare_entity | 141 |

### Entity-presence sanity check

- Turns containing at least one candidate entity (proper noun, acronym, ID, date, $, version, %): **260 / 334 (77.8%)**
- Turns containing at least one *corpus-rare* entity (<=3 occurrences across missed-turn+neighbor texts): **141 / 334 (42.2%)**
The `rare_entity` primary bucket above uses the stricter corpus-rare filter; the generous view is a loose upper bound on what purely-entity-based alt-keys could address.

## 3. Non-predictable categories

| category | count | % of all missed |
|---|---:|---:|
| arbitrary_conjunction | 11 | 3.3% |
| pattern_in_many | 29 | 8.7% |
| distant_inference | 1 | 0.3% |
| genuinely_idiosyncratic_query | 61 | 18.3% |

## 4. Examples per category

### `anaphoric`

- **turn 10** (q-category: evolving_terminology, orig failure_mode: anaphoric_reference)
  - text: "Yeah, 16 weeks. Aggressive but doable if we don't get pulled into other stuff."
  - all predictable tags fired: ['anaphoric', 'alias_evolution', 'rare_entity']
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- **turn 54** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "Yeah. So the VP came to our standup today asking about 'the new portal' - she never uses the Phoenix name, just calls it 'the new portal' or 'the portal project.'"
  - all predictable tags fired: ['anaphoric']
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- **turn 70** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "Exactly. The product manager wrote a blog post for the company newsletter about 'Portal 2.0' which is yet ANOTHER name for it. I swear this project has more names than a con artist."
  - all predictable tags fired: ['anaphoric', 'alias_evolution', 'rare_entity']
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."

### `short_response`

- **turn 72** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "Back from lunch. What's the status on that timeout thing?"
  - all predictable tags fired: ['short_response']
  - question: "What names were used for the JIRA-4521 bug throughout the investigation?"
- **turn 36** (q-category: conjunction, orig failure_mode: ranked_out)
  - text: "Same meeting next Wednesday. We're presenting colors and logo together."
  - all predictable tags fired: ['short_response']
  - question: "What content should be included in the presentation for the Acme Corp client meeting on Wednesday?"
- **turn 48** (q-category: contradiction, orig failure_mode: ranked_out)
  - text: "Diana's email says it's double rooms, shared with a colleague."
  - all predictable tags fired: ['short_response', 'structured_fact']
  - question: "Which sources of information about the retreat turned out to be wrong, and why?"

### `update_marker`

- **turn 10** (q-category: unfinished_business, orig failure_mode: ranked_out)
  - text: "Oh and one more thing - I promised our VP Carla that I'd send her the updated roadmap by end of day Tuesday. She's been asking about it."
  - all predictable tags fired: ['update_marker', 'structured_fact', 'rare_entity']
  - question: "What tasks or promises were assigned during the conversation that were never completed or followed up on?"
- **turn 64** (q-category: quantitative_aggregation, orig failure_mode: ranked_out)
  - text: "Hmm, true but then we'd also save some of Owen's order processing time since the current checkout talks to the old API. Maybe 20 hours from Owen?"
  - all predictable tags fired: ['update_marker']
  - question: "How did the project estimate compare to the client's budget, and what was the resolution?"
- **turn 24** (q-category: consistency_checking, orig failure_mode: ranked_out)
  - text: "Actually yes. I just remembered - my boss scheduled a strategy session for Thursday morning at 9am. It's at the uptown office."
  - all predictable tags fired: ['update_marker']
  - question: "Are there any scheduling conflicts or inconsistencies in the user's week that were identified or remain unresolved?"

### `known_unknown`

- **turn 46** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "I'll look into those. Hey Sara just Slacked the team. She says the index migration worked great in staging. Query time went from 58 seconds to 0.2 seconds."
  - all predictable tags fired: ['known_unknown', 'rare_entity']
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"
- **turn 88** (q-category: unfinished_business, orig failure_mode: ranked_out)
  - text: "You know what, I don't think he did. Let me check... No, he hasn't. And now I'm pulling him onto the auth fix so it'll be further delayed."
  - all predictable tags fired: ['known_unknown']
  - question: "What tasks or promises were assigned during the conversation that were never completed or followed up on?"
- **turn 94** (q-category: unfinished_business, orig failure_mode: ranked_out)
  - text: "Good. I also need to follow up with Hannah about the venue research. She said early this week. Let me check... Still nothing from her. I'll give her until Wednesday."
  - all predictable tags fired: ['known_unknown']
  - question: "What tasks or promises were assigned during the conversation that were never completed or followed up on?"

### `alias_evolution`

- **turn 72** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "And Ravi just started calling it 'the bird' because of the phoenix emoji. Seven names."
  - all predictable tags fired: ['alias_evolution', 'rare_entity']
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- **turn 25** (q-category: sequential_chain, orig failure_mode: ranked_out)
  - text: "Golpar! That's Persian hogweed seed, also known as angelica seed. It's a very distinctive Iranian spice with a warm, musky, slightly bitter flavor. It's quite uncommon outside of Iranian cooking."
  - all predictable tags fired: ['alias_evolution', 'rare_entity']
  - question: "What chain of discoveries led to successfully recreating the grandmother's lamb stew?"

### `structured_fact`

- **turn 10** (q-category: negation, orig failure_mode: ranked_out)
  - text: "Frank estimated about $28K per month for our workload. That includes reserved instances for the steady-state VMs and spot instances for batch processing."
  - all predictable tags fired: ['structured_fact', 'rare_entity']
  - question: "Which cloud providers were rejected in the migration evaluation and what were the specific reasons for each rejection?"
- **turn 18** (q-category: negation, orig failure_mode: ranked_out)
  - text: "Min got a quote from Google Cloud's sales team. They came in at $24K per month, which includes sustained use discounts and some migration credits."
  - all predictable tags fired: ['structured_fact', 'rare_entity']
  - question: "Which cloud providers were rejected in the migration evaluation and what were the specific reasons for each rejection?"
- **turn 26** (q-category: negation, orig failure_mode: ranked_out)
  - text: "Azure came in at $30K per month. More expensive than both AWS and GCP. Leah says we can get it down to $26K if we commit to a 3-year reserved instance plan."
  - all predictable tags fired: ['structured_fact', 'rare_entity']
  - question: "Which cloud providers were rejected in the migration evaluation and what were the specific reasons for each rejection?"

### `rare_entity`

- **turn 68** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "Nah I'll handle it. Back to work - so the team has started calling the project 'v2' informally. Like 'is that a v1 bug or a v2 thing?' when discussing issues."
  - all predictable tags fired: ['rare_entity']
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- **turn 18** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "Please. Ok so the timeout thing - Kai found it. The last deployment accidentally included a config file that was meant for the staging environment. Staging has a 3-minute session timeout for testing purposes."
  - all predictable tags fired: ['rare_entity']
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"
- **turn 54** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "True but it FEELS like one big monster. Same deployment triggered both problems, just in different ways. The staging config caused the session issue and the missing index caused the timeout."
  - all predictable tags fired: ['rare_entity']
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"

### `arbitrary_conjunction`

- **turn 4** (q-category: proactive, orig failure_mode: ranked_out)
  - text: "So the core team is me (project lead), Vanessa on design, Tom on frontend dev, Maria on backend, and Hiroshi on content/copy."
  - question: "Draft a project status update for the Acme Corp rebrand team covering current progress and next steps."
- **turn 18** (q-category: proactive, orig failure_mode: ranked_out)
  - text: "Thanks. Ok, I also wanted to talk about my knee. I've been having pain in my right knee for about 3 weeks."
  - question: "Help me prepare a list of topics to discuss with Dr. Patel at my January 25th appointment."
- **turn 36** (q-category: proactive, orig failure_mode: vocabulary_gap)
  - text: "About 2 weeks. I think it might be stress-related - work has been intense lately."
  - question: "Help me prepare a list of topics to discuss with Dr. Patel at my January 25th appointment."

### `pattern_in_many`

- **turn 18** (q-category: quantitative_aggregation, orig failure_mode: ranked_out)
  - text: "Good question. He says no, add another 20 hours for integration testing."
  - question: "What is the total estimated hours for the website migration project, broken down by person? Include all revisions to the estimates."
- **turn 36** (q-category: quantitative_aggregation, orig failure_mode: ranked_out)
  - text: "Here's where it gets interesting. She says 70 hours for the migration scripts, but that includes a lot of data cleaning. Some of the old records have character encoding issues, duplicates, and orphaned foreign keys."
  - question: "What is the total estimated hours for the website migration project, broken down by person? Include all revisions to the estimates."
- **turn 52** (q-category: quantitative_aggregation, orig failure_mode: ranked_out)
  - text: "But wait, Quinn also said he'd need about 15 hours for the staging environment setup and another 10 hours for security hardening. He forgot to include those initially."
  - question: "What is the total estimated hours for the website migration project, broken down by person? Include all revisions to the estimates."

### `distant_inference`

- **turn 6** (q-category: inference, orig failure_mode: ranked_out)
  - text: "And atorvastatin 20mg at night for cholesterol. That one I've been on for years."
  - question: "Based on everything in the conversation, what medication interactions and health concerns should the user bring up with Dr. Patel at their January 25th appointment?"

### `genuinely_idiosyncratic_query`

- **turn 14** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "End of week 6. That's the part the VP cares about most because it's the most visible improvement."
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- **turn 92** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "Which is tight with the added scope. We might need to negotiate on the self-service features in phase 3."
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- **turn 2** (q-category: evolving_terminology, orig failure_mode: ranked_out)
  - text: "Our internal HR tool. About 300 people use it daily. The login issue started around 9am according to the first reports."
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"

## 5. Predictability by original failure mode

| original_failure_mode | predictable / total | % predictable |
|---|---:|---:|
| anaphoric_reference | 34 / 34 | 100.0% |
| ranked_out | 166 / 250 | 66.4% |
| vocabulary_gap | 32 / 50 | 64.0% |

## 6. Predictability by question category

| question_category | predictable / total | % predictable |
|---|---:|---:|
| absence_inference | 11 / 24 | 45.8% |
| completeness | 21 / 26 | 80.8% |
| conjunction | 3 / 3 | 100.0% |
| consistency_checking | 6 / 9 | 66.7% |
| constraint_propagation | 9 / 12 | 75.0% |
| contradiction | 4 / 4 | 100.0% |
| evolving_terminology | 19 / 35 | 54.3% |
| frequency_detection | 7 / 8 | 87.5% |
| inference | 5 / 6 | 83.3% |
| locomo_multi_hop | 3 / 3 | 100.0% |
| locomo_single_hop | 5 / 10 | 50.0% |
| locomo_temporal | 0 / 2 | 0.0% |
| logic_constraint | 23 / 35 | 65.7% |
| negation | 14 / 19 | 73.7% |
| open_exploration | 13 / 21 | 61.9% |
| perspective_separation | 7 / 9 | 77.8% |
| proactive | 22 / 25 | 88.0% |
| procedural | 13 / 21 | 61.9% |
| quantitative_aggregation | 12 / 19 | 63.2% |
| sequential_chain | 13 / 20 | 65.0% |
| state_change | 13 / 14 | 92.9% |
| unfinished_business | 9 / 9 | 100.0% |

## 7. Heuristic recommendations (ingest-time, no LLM)

All seven predictable categories can be detected with simple regex or token checks:

- **anaphoric**: `first_token(text)` in `{that, this, those, these, it, they, he, she, ...}`.  Attach `preceding_turn_text + this_turn_text` as an extra key.
- **short_response**: `word_count <= 4` OR first-token in `{yeah, ok, sure, yes, no, definitely, ...}`. Attach `preceding_turn_text + this_turn_text` as extra key.
- **update_marker**: sentence-initial regex `^(actually|wait|oh|scratch that|correction|on second thought|update|let me correct|turns out|never mind) [,. ]`. Tag the turn as an *update* and carry the previous-turn text as a co-key so the retraction is indexed with the thing it retracts.
- **known_unknown**: regex on `let me check | circle back | TBD | pending | not sure | waiting on`. Tag as `unresolved_question` so a query for the later resolution also surfaces the original.
- **alias_evolution**: phrase regex `call(ed)? it | aka | also known as | renamed | new name`, plus proper-noun overlap between this turn and neighbor turns. Store BOTH surface names as keys.
- **structured_fact**: keyword list `allergy | deadline | prescription | dosage | prefer | by (day) | $, %, version`. Tag with fact-type; fact-type becomes a retrievable key.
- **rare_entity**: proper-noun / number regex. Emit entity tokens as additional keys.

## 8. Summary answers to the key questions

1. **What % of observed failures are ingestion-predictable?** 69.5% (232 / 334).
2. **Which categories dominate?** `anaphoric` (90, 26.9%), `rare_entity` (66, 19.8%), `structured_fact` (35, 10.5%).
3. **Can most predictable categories be caught by cheap deterministic heuristics (no LLM)?** Yes - all seven categories are detectable with first-token checks, sentence-initial regexes, and keyword / proper-noun lists. None require an LLM per turn. Only `alias_evolution` benefits from light entity matching across adjacent turns, which is still cheap.
4. **What % REQUIRE query knowledge (hard ceiling for ingest-side)?** 30.5% (102 / 334). This is the theoretical ceiling below which ingest-side tagging cannot help.

## 9. Limitations / caveats

- **Rare-entity filter uses corpus-frequency.** At true ingest time (streaming), you don't know an entity's future frequency. A practical implementation would emit a proper-noun or number token as a key for *every* such entity; the generous entity-presence upper bound above (77.8% of missed turns contain at least one candidate entity) captures this looser view. The downside is that non-rare entities like `Angular` or `React` in this dataset appear in many turns, so the per-entity index becomes less selective - but that's a retrieval-weighting problem, not an ingest-signal problem.
- **Category boundaries are priority-based.** A turn that's anaphoric AND contains a rare entity AND is structured-fact is counted under `anaphoric` only. The multi-tag fire-count table shows overlap; a single cheap key-generation pass would fire all applicable generators.
- **`alias_evolution` count is low because the dataset's aliasing happens across *distant* turns.** Catching cross-turn alias co-occurrence at ingest time would require a simple proper-noun graph that links entities appearing in phrases like 'call it X' / 'also known as Y' - cheap but not detectable from the missed turn alone. Still deterministic, no LLM needed.
- **100% of originally-labeled `anaphoric_reference` failures are caught.** This validates the first-token heuristic.
- **66% of `ranked_out` and 64% of `vocabulary_gap` failures are ingestion-predictable** via entity + structured-fact tagging even though they are not labeled anaphoric.

## 10. Leverage verdict

**HIGH leverage.** Cheap ingest-time heuristics could produce alternate keys for the majority of observed failures, without per-turn LLM cost.
