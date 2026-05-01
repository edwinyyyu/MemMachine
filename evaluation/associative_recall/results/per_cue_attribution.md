# Per-Cue Attribution Analysis

- Total (question, cue) pairs analyzed: **4524**
- Unique questions: **118**
- Top-K used for retrieval: **20**
- Winners (≥1 gold hit exclusive vs baseline): **2725** (60.2%)
- Any-gold hitters: **3946**
- Losers (0 gold in top-20): **578** (12.8%)

## Winners vs Losers — Feature Distribution

| feature | all | winners | hitters (any gold) | losers |
|---|---|---|---|---|
| mean_len_words | 23.138 | 25.731 | 24.044 | 16.958 |
| median_len_words | 20.000 | 23.000 | 21.000 | 16.000 |
| mean_jaccard | 0.102 | 0.067 | 0.100 | 0.115 |
| pct_is_question | 4.2% | 3.4% | 3.9% | 6.7% |
| mean_entity_count | 2.803 | 3.247 | 3.015 | 1.358 |
| mean_number_count | 0.779 | 0.956 | 0.866 | 0.190 |
| mean_cue_q_cos | 0.473 | 0.436 | 0.475 | 0.461 |
| mean_best_gold_cos | 0.533 | 0.575 | 0.574 | 0.258 |

## Top features distinguishing winners from losers

| feature | winners | losers | delta | rel Δ |
|---|---|---|---|---|
| mean_number_count | 0.956 | 0.190 | +0.766 | +4.02 |
| mean_entity_count | 3.247 | 1.358 | +1.888 | +1.39 |
| mean_best_gold_cos | 0.575 | 0.258 | +0.317 | +1.23 |
| mean_len_words | 25.731 | 16.958 | +8.773 | +0.52 |
| pct_is_question | 3.4% | 6.7% | -3.3 | -0.49 |
| mean_jaccard | 0.067 | 0.115 | -0.048 | -0.42 |
| mean_cue_q_cos | 0.436 | 0.461 | -0.025 | -0.05 |

## Category-specific patterns

| category | n | winner rate | hitter rate | win_len | lose_len | win_cos | lose_cos |
|---|---|---|---|---|---|---|---|
| locomo_temporal | 894 | 0.25 | 0.72 | 15.8 | 14.4 | 0.440 | 0.229 |
| locomo_single_hop | 594 | 0.68 | 0.69 | 16.7 | 16.3 | 0.436 | 0.263 |
| locomo_multi_hop | 249 | 0.28 | 0.71 | 26.0 | 19.8 | 0.466 | 0.277 |
| evolving_terminology | 231 | 0.80 | 1.00 | 25.4 | 26.0 | 0.578 | 0.240 |
| proactive | 201 | 0.75 | 0.94 | 32.2 | 39.5 | 0.566 | 0.363 |
| completeness | 198 | 0.74 | 0.99 | 32.5 | 17.0 | 0.603 | 0.333 |
| perspective_separation | 175 | 0.67 | 0.99 | 32.2 | 17.0 | 0.687 | 0.228 |
| sequential_chain | 144 | 0.91 | 1.00 | 27.6 | 0.0 | 0.610 | 0.000 |
| inference | 143 | 0.62 | 0.98 | 31.5 | 12.7 | 0.626 | 0.155 |
| logic_constraint | 142 | 0.77 | 1.00 | 31.9 | 0.0 | 0.630 | 0.000 |
| conjunction | 139 | 0.65 | 0.94 | 35.5 | 18.4 | 0.579 | 0.268 |
| negation | 138 | 0.96 | 0.99 | 28.5 | 20.5 | 0.658 | 0.276 |
| state_change | 132 | 0.84 | 1.00 | 26.5 | 0.0 | 0.691 | 0.000 |
| absence_inference | 132 | 0.95 | 1.00 | 25.0 | 0.0 | 0.602 | 0.000 |
| control | 130 | 0.00 | 0.88 | 0.0 | 16.5 | 0.000 | 0.266 |
| unfinished_business | 128 | 0.66 | 0.97 | 22.2 | 16.5 | 0.654 | 0.237 |
| quantitative_aggregation | 128 | 0.79 | 1.00 | 26.5 | 0.0 | 0.650 | 0.000 |
| procedural | 99 | 0.78 | 0.99 | 34.6 | 20.0 | 0.543 | 0.409 |
| constraint_propagation | 94 | 0.98 | 1.00 | 27.3 | 0.0 | 0.650 | 0.000 |
| contradiction | 93 | 0.94 | 1.00 | 22.4 | 0.0 | 0.648 | 0.000 |

## 10 Winning cue examples

### Winner #1  [open_exploration]
- **Question:** Based on all the patterns in our conversations, are there any health concerns about the user I should flag?
- **Cue:** Oh, and btw I definitely feel more stressed on Mondays, which might be making it worse.
- gold_hit=12, gold_exclusive=9, cue_q_cos=0.234, best_gold_cos=0.574, len=16
- **Retrieved gold turn:** Yeah, same thing. Kind of a pressure behind my eyes and temples. Took some Advil and it helped. Probably just the Sunday-to-Monday stress transition.

### Winner #2  [evolving_terminology]
- **Question:** What was the root cause of the monster bug and how was each aspect resolved?
- **Cue:** Added the missing DB index, Sara finished the payroll queue cleanup, and we're updating the deployment process to prevent env-specific configs leaking
- gold_hit=11, gold_exclusive=9, cue_q_cos=0.250, best_gold_cos=0.606, len=24
- **Retrieved gold turn:** I'll look into those. Hey Sara just Slacked the team. She says the index migration worked great in staging. Query time went from 58 seconds to 0.2 seconds.

### Winner #3  [sequential_chain]
- **Question:** What chain of discoveries led to successfully recreating the grandmother's lamb stew?
- **Cue:** Hamid taught me to treat the limoo amani by piercing or cracking them so the pulp infuses the broth, and to either dry‑toast golpar briefly or bloom it in hot oil; then brown the meat well and simmer everything slowly in a heavy Dutch oven.
- gold_hit=11, gold_exclusive=9, cue_q_cos=0.351, best_gold_cos=0.557, len=46
- **Retrieved gold turn:** She mentions toasting the spices in oil before adding the meat. Dariush said the exact phrase was something like 'bloom the aromatics in hot fat until the kitchen sings.'

### Winner #4  [frequency_detection]
- **Question:** How does the user relate to coffee throughout the conversation? Track all mentions and the contexts they appear in.
- **Cue:** I'm at my desk by the window with my mug; they moved the coffee machine next to the printers.
- gold_hit=14, gold_exclusive=8, cue_q_cos=0.275, best_gold_cos=0.505, len=20
- **Retrieved gold turn:** Smart. Adding that now. By the way, the office coffee machine is still broken. Day 3. People are getting grumpy.

### Winner #5  [constraint_propagation]
- **Question:** How did the budget cut cascade through the CRM project and what other projects were affected?
- **Cue:** Dropping the quoting tool and pulling the internal team off the mobile app could delay the mobile app and cost around $200K in lost mobile revenue; the VP demanded the quoting tool be restored immediately.
- gold_hit=13, gold_exclusive=8, cue_q_cos=0.366, best_gold_cos=0.694, len=35
- **Retrieved gold turn:** He agreed to drop the quoting tool requirement for now. We'll stick with the reduced CRM scope and not pull the dev team off the mobile app. The quoting tool goes on the wish list for Q1 next year when hopefully there's budget.

### Winner #6  [absence_inference]
- **Question:** Based on the conversation, does the user follow any specific dietary pattern? What evidence supports your conclusion?
- **Cue:** "grain bowls are my staple" quinoa rice roasted vegetables beans rotate tahini chimichurri peanut sauce
- gold_hit=12, gold_exclusive=7, cue_q_cos=0.285, best_gold_cos=0.879, len=15
- **Retrieved gold turn:** Grain bowls are my staple. Like, quinoa or rice with roasted vegetables, beans, and some kind of dressing. I rotate between tahini, chimichurri, and peanut sauce.

### Winner #7  [consistency_checking]
- **Question:** Are there any scheduling conflicts or inconsistencies in the user's week that were identified or remain unresolved?
- **Cue:** Boss wants to move Friday 1:1 to Thursday 10am but the strategy session is 9am–11am uptown, so 10am would overlap; suggested moving the 1:1 to 11:15am instead.
- gold_hit=11, gold_exclusive=7, cue_q_cos=0.305, best_gold_cos=0.736, len=31
- **Retrieved gold turn:** If the strategy session runs 9am-11am as planned, a 10am 1:1 would overlap. Can she do 11:15am instead?

### Winner #8  [evolving_terminology]
- **Question:** What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?
- **Cue:** JIRA-4521 timesheet timeout missing index fix resulted in ~200 corrupted records in the payroll queue Sara cleanup script Kai monitoring
- gold_hit=10, gold_exclusive=7, cue_q_cos=0.444, best_gold_cos=0.617, len=21
- **Retrieved gold turn:** So Sara's migration is done. Timesheet submissions are working again. But now she noticed something weird in the logs - there are a bunch of failed API calls to the payroll service.

### Winner #9  [negation]
- **Question:** Which cloud providers were rejected in the migration evaluation and what were the specific reasons for each rejection?
- **Cue:** We rejected GCP and chose AWS — Frank's AWS expertise, migration risk mitigation, and confidence in hitting the 18-month deadline were the deciding factors.
- gold_hit=10, gold_exclusive=7, cue_q_cos=0.520, best_gold_cos=0.757, len=25
- **Retrieved gold turn:** We went with AWS. Frank's expertise and the risk mitigation won out. The $7K monthly savings over on-prem is sufficient, and we believe we can hit the 18-month deadline.

### Winner #10  [evolving_terminology]
- **Question:** What is the current status of Project Phoenix? Include any milestones reached and upcoming work.
- **Cue:** "phase 1 is complete" "dashboard loading 4x faster" "phase 2 reporting module" week 11 "phase 3 notification center" "subscription management post-launch" "project charter" "steering committee"
- gold_hit=9, gold_exclusive=7, cue_q_cos=0.407, best_gold_cos=0.833, len=26
- **Retrieved gold turn:** Key points: phase 1 is complete with the dashboard loading 4x faster, phase 2 reporting module is on track for week 11, scope expanded to include a notification center in phase 3, and you may need to defer subscription management to a post-

## 10 Losing cue examples

### Loser #1  [locomo_temporal]
- **Question:** How long ago was Caroline's 18th birthday?
- **Cue:** When was Caroline's 18th birthday? ten years ago
- gold_hit=0, cue_q_cos=0.916, best_gold_cos=0.296, len=9, entities=1, is_question=True

### Loser #2  [locomo_temporal]
- **Question:** When did Melanie paint a sunrise?
- **Cue:** Melanie painted that sunrise last year
- gold_hit=0, cue_q_cos=0.870, best_gold_cos=0.082, len=6, entities=0, is_question=False

### Loser #3  [locomo_temporal]
- **Question:** When did Caroline have a picnic?
- **Cue:** Caroline picnic last weekend last month family park outing when did Caroline have a picnic
- gold_hit=0, cue_q_cos=0.848, best_gold_cos=0.292, len=15, entities=1, is_question=False

### Loser #4  [locomo_single_hop]
- **Question:** Where did Caroline move from 4 years ago?
- **Cue:** "Where did Caroline move from 4 years ago? moved here from her grandma's home country Sweden"
- gold_hit=0, cue_q_cos=0.817, best_gold_cos=0.317, len=17, entities=4, is_question=False

### Loser #5  [locomo_temporal]
- **Question:** When did Melanie read the book "nothing is impossible"?
- **Cue:** Melanie read "Nothing Is Impossible" when? Melanie read Nothing Is Impossible read Mel when last week last month last year two days ago
- gold_hit=0, cue_q_cos=0.811, best_gold_cos=0.319, len=23, entities=9, is_question=False

### Loser #6  [locomo_temporal]
- **Question:** How long has Caroline had her current group of friends for?
- **Cue:** "How long has Caroline been friends with her current group? friends from activism/volunteering connected LGBTQ activists years together"
- gold_hit=0, cue_q_cos=0.809, best_gold_cos=0.380, len=19, entities=2, is_question=False

### Loser #7  [beam_instruction_following]
- **Question:** How do I calculate the chance of drawing a red card from a standard deck?
- **Cue:** "In a standard 52-card deck (no jokers), how many red cards are there and what's the probability of drawing one on a single draw?"
- gold_hit=0, cue_q_cos=0.791, best_gold_cos=0.273, len=26, entities=2, is_question=False

### Loser #8  [locomo_temporal]
- **Question:** When did Melanie go to the museum?
- **Cue:** "Melanie went to the museum last weekend"
- gold_hit=0, cue_q_cos=0.789, best_gold_cos=0.417, len=7, entities=1, is_question=False

### Loser #9  [locomo_multi_hop]
- **Question:** Would Caroline pursue writing as a career option?
- **Cue:** Caroline consider writing as a career? "What kinda jobs are you thinkin' of?" continue education, explore careers, counseling vs writing
- gold_hit=0, cue_q_cos=0.781, best_gold_cos=0.363, len=20, entities=2, is_question=False

### Loser #10  [locomo_single_hop]
- **Question:** What books has Melanie read?
- **Cue:** Which books has Melanie read? Melanie favorite reads trans memoirs adoption parenting art activism volunteering reading list
- gold_hit=0, cue_q_cos=0.773, best_gold_cos=0.341, len=17, entities=2, is_question=True

## Within-category deltas (winners − losers, averaged across categories)

Controls for the confound that some categories have inherently longer/shorter cues.

| feature | avg Δ (winner − loser) | #cats with data |
|---|---|---|
| cue_len_words | +5.419 | 9 |
| jaccard_with_q | -0.022 | 9 |
| is_question | +0.042 | 9 |
| entity_count | +1.066 | 9 |
| number_count | +0.139 | 9 |
| cue_q_cos | +0.049 | 9 |
| best_gold_cos | +0.262 | 9 |

## Actionable summary

Based on winner-vs-loser distributions and within-category patterns:

- **Length:** winners average 25.7 words vs losers 17.0. Longer cues win.
- **Embedding distance from question:** winners' cues sit at cos=0.436 from the question vs losers' 0.461. Winners probe further from the query.
- **Question-form cues:** 3.4% of winners are questions vs 6.7% of losers. Statement form wins.
- **Entity density:** winners 3.25 entities/cue vs losers 1.36. More entity-dense cues win.
- **Lexical overlap with question:** winners Jaccard=0.067 vs losers 0.115. Counterintuitively, high question-token overlap is a loser signal — good cues probe *around* the question with chat-style text, not by echoing it.
- **Best-gold cosine (how close cue got to any gold turn):** winners 0.575 vs losers 0.258. This is the strongest signal — cues that geometrically approach a gold turn succeed.

### Loser archetype (from top 10)

The dominant failure mode is the **"interrogative paraphrase"**: cues like `"Melanie painted that sunrise last year"` or `"When was Caroline's 18th birthday? ten years ago"`. They:
- Are short declarative paraphrases of the question or question+guessed-answer.
- Sit at high cosine to the question (0.78–0.92) but low cosine to the actual gold turn (<0.35).
- Hallucinate/guess the answer inline, polluting the embedding with tokens not in the chat log.
- Concentrate in locomo_temporal / locomo_single_hop where gold is a short chat turn whose vocabulary the LLM cannot predict from the question alone.

### Winner archetype (from top 10)

- Chat-message text with named entities + specific nouns + timestamps or numbers. Sequential-chain, logic-constraint, and evolving-terminology questions produce the best cues because the LLM can draw vocabulary from the already-retrieved context rather than guessing.
- Notable: many winners have LOW cue_q_cos (0.23–0.37). The LLM is productively *pivoting* — inventing new vocabulary that matches the gold turn's voice rather than the question's.

### v2f-successor variants to test

1. **Anti-paraphrase prompt hardening.** The existing v2f prompt already says "Do NOT write questions", but losers still echo question tokens. Add: "Do NOT restate the question. Do NOT guess an answer. Write a quote that might appear verbatim in the chat." Test: expect biggest lift on locomo_temporal (25% winner rate currently).
2. **Context-anchored cues for sparse retrieval.** When the context_section is empty or weak (locomo_temporal, locomo_single_hop), v2f has nothing to seed cues with and falls back to paraphrasing. Add a mandatory 2-stage: hop0 retrieves 10, then force the LLM to *quote* 1–2 phrases from the retrieved context verbatim and extend each into a cue. Gold-cosine lift should be large because cues inherit real chat vocabulary.
3. **Entity/number injection for proactive and completeness categories.** Entity density is the 2nd-strongest feature (+1.9 per cue; rel Δ +1.4). For categories where recall is already >80% (constraint_propagation, negation), focus instead on recall@all by generating multiple entity-specific cue variants.
