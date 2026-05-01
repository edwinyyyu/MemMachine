# Analysis A: Genuinely-Idiosyncratic Missed Turns
Operational bucket: non-predictable by 7-regex heuristics AND q-category not in {proactive, quantitative_aggregation, inference}.
- Total missed turns: **334**
- Non-predictable (no regex heuristic fires): **177**
- Idiosyncratic bucket (this analysis): **148**
- Source report's claim was 61; our count differs because the source used a slightly different entity/corpus-rarity threshold.
## 1. Surface features
### Word-count distribution (missed turn text)
- mean=27.5, median=26, p10=18, p90=37, min=12, max=60
### Role distribution
- user: 129 (87.2%)
- assistant: 19 (12.8%)
### Original failure-mode label
- ranked_out: 121 (81.8%)
- vocabulary_gap: 27 (18.2%)
### Adjacency to retrieved turns
- adjacent-to-retrieved at radius 1: 60 / 148 (40.5%)
- adjacent-to-retrieved at radius 2: 81 / 148 (54.7%)
### Q-category distribution
- evolving_terminology: 23 (15.5%)
- logic_constraint: 19 (12.8%)
- absence_inference: 19 (12.8%)
- open_exploration: 15 (10.1%)
- procedural: 12 (8.1%)
- sequential_chain: 12 (8.1%)
- completeness: 9 (6.1%)
- locomo_single_hop: 7 (4.7%)
- perspective_separation: 5 (3.4%)
- constraint_propagation: 5 (3.4%)
- negation: 4 (2.7%)
- frequency_detection: 4 (2.7%)
- consistency_checking: 4 (2.7%)
- unfinished_business: 3 (2.0%)
- locomo_temporal: 2 (1.4%)
- locomo_multi_hop: 2 (1.4%)
- conjunction: 1 (0.7%)
- state_change: 1 (0.7%)
- contradiction: 1 (0.7%)
## 2. Embedding-distance story
- cosine(missed_turn, question): mean=0.294, median=0.287, p25=0.236, p75=0.347
- cosine(retrieved_gold, question) (same question): mean=0.388, median=0.397
- gap (retrieved_gold mean - missed mean): +0.104
- rank of missed turn within its own conversation (1536-D cosine to q): mean=40.5, median=36, p75=49, max=331
- Jaccard(missed_tokens, question_tokens): mean=0.013, median=0.000
- Jaccard(missed_tokens, retrieved-sibling-gold_tokens): mean=0.041, median=0.036
## 3. Clusters (k-means on missed-turn embeddings)
### Cluster 0 (20 turns)
**Top lift terms:** `kai` (51.8), `payroll` (40.3), `deployment` (37.0), `timesheet` (37.0), `records` (37.0), `login` (29.6), `queue` (29.6), `data` (23.7), `issue` (22.2), `config` (22.2)
**Examples:**
- [evolving_terminology] (conv `adv_evolving_term_2`, turn 2, user): "Our internal HR tool. About 300 people use it daily. The login issue started around 9am according to the first reports."
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"
- [evolving_terminology] (conv `adv_evolving_term_2`, turn 10, user): "That's what Kai thinks too. He just pinged me - he found something. The session tokens are expiring way too fast. Like, after 2-3 minutes instead of the normal 30 minutes."
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"
- [evolving_terminology] (conv `adv_evolving_term_2`, turn 18, user): "Please. Ok so the timeout thing - Kai found it. The last deployment accidentally included a config file that was meant for the staging environment. Staging has a 3-minute session t"
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"
### Cluster 1 (48 turns)
**Top lift terms:** `it's` (19.3), `like` (15.2), `i'll` (14.2), `ha` (12.8), `eat` (12.8), `cooking` (12.8), `sauce` (12.8), `i'm` (11.4), `he's` (11.4), `weekend` (10.2)
**Examples:**
- [negation] (conv `adv_negation_1`, turn 36, user): "She also showed a prototype she built over the weekend. A mini dashboard with 500 updating data points. It was buttery smooth even on her old phone."
  - question: "What frontend frameworks were considered and rejected for the dashboard project, and why was each one eliminated?"
- [negation] (conv `adv_negation_1`, turn 52, user): "Tomoko was visibly frustrated. She thinks we're making a safe but suboptimal choice. She says in two years we'll wish we'd picked Svelte when we're fighting React re-render issues."
  - question: "What was the final technology decision and what mitigation strategies were agreed upon to address known weaknesses?"
- [perspective_separation] (conv `adv_perspective_1`, turn 80, user): "She said it's the least bad option. Still thinks we'll be cleaning up in July but at least the scope is more realistic now. She's not happy but she's accepted it."
  - question: "What is Alice's position on the June 15th Meridian launch deadline, and how has it evolved?"
### Cluster 2 (11 turns)
**Top lift terms:** `caroline` (60.2), `help` (32.1), `understanding` (30.1), `great` (22.8), `support` (22.6), `counselor` (20.1), `empathy` (20.1), `you'd` (20.1), `reminds` (20.1), `ago` (20.1)
**Examples:**
- [locomo_temporal] (conv `locomo_conv-26`, turn 11, assistant): "You'd be a great counselor! Your empathy and understanding will really help the people you work with. By the way, take a look at this."
  - question: "When did Melanie paint a sunrise?"
- [locomo_temporal] (conv `locomo_conv-26`, turn 62, user): "Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome--"
  - question: "How long ago was Caroline's 18th birthday?"
- [locomo_multi_hop] (conv `locomo_conv-26`, turn 39, user): "Thanks Mel! Your kind words mean a lot. Sharing our experiences isn't always easy, but I feel it's important to help promote understanding and acceptance. I've been blessed with lo"
  - question: "Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?"
### Cluster 3 (13 turns)
**Top lift terms:** `features` (51.5), `salesforce` (41.2), `th` (33.0), `tight` (30.9), `weeks` (30.9), `implementation` (30.9), `cloudforce` (30.9), `integration` (30.9), `quoting` (30.9), `erp` (30.9)
**Examples:**
- [evolving_terminology] (conv `adv_evolving_term_1`, turn 92, user): "Which is tight with the added scope. We might need to negotiate on the self-service features in phase 3."
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- [negation] (conv `adv_negation_1`, turn 48, user): "Marcus said we couldn't afford 2 weeks of exploration on a 12-week timeline. And if the spike fails, we've wasted that time and still have to use React."
  - question: "What frontend frameworks were considered and rejected for the dashboard project, and why was each one eliminated?"
- [negation] (conv `adv_negation_2`, turn 38, user): "And he added that during the learning curve period, we'd be more likely to make configuration mistakes that could cause outages or security incidents. The compliance team would not"
  - question: "Which cloud providers were rejected in the migration evaluation and what were the specific reasons for each rejection?"
### Cluster 4 (44 turns)
**Top lift terms:** `room` (33.5), `pm` (22.3), `day` (19.1), `needs` (19.1), `eve` (18.6), `desk` (18.6), `team` (16.7), `next` (15.5), `ends` (14.9), `floor` (14.9)
**Examples:**
- [evolving_terminology] (conv `adv_evolving_term_1`, turn 14, user): "End of week 6. That's the part the VP cares about most because it's the most visible improvement."
  - question: "What is the current status of Project Phoenix? Include any milestones reached and upcoming work."
- [evolving_terminology] (conv `adv_evolving_term_2`, turn 84, user): "My team. It'll be Dev since he does most of our marketing pages. Anyway, back to the monster - Sara just finished the payroll cleanup."
  - question: "What is the full story of the JIRA-4521 bug? What caused it and what were all its effects?"
- [unfinished_business] (conv `adv_unfinished`, turn 8, user): "Late October. Hannah volunteered to research venues and send out options by this Friday. She's really into that kind of thing, always organizing team events."
  - question: "What tasks or promises were assigned during the conversation that were never completed or followed up on?"
### Cluster 5 (12 turns)
**Top lift terms:** `smart` (71.2), `nest` (58.1), `home` (37.2), `google` (34.9), `doorbell` (34.9), `camera` (34.9), `off` (26.1), `plug` (23.2), `see` (23.2), `who's` (23.2)
**Examples:**
- [completeness] (conv `synth_technical`, turn 26, user): "For the bedroom I want smart lighting too. Just the bedside lamps - two of them. And I want motion sensors."
  - question: "List all smart home devices the user plans to purchase, organized by room and phase."
- [completeness] (conv `synth_technical`, turn 33, assistant): "Smart plugs for the coffee maker are a classic automation - start brewing before you even get out of bed. For the slow cooker, smart plugs are great but make sure you use a plug wi"
  - question: "List all smart home devices the user plans to purchase, organized by room and phase."
- [completeness] (conv `synth_technical`, turn 41, assistant): "A doorbell camera is one of the most practical smart home additions. The Nest Doorbell integrates perfectly with your Google ecosystem - you can see who's at the door on your phone"
  - question: "List all smart home devices the user plans to purchase, organized by room and phase."
## 4. Failure-type classification
Rule-based from computed features (lexical overlap, cosine, length, pronouns):
- topic_drift: 80 (54.1%)
- implicit_reference: 51 (34.5%)
- other: 10 (6.8%)
- lexical_mismatch: 7 (4.7%)
## 5. Question concentration
- 42 distinct questions contain at least one idiosyncratic miss.
- Top 10 questions hold 75 / 148 idiosyncratic misses.
| conv | q_idx | count |
|---|---:|---:|
| puzzle_explore_1 | 11 | 12 |
| adv_evolving_term_2 | 19 | 10 |
| adv_evolving_term_2 | 2 | 8 |
| puzzle_logic_2 | 2 | 8 |
| puzzle_absence_1 | 13 | 7 |
| synth_planning | 17 | 6 |
| synth_technical | 18 | 6 |
| puzzle_logic_1 | 1 | 6 |
| puzzle_absence_2 | 14 | 6 |
| puzzle_absence_2 | 15 | 6 |
## 6. Verdict
- Plausibly fixable via existing cue-generation architectures (v2f / chain_with_scratchpad): **58 / 148** (39.2%)
  (lexical_mismatch has sibling-gold signal to piggy-back; implicit_reference carries deictics that a small expand-with-prev-turn rule would catch.)
- Structural-role turns: **0** — short / transitional / meta-comments, best caught by always-attach-preceding-turn expansion.
- Likely retrieval ceiling (topic_drift + other, no surface or paraphrase handle): **90 / 148** (60.8%)

**Key finding:** see mean cosine(missed, q) vs mean cosine(retrieved_gold, q) above. If the gap is large (>0.1), embeddings alone cannot rank these turns competitively — a better embedding or query-side expansion is required.
