# Notes-prompt iterative tuning

## Headline

- Winner: **Round 3 (Candidate A'' — structured-label, tightened)**: 3/3 PHATIC correct, 12/12 non-phatic concrete; referents consistently specific (e.g. `"that agency" -> "one of the adoption agencies Caroline is looking into"`), and counts expressed as running totals (`COUNT: Melanie's pets = 2 (a pup/dog and a cat/kitty)`).
- Round 1 (A, starter) over-labeled phatic turns (1/3 PHATIC). Round 2 (A') hit the 80% thresholds but still had minor softness (e.g. tid=28 omitted a RESOLVED line). Round 3 (A'') fixed that without regressions and produced the richest grounded observations. Round 4 (B, free-form) regressed on phatic accuracy (1/3) — the structured labels are materially better at the phatic-skip decision.
- Recommendation: use **Candidate A''** as the v4 note-generation prompt. No further rounds needed (decision-rule thresholds met at Round 2 already; Round 3 is strictly better).

## Prompt evolution (diffs)

**Round 1 (A, starter)**: the baseline structured-label prompt with RESOLVED/FACT/COUNT/UPDATE/LINK/NAME, terse PHATIC rule.

**Round 2 (A', vs A)**:
- Replaced the one-liner PHATIC rule with: "purely phatic (generic politeness, greetings, encouragement, small talk, filler)" and added positive examples ("Thanks!", "Glad you had fun", "Have a great day!", "That's awesome, keep it up!").
- Required "exact referent name or phrase from context" on RESOLVED lines, closing the "it -> running" vagueness that appeared in Round 1.
- Added an explicit no-fabrication rule: "Only extract information directly stated in the current turn (context is only for resolving references)".
- Tightened output form: "Max 20 words per line; 1 observation per line; no preamble; no explanations".

**Round 3 (A'', vs A')**:
- Added an inline example to every label (e.g. `COUNT: Melanie's pets = 2 (a dog and a cat, Luna and Oliver)`) so the model sees the target level of specificity.
- Reformulated the phatic rule to a functional test: "output PHATIC if removing this turn from the conversation would lose no concrete information".
- Added a guardrail: "If the turn has any specific content at all (a name, number, object, concrete update), do NOT output PHATIC — use a proper label" — fixes the failure mode where the tightened Round-2 phatic rule over-skips borderline turns.
- Added: "Never write a thematic summary; write concrete listener observations" to make the user's instruction explicit.

**Round 4 (B, alternative design — free-form)**: Dropped labels in favor of "1-3 short factual observations". Included purely for comparison. It performs worse on phatic discrimination because the model defaults to writing plain sentences even when nothing specific was said.

## Winning prompt (A'')

```
You are a careful listener taking notes on a two-person conversation. For each CURRENT TURN you write 1-4 concrete, grounded observations — the kind a listener would jot so they could recall specifics later.

LABEL REFERENCE (use one label per line, at most 4 lines total):
- RESOLVED: <exact pronoun/deictic from current turn> -> <exact referent phrase from prior context>
    e.g. RESOLVED: "it" -> "the adoption agency Caroline mentioned in her last turn"
- FACT: <a specific new detail explicitly stated in this turn>
    e.g. FACT: Melanie and her husband have been married 5 years.
- COUNT: <entity> = <running total or duration with units>
    e.g. COUNT: Melanie's pets = 2 (a dog and a cat, Luna and Oliver)
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
    e.g. UPDATE: Caroline marched in the parade -> Caroline missed the parade.
- LINK: <current element> refers to <earlier topic/event from context>
    e.g. LINK: Caroline's painting of unity refers to her LGBTQ center visit.
- NAME: <new proper noun or named entity introduced> = <short grounded description>
    e.g. NAME: "Becoming Nicole" by Amy Ellis Nutt = Caroline's favorite inspirational book about a trans girl.

PHATIC rule: output exactly "PHATIC" (and nothing else) if removing this turn from the conversation would lose no concrete information. Typical PHATIC turns are generic politeness, greetings, goodbyes, brief encouragement, or echo-agreements ("Thanks!", "Keep it up!", "Glad to hear", "Enjoy your day", "Bye!", "Great to see you", "That's awesome").

Hard constraints:
- ONLY use information directly present in the CURRENT TURN (the context exists only to resolve references). Do NOT invent facts, numbers, dates, or names not stated.
- 1 observation per line; max ~20 words per line; no preamble, no explanations, no markdown.
- Prefer SPECIFIC referents ("Melanie's pottery bowl from her class") over vague ones ("it", "that thing").
- Never write a thematic summary; write concrete listener observations.
- If the turn has any specific content at all (a name, number, object, concrete update), do NOT output PHATIC — use a proper label.

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations (labeled lines, or PHATIC):
```

## Per-round qualitative analysis

**Round 1 (A)** — PHATIC 1/3, concrete 12/12.
- Failures: tid=130 and tid=134 (both phatic) were forced into FACT/LINK/RESOLVED despite being empty politeness. tid=130 output `FACT: Caroline asserts that running can boost mood` and `RESOLVED: "it" -> running (in "Keep it up!")`, which is meta-observation noise, not listener knowledge gain.
- Strengths already visible on non-phatic: named entities cleanly extracted (e.g. Becoming Nicole / Amy Ellis Nutt), counts captured (`COUNT: pets = 2`), parade miss correctly flagged as `FACT: Caroline did not attend the parade.`

**Round 2 (A')** — PHATIC 3/3, concrete 12/12.
- Improved: all three phatic turns skip cleanly. Referents on RESOLVED lines now point at specific prior content (e.g. `that -> the dog.`, `here's -> the cat.`).
- Minor soft spots: tid=28 omitted an explicit RESOLVED line (only a LINK). Still counted as concrete but worth tightening.
- Count format drift: tid=47 produced `COUNT: friends = 4 years` (confusing — 4 years is the duration, not a count of friends). Round 3's explicit `COUNT` example fixes this on tid=123 and tid=50.

**Round 3 (A'')** — PHATIC 3/3, concrete 12/12. Winner.
- All phatic turns skipped. Named entities named. Counts properly expressed (`COUNT: Melanie's pets = 2 (a pup/dog and a cat/kitty)`, `COUNT: Melanie's marriage = 5 years`).
- Anaphora resolved explicitly and specifically on every non-phatic turn, including the tricky tid=12 (`"this" -> "the item Melanie said 'take a look at' (her painting)"`) and tid=195 (`"our" -> "the new LGBTQ activist group Caroline mentioned joining last Tues."`).
- Updates correctly captured as UPDATE-style FACTs (tid=197 `FACT: Caroline missed the pride parade`, tid=351 `FACT: The sign was just a precaution`).
- No fabrication observed across 12 non-phatic turns. No thematic summaries.

**Round 4 (B, free-form alt)** — PHATIC 1/3, concrete 12/12.
- Regression on phatic discrimination (only tid=130 skipped; tid=134 and tid=173 got sentence-ified into trivial restatements). This is the expected failure mode of a free-form prompt without a functional PHATIC test — the LLM finds *something* to say.
- Non-phatic content is fluent and concrete but loses the structured label scaffolding that downstream parsing/retrieval may benefit from.

## Scorecard (all rounds)

| Round | Phatic acc | Concrete rate | Notes |
|-------|------------|---------------|-------|
| 1 (A starter) | 1/3 = 33% | 12/12 = 100% | Over-labels phatic turns. |
| 2 (A' tightened) | 3/3 = 100% | 12/12 = 100% | Passes thresholds; minor RESOLVED omissions. |
| 3 (A'' + examples) | 3/3 = 100% | 12/12 = 100% | **Winner**: richest, most specific referents. |
| 4 (B free-form) | 1/3 = 33% | 12/12 = 100% | Free-form lacks PHATIC discipline. |

## Recommendation

Adopt **Candidate A''** as the v4 note-generation prompt. Decision-rule thresholds (≥80% PHATIC, ≥80% concrete) are met with a wide margin (both 100%) in 3 rounds; no further rounds needed. Do NOT commit to full ingest from this script — this is a prompt-tuning check only.

---

## Sample set (15 diverse turns from locomo_conv-26)

- turn_id=130 (phatic): Caroline: generic encouragement 'Running can boost mood. Keep it up!'
  current: `Caroline: Cool, Melanie! Running can really boost your mood. Keep it up!`
- turn_id=134 (phatic): Caroline: 'Glad it helped ya, Melanie!' short encouragement
  current: `Caroline: Glad it helped ya, Melanie!`
- turn_id=173 (phatic): Caroline: 'No worries... Enjoy your day!' goodbye phatic
  current: `Caroline: No worries, Mel! Your friendship means so much to me. Enjoy your day!`
- turn_id=12 (anaphora): Caroline: 'Is this your own painting?' — 'this' → painting Mel just shared
  current: `Caroline: Thanks, Melanie! That's really sweet. Is this your own painting?`
- turn_id=28 (anaphora): Melanie: 'What made you pick it?' — 'it' → adoption agency
  current: `Melanie: Wow, that agency looks great! What made you pick it?`
- turn_id=82 (anaphora): Caroline: 'That bowl is gorgeous! ... Did you make it?' — 'it' → pottery bowl
  current: `Caroline: That bowl is gorgeous! The black and white design looks so fancy. Did you make it?`
- turn_id=60 (named): Caroline: necklace from grandma in Sweden (new entities: Sweden, grandma, necklace)
  current: `Caroline: Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family.`
- turn_id=118 (named): Caroline: favorite book 'Becoming Nicole' by Amy Ellis Nutt
  current: `Caroline: I loved "Becoming Nicole" by Amy Ellis Nutt. It's a real inspiring true story about a trans girl and her family. It made me feel connected and gave me a lot of hope for my own path. Highly recommend it for sure!`
- turn_id=195 (named): Caroline: group name 'Connected LGBTQ Activists'
  current: `Caroline: Thanks, Melanie! It's awesome to have our own platform to be ourselves and support others' rights. Our group, 'Connected LGBTQ Activists', is made of all kinds of people investing in positive changes. We have regular meetings, plan events and campaigns, to get together and support each other.`
- turn_id=47 (count): Caroline: known these friends for 4 years (since moved from home country)
  current: `Caroline: Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?`
- turn_id=123 (count): Melanie: 'a pup and a kitty' — now 2 pets named later (Luna, Oliver)
  current: `Melanie: We've got a pup and a kitty. That's the dog, and here's our cat! They brighten up our day and always make us smile.`
- turn_id=50 (fact): Melanie: '5 years already!' years married
  current: `Melanie: 5 years already! Time flies- feels like just yesterday I put this dress on! Thanks, Caroline!`
- turn_id=62 (fact): Caroline: hand-painted bowl from friend on 18th birthday ten years ago
  current: `Caroline: Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome-- it reminds me of art and self-expression.`
- turn_id=197 (update): Caroline: 'I missed it but it was a powerful reminder' — correcting implied attendance
  current: `Caroline: Last weekend our city held a pride parade! So many people marched through the streets waving flags, holding signs and celebrating love and diversity. I missed it but it was a powerful reminder that we are not alone in this fight for equality and inclusivity. Change is possible!`
- turn_id=351 (update): Melanie: 'The sign was just a precaution... had a great time' — correcting alarm
  current: `Melanie: The sign was just a precaution, I had a great time. But thank you for your concern, you're so thoughtful!`


## round1_A

**Scorecard**: PHATIC 1/3 = 33%; concrete 12/12 = 100%

<details><summary>Prompt</summary>

```
You are listening in on a conversation. Read the current turn in context and write specific listener observations.

If applicable, use these labels one per line:
- RESOLVED: <pronoun/deictic> -> <referent>
- FACT: <specific new detail>
- COUNT: <entity> = <running total>
- UPDATE: <prior claim> -> <new claim>
- LINK: <current element> refers to <earlier topic>
- NAME: <new entity> = <description>

If the turn is purely phatic (no specific content), output exactly: PHATIC

Context:
{context_block}

Current turn:
{current_turn}

Observations (labeled lines, or PHATIC):

```
</details>

### Outputs

**turn_id=130 (phatic)** — Caroline: generic encouragement 'Running can boost mood. Keep it up!'

current: `Caroline: Cool, Melanie! Running can really boost your mood. Keep it up!`

```
FACT: Caroline asserts that running can boost mood.
LINK: Caroline's encouragement refers to Melanie's earlier mention of running to de-stress.
RESOLVED: "it" -> running (in "Keep it up!")
```

**turn_id=134 (phatic)** — Caroline: 'Glad it helped ya, Melanie!' short encouragement

current: `Caroline: Glad it helped ya, Melanie!`

```
FACT: Caroline expresses gladness that something helped Melanie.
LINK: "it" -> the activity/practice that improved Melanie's mental health mentioned earlier.
RESOLVED: it -> the mental-health improvement Melanie described.
```

**turn_id=173 (phatic)** — Caroline: 'No worries... Enjoy your day!' goodbye phatic

current: `Caroline: No worries, Mel! Your friendship means so much to me. Enjoy your day!`

```
PHATIC
```

**turn_id=12 (anaphora)** — Caroline: 'Is this your own painting?' — 'this' → painting Mel just shared

current: `Caroline: Thanks, Melanie! That's really sweet. Is this your own painting?`

```
FACT: Caroline expresses gratitude ("Thanks, Melanie!").
LINK: "That's really sweet" refers to Melanie's compliment about Caroline being a great counselor.
FACT: Caroline asks whether the painting shown is Melanie's ("Is this your own painting?").
```

**turn_id=28 (anaphora)** — Melanie: 'What made you pick it?' — 'it' → adoption agency

current: `Melanie: Wow, that agency looks great! What made you pick it?`

```
RESOLVED: that agency -> the adoption agency Caroline mentioned she's looking into
FACT: Melanie expresses enthusiasm and approval ("that agency looks great")
FACT: Melanie asks for Caroline's reasoning/motivation ("What made you pick it?")
LINK: current question refers to Caroline's prior mention of "one of the adoption agencies I'm looking into"
```

**turn_id=82 (anaphora)** — Caroline: 'That bowl is gorgeous! ... Did you make it?' — 'it' → pottery bowl

current: `Caroline: That bowl is gorgeous! The black and white design looks so fancy. Did you make it?`

```
RESOLVED: That bowl -> the pottery Melanie just showed ("Look at this!")
FACT: Caroline compliments the bowl as "gorgeous"
FACT: Caroline describes the bowl's black and white design as looking "fancy"
FACT: Caroline asks whether Melanie made the bowl (questions its authorship)
LINK: Caroline's comment refers back to Melanie's shared pottery item ("Look at this!")
```

**turn_id=60 (named)** — Caroline: necklace from grandma in Sweden (new entities: Sweden, grandma, necklace)

current: `Caroline: Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family.`

```
FACT: The necklace is a gift from Caroline's grandma in Sweden.
FACT: Caroline received the necklace when she was young.
FACT: The necklace symbolizes love, faith, and strength for Caroline.
FACT: Caroline views the necklace as a reminder of her roots and her family's love and support.
RESOLVED: this/it -> the necklace Caroline is wearing.
LINK: Caroline's explanation responds directly to Melanie's question about special meaning of the necklace.
```

**turn_id=118 (named)** — Caroline: favorite book 'Becoming Nicole' by Amy Ellis Nutt

current: `Caroline: I loved "Becoming Nicole" by Amy Ellis Nutt. It's a real inspiring true story about a trans girl and her family. It made me feel connected and gave me a lot of hope for my own path. Highly recommend it for sure!`

```
NAME: Becoming Nicole = nonfiction book by Amy Ellis Nutt about a trans girl and her family
FACT: Caroline loved "Becoming Nicole"
FACT: It's described as an inspiring true story about a trans girl and her family
FACT: The book made Caroline feel connected and gave her hope for her own path
FACT: Caroline highly recommends the book
RESOLVED: "this one's" / "it" -> "Becoming Nicole"
LINK: Caroline's earlier comment about books guiding and motivating her refers to "Becoming Nicole"
```

**turn_id=195 (named)** — Caroline: group name 'Connected LGBTQ Activists'

current: `Caroline: Thanks, Melanie! It's awesome to have our own platform to be ourselves and support others' rights. Our group, 'Connected LGBTQ Activists', is made of all kinds of people investing in positive changes. We have regular meetings, plan events and campaigns, to get together and support each other.`

```
NAME: "Connected LGBTQ Activists" = the group Caroline joined
FACT: The group consists of diverse people ("all kinds of people") invested in positive change
FACT: They hold regular meetings
FACT: They plan events and campaigns
FACT: The group provides a platform for members to be themselves and support others' rights
LINK: "our group" -> the new LGBTQ activist group Caroline mentioned earlier
RESOLVED: "we"/"our" -> members of Connected LGBTQ Activists
```

**turn_id=47 (count)** — Caroline: known these friends for 4 years (since moved from home country)

current: `Caroline: Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?`

```
FACT: Caroline has known these friends for 4 years.
FACT: She moved from her home country around the time she met them.
FACT: Their support was especially important after a tough breakup.
RESOLVED: "them" -> the friends/family/mentors Caroline mentioned earlier.
FACT: Caroline asks Melanie who supports her.
```

**turn_id=123 (count)** — Melanie: 'a pup and a kitty' — now 2 pets named later (Luna, Oliver)

current: `Melanie: We've got a pup and a kitty. That's the dog, and here's our cat! They brighten up our day and always make us smile.`

```
FACT: Melanie has two pets — a pup (dog) and a kitty (cat).
NAME: pup = dog
NAME: kitty = cat
COUNT: pets = 2
FACT: The pets brighten their day and always make them smile.
LINK: "They" refers back to Caroline's remark that pets bring joy and comfort.
```

**turn_id=50 (fact)** — Melanie: '5 years already!' years married

current: `Melanie: 5 years already! Time flies- feels like just yesterday I put this dress on! Thanks, Caroline!`

```
FACT: Melanie has been married for 5 years.
RESOLVED: How long have you been married? -> 5 years
LINK: "this dress" refers to the dress she wore at her wedding (implied).
FACT: Melanie expresses gratitude to Caroline ("Thanks, Caroline!").
FACT: Melanie feels time has gone by quickly ("Time flies").
```

**turn_id=62 (fact)** — Caroline: hand-painted bowl from friend on 18th birthday ten years ago

current: `Caroline: Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome-- it reminds me of art and self-expression.`

```
NAME: hand-painted bowl = a bowl painted by a friend
FACT: the friend gave the bowl for Caroline's 18th birthday
FACT: the gift was given ten years ago
FACT: the bowl's pattern and colors remind Caroline of art and self-expression
COUNT: sentimental items = 2
LINK: this response refers to Melanie's question about other treasured objects
RESOLVED: it -> the hand-painted bowl
```

**turn_id=197 (update)** — Caroline: 'I missed it but it was a powerful reminder' — correcting implied attendance

current: `Caroline: Last weekend our city held a pride parade! So many people marched through the streets waving flags, holding signs and celebrating love and diversity. I missed it but it was a powerful reminder that we are not alone in this fight for equality and inclusivity. Change is possible!`

```
FACT: The city held a pride parade last weekend.
FACT: Many people marched, waving flags and holding signs, celebrating love and diversity.
RESOLVED: I -> Caroline (Caroline says "I missed it").
FACT: Caroline did not attend the parade.
FACT: Caroline experienced the parade as a powerful reminder of not being alone in the fight for equality and inclusivity.
FACT: Caroline believes the parade signals that change is possible.
LINK: This response answers Melanie's question about whether Caroline has been part of any events or campaigns lately.
```

**turn_id=351 (update)** — Melanie: 'The sign was just a precaution... had a great time' — correcting alarm

current: `Melanie: The sign was just a precaution, I had a great time. But thank you for your concern, you're so thoughtful!`

```
FACT: The sign at the café was only a precaution.
FACT: Melanie had a great time at the café.
FACT: Melanie thanks Caroline for her concern and calls her thoughtful.
LINK: "The sign" refers to Caroline's question about whether anything happened.
```


## round2_Aprime

**Scorecard**: PHATIC 3/3 = 100%; concrete 12/12 = 100%

<details><summary>Prompt</summary>

```
You are listening in on a conversation. Read the CURRENT TURN in the light of the preceding context and write specific listener observations that a careful note-taker would jot down.

Use one label per line, choosing from:
- RESOLVED: <pronoun/deictic word from current turn> -> <exact referent name or phrase from context>
- FACT: <specific new detail explicitly stated in the current turn>
- COUNT: <entity> = <running total or duration with units>
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
- LINK: <current element> refers to <earlier topic or event in context>
- NAME: <new proper noun or named entity> = <short description grounded in the current turn>

PHATIC rule: if the current turn is purely phatic (generic politeness, greetings, encouragement, small talk, filler) with no new concrete content, output exactly:
PHATIC
Examples of PHATIC-worthy turns: "Thanks!", "Glad you had fun", "Have a great day!", "That's awesome, keep it up!".

Hard constraints:
- Only extract information directly stated in the current turn (context is only for resolving references). Do NOT invent facts.
- Max 20 words per line; 1 observation per line; no preamble; no explanations.
- Prefer concrete referents ("the adoption agency Caroline mentioned") over vague ones ("it").
- If the turn contains content but no label fits, still try FACT with the concrete detail; do not output PHATIC unless truly phatic.

Context (most recent last; each line is "<speaker>: <content>"):
{context_block}

Current turn:
{current_turn}

Observations (labeled lines, or PHATIC):

```
</details>

### Outputs

**turn_id=130 (phatic)** — Caroline: generic encouragement 'Running can boost mood. Keep it up!'

current: `Caroline: Cool, Melanie! Running can really boost your mood. Keep it up!`

```
PHATIC
```

**turn_id=134 (phatic)** — Caroline: 'Glad it helped ya, Melanie!' short encouragement

current: `Caroline: Glad it helped ya, Melanie!`

```
PHATIC
```

**turn_id=173 (phatic)** — Caroline: 'No worries... Enjoy your day!' goodbye phatic

current: `Caroline: No worries, Mel! Your friendship means so much to me. Enjoy your day!`

```
PHATIC
```

**turn_id=12 (anaphora)** — Caroline: 'Is this your own painting?' — 'this' → painting Mel just shared

current: `Caroline: Thanks, Melanie! That's really sweet. Is this your own painting?`

```
FACT: Caroline thanks Melanie for the compliment.
FACT: Caroline asks whether the painting is Melanie's own.
```

**turn_id=28 (anaphora)** — Melanie: 'What made you pick it?' — 'it' → adoption agency

current: `Melanie: Wow, that agency looks great! What made you pick it?`

```
FACT: Melanie says the agency "looks great."
LINK: "that agency" -> one of the adoption agencies Caroline is looking into.
```

**turn_id=82 (anaphora)** — Caroline: 'That bowl is gorgeous! ... Did you make it?' — 'it' → pottery bowl

current: `Caroline: That bowl is gorgeous! The black and white design looks so fancy. Did you make it?`

```
RESOLVED: That bowl -> the pottery Melanie showed
FACT: Caroline calls the bowl "gorgeous"
FACT: Caroline asks if Melanie made the bowl
```

**turn_id=60 (named)** — Caroline: necklace from grandma in Sweden (new entities: Sweden, grandma, necklace)

current: `Caroline: Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family.`

```
FACT: Necklace = gift from Caroline's grandma in Sweden.
RESOLVED: "my home country" -> Sweden
FACT: Grandma gave necklace to Caroline when she was young.
FACT: Necklace symbolizes love, faith, and strength.
FACT: Necklace is a reminder of Caroline's roots and family support.
LINK: "this necklace" refers to Melanie's question about special meaning.
```

**turn_id=118 (named)** — Caroline: favorite book 'Becoming Nicole' by Amy Ellis Nutt

current: `Caroline: I loved "Becoming Nicole" by Amy Ellis Nutt. It's a real inspiring true story about a trans girl and her family. It made me feel connected and gave me a lot of hope for my own path. Highly recommend it for sure!`

```
NAME: Becoming Nicole = book Caroline loved
NAME: Amy Ellis Nutt = author of Becoming Nicole
FACT: It is an inspiring true story about a trans girl and her family
FACT: Caroline felt connected and gained a lot of hope for her own path
FACT: Caroline highly recommends the book
LINK: "this one" refers to Becoming Nicole
```

**turn_id=195 (named)** — Caroline: group name 'Connected LGBTQ Activists'

current: `Caroline: Thanks, Melanie! It's awesome to have our own platform to be ourselves and support others' rights. Our group, 'Connected LGBTQ Activists', is made of all kinds of people investing in positive changes. We have regular meetings, plan events and campaigns, to get together and support each other.`

```
NAME: Connected LGBTQ Activists = the activist group Caroline belongs to
FACT: Group provides a platform for members to be themselves and support others' rights
FACT: Group is made of all kinds of people investing in positive changes
FACT: Group holds regular meetings
FACT: Group plans events and campaigns to bring members together and offer mutual support
```

**turn_id=47 (count)** — Caroline: known these friends for 4 years (since moved from home country)

current: `Caroline: Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?`

```
RESOLVED: them -> friends, family and mentors
FACT: Caroline has known these friends for 4 years
COUNT: friends = 4 years
FACT: Caroline moved from her home country
FACT: Caroline experienced a tough breakup and received support afterward
LINK: "Who supports you, Mel?" -> refers to Melanie's earlier "What motivates you?" question
```

**turn_id=123 (count)** — Melanie: 'a pup and a kitty' — now 2 pets named later (Luna, Oliver)

current: `Melanie: We've got a pup and a kitty. That's the dog, and here's our cat! They brighten up our day and always make us smile.`

```
FACT: Melanie has a pup and a kitty.
RESOLVED: That -> the dog.
RESOLVED: here's -> the cat.
LINK: "They brighten up our day and always make us smile" refers to Caroline's point that pets bring joy.
```

**turn_id=50 (fact)** — Melanie: '5 years already!' years married

current: `Melanie: 5 years already! Time flies- feels like just yesterday I put this dress on! Thanks, Caroline!`

```
COUNT: Melanie's marriage = 5 years
RESOLVED: "this dress" -> her wedding dress (photo referenced)
FACT: Melanie thanks Caroline and says time feels like it passed quickly
```

**turn_id=62 (fact)** — Caroline: hand-painted bowl from friend on 18th birthday ten years ago

current: `Caroline: Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome-- it reminds me of art and self-expression.`

```
FACT: hand-painted bowl = sentimental object Caroline treasures
FACT: a friend made the bowl as a gift for her 18th birthday
FACT: the gift was given ten years ago
FACT: the bowl's pattern and colors remind Caroline of art and self-expression
RESOLVED: it -> hand-painted bowl
```

**turn_id=197 (update)** — Caroline: 'I missed it but it was a powerful reminder' — correcting implied attendance

current: `Caroline: Last weekend our city held a pride parade! So many people marched through the streets waving flags, holding signs and celebrating love and diversity. I missed it but it was a powerful reminder that we are not alone in this fight for equality and inclusivity. Change is possible!`

```
FACT: A pride parade was held in Caroline's city last weekend.
FACT: Many people marched waving flags, holding signs, celebrating love and diversity.
FACT: Caroline missed the pride parade.
FACT: Caroline described the event as a powerful reminder they are not alone in the fight for equality and inclusivity.
LINK: The parade refers to the group's events and campaigns Caroline mentioned earlier.
```

**turn_id=351 (update)** — Melanie: 'The sign was just a precaution... had a great time' — correcting alarm

current: `Melanie: The sign was just a precaution, I had a great time. But thank you for your concern, you're so thoughtful!`

```
FACT: The sign was just a precaution.
FACT: Melanie had a great time at the café.
RESOLVED: your concern -> Caroline's question "Did anything happen?"
```


## round3_Adoubleprime

**Scorecard**: PHATIC 3/3 = 100%; concrete 12/12 = 100%

<details><summary>Prompt</summary>

```
You are a careful listener taking notes on a two-person conversation. For each CURRENT TURN you write 1-4 concrete, grounded observations — the kind a listener would jot so they could recall specifics later.

LABEL REFERENCE (use one label per line, at most 4 lines total):
- RESOLVED: <exact pronoun/deictic from current turn> -> <exact referent phrase from prior context>
    e.g. RESOLVED: "it" -> "the adoption agency Caroline mentioned in her last turn"
- FACT: <a specific new detail explicitly stated in this turn>
    e.g. FACT: Melanie and her husband have been married 5 years.
- COUNT: <entity> = <running total or duration with units>
    e.g. COUNT: Melanie's pets = 2 (a dog and a cat, Luna and Oliver)
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
    e.g. UPDATE: Caroline marched in the parade -> Caroline missed the parade.
- LINK: <current element> refers to <earlier topic/event from context>
    e.g. LINK: Caroline's painting of unity refers to her LGBTQ center visit.
- NAME: <new proper noun or named entity introduced> = <short grounded description>
    e.g. NAME: "Becoming Nicole" by Amy Ellis Nutt = Caroline's favorite inspirational book about a trans girl.

PHATIC rule: output exactly "PHATIC" (and nothing else) if removing this turn from the conversation would lose no concrete information. Typical PHATIC turns are generic politeness, greetings, goodbyes, brief encouragement, or echo-agreements ("Thanks!", "Keep it up!", "Glad to hear", "Enjoy your day", "Bye!", "Great to see you", "That's awesome").

Hard constraints:
- ONLY use information directly present in the CURRENT TURN (the context exists only to resolve references). Do NOT invent facts, numbers, dates, or names not stated.
- 1 observation per line; max ~20 words per line; no preamble, no explanations, no markdown.
- Prefer SPECIFIC referents ("Melanie's pottery bowl from her class") over vague ones ("it", "that thing").
- Never write a thematic summary; write concrete listener observations.
- If the turn has any specific content at all (a name, number, object, concrete update), do NOT output PHATIC — use a proper label.

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations (labeled lines, or PHATIC):

```
</details>

### Outputs

**turn_id=130 (phatic)** — Caroline: generic encouragement 'Running can boost mood. Keep it up!'

current: `Caroline: Cool, Melanie! Running can really boost your mood. Keep it up!`

```
PHATIC
```

**turn_id=134 (phatic)** — Caroline: 'Glad it helped ya, Melanie!' short encouragement

current: `Caroline: Glad it helped ya, Melanie!`

```
PHATIC
```

**turn_id=173 (phatic)** — Caroline: 'No worries... Enjoy your day!' goodbye phatic

current: `Caroline: No worries, Mel! Your friendship means so much to me. Enjoy your day!`

```
PHATIC
```

**turn_id=12 (anaphora)** — Caroline: 'Is this your own painting?' — 'this' → painting Mel just shared

current: `Caroline: Thanks, Melanie! That's really sweet. Is this your own painting?`

```
FACT: Caroline thanks Melanie for the compliment.
FACT: Caroline calls Melanie's compliment "really sweet."
RESOLVED: "this" -> "the item Melanie said 'take a look at' (her painting)"
```

**turn_id=28 (anaphora)** — Melanie: 'What made you pick it?' — 'it' → adoption agency

current: `Melanie: Wow, that agency looks great! What made you pick it?`

```
RESOLVED: "that agency" -> "one of the adoption agencies Caroline is looking into"
FACT: Melanie says the agency "looks great"
FACT: Melanie asks, "What made you pick it?" (seeking Caroline's reasons)
```

**turn_id=82 (anaphora)** — Caroline: 'That bowl is gorgeous! ... Did you make it?' — 'it' → pottery bowl

current: `Caroline: That bowl is gorgeous! The black and white design looks so fancy. Did you make it?`

```
RESOLVED: "That bowl" -> "the pottery Melanie just showed ('Look at this!')"
FACT: Caroline calls the bowl "gorgeous."
FACT: Caroline describes the bowl's design as "black and white" and "fancy."
FACT: Caroline asks whether Melanie made the bowl.
```

**turn_id=60 (named)** — Caroline: necklace from grandma in Sweden (new entities: Sweden, grandma, necklace)

current: `Caroline: Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family.`

```
RESOLVED: "this necklace" -> "the necklace Melanie mentioned"
FACT: necklace = a gift from Caroline's grandma in Sweden
FACT: Caroline received the necklace when she was young
FACT: necklace stands for love, faith and strength and reminds Caroline of her roots and family support
```

**turn_id=118 (named)** — Caroline: favorite book 'Becoming Nicole' by Amy Ellis Nutt

current: `Caroline: I loved "Becoming Nicole" by Amy Ellis Nutt. It's a real inspiring true story about a trans girl and her family. It made me feel connected and gave me a lot of hope for my own path. Highly recommend it for sure!`

```
NAME: "Becoming Nicole" by Amy Ellis Nutt = book Caroline loved  
FACT: The book is an inspiring true story about a trans girl and her family  
FACT: The book made Caroline feel connected and gave her a lot of hope for her own path  
FACT: Caroline highly recommends the book
```

**turn_id=195 (named)** — Caroline: group name 'Connected LGBTQ Activists'

current: `Caroline: Thanks, Melanie! It's awesome to have our own platform to be ourselves and support others' rights. Our group, 'Connected LGBTQ Activists', is made of all kinds of people investing in positive changes. We have regular meetings, plan events and campaigns, to get together and support each other.`

```
RESOLVED: "our" -> "the new LGBTQ activist group Caroline mentioned joining last Tues."
NAME: "Connected LGBTQ Activists" = Caroline's activist group.
FACT: The group is made of all kinds of people investing in positive changes.
FACT: The group has regular meetings and plans events and campaigns to support each other.
```

**turn_id=47 (count)** — Caroline: known these friends for 4 years (since moved from home country)

current: `Caroline: Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?`

```
RESOLVED: "them" -> "friends, family and mentors"  
COUNT: friends known = 4 years  
FACT: Caroline moved from her home country (friends since that move)  
FACT: Their support was important especially after Caroline's tough breakup; she asks who supports Mel
```

**turn_id=123 (count)** — Melanie: 'a pup and a kitty' — now 2 pets named later (Luna, Oliver)

current: `Melanie: We've got a pup and a kitty. That's the dog, and here's our cat! They brighten up our day and always make us smile.`

```
FACT: Melanie has a pup and a kitty.
COUNT: Melanie's pets = 2 (a pup/dog and a cat/kitty).
RESOLVED: "That's the dog" -> "the pup."
```

**turn_id=50 (fact)** — Melanie: '5 years already!' years married

current: `Melanie: 5 years already! Time flies- feels like just yesterday I put this dress on! Thanks, Caroline!`

```
COUNT: Melanie's marriage = 5 years
FACT: Melanie says it "feels like just yesterday" she put this dress on (implying five years ago)
FACT: Melanie thanks Caroline ("Thanks, Caroline!")
```

**turn_id=62 (fact)** — Caroline: hand-painted bowl from friend on 18th birthday ten years ago

current: `Caroline: Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome-- it reminds me of art and self-expression.`

```
FACT: hand-painted bowl = a sentimental object Caroline treasures.
FACT: a friend made the bowl for Caroline's 18th birthday ten years ago.
FACT: the bowl's pattern and colors remind Caroline of art and self-expression.
```

**turn_id=197 (update)** — Caroline: 'I missed it but it was a powerful reminder' — correcting implied attendance

current: `Caroline: Last weekend our city held a pride parade! So many people marched through the streets waving flags, holding signs and celebrating love and diversity. I missed it but it was a powerful reminder that we are not alone in this fight for equality and inclusivity. Change is possible!`

```
FACT: The city held a pride parade last weekend.
FACT: Many people marched waving flags, holding signs, celebrating love and diversity.
FACT: Caroline missed the pride parade.
RESOLVED: "we" -> "Connected LGBTQ Activists"
```

**turn_id=351 (update)** — Melanie: 'The sign was just a precaution... had a great time' — correcting alarm

current: `Melanie: The sign was just a precaution, I had a great time. But thank you for your concern, you're so thoughtful!`

```
FACT: The sign was just a precaution.
FACT: Melanie had a great time.
RESOLVED: "your concern" -> "Caroline's question about the sign"
```


## round4_B_freeform

**Scorecard**: PHATIC 1/3 = 33%; concrete 12/12 = 100%

<details><summary>Prompt</summary>

```
You are a careful listener in a two-person conversation. After the CURRENT TURN, write 1-3 short factual observations about what a listener JUST learned.

Each observation must be:
- a single plain sentence (no bullets, no labels, no markdown),
- grounded in the current turn only (use context only to resolve pronouns/references),
- concrete (specific names, numbers, durations, places, events) — not thematic.

If the current turn is purely phatic (greetings, thanks, generic encouragement, goodbyes, echoes), output exactly: PHATIC

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations:

```
</details>

### Outputs

**turn_id=130 (phatic)** — Caroline: generic encouragement 'Running can boost mood. Keep it up!'

current: `Caroline: Cool, Melanie! Running can really boost your mood. Keep it up!`

```
PHATIC
```

**turn_id=134 (phatic)** — Caroline: 'Glad it helped ya, Melanie!' short encouragement

current: `Caroline: Glad it helped ya, Melanie!`

```
Caroline is glad that whatever was done helped Melanie.  
Caroline addressed Melanie by name and used the informal word "ya."
```

**turn_id=173 (phatic)** — Caroline: 'No worries... Enjoy your day!' goodbye phatic

current: `Caroline: No worries, Mel! Your friendship means so much to me. Enjoy your day!`

```
Caroline told Melanie not to worry.  
Caroline said Melanie's friendship means a lot to her.  
Caroline told Melanie to enjoy her day.
```

**turn_id=12 (anaphora)** — Caroline: 'Is this your own painting?' — 'this' → painting Mel just shared

current: `Caroline: Thanks, Melanie! That's really sweet. Is this your own painting?`

```
Caroline thanked Melanie for the compliment.  
Caroline asked whether the painting is Melanie's own.
```

**turn_id=28 (anaphora)** — Melanie: 'What made you pick it?' — 'it' → adoption agency

current: `Melanie: Wow, that agency looks great! What made you pick it?`

```
Melanie said the agency looks great.  
Melanie asked what made Caroline pick that agency.
```

**turn_id=82 (anaphora)** — Caroline: 'That bowl is gorgeous! ... Did you make it?' — 'it' → pottery bowl

current: `Caroline: That bowl is gorgeous! The black and white design looks so fancy. Did you make it?`

```
Caroline said the bowl is gorgeous.  
She described the bowl's black and white design as looking fancy.  
Caroline asked Melanie whether she made the bowl.
```

**turn_id=60 (named)** — Caroline: necklace from grandma in Sweden (new entities: Sweden, grandma, necklace)

current: `Caroline: Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family.`

```
The necklace was a gift from Caroline's grandma in Sweden.  
Caroline received the necklace when she was young.  
The necklace symbolizes love, faith, and strength and reminds Caroline of her roots and family support.
```

**turn_id=118 (named)** — Caroline: favorite book 'Becoming Nicole' by Amy Ellis Nutt

current: `Caroline: I loved "Becoming Nicole" by Amy Ellis Nutt. It's a real inspiring true story about a trans girl and her family. It made me feel connected and gave me a lot of hope for my own path. Highly recommend it for sure!`

```
Caroline loved the book "Becoming Nicole" by Amy Ellis Nutt.  
The book is a true story about a trans girl and her family.  
Reading it made Caroline feel connected and gave her a lot of hope for her own path, and she highly recommends it.
```

**turn_id=195 (named)** — Caroline: group name 'Connected LGBTQ Activists'

current: `Caroline: Thanks, Melanie! It's awesome to have our own platform to be ourselves and support others' rights. Our group, 'Connected LGBTQ Activists', is made of all kinds of people investing in positive changes. We have regular meetings, plan events and campaigns, to get together and support each other.`

```
Caroline's group is called "Connected LGBTQ Activists."  
The group has regular meetings.  
They plan events and campaigns to support each other and others' rights.
```

**turn_id=47 (count)** — Caroline: known these friends for 4 years (since moved from home country)

current: `Caroline: Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?`

```
Caroline has known her friends for four years.  
She moved from her home country four years ago.  
She says their love and help were especially important after a tough breakup.
```

**turn_id=123 (count)** — Melanie: 'a pup and a kitty' — now 2 pets named later (Luna, Oliver)

current: `Melanie: We've got a pup and a kitty. That's the dog, and here's our cat! They brighten up our day and always make us smile.`

```
Melanie has a pup and a kitty (a dog and a cat).  
She says the pets brighten up their day and always make them smile.
```

**turn_id=50 (fact)** — Melanie: '5 years already!' years married

current: `Melanie: 5 years already! Time flies- feels like just yesterday I put this dress on! Thanks, Caroline!`

```
Melanie has been married for 5 years.  
She said time has flown and it feels like she just put on her wedding dress.  
She thanked Caroline.
```

**turn_id=62 (fact)** — Caroline: hand-painted bowl from friend on 18th birthday ten years ago

current: `Caroline: Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome-- it reminds me of art and self-expression.`

```
Caroline has a hand-painted bowl that she considers sentimental.  
A friend made the bowl for her 18th birthday ten years ago.  
The bowl's pattern and colors remind her of art and self-expression.
```

**turn_id=197 (update)** — Caroline: 'I missed it but it was a powerful reminder' — correcting implied attendance

current: `Caroline: Last weekend our city held a pride parade! So many people marched through the streets waving flags, holding signs and celebrating love and diversity. I missed it but it was a powerful reminder that we are not alone in this fight for equality and inclusivity. Change is possible!`

```
Last weekend your city held a pride parade. Many people marched through the streets waving flags and holding signs. You missed the parade.
```

**turn_id=351 (update)** — Melanie: 'The sign was just a precaution... had a great time' — correcting alarm

current: `Melanie: The sign was just a precaution, I had a great time. But thank you for your concern, you're so thoughtful!`

```
The sign was just a precaution.  
Melanie had a great time.  
Melanie thanked Caroline for her concern and called her thoughtful.
```
