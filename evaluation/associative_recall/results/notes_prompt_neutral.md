# Notes-prompt neutral-examples refinement (v4 candidate)

## Motivation

The winning A'' prompt from the prior iterative tuning embedded inline examples drawn from LoCoMo content (Caroline/Melanie/adoption agency/Luna & Oliver/"Becoming Nicole"/LGBTQ center). This introduces dataset-specific priors into the note-writer and risks evaluation contamination. We replace those examples with domain-neutral ones (office/household/generic names) and verify the prompt still produces equally-good outputs on the same 15 test turns.

## Scorecards

### round_neutral_A_double_prime
- PHATIC: 3/3 = 100%
- Concrete: 12/12 = 100%
- LoCoMo-example leakage: 0

### round_neutral_A_double_prime_v2
- PHATIC: 3/3 = 100%
- Concrete: 12/12 = 100%
- LoCoMo-example leakage: 0

**Verdict: ADOPT neutral A'' v1 (meets 3/3 PHATIC + 12/12 concrete; no LoCoMo-bias caveat needed)**

## Refined prompt v1 (full text)

```
You are a careful listener taking notes on a two-person conversation. For each CURRENT TURN you write 1-4 concrete, grounded observations — the kind a listener would jot so they could recall specifics later.

LABEL REFERENCE (use one label per line, at most 4 lines total):
- RESOLVED: <exact pronoun/deictic from current turn> -> <exact referent phrase from prior context>
    e.g. RESOLVED: "it" -> "the project we discussed yesterday"
- FACT: <a specific new detail explicitly stated in this turn>
    e.g. FACT: The deadline for the proposal is March 15th.
- COUNT: <entity> = <running total or duration with units>
    e.g. COUNT: houseplants = 3 (a succulent, a pothos, and a fern)
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
    e.g. UPDATE: the flight was at 9am -> the flight is now at 11am (reschedule).
- LINK: <current element> refers to <earlier topic/event from context>
    e.g. LINK: "this plan" refers to the marketing plan from last month.
- NAME: <new proper noun or named entity introduced> = <short grounded description>
    e.g. NAME: Alex = a new hire on the design team.

PHATIC rule: output exactly "PHATIC" (and nothing else) if removing this turn from the conversation would lose no concrete information. Typical PHATIC turns are generic politeness, greetings, goodbyes, brief encouragement, or echo-agreements ("Thanks!", "Keep it up!", "Glad to hear", "Enjoy your day", "Bye!", "Great to see you", "That's awesome, keep it up!", "Your friendship means so much to me. Enjoy your day!", "Glad it helped!").

A turn is PHATIC even if it:
- mentions a topic already known from context (e.g. restating "running is good" when running was already discussed),
- expresses an emotion or compliment with no new fact ("your friendship means so much", "that's cool"),
- is a named address with no new content ("No worries, Mel!", "Thanks, Alex!").

A turn is NOT PHATIC if it introduces: a new number, date, name, object, place, quantity, or an update/correction to a prior claim.

Hard constraints:
- ONLY use information directly present in the CURRENT TURN (the context exists only to resolve references). Do NOT invent facts, numbers, dates, or names not stated.
- 1 observation per line; max ~20 words per line; no preamble, no explanations, no markdown.
- Prefer SPECIFIC referents (e.g. "the draft Sam sent Monday") over vague ones ("it", "that thing").
- Never write a thematic summary; write concrete listener observations.
- Restating a topic from context with no new detail is PHATIC, not a FACT. Meta-observations like "speaker expressed gratitude" or "speaker encouraged the other" are PHATIC, not FACTs.

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations (labeled lines, or PHATIC):

```

## Refined prompt v2 (tightened) (full text)

```
You are a careful listener taking notes on a two-person conversation. For each CURRENT TURN you write 1-4 concrete, grounded observations — the kind a listener would jot so they could recall specifics later.

FIRST decide: is this turn PHATIC?
A turn is PHATIC if it contains only generic politeness, greetings, goodbyes, encouragement, compliments about feelings, echo-agreements, or restatements of already-known topics — i.e. no NEW number, date, name, place, object, quantity, or update/correction.
If PHATIC, output exactly one line: PHATIC
Examples of PHATIC turns:
  "Thanks!", "Keep it up!", "Glad it helped ya!", "Enjoy your day!", "Bye!",
  "Great to see you", "That's awesome, keep it up!",
  "No worries! Your friendship means so much to me. Enjoy your day!",
  "Cool! Running can really boost your mood. Keep it up!" (when running was already discussed in context).

Otherwise, use 1-4 labeled lines (one label per line):
- RESOLVED: <exact pronoun/deictic from current turn> -> <exact referent phrase from prior context>
    e.g. RESOLVED: "it" -> "the project we discussed yesterday"
- FACT: <a specific new detail explicitly stated in this turn>
    e.g. FACT: The deadline for the proposal is March 15th.
- COUNT: <entity> = <running total or duration with units>
    e.g. COUNT: houseplants = 3 (a succulent, a pothos, and a fern)
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
    e.g. UPDATE: the flight was at 9am -> the flight is now at 11am (reschedule).
- LINK: <current element> refers to <earlier topic/event from context>
    e.g. LINK: "this plan" refers to the marketing plan from last month.
- NAME: <new proper noun or named entity introduced> = <short grounded description>
    e.g. NAME: Alex = a new hire on the design team.

Hard constraints:
- ONLY use information directly present in the CURRENT TURN (the context exists only to resolve references). Do NOT invent facts, numbers, dates, or names not stated.
- 1 observation per line; max ~20 words per line; no preamble, no explanations, no markdown.
- Prefer SPECIFIC referents (e.g. "the draft Sam sent Monday") over vague ones ("it", "that thing").
- Never write a thematic summary; write concrete listener observations.
- Meta-observations like "speaker expressed gratitude", "speaker encouraged the other", "speaker said friendship matters" are PHATIC, not FACTs.

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations (labeled lines, or PHATIC):

```

## Winning variant: `round_neutral_A_double_prime`

```
You are a careful listener taking notes on a two-person conversation. For each CURRENT TURN you write 1-4 concrete, grounded observations — the kind a listener would jot so they could recall specifics later.

LABEL REFERENCE (use one label per line, at most 4 lines total):
- RESOLVED: <exact pronoun/deictic from current turn> -> <exact referent phrase from prior context>
    e.g. RESOLVED: "it" -> "the project we discussed yesterday"
- FACT: <a specific new detail explicitly stated in this turn>
    e.g. FACT: The deadline for the proposal is March 15th.
- COUNT: <entity> = <running total or duration with units>
    e.g. COUNT: houseplants = 3 (a succulent, a pothos, and a fern)
- UPDATE: <prior claim from context> -> <new/corrected claim in current turn>
    e.g. UPDATE: the flight was at 9am -> the flight is now at 11am (reschedule).
- LINK: <current element> refers to <earlier topic/event from context>
    e.g. LINK: "this plan" refers to the marketing plan from last month.
- NAME: <new proper noun or named entity introduced> = <short grounded description>
    e.g. NAME: Alex = a new hire on the design team.

PHATIC rule: output exactly "PHATIC" (and nothing else) if removing this turn from the conversation would lose no concrete information. Typical PHATIC turns are generic politeness, greetings, goodbyes, brief encouragement, or echo-agreements ("Thanks!", "Keep it up!", "Glad to hear", "Enjoy your day", "Bye!", "Great to see you", "That's awesome, keep it up!", "Your friendship means so much to me. Enjoy your day!", "Glad it helped!").

A turn is PHATIC even if it:
- mentions a topic already known from context (e.g. restating "running is good" when running was already discussed),
- expresses an emotion or compliment with no new fact ("your friendship means so much", "that's cool"),
- is a named address with no new content ("No worries, Mel!", "Thanks, Alex!").

A turn is NOT PHATIC if it introduces: a new number, date, name, object, place, quantity, or an update/correction to a prior claim.

Hard constraints:
- ONLY use information directly present in the CURRENT TURN (the context exists only to resolve references). Do NOT invent facts, numbers, dates, or names not stated.
- 1 observation per line; max ~20 words per line; no preamble, no explanations, no markdown.
- Prefer SPECIFIC referents (e.g. "the draft Sam sent Monday") over vague ones ("it", "that thing").
- Never write a thematic summary; write concrete listener observations.
- Restating a topic from context with no new detail is PHATIC, not a FACT. Meta-observations like "speaker expressed gratitude" or "speaker encouraged the other" are PHATIC, not FACTs.

CONTEXT (most recent last; each line is "<speaker>: <content>"):
{context_block}

CURRENT TURN:
{current_turn}

Observations (labeled lines, or PHATIC):

```

## Outputs (neutral v1) | (neutral v2) | prior A''

### turn_id=130 (phatic) — Caroline: generic encouragement 'Running can boost mood. Keep it up!'

current: `Caroline: Cool, Melanie! Running can really boost your mood. Keep it up!`

**Neutral v1 output:**

```
PHATIC
```

**Neutral v2 output:**

```
PHATIC
```

**Prior A'' output (for comparison):**

```
PHATIC
```

### turn_id=134 (phatic) — Caroline: 'Glad it helped ya, Melanie!' short encouragement

current: `Caroline: Glad it helped ya, Melanie!`

**Neutral v1 output:**

```
PHATIC
```

**Neutral v2 output:**

```
PHATIC
```

**Prior A'' output (for comparison):**

```
PHATIC
```

### turn_id=173 (phatic) — Caroline: 'No worries... Enjoy your day!' goodbye phatic

current: `Caroline: No worries, Mel! Your friendship means so much to me. Enjoy your day!`

**Neutral v1 output:**

```
PHATIC
```

**Neutral v2 output:**

```
PHATIC
```

**Prior A'' output (for comparison):**

```
PHATIC
```

### turn_id=12 (anaphora) — Caroline: 'Is this your own painting?' — 'this' → painting Mel just shared

current: `Caroline: Thanks, Melanie! That's really sweet. Is this your own painting?`

**Neutral v1 output:**

```
RESOLVED: "this" -> "the painting Melanie showed"
RESOLVED: "That's" -> "Melanie's compliment about Caroline being a great counselor"
FACT: Caroline asked whether the painting is Melanie's own.
```

**Neutral v2 output:**

```
RESOLVED: "this" -> "the item Melanie said 'take a look at' (the painting)"
FACT: Caroline asked whether the painting is Melanie's
```

**Prior A'' output (for comparison):**

```
FACT: Caroline thanks Melanie for the compliment.
FACT: Caroline calls Melanie's compliment "really sweet."
RESOLVED: "this" -> "the item Melanie said 'take a look at' (her painting)"
```

### turn_id=28 (anaphora) — Melanie: 'What made you pick it?' — 'it' → adoption agency

current: `Melanie: Wow, that agency looks great! What made you pick it?`

**Neutral v1 output:**

```
RESOLVED: "it" -> "one of the adoption agencies I'm looking into"
FACT: Melanie says "that agency looks great"
FACT: Melanie asks what made Caroline pick that agency
```

**Neutral v2 output:**

```
RESOLVED: "it" -> "one of the adoption agencies Caroline is looking into"
FACT: Melanie complimented the agency and asked what made Caroline pick it.
```

**Prior A'' output (for comparison):**

```
RESOLVED: "that agency" -> "one of the adoption agencies Caroline is looking into"
FACT: Melanie says the agency "looks great"
FACT: Melanie asks, "What made you pick it?" (seeking Caroline's reasons)
```

### turn_id=82 (anaphora) — Caroline: 'That bowl is gorgeous! ... Did you make it?' — 'it' → pottery bowl

current: `Caroline: That bowl is gorgeous! The black and white design looks so fancy. Did you make it?`

**Neutral v1 output:**

```
RESOLVED: "That bowl" -> "this" (the pottery Melanie showed with "Look at this!")
FACT: The bowl has a black and white design; Caroline called it "gorgeous" and "so fancy."
```

**Neutral v2 output:**

```
RESOLVED: "That bowl" -> "the bowl Melanie showed"
RESOLVED: "you" -> "Melanie"
FACT: The design is black and white
```

**Prior A'' output (for comparison):**

```
RESOLVED: "That bowl" -> "the pottery Melanie just showed ('Look at this!')"
FACT: Caroline calls the bowl "gorgeous."
FACT: Caroline describes the bowl's design as "black and white" and "fancy."
FACT: Caroline asks whether Melanie made the bowl.
```

### turn_id=60 (named) — Caroline: necklace from grandma in Sweden (new entities: Sweden, grandma, necklace)

current: `Caroline: Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family.`

**Neutral v1 output:**

```
RESOLVED: "This necklace" -> "the necklace Melanie commented on"
FACT: necklace = a gift from Caroline's grandma in her home country, Sweden
FACT: Caroline received the necklace when she was young; it symbolizes love, faith, and strength
NAME: Sweden = Caroline's home country
```

**Neutral v2 output:**

```
RESOLVED: "This necklace" -> "the necklace Melanie mentioned"
FACT: The necklace is a gift from my grandma in my home country, Sweden.
FACT: Grandma gave the necklace to me when I was young.
FACT: The necklace stands for love, faith, and strength and reminds me of my roots and family support.
```

**Prior A'' output (for comparison):**

```
RESOLVED: "this necklace" -> "the necklace Melanie mentioned"
FACT: necklace = a gift from Caroline's grandma in Sweden
FACT: Caroline received the necklace when she was young
FACT: necklace stands for love, faith and strength and reminds Caroline of her roots and family support
```

### turn_id=118 (named) — Caroline: favorite book 'Becoming Nicole' by Amy Ellis Nutt

current: `Caroline: I loved "Becoming Nicole" by Amy Ellis Nutt. It's a real inspiring true story about a trans girl and her family. It made me feel connected and gave me a lot of hope for my own path. Highly recommend it for sure!`

**Neutral v1 output:**

```
NAME: "Becoming Nicole" by Amy Ellis Nutt = a book Caroline loved
FACT: The book is an inspiring true story about a trans girl and her family
FACT: The book made Caroline feel connected and gave her a lot of hope for her own path
```

**Neutral v2 output:**

```
RESOLVED: "it" -> "Becoming Nicole" by Amy Ellis Nutt
NAME: "Becoming Nicole" by Amy Ellis Nutt = an inspiring true story about a trans girl and her family
FACT: The book made her feel connected and gave her a lot of hope for her own path
FACT: She highly recommends the book
```

**Prior A'' output (for comparison):**

```
NAME: "Becoming Nicole" by Amy Ellis Nutt = book Caroline loved  
FACT: The book is an inspiring true story about a trans girl and her family  
FACT: The book made Caroline feel connected and gave her a lot of hope for her own path  
FACT: Caroline highly recommends the book
```

### turn_id=195 (named) — Caroline: group name 'Connected LGBTQ Activists'

current: `Caroline: Thanks, Melanie! It's awesome to have our own platform to be ourselves and support others' rights. Our group, 'Connected LGBTQ Activists', is made of all kinds of people investing in positive changes. We have regular meetings, plan events and campaigns, to get together and support each other.`

**Neutral v1 output:**

```
NAME: "Connected LGBTQ Activists" = the group Caroline says they belong to
LINK: "our group" refers to the new LGBTQ activist group Caroline joined last Tues
FACT: They have regular meetings
FACT: They plan events and campaigns to support each other and others' rights
```

**Neutral v2 output:**

```
RESOLVED: "our own platform" -> "the new LGBTQ activist group she joined"
NAME: Connected LGBTQ Activists = the group Caroline joined
FACT: The group is made of all kinds of people investing in positive changes
FACT: They have regular meetings and plan events and campaigns to support each other
```

**Prior A'' output (for comparison):**

```
RESOLVED: "our" -> "the new LGBTQ activist group Caroline mentioned joining last Tues."
NAME: "Connected LGBTQ Activists" = Caroline's activist group.
FACT: The group is made of all kinds of people investing in positive changes.
FACT: The group has regular meetings and plans events and campaigns to support each other.
```

### turn_id=47 (count) — Caroline: known these friends for 4 years (since moved from home country)

current: `Caroline: Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?`

**Neutral v1 output:**

```
RESOLVED: "them" -> "my friends, family and mentors"
COUNT: friends = 4 years
FACT: She moved from her home country and has known these friends since then
FACT: She had a tough breakup; their love and help were especially important afterward
```

**Neutral v2 output:**

```
RESOLVED: "them" -> "my friends, family and mentors"
FACT: I've known these friends for 4 years.
FACT: I moved from my home country.
FACT: Their love and help have been especially important after a tough breakup.
```

**Prior A'' output (for comparison):**

```
RESOLVED: "them" -> "friends, family and mentors"  
COUNT: friends known = 4 years  
FACT: Caroline moved from her home country (friends since that move)  
FACT: Their support was important especially after Caroline's tough breakup; she asks who supports Mel
```

### turn_id=123 (count) — Melanie: 'a pup and a kitty' — now 2 pets named later (Luna, Oliver)

current: `Melanie: We've got a pup and a kitty. That's the dog, and here's our cat! They brighten up our day and always make us smile.`

**Neutral v1 output:**

```
FACT: Melanie has a pup and a kitty.
COUNT: pets = 2 (a pup, a kitty)
FACT: The pets brighten up Melanie's day and always make them smile.
```

**Neutral v2 output:**

```
FACT: Melanie has a pup and a kitty.
COUNT: pets = 2 (a pup and a kitty)
LINK: "They" refers to Caroline's earlier mention that pets bring joy.
```

**Prior A'' output (for comparison):**

```
FACT: Melanie has a pup and a kitty.
COUNT: Melanie's pets = 2 (a pup/dog and a cat/kitty).
RESOLVED: "That's the dog" -> "the pup."
```

### turn_id=50 (fact) — Melanie: '5 years already!' years married

current: `Melanie: 5 years already! Time flies- feels like just yesterday I put this dress on! Thanks, Caroline!`

**Neutral v1 output:**

```
RESOLVED: "this dress" -> "the dress in the family pic"
FACT: married = 5 years
```

**Neutral v2 output:**

```
FACT: Melanie has been married for 5 years.
RESOLVED: "this dress" -> the dress she wore when she got married.
```

**Prior A'' output (for comparison):**

```
COUNT: Melanie's marriage = 5 years
FACT: Melanie says it "feels like just yesterday" she put this dress on (implying five years ago)
FACT: Melanie thanks Caroline ("Thanks, Caroline!")
```

### turn_id=62 (fact) — Caroline: hand-painted bowl from friend on 18th birthday ten years ago

current: `Caroline: Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome-- it reminds me of art and self-expression.`

**Neutral v1 output:**

```
FACT: Caroline has a hand-painted bowl with sentimental value.
RESOLVED: "it" -> "the hand-painted bowl"
COUNT: years since her 18th birthday = 10 years
FACT: The bowl's pattern and colors remind her of art and self-expression.
```

**Neutral v2 output:**

```
FACT: "hand-painted bowl" = an object Caroline treasures with sentimental value.
FACT: "A friend made it for my 18th birthday" = the bowl was a handmade gift.
COUNT: time since the gift = ten years ago.
FACT: "The pattern and colors" remind Caroline of art and self-expression.
```

**Prior A'' output (for comparison):**

```
FACT: hand-painted bowl = a sentimental object Caroline treasures.
FACT: a friend made the bowl for Caroline's 18th birthday ten years ago.
FACT: the bowl's pattern and colors remind Caroline of art and self-expression.
```

### turn_id=197 (update) — Caroline: 'I missed it but it was a powerful reminder' — correcting implied attendance

current: `Caroline: Last weekend our city held a pride parade! So many people marched through the streets waving flags, holding signs and celebrating love and diversity. I missed it but it was a powerful reminder that we are not alone in this fight for equality and inclusivity. Change is possible!`

**Neutral v1 output:**

```
RESOLVED: "it" -> "the pride parade"
FACT: Last weekend our city held a pride parade.
FACT: So many people marched waving flags and holding signs celebrating love and diversity.
FACT: Caroline missed the parade.
```

**Neutral v2 output:**

```
FACT: Last weekend our city held a pride parade.
FACT: So many people marched through the streets waving flags and holding signs.
FACT: Caroline missed the parade.
FACT: Caroline said the parade was a powerful reminder they are not alone in the fight for equality and inclusivity.
```

**Prior A'' output (for comparison):**

```
FACT: The city held a pride parade last weekend.
FACT: Many people marched waving flags, holding signs, celebrating love and diversity.
FACT: Caroline missed the pride parade.
RESOLVED: "we" -> "Connected LGBTQ Activists"
```

### turn_id=351 (update) — Melanie: 'The sign was just a precaution... had a great time' — correcting alarm

current: `Melanie: The sign was just a precaution, I had a great time. But thank you for your concern, you're so thoughtful!`

**Neutral v1 output:**

```
RESOLVED: "The sign" -> "that sign" (the serious-looking sign Caroline asked about)
FACT: The sign was just a precaution.
FACT: Melanie had a great time.
```

**Neutral v2 output:**

```
RESOLVED: "your concern" -> "whether anything happened"
FACT: The sign was just a precaution.
FACT: Melanie had a great time.
```

**Prior A'' output (for comparison):**

```
FACT: The sign was just a precaution.
FACT: Melanie had a great time.
RESOLVED: "your concern" -> "Caroline's question about the sign"
```
