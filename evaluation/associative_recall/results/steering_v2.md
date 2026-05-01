# Active Embedding Steering V2 (evidence-grounded)

V2 fix: SUBTRACT uses embeddings of LLM-classified distractor turns (actual retrieved text), not imagined opposite concepts. ADD phrases grounded in query vocabulary or extracted from gold-likely retrieved turns; fabrication of specific details forbidden.

Fixed: alpha=beta=0.1, text-embedding-3-small, gpt-5-mini, LME-hard-30 POC subset, up to 3 rounds with early-stop.

## Recall matrix (LME-hard-30)

| Variant | n | R@20 | R@50 | time (s) |
| --- | --- | --- | --- | --- |
| `steerv2_full` | 30 | 0.5944 | 0.7360 | 134.9 |
| `steerv2_subonly` | 30 | 0.4432 | 0.5649 | 44.3 |
| `steerv2_addonly` | 30 | 0.6515 | 0.8040 | 53.1 |
| `baseline_v2f_direct` | 30 | 0.6303 | 0.8169 | 2.8 |

## V2 vs V1 (LME R@50)

| Variant | V1 R@50 | V2 R@50 | Δ |
| --- | --- | --- | --- |
| `steerv2_full` (vs `steer_v2f_2round`) | 0.7994 | 0.7360 | -0.0634 |
| `steerv2_subonly` (vs `steer_v2f_subonly`) | 0.8128 | 0.5649 | -0.2479 |
| `steerv2_addonly` (vs `steer_v2f_addonly`) | 0.8085 | 0.8040 | -0.0045 |
| `baseline_v2f_direct` (vs `baseline_v2f_direct`) | 0.8169 | 0.8169 | +0.0000 |

## LME category recall (R@20)

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `steerv2_full` | 0.4841 | 0.8250 | 0.4742 |
| `steerv2_subonly` | 0.3721 | 0.6646 | 0.2928 |
| `steerv2_addonly` | 0.5379 | 0.8562 | 0.5603 |
| `baseline_v2f_direct` | 0.5452 | 0.8104 | 0.5353 |

## LME category recall (R@50)

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `steerv2_full` | 0.6230 | 0.8875 | 0.6976 |
| `steerv2_subonly` | 0.5392 | 0.7583 | 0.3971 |
| `steerv2_addonly` | 0.7021 | 0.9187 | 0.7910 |
| `baseline_v2f_direct` | 0.7398 | 0.9021 | 0.8089 |

## LLM classification quality

Precision = fraction of LLM-flagged turns where the classification matches ground truth (distractor = not-gold, gold-likely = is-gold).

| Variant | flagged_distractors | distractor_precision | flagged_gold_likely | gold_likely_precision |
| --- | --- | --- | --- | --- |
| `steerv2_full` | 300 | 0.2733 | 104 | 0.9615 |
| `steerv2_subonly` | 300 | 0.32 | 94 | 0.9255 |
| `steerv2_addonly` | 0 | -- | 94 | 0.9681 |
| `baseline_v2f_direct` | 0 | -- | 0 | -- |

## Round-by-round R@50

| Variant | rd 0 | rd 1 | rd 2 | rd 3 | drift@final |
| --- | --- | --- | --- | --- | --- |
| `steerv2_full` | 0.8169 | 0.8064 | 0.7860 | 0.7360 | 0.7783 |
| `steerv2_subonly` | 0.8169 | 0.7837 | 0.7046 | 0.5649 | 0.7605 |
| `steerv2_addonly` | 0.8169 | 0.8143 | 0.8067 | 0.8040 | 0.9119 |
| `baseline_v2f_direct` | 0.8169 | -- | -- | -- | 1.0000 |

## Sample classifications (steerv2_full, first 3 questions)

### Q `ba358f49` (multi-session)

- question: How many years will I be when my friend Rachel gets married?
- initial cue: `User: I'm looking for some advice on skincare routines for my age group. I've been noticing some fine lines and wrinkles lately, and I want to start taking better care of my skin. By the way, my friend Rachel's getting married next year, and it's got me th`
- round 0: R@50=0.6667, top-5 gold mask: [False, True, True, False, True]
- round 1: R@50=0.625, drift=0.971
  - distractor_indices=[0, 1, 2, 3, 4] (correct: 1/5), gold_likely=[] (correct: 0/0)
  - top-5 gold mask: [False, True, True, True, True]
  - ADD phrases: ['friend Rachel getting married next year', 'my age group skincare', 'fine lines and wrinkles']
  - SUBTRACTED (previews): ['* How do you envision the Memory Palace looking and feeling? Is it a grand, ornate structure, or a m', "I'm looking for some advice on skincare routines for my age group. I've been noticing some fine line", "Congratulations to Rachel on her upcoming wedding!\n\nNow, let's talk about your skincare concerns! Fi", '* Cyberpunk 2077\n* Ghost of Tsushima\n* Resident Evil Village\n* Call of Duty: Black Ops Cold War\n* As', "I've been researching skincare routines and products online, and I'm not sure what to choose. Can yo"]
  - reasoning: No retrieved turn calculates your age at Rachel's wedding; all turns are unrelated or only discuss skincare/other topics.
- round 2: R@50=0.5417, drift=0.914
  - distractor_indices=[0, 2, 3] (correct: 1/3), gold_likely=[1, 4] (correct: 2/2)
  - top-5 gold mask: [False, True, True, True, True]
  - ADD phrases: ['getting married next year', "I'm 32", 'friend Rachel']
  - SUBTRACTED (previews): ['* How do you envision the Memory Palace looking and feeling? Is it a grand, ornate structure, or a m', "Congratulations to Rachel on her upcoming wedding!\n\nNow, let's talk about your skincare concerns! Fi", 'Once I have a better understanding of your skin profile and preferences, I can suggest some effectiv']
  - reasoning: Turns 1 and 4 state the wedding timing and the user's age, while the others are unrelated to the question.
- round 3: R@50=0.5417, drift=0.818
  - distractor_indices=[0, 2, 3] (correct: 1/3), gold_likely=[1, 4] (correct: 2/2)
  - top-5 gold mask: [False, True, True, True, True]
  - ADD phrases: ['getting married next year', "I'm 32", 'friend Rachel']
  - SUBTRACTED (previews): ['* How do you envision the Memory Palace looking and feeling? Is it a grand, ornate structure, or a m', "Congratulations to Rachel on her upcoming wedding!\n\nNow, let's talk about your skincare concerns! Fi", 'Once I have a better understanding of your skin profile and preferences, I can suggest some effectiv']
  - reasoning: Turns 1 and 4 state the wedding timing and the user's age, while the others are unrelated to the question.

### Q `5a7937c8` (multi-session)

- question: How many days did I spend participating in faith-related activities in December?
- initial cue: `User: I'm looking for some volunteer opportunities in my community, preferably something related to food banks or pantries. I actually helped out at the church's annual holiday food drive on December 10th, sorting donations and packing boxes for families i`
- round 0: R@50=0.5625, top-5 gold mask: [False, True, True, True, True]
- round 1: R@50=0.5625, drift=0.963
  - distractor_indices=[0, 2, 3, 4] (correct: 1/4), gold_likely=[1] (correct: 1/1)
  - top-5 gold mask: [False, True, True, True, True]
  - ADD phrases: ["church's annual holiday food drive on December 10th", 'sorting donations', 'packing boxes for families in need']
  - SUBTRACTED (previews): ['Good luck on your adoption journey, and I hope you find the perfect companion soon!', "That's great to hear! It's wonderful that you had a positive experience volunteering at the church's", "By volunteering in a food bank or pantry, you'll be making a tangible difference in your community. ", "I'm interested in volunteering at a food bank or pantry that focuses on serving families with childr"]
  - reasoning: Turn 1 explicitly mentions volunteering on December 10th; the others are related but do not state specific December faith activity days.
- round 2: R@50=0.5625, drift=0.876
  - distractor_indices=[0, 2, 3, 4] (correct: 1/4), gold_likely=[1] (correct: 1/1)
  - top-5 gold mask: [False, True, True, True, True]
  - ADD phrases: ["church's annual holiday food drive", 'December 10', 'sorting donations and packing boxes']
  - SUBTRACTED (previews): ['Good luck on your adoption journey, and I hope you find the perfect companion soon!', "That's great to hear! It's wonderful that you had a positive experience volunteering at the church's", '**Variations:** While the traditional format is often followed, some churches may adapt the service ', "That sounds wonderful. I think I'll definitely look into attending a Lessons and Carols service this"]
  - reasoning: Turn 1 provides the December 10 church volunteering detail needed to count faith-related days; the others are topically related but don't supply date-specific participation.
- round 3: R@50=0.5938, drift=0.777
  - distractor_indices=[0, 2, 3, 4] (correct: 1/4), gold_likely=[1] (correct: 1/1)
  - top-5 gold mask: [False, True, True, True, True]
  - ADD phrases: ["church's annual holiday food drive on December 10th", 'sorting donations and packing boxes', 'helped out at the church']
  - SUBTRACTED (previews): ['Good luck on your adoption journey, and I hope you find the perfect companion soon!', "That's great to hear! It's wonderful that you had a positive experience volunteering at the church's", "I'm particularly interested in food banks and pantries. Can you tell me more about what's involved i", "Sorting and packing donations is a crucial step in the food bank or pantry process, and it's a great"]
  - reasoning: Turn 1 explicitly states participation on December 10th, while the others are unrelated or general volunteering content.

### Q `6c49646a` (multi-session)

- question: What is the total distance I covered in my four road trips?
- initial cue: `User: I'm glad I could fit in Maroon Lake. Since I've covered a total of 1,800 miles on my recent three road trips, including a solo trip to Durango, a weekend trip to Breckenridge, and a family trip to Santa Fe, I'm comfortable with the drive and explorin`
- round 0: R@50=0.875, top-5 gold mask: [True, True, True, True, True]
- round 1: R@50=0.7917, drift=0.925
  - distractor_indices=[0, 1, 2, 3, 4] (correct: 1/5), gold_likely=[] (correct: 0/0)
  - top-5 gold mask: [True, True, True, False, True]
  - ADD phrases: ['total distance for four road trips', '1,800 miles for recent three road trips', 'solo trip to Durango, weekend trip to Breckenridge, family trip to Santa Fe']
  - SUBTRACTED (previews): ['By planning ahead and being prepared, you can minimize the hassle of parking and make the most of yo', "I'm glad I could fit in Maroon Lake. Since I've covered a total of 1,800 miles on my recent three ro", "With your experience and comfort level with road trips, you'll be able to enjoy the drive and scenic", "I'll definitely check the road conditions before I leave. I'm also thinking of stopping at Maroon La", "Maroon Lake is a fantastic spot, and I'm happy to help you decide if it's worth the detour.\n\n**Getti"]
  - reasoning: None of the retrieved turns computes or states the total for all four trips; they only provide context or unrelated advice.
- round 2: R@50=0.6667, drift=0.831
  - distractor_indices=[0, 2, 3] (correct: 1/3), gold_likely=[1, 4] (correct: 2/2)
  - top-5 gold mask: [True, True, True, False, True]
  - ADD phrases: ['covered a total of 1,800 miles', 'covered a total of 1,200 miles', 'four road trips']
  - SUBTRACTED (previews): ['By planning ahead and being prepared, you can minimize the hassle of parking and make the most of yo', "With your experience and comfort level with road trips, you'll be able to enjoy the drive and scenic", 'Overall, I think this bonus challenge can be a great addition to your scavenger hunt, as long as you']
  - reasoning: Turns 1 and 4 contain the mileage amounts needed to compute the four-trip total; the others are assistant commentary.
- round 3: R@50=0.6667, drift=0.710
  - distractor_indices=[0, 2, 3] (correct: 1/3), gold_likely=[1, 4] (correct: 2/2)
  - top-5 gold mask: [True, True, True, False, True]
  - ADD phrases: ['covered a total of 1,800 miles', 'covered a total of 1,200 miles', 'four road trips']
  - SUBTRACTED (previews): ['By planning ahead and being prepared, you can minimize the hassle of parking and make the most of yo', "With your experience and comfort level with road trips, you'll be able to enjoy the drive and scenic", 'Overall, I think this bonus challenge can be a great addition to your scavenger hunt, as long as you']
  - reasoning: Turns 1 and 4 contain the mileage amounts needed to compute the four-trip total; the others are assistant commentary.

## Verdict

- baseline (v2f speaker-format direct): R@50 = 0.8169
- steerv2_full: R@50 = 0.7360 (Δ vs baseline = -0.0809)
- steerv2_addonly: R@50 = 0.8040 (Δ = -0.0129)
- steerv2_subonly: R@50 = 0.5649 (Δ = -0.2520)

**No lift (best Δ = -0.0129) and classification precision = 0.2733**; classification quality could be the bottleneck.

## Outputs

- JSON: `results/steering_v2.json`
- Sources: `active_steering_v2.py`, `steerv2_eval.py`
- Caches: `cache/steerv2_llm_cache.json`, `cache/steerv2_embedding_cache.json`
