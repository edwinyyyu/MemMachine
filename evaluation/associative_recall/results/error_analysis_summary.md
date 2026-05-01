# Error analysis: meta_v2f retrieval failures

## Scope
- Analyzed **55** questions where meta_v2f `arch_r@20 < 0.8` across 4 datasets (advanced, synthetic, puzzle, locomo).
- Total **334** source turns were missed at K=20.

## Headline findings

- **50.0%** of missed turns are adjacent (±1) to a turn that WAS retrieved.
- **64.1%** of missed turns are within ±2 of a retrieved turn.
- **~75%** of missed turns (labelled `ranked_out`) contain vocabulary overlap with the question or a v2f-generated cue — they lost the cosine competition to distractors, not to a true vocab gap.

## Failure mode counts (aggregate)

| Mode | Count | Share |
|---|---:|---:|
| ranked_out | 250 | 74.9% |
| vocabulary_gap | 50 | 15.0% |
| anaphoric_reference | 34 | 10.2% |

**Legend**
- `ranked_out` — turn has some content term overlap with the question or a cue, but wasn't in the top-K
- `vocabulary_gap` — no shared content terms with question or cues, and > 0 content tokens
- `anaphoric_reference` — starts with a pronoun/acknowledger ("that", "yeah", "sure", …) and shares no question terms
- `implicit_short_response` — ≤10 words AND no shared terms with question/cues (subset of the others)

## Per-dataset breakdown

| Dataset | Failing Qs | Missed turns | adj_r1% | adj_r2% | short% | Modes |
|---|---:|---:|---:|---:|---:|---|
| advanced_23q | 17 | 120 | 52.5% | 74.2% | 0.8% | ranked_out=97, anaphoric_reference=12, vocabulary_gap=11 |
| locomo_30q | 10 | 15 | 46.7% | 53.3% | 0.0% | ranked_out=9, vocabulary_gap=5, anaphoric_reference=1 |
| puzzle_16q | 15 | 118 | 49.2% | 61.9% | 1.7% | ranked_out=94, vocabulary_gap=14, anaphoric_reference=10 |
| synthetic_19q | 13 | 81 | 48.1% | 54.3% | 1.2% | ranked_out=50, vocabulary_gap=20, anaphoric_reference=11 |

## Per-category patterns

| Category | Missed | adj_r1% | adj_r2% | Dominant modes |
|---|---:|---:|---:|---|
| evolving_terminology | 35 | 51.4% | 65.7% | ranked_out=31, anaphoric_reference=2, vocabulary_gap=2 |
| logic_constraint | 35 | 68.6% | 71.4% | ranked_out=31, anaphoric_reference=2, vocabulary_gap=2 |
| completeness | 26 | 65.4% | 73.1% | ranked_out=18, anaphoric_reference=6, vocabulary_gap=2 |
| proactive | 25 | 36.0% | 44.0% | ranked_out=17, vocabulary_gap=6, anaphoric_reference=2 |
| absence_inference | 24 | 8.3% | 41.7% | ranked_out=15, vocabulary_gap=6, anaphoric_reference=3 |
| procedural | 21 | 28.6% | 33.3% | ranked_out=10, vocabulary_gap=9, anaphoric_reference=2 |
| open_exploration | 21 | 38.1% | 57.1% | ranked_out=17, vocabulary_gap=4 |
| sequential_chain | 20 | 70.0% | 75.0% | ranked_out=16, vocabulary_gap=2, anaphoric_reference=2 |
| negation | 19 | 31.6% | 84.2% | ranked_out=13, anaphoric_reference=3, vocabulary_gap=3 |
| quantitative_aggregation | 19 | 78.9% | 94.7% | ranked_out=18, anaphoric_reference=1 |
| state_change | 14 | 57.1% | 64.3% | ranked_out=12, anaphoric_reference=2 |
| constraint_propagation | 12 | 33.3% | 50.0% | ranked_out=9, anaphoric_reference=2, vocabulary_gap=1 |
| locomo_single_hop | 10 | 40.0% | 50.0% | ranked_out=5, vocabulary_gap=4, anaphoric_reference=1 |
| unfinished_business | 9 | 33.3% | 66.7% | ranked_out=7, vocabulary_gap=2 |
| perspective_separation | 9 | 77.8% | 77.8% | ranked_out=9 |
| consistency_checking | 9 | 66.7% | 77.8% | ranked_out=6, vocabulary_gap=2, anaphoric_reference=1 |
| frequency_detection | 8 | 50.0% | 75.0% | ranked_out=4, anaphoric_reference=3, vocabulary_gap=1 |
| inference | 6 | 100.0% | 100.0% | ranked_out=4, vocabulary_gap=2 |

## Representative failure examples

### Dataset: `advanced_23q`

### adv_quantitative / q13 — quantitative_aggregation (r@20=0.17)
**Question**: How did the project estimate compare to the client's budget, and what was the resolution?
**Source turns**: [44, 55, 59, 62, 63, 64, 66, 67, 74, 75, 77, 80] (12 total)
**Retrieved source turns**: [74, 80]
**Missed source turns**: [44, 55, 59, 62, 63, 64, 66, 67, 75, 77]
**Cues v2f generated**:
  - _Which exceeds the client's budget. This is going to be a hard sell._
  - _Client got back to me. They approved phase 1 at 630 hours and want a separate quote for phase 2. We're a go!_
**Missed turn samples** (up to 3):
  - **Turn 44** (user) mode=`anaphoric_reference` [ADJ_r1,ANAPH]
    > They said they were thinking around 600-700 hours total. We're already at 480 with two people not yet counted.
    cue_overlap: ['hours', 're']
      before(turn 43,HIT): Setting expectations early is important. Do you have a sense of the client's budget?
      after(turn 45,HIT): That's going to be a tough conversation. You're likely looking at 700-900 hours total.
  - **Turn 55** (assistant) mode=`ranked_out` [ADJ_r1]
    > You at 80 hours. Grand total: Nina 180h, Owen 155h, Patricia 145h, Quinn 105h, You 80h = 665 hours.
    cue_overlap: ['hours']
      before(turn 54,HIT): And then there's me. Project management, client communication, sprint planning, and oversight. I estimate about 80 hours across the project.
      after(turn 56,HIT): Hmm, that's right at the top of the client's budget. But we still have Nina's browser testing.
  - **Turn 59** (assistant) mode=`ranked_out` [ADJ_r1]
    > Nina's revised total: 235 hours (original 180 + 30 browser testing + 25 responsive design). Grand total is now 720 hours.
    question_overlap: ['s']
    cue_overlap: ['hours', 's']
      before(turn 58,other): Just now actually. She says browser testing across Chrome, Firefox, Safari, and Edge will be 30 hours. She also realized she forgot to account for responsive design adjustments - a…
      after(turn 60,HIT): Which exceeds the client's budget. This is going to be a hard sell.

### adv_evolving_term_2 / q19 — evolving_terminology (r@20=0.29)
**Question**: What was the root cause of the monster bug and how was each aspect resolved?
**Source turns**: [4, 18, 20, 36, 38, 46, 52, 54, 58, 60, 62, 64, 66, 68, 84, 86, 91] (17 total)
**Retrieved source turns**: [4, 18, 52, 54, 60]
**Missed source turns**: [20, 36, 38, 46, 58, 62, 64, 66, 68, 84, 86, 91]
**Cues v2f generated**:
  - _Same deployment process was the root cause — staging config caused the session/login issue and a missing DB index caused the timesheet timeout; include notes from the patch author …_
  - _Timesheet timeout led to data corruption in the payroll queue; Sara updated the ticket documenting payroll data cleanup, confirmation that overtime numbers are correct, and that al…_
**Missed turn samples** (up to 3):
  - **Turn 20** (user) mode=`ranked_out` [ADJ_r1]
    > Because the deployment went to 2 of our 4 app server pods. The load balancer round-robins between them. So roughly half the requests hit the bad config.
    cue_overlap: ['config', 'deployment']
      before(turn 19,HIT): Ah, a classic config leak from staging to production. That explains the short timeouts. But why is it only affecting some users?
      after(turn 21,other): That perfectly explains the coin-flip behavior. Fix would be to redeploy the correct config to those two pods.
  - **Turn 36** (user) mode=`ranked_out` [ADJ_r1]
    > Sara says the query she added for the overtime calculation is doing a full table scan on the timesheet_entries table. That table has 2 million rows.
    cue_overlap: ['overtime', 'sara', 'timesheet']
      before(turn 35,HIT): Makes sense to have the patch author investigate. Any initial findings?
      after(turn 37,other): A full table scan on 2 million rows would absolutely cause a 60-second timeout. Missing index?
  - **Turn 38** (user) mode=`anaphoric_reference` [ANAPH]
    > Exactly. She forgot to add an index for the new query pattern. She's writing the migration now.
    cue_overlap: ['index']
      before(turn 37,other): A full table scan on 2 million rows would absolutely cause a 60-second timeout. Missing index?
      after(turn 39,other): A simple index should fix the performance. How long until it can be deployed?

### adv_consistency / q17 — consistency_checking (r@20=0.36)
**Question**: Are there any scheduling conflicts or inconsistencies in the user's week that were identified or remain unresolved?
**Source turns**: [10, 24, 34, 35, 36, 44, 45, 48, 49, 50, 58, 59, 64, 65] (14 total)
**Retrieved source turns**: [10, 35, 44, 45, 49]
**Missed source turns**: [24, 34, 36, 48, 50, 58, 59, 64, 65]
**Cues v2f generated**:
  - _Thursday board presentation 2pm-3:30pm estimated overlaps with team retro at 3pm — someone else to facilitate the retro_
  - _Design review requires you at the midtown office on Wednesday; marketing sync 3pm is virtual ~45min and API migration check-in scheduled Wednesday 4pm (confirm virtual)_
**Missed turn samples** (up to 3):
  - **Turn 24** (user) mode=`ranked_out` [ADJ_r1]
    > Actually yes. I just remembered - my boss scheduled a strategy session for Thursday morning at 9am. It's at the uptown office.
    question_overlap: ['s']
    cue_overlap: ['office', 'scheduled', 'thursday']
      before(turn 23,HIT): Great choice! Now, back to the week - anything else to plan for?
      after(turn 25,HIT): Thursday is getting busy now - strategy session at 9am uptown, then board presentation at 2pm downtown. Are those offices far apart?
  - **Turn 34** (user) mode=`ranked_out` [ADJ_r1]
    > Yes, all virtual. Good. Let me see about other things... Oh, I told the Denver office I'd join their all-hands on Thursday at 11am Mountain Time. That's 1pm Eastern.
    question_overlap: ['s']
    cue_overlap: ['office', 'thursday', 'virtual']
      before(turn 33,HIT): Wednesday at 4pm works since the marketing sync at 3pm is virtual and usually runs about 45 minutes. Would the API migration check-in be virtual too?
      after(turn 35,HIT): Wait - Thursday at 1pm Eastern you'd be joining the Denver all-hands, but the board presentation is at 2pm downtown. Will the Denver all-hands be a quick thing? That's only a 1-hou…
  - **Turn 36** (user) mode=`ranked_out` [ADJ_r1]
    > Hmm, you're right. The Denver all-hands usually runs an hour. So it would go until 2pm which is exactly when the board presentation starts. That's a conflict.
    question_overlap: ['s']
    cue_overlap: ['2pm', 'board', 'presentation']
      before(turn 35,HIT): Wait - Thursday at 1pm Eastern you'd be joining the Denver all-hands, but the board presentation is at 2pm downtown. Will the Denver all-hands be a quick thing? That's only a 1-hou…
      after(turn 37,other): You'd need to either skip the Denver all-hands, join for just the first portion, or see if you can present at the board meeting slightly later. Which is more important?

### adv_perspective_1 / q10 — perspective_separation (r@20=0.36)
**Question**: What was Bob's position on the launch deadline and how did it differ from Carol's?
**Source turns**: [12, 14, 20, 22, 24, 26, 40, 62, 64, 68, 78] (11 total)
**Retrieved source turns**: [12, 20, 40, 78]
**Missed source turns**: [14, 22, 24, 26, 62, 64, 68]
**Cues v2f generated**:
  - _Carol was indifferent as long as_
  - _Carol's position is that the timeline isn't just about features - even the minimum scope has a QA bottleneck; she's nervous but willing to meet June 15 with caveats_
**Missed turn samples** (up to 3):
  - **Turn 14** (user) mode=`ranked_out` [ADJ_r1]
    > He wants to cut features to hit the date. Specifically, he'd drop the advanced analytics module and the custom workflow builder from the initial release. Those could come in a July update. He said, 'Core functionality on…
    cue_overlap: ['features']
      before(turn 13,other): Bob's bringing the business reality. 200 enterprise customers in the pipeline is serious. What's his proposed solution?
      after(turn 15,HIT): Interesting - Bob is prioritizing the date over completeness. That's a product manager's typical lean. How did others react?
  - **Turn 22** (user) mode=`ranked_out` [ADJ_r1]
    > The bug count has been rising steadily. There are currently 47 open bugs, 12 of which she classifies as release-blockers. She said we can't ship until those 12 are fixed regardless of when the deadline is. At current fix…
    question_overlap: ['deadline']
    cue_overlap: ['t']
      before(turn 21,other): She brought data to the meeting too. What did it show?
      after(turn 23,HIT): So Carol's position is that the timeline isn't just about features - even the minimum scope has a QA bottleneck. Where does that leave June 15th?
  - **Turn 24** (user) mode=`ranked_out` [ADJ_r1]
    > She said June 15th is theoretically possible if we stop adding new features immediately and put the whole team on bug fixes. But that means nothing new gets done for 2 weeks.
    cue_overlap: ['features', 'june']
      before(turn 23,HIT): So Carol's position is that the timeline isn't just about features - even the minimum scope has a QA bottleneck. Where does that leave June 15th?
      after(turn 25,HIT): A bug fix sprint as the path to June 15th. That would address Carol's concerns but neither Alice's nor Dave's.

### Dataset: `locomo_30q`

### locomo_conv-26 / q15 — locomo_single_hop (r@20=0.00)
**Question**: What activities does Melanie partake in?
**Source turns**: [11, 17, 79, 174] (4 total)
**Retrieved source turns**: []
**Missed source turns**: [11, 17, 79, 174]
**Cues v2f generated**:
  - _I play guitar and piano, and I sing at open mics and family gatherings_
  - _We like hiking and game nights, I bake sometimes, and I volunteer with local LGBTQ youth programs and do yoga/therapy for my mental health_
**Missed turn samples** (up to 3):
  - **Turn 11** (assistant) mode=`vocabulary_gap`
    > You'd be a great counselor! Your empathy and understanding will really help the people you work with. By the way, take a look at this.
      before(turn 10,other): I'm keen on counseling or working in mental health - I'd love to support those with similar issues.
      after(turn 12,other): Thanks, Melanie! That's really sweet. Is this your own painting?
  - **Turn 17** (assistant) mode=`vocabulary_gap`
    > Yep, Caroline. Taking care of ourselves is vital. I'm off to go swimming with the kids. Talk to you soon!
      before(turn 16,other): Totally agree, Mel. Relaxing and expressing ourselves is key. Well, I'm off to go do some research.
      after(turn 18,other): Hey Caroline, since we last chatted, I've had a lot of things happening to me. I ran a charity race for mental health last Saturday – it was really rewarding. Really made me think …
  - **Turn 79** (assistant) mode=`ranked_out` [ADJ_r1]
    > Wow, Caroline! That's great! I just signed up for a pottery class yesterday. It's like therapy for me, letting me express myself and get creative. Have you found any activities that make you feel the same way?
    question_overlap: ['activities']
    cue_overlap: ['therapy']
      before(turn 78,other): Thanks, Mel! It really motivated me for sure. Talking to the community made me want to use my story to help others too - I'm still thinking that counseling and mental health is the…
      after(turn 80,HIT): Wow, Melanie! I'm getting creative too, just learning the piano. What made you try pottery?

### locomo_conv-26 / q27 — locomo_multi_hop (r@20=0.00)
**Question**: Would Caroline pursue writing as a career option?
**Source turns**: [112, 116] (2 total)
**Retrieved source turns**: []
**Missed source turns**: [112, 116]
**Cues v2f generated**:
  - _I've been thinking about pursuing writing as a career—considering freelance writing, journalism, or becoming a fiction author and wondering how to get started with a portfolio and …_
  - _I want to be a writer and am weighing options like self-publishing a novel, applying to creative writing programs, or combining writing with my counseling work as a career path._
**Missed turn samples** (up to 3):
  - **Turn 112** (user) mode=`ranked_out` [ADJ_r1]
    > Thanks, Mell! I'm still looking into counseling and mental health jobs. It's important to me that people have someone to talk to, and I want to help make that happen.
    cue_overlap: ['counseling', 'want']
      before(turn 111,other): Wow, Caroline. We've come so far, but there's more to do. Your drive to help is awesome! What's your plan to pitch in?
      after(turn 113,HIT): Wow, Caroline! You're so inspiring for wanting to help others with their mental health. What's pushing you to keep going forward with it?
  - **Turn 116** (user) mode=`ranked_out` [ADJ_r1]
    > Thanks so much, Mel! Seeing this pic just made me appreciate my love of reading even more. Books guide me, motivate me and help me discover who I am. They're a huge part of my journey, and this one's reminding me to keep…
    cue_overlap: ['am']
      before(turn 115,HIT): Caroline, so glad you got the support! Your experience really brought you to where you need to be. You're gonna make a huge difference! This book I read last year reminds me to alw…
      after(turn 117,other): Wow, Caroline! Books have such an awesome power! Which one has been your favorite guide?

### locomo_conv-26 / q23 — locomo_single_hop (r@20=0.00)
**Question**: What books has Melanie read?
**Source turns**: [101, 115] (2 total)
**Retrieved source turns**: []
**Missed source turns**: [101, 115]
**Cues v2f generated**:
  - _Melanie: I’ve read several books recently — I just finished reading and loved [book title] and [book title], and I’ve also read..._
  - _Melanie: My favorite book/guide is..., I’ve been reading..., books I’ve read include..., would recommend..._
**Missed turn samples** (up to 3):
  - **Turn 101** (assistant) mode=`ranked_out` [ADJ_r2]
    > I loved reading "Charlotte's Web" as a kid. It was so cool seeing how friendship and compassion can make a difference.
    cue_overlap: ['loved', 'reading']
      before(turn 100,other): I've got lots of kids' books- classics, stories from different cultures, educational books, all of that. What's a favorite book you remember from your childhood?
      after(turn 102,other): Wow, that's great! It sure shows how important friendship and compassion are. It's made me appreciate how lucky I am to have my friends and family helping with my transition. They …
  - **Turn 115** (assistant) mode=`ranked_out` [ADJ_r1]
    > Caroline, so glad you got the support! Your experience really brought you to where you need to be. You're gonna make a huge difference! This book I read last year reminds me to always pursue my dreams, just like you are …
    question_overlap: ['read']
    cue_overlap: ['book', 'read']
      before(turn 114,other): I struggled with mental health, and support I got was really helpful. It made me realize how important it is for others to have a support system. So, I started looking into counsel…
      after(turn 116,HIT): Thanks so much, Mel! Seeing this pic just made me appreciate my love of reading even more. Books guide me, motivate me and help me discover who I am. They're a huge part of my jour…

### Dataset: `puzzle_16q`

### puzzle_logic_1 / q1 — logic_constraint (r@20=0.09)
**Question**: What were all the constraints for the desk arrangement, including which ones were eventually resolved or became irrelevant?
**Source turns**: [6, 12, 24, 28, 36, 44, 50, 72, 78, 80, 82] (11 total)
**Retrieved source turns**: [6]
**Missed source turns**: [12, 24, 28, 36, 44, 50, 72, 78, 80, 82]
**Cues v2f generated**:
  - _Summarize all desk-arrangement constraints mentioned (Alice and Dave not adjacent; desk 1 preferred for the whiteboard; all desks by the windows; Bob may move) and explicitly state…_
  - _Any messages that list final or confirmed desk assignments, note "resolved" or "irrelevant", or mention Eve/Carol/Dave preferences (including references to the Henderson project or…_
**Missed turn samples** (up to 3):
  - **Turn 12** (user) mode=`anaphoric_reference` [ADJ_r1,ANAPH]
    > Maybe later. Oh, I just got a message from Alice. She says she absolutely cannot sit next to Dave. They had that whole argument about the Henderson project and things are still tense.
    cue_overlap: ['alice', 'dave', 'henderson', 'later', 'project']
      before(turn 11,other): Absolutely! A classic is pulled pork - you can do it with just a pork shoulder, BBQ sauce, and some seasonings. About 8 hours on low. Want details?
      after(turn 13,HIT): Understood. So Alice and Dave need at least one desk between them. That's constraint number one.
  - **Turn 24** (user) mode=`ranked_out` [ADJ_r1,ANAPH]
    > Right. And I want desk 1 since I use the whiteboard constantly for sketching out architectures.
    question_overlap: ['desk']
    cue_overlap: ['1', 'desk', 'whiteboard']
      before(turn 23,HIT): So desk 1 is the leftmost, next to the whiteboard, and desk 6 is the rightmost. Got it.
      after(turn 25,HIT): Makes sense. So you're in desk 1. That leaves desks 2-6 for Alice, Bob, Carol, Dave, and Eve.
  - **Turn 28** (user) mode=`ranked_out` [ADJ_r1]
    > Good point, they're all window desks. But the ones at the ends get the most light because of how the building is angled. So desk 1 or desk 6 would be best for him.
    question_overlap: ['desk', 'ones']
    cue_overlap: ['1', 'desk', 'desks']
      before(turn 27,HIT): All the desks are by the windows though, right? Does he have a preference for a specific position?
      after(turn 29,HIT): Since you're taking desk 1, that means Bob would prefer desk 6 for the best natural light. Want to tentatively put him there?

### puzzle_chain_2 / q5 — sequential_chain (r@20=0.31)
**Question**: What chain of discoveries led to successfully recreating the grandmother's lamb stew?
**Source turns**: [2, 4, 14, 16, 24, 25, 27, 32, 33, 38, 39, 41, 50, 52, 62, 64] (16 total)
**Retrieved source turns**: [2, 24, 32, 33, 64]
**Missed source turns**: [4, 14, 16, 25, 27, 38, 39, 41, 50, 52, 62]
**Cues v2f generated**:
  - _It really was like detective work! From the journal discovery to identifying golpar, to the dried limes memory, to the cooking technique. Each clue led to the next._
  - _Ok so the chain is: get golpar and dried limes from Sahel Market, bloom the spices in oil first, then build the stew in the Dutch oven. I'm going to try this Sunday._
**Missed turn samples** (up to 3):
  - **Turn 4** (user) mode=`ranked_out` [ADJ_r1]
    > She always used this big cast iron Dutch oven. But I have one too and it doesn't taste the same. I think it's a missing ingredient.
    question_overlap: ['s']
    cue_overlap: ['dutch', 'oven']
      before(turn 3,HIT): That 'smoky warmth' is a good clue. Could be a specific spice or a cooking technique. Did she use a particular kind of pot or cooking method?
      after(turn 5,other): Cast iron can add some flavor over decades of seasoning, but you're probably right that it's a specific ingredient. Did she ever share the recipe?
  - **Turn 14** (user) mode=`ranked_out`
    > An old leather journal that belonged to my grandmother. It's mostly in Farsi - she was Iranian. She immigrated here in the 70s.
    question_overlap: ['grandmother', 's']
    cue_overlap: ['journal']
      before(turn 13,other): What did you find?
      after(turn 15,other): What a treasure! Can you read Farsi?
  - **Turn 16** (user) mode=`ranked_out` [ADJ_r1]
    > Not well, but my cousin Dariush can. He's going to help me translate some of it. There are a few pages that look like they have recipes.
    question_overlap: ['s']
    cue_overlap: ['going']
      before(turn 15,other): What a treasure! Can you read Farsi?
      after(turn 17,HIT): That could be the breakthrough you need! If her Sunday stew recipe is in there, you might finally crack the mystery ingredient.

### puzzle_explore_1 / q11 — open_exploration (r@20=0.38)
**Question**: Based on all the patterns in our conversations, are there any health concerns about the user I should flag?
**Source turns**: [0, 2, 10, 12, 14, 16, 28, 30, 38, 40, 42, 43, 52, 54, 55, 72, 74, 75, 77, 82, 84] (21 total)
**Retrieved source turns**: [0, 28, 38, 40, 43, 74, 75, 82]
**Missed source turns**: [2, 10, 12, 14, 16, 30, 42, 52, 54, 55, 72, 77, 84]
**Cues v2f generated**:
  - _Third Monday in a row I get neck pain that sometimes radiates to my shoulder and gives me headaches, and occasionally my arm feels numb or tingly_
  - _I sleep later on weekends, drink more caffeine on Mondays, haven't changed my desk setup or taken meds, and I said I'd try stretches and see my doctor if it doesn't improve_
**Missed turn samples** (up to 3):
  - **Turn 2** (user) mode=`ranked_out` [ADJ_r2]
    > Sleep was fine, about 7 hours. I had plenty of water yesterday. Probably just stress from the new project that started last week.
    cue_overlap: ['sleep']
      before(turn 1,other): Sorry to hear that! Did you sleep ok? Headaches can come from poor sleep, dehydration, or stress.
      after(turn 3,other): New projects can definitely bring stress headaches. What's the project about?
  - **Turn 10** (user) mode=`ranked_out`
    > Good idea, I'll show her some swatches. Hey, we changed floors at work last month. Did I mention that?
    cue_overlap: ['changed']
      before(turn 9,other): Sage green is really popular right now and creates a calming atmosphere. You could compromise with a very light sage that reads as a soft neutral from a distance.
      after(turn 11,other): No, you didn't! What prompted the move?
  - **Turn 12** (user) mode=`vocabulary_gap`
    > Our team got consolidated. We used to be on floor 3, now we're on floor 7. The views are better but the lighting is different - these super bright overhead fluorescents. Kind of harsh.
      before(turn 11,other): No, you didn't! What prompted the move?
      after(turn 13,other): Harsh fluorescent lighting can be unpleasant. Some people find it causes eye strain or headaches. Have you noticed any difference since the move?

### puzzle_absence_1 / q13 — absence_inference (r@20=0.40)
**Question**: Why did the user choose an indoor restaurant venue for the December holiday party instead of their friend Marcus's backyard?
**Source turns**: [2, 4, 8, 10, 12, 14, 18, 20, 22, 40, 42, 50, 52, 54, 60] (15 total)
**Retrieved source turns**: [12, 40, 42, 50, 54, 60]
**Missed source turns**: [2, 4, 8, 10, 14, 18, 20, 22, 52]
**Cues v2f generated**:
  - _Honestly... I think it would just be easier for everyone. Less setup, less cleanup. Plus the restaurant has a great atmosphere. The place I'm looking at has this gorgeous private r…_
  - _Marcus was a little disappointed we're not doing his backyard but he understands. He said 'yeah you hate the cold, makes sense.' I can't help it!_
**Missed turn samples** (up to 3):
  - **Turn 2** (user) mode=`ranked_out`
    > All sorts of stuff. Last June we did a barbecue at the park and it was perfect. Great weather, everyone loved it.
    cue_overlap: ['everyone', 'great']
      before(turn 1,other): That's a great idea! What kinds of activities do you and your friends enjoy?
      after(turn 3,other): A park barbecue sounds lovely! Do you do that often?
  - **Turn 4** (user) mode=`vocabulary_gap`
    > We try to do something every month or two. In July we went to an outdoor concert series at the amphitheater downtown. That was awesome.
      before(turn 3,other): A park barbecue sounds lovely! Do you do that often?
      after(turn 5,other): Outdoor concerts in summer are the best. What kind of music?
  - **Turn 8** (user) mode=`vocabulary_gap`
    > Let me think... In August we did a hiking day trip to the gorge. About 8 of us went. Then September we had a big soccer game in the park.
      before(turn 7,other): Sounds wonderful! What other outings have you done?
      after(turn 9,other): So you're an active, outdoorsy group! What about fall and winter activities?

### Dataset: `synthetic_19q`

### synth_work / q14 — proactive (r@20=0.23)
**Question**: Draft a project status update for the Acme Corp rebrand team covering current progress and next steps.
**Source turns**: [4, 6, 10, 20, 48, 52, 58, 63, 66, 68, 84, 86, 88] (13 total)
**Retrieved source turns**: [10, 63, 88]
**Missed source turns**: [4, 6, 20, 48, 52, 58, 66, 68, 84, 86]
**Cues v2f generated**:
  - _Status of logo concepts — three options from Vanessa — latest feedback, approvals, and chosen direction_
  - _Hiroshi brief on content strategy and brand voice guide (tone refresh); Phase 1 (Jan 5–Jan 31) brand identity status and Phase 2 (Feb 1–Feb 21) website design/dev progress; name of…_
**Missed turn samples** (up to 3):
  - **Turn 4** (user) mode=`ranked_out` [ADJ_r2]
    > So the core team is me (project lead), Vanessa on design, Tom on frontend dev, Maria on backend, and Hiroshi on content/copy.
    question_overlap: ['project', 'team']
    cue_overlap: ['content', 'design', 'dev', 'hiroshi', 'vanessa']
      before(turn 3,other): That's a big project! Who's on the team?
      after(turn 5,other): Good team. What's the timeline looking like?
  - **Turn 6** (user) mode=`vocabulary_gap`
    > Deadline is March 15th. Non-negotiable, they have a product launch on March 20th.
      before(turn 5,other): Good team. What's the timeline looking like?
      after(turn 7,other): That's a tight turnaround. When did the contract start?
  - **Turn 20** (user) mode=`ranked_out` [ANAPH]
    > Right, let me check. Actually Tom talked to their IT team and they're on WordPress with a custom theme. We're proposing to migrate to a headless CMS - probably Strapi with a Next.js frontend.
    question_overlap: ['next', 'team']
      before(turn 19,other): You haven't mentioned their current CMS to me. Did they share that in the brand book or in a separate technical document?
      after(turn 21,other): Strapi + Next.js is a solid stack for a modern website. Is the client on board with the migration?

### synth_planning / q17 — procedural (r@20=0.29)
**Question**: Create a complete checklist of remaining tasks for the June 15th anniversary party, organized by timeline.
**Source turns**: [2, 46, 48, 50, 54, 56, 58, 62, 64, 69, 78, 80, 82, 86, 87, 90, 92] (17 total)
**Retrieved source turns**: [2, 48, 69, 86, 87]
**Missed source turns**: [46, 50, 54, 56, 58, 62, 64, 78, 80, 82, 90, 92]
**Cues v2f generated**:
  - _Complete checklist of remaining tasks for the June 15th surprise 40th anniversary party organized by timeline, working backwards from June 15 (By May 1 send digital invitations; By…_
  - _Master checklist and day-of timeline with vendor confirmations and deadlines: invitations and RSVP deadlines, confirm DJ Marco, Aunt Rosa catering, photographer, videographer Jason…_
**Missed turn samples** (up to 3):
  - **Turn 46** (user) mode=`anaphoric_reference` [ADJ_r1,ANAPH]
    > Yeah I think $150 covers it. Ok so let me think about what else... Oh! We need to figure out how to actually GET my parents there without spoiling the surprise.
    cue_overlap: ['surprise']
      before(turn 45,other): Homemade hot sauce with custom labels is such a personal and unique favor! The bottles and labels might cost around $100-150 for 70. That's a great use of budget.
      after(turn 47,HIT): The logistics of the surprise! This is crucial. What are you thinking?
  - **Turn 50** (user) mode=`vocabulary_gap` [ADJ_r1]
    > I'll be at the venue. Karen will drive them. She'll text me when they're 10 minutes away so everyone can get in position.
      before(turn 49,HIT): Using Karen's actual birthday as the cover story is clever - it's believable and close to the real date. They'll already be dressed nicely for a birthday dinner. Who will the point…
      after(turn 51,HIT): Good plan. Make sure DJ Marco knows the signal too - he'll need to cue the right music or announcement. Now, is there a Plan B if something goes wrong? Like if your parents want to…
  - **Turn 54** (user) mode=`vocabulary_gap`
    > Let me think... Actually, can we talk about the seating arrangement? With 60 people there are some dynamics to manage.
      before(turn 53,other): That's a reasonable fallback. OK so where are we on the checklist?
      after(turn 55,other): Absolutely. Any family dynamics or considerations?

### synth_personal / q6 — completeness (r@20=0.31)
**Question**: List ALL dietary restrictions and food preferences for every guest at the Saturday dinner party, including any updates or corrections.
**Source turns**: [14, 16, 18, 28, 32, 40, 42, 50, 62, 64, 84, 91, 93] (13 total)
**Retrieved source turns**: [32, 84, 91, 93]
**Missed source turns**: [14, 16, 18, 28, 40, 42, 50, 62, 64]
**Cues v2f generated**:
  - _Final restriction summary: Bob — peanut allergy, lactose intolerant (aged cheese/butter OK), keto preference; Emma — tree nut allergy; Sarah — vegan and soy allergy; Priya & Dev Pa…_
  - _Reminder/decision: blanket no‑nut policy for the dinner (no peanuts, tree nuts, or nut‑derived oils) and any updates or corrections to individual restrictions should be listed (e.g…_
**Missed turn samples** (up to 3):
  - **Turn 14** (user) mode=`ranked_out` [ADJ_r1,ANAPH]
    > Yeah but first let me tell you the dietary stuff. Bob is allergic to peanuts. He's been allergic since childhood.
    question_overlap: ['dietary']
    cue_overlap: ['bob', 'peanuts']
      before(turn 13,other): Nice group! That's 8 including you. Should I help with the menu?
      after(turn 15,HIT): Noted - peanut allergy for Bob. Anyone else?
  - **Turn 16** (user) mode=`ranked_out` [ADJ_r1]
    > Sarah is vegetarian. She's been vegetarian for about 5 years now. Oh and she doesn't eat eggs either actually, so more like vegan for cooking purposes.
    cue_overlap: ['sarah', 'vegan']
      before(turn 15,HIT): Noted - peanut allergy for Bob. Anyone else?
      after(turn 17,other): Got it - Sarah is effectively vegan. I'll keep that in mind for the menu.
  - **Turn 18** (user) mode=`ranked_out`
    > The Patels don't eat beef - they're Hindu. But everything else is fine for them.
    cue_overlap: ['beef']
      before(turn 17,other): Got it - Sarah is effectively vegan. I'll keep that in mind for the menu.
      after(turn 19,other): Understood. No beef for Priya and Dev Patel. What about James and Linda?

### synth_work / q4 — conjunction (r@20=0.57)
**Question**: What content should be included in the presentation for the Acme Corp client meeting on Wednesday?
**Source turns**: [12, 14, 34, 36, 68, 86, 88] (7 total)
**Retrieved source turns**: [12, 14, 34, 68]
**Missed source turns**: [36, 86, 88]
**Cues v2f generated**:
  - _Brand color palette (primary Pantone 2945 C blue, proposed new secondaries: cool gray + teal). Logo concepts (three options from Vanessa). Typography (Acme Sans_
  - _Acme wants a complete tone refresh to be more approachable and human; new CEO pushing for a startup-y feel. CMO Patricia requested adding an animated hero video to the website (sco…_
**Missed turn samples** (up to 3):
  - **Turn 36** (user) mode=`ranked_out` [ADJ_r1,SHORT]
    > Same meeting next Wednesday. We're presenting colors and logo together.
    question_overlap: ['meeting', 'wednesday']
    cue_overlap: ['logo']
      before(turn 35,other): Smart to give them options. When will those be presented?
      after(turn 37,HIT): Big meeting then. Color palette and logo concepts both. Should Vanessa present, or will you?
  - **Turn 86** (user) mode=`vocabulary_gap`
    > He went with 'confident but conversational.' Think Slack or Stripe's marketing copy. Professional but not stuffy.
      before(turn 85,other): Great, let's add that after typography then: Brand Voice Guide overview. What direction did Hiroshi settle on?
      after(turn 87,other): 'Confident but conversational' - that fits perfectly with Daniel Park's vision. Slack and Stripe are excellent reference points for that tone. Hiroshi clearly understood the brief.
  - **Turn 88** (user) mode=`anaphoric_reference` [ANAPH]
    > Yeah he nailed it. Ok I'll work on the deck using these sections. One more thing - Vanessa just told me she wants to add a motion guidelines section to the brand book. Like how things should animate on the website.
    cue_overlap: ['brand', 'more', 'vanessa', 'wants', 'website']
      before(turn 87,other): 'Confident but conversational' - that fits perfectly with Daniel Park's vision. Slack and Stripe are excellent reference points for that tone. Hiroshi clearly understood the brief.
      after(turn 89,other): Motion guidelines are a smart addition to the brand book - they ensure consistency in how elements move across the website and future applications. Things like easing curves, durat…


## Failure modes and proposed interventions

### 1. Adjacent-to-found (mechanical fix)
**Prevalence**: 50.0% of all missed turns are ±1 from a retrieved turn; 64.1% are ±2.

**Why**: The cosine similarity landscape is not smooth across a multi-turn exchange. A question turn ("OK, going with React.") ranks high, but the user's reply ("sure, whatever you want") or the preceding setup ("so you're rejecting Vue then") is semantically adjacent content that wouldn't embed close to the question independently.

**Intervention**: Post-retrieval neighbor expansion at r=1 or r=2. This is purely mechanical — no LLM calls. If K=20 and we retrieve 20 segments, expand each to include ±1 neighbors (up to some cap), then trim back. By the adj_r1% numbers alone, this could lift recall by 5-15 percentage points per dataset with zero new LLM work.

### 2. Ranked-out (the big one: ~75% of failures)
**What it looks like**: The missed turn contains vocabulary from the question or cue, but ranked below the K=20 cutoff. Common subcases:
- **Diluted by better-matching distractors**: A conversation about React vs Vue contains 30 turns with "Vue" in them but only 7 are the "why rejected" turns. Question/cue about rejection matches all 30.
- **Chronic long-tail**: Questions asking "all X" or "every Y" have 10-20 source turns; even a perfect cosine would struggle when the target set is larger than the budget.
- **Short high-overlap lost to longer high-overlap**: Short turns ("yeah drop Vue") have high IDF-style signal but lose to longer paraphrases that happen to have more keyword cover.

**Intervention**:
- **Multi-cue diversification**: Instead of 2 cues, generate 4-5 cues and spread K across them so each cue contributes fewer top segments. v2f currently shoves 10 into the question and 10 into each of 2 cues. Spreading to 5 cues × 4-top each would surface more long-tail source turns.
- **Reciprocal-rank fusion** between question and cues rather than dedup-and-append, so each cue has a fair shot at pushing a segment into the top-K.
- **Adaptive cue generation for multi-item questions**: v2f already has a completeness hint; the data suggests the hint isn't producing ENOUGH cues for "all X" questions. Detect the cardinality implied by the question and scale cue count accordingly (e.g., N_cues = max(2, K_remaining / 5)).

### 3. Vocabulary gap (~15% of failures)
**What it looks like**: The missed turn uses terminology that never appears in the question or cues. Seen frequently in `evolving_terminology` and `procedural` categories where the same concept gets renamed across the conversation. E.g., a bug referred to as "that orderbook issue" early on and "JIRA-4521" later — cues mention JIRA-4521 and miss the early orderbook turns.

**Intervention**:
- **Two-pass retrieval with entity bridging**: First pass finds the "JIRA-4521" turns. Extract entities/aliases from those turns ("orderbook", "price sync") and issue a second pass with those. This is a pure retrieval-layer change (1 extra LLM call to extract aliases).
- **Ingestion-side: context-enriched derivatives**. Each turn's embedding could be augmented with a short summary of its 3-turn context window. A reply like "sure" would then embed with the question it answers baked in, so vocabulary from the preceding turn becomes findable via the reply's embedding.

### 4. Anaphoric / implicit short responses (~10% of failures)
**What it looks like**: Turns like "yeah, exactly", "agreed — let's do that", "probably fine". They carry the user's endorsement but no retrievable content. In "rejected frameworks" questions, the actual rejection is often a short "no, drop Vue" — the reasoning lives in the prior turn.

**Intervention**:
- **Speaker-prefixed + context-enriched embeddings** at ingestion: each turn's stored embedding concatenates the prior 1-2 turns. Short responses become retrievable via the decision they're acknowledging.
- **Neighbor expansion (r=1)** is a cheaper retrieval-side approximation that recovers most of these.

## Meta-question: retrieval-layer vs ingestion-side

| Failure mode | Share | Retrieval-layer fix | Ingestion-side fix |
|---|---:|---|---|
| Adjacent-to-found | ~50% of missed | **YES** — r=1/2 neighbor expansion post-retrieval | context-enriched embeddings also work |
| Ranked-out | ~75% (overlaps with above) | **YES** — more cues + RRF + adaptive N_cues | richer per-turn derivatives |
| Vocabulary gap | ~15% | partial — entity-bridge 2nd pass needs ingestion index over entities | **YES** — summary/alias augmentation at ingestion |
| Anaphoric/short | ~10% | partial — neighbor expansion recovers most | **YES** — context-enriched embeddings fundamentally fix it |


### Net recommendation (ordered by cost × impact):

1. **Neighbor expansion at r=1 (retrieval layer)** — free-ish mechanical fix; ~50% of misses live within r=1.
2. **Increase cue count and use RRF fusion (retrieval layer)** — addresses the dominant `ranked_out` mode.
3. **Context-enriched embeddings (ingestion side)** — addresses both anaphoric and adjacency failures at their root; more invasive but the right long-term fix.
4. **Alias extraction pass (retrieval layer, 1 extra LLM call)** — for evolving_terminology specifically.
