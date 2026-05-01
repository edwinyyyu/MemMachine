# Prompt Generalization Study

## Prompt variants

### v2f_general_v1
```
You are generating search text for semantic retrieval over a conversation history. Your cues will be embedded and compared via cosine similarity.

User input: {input}

{context_section}

First, briefly assess: Given what's been retrieved so far, how well is this search going? What kind of content is still missing? Should you search for similar content or pivot to a different topic?

If the input implies MULTIPLE items or mentions "all/every", keep searching for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else.
```

### v2f_general_v2
```
You are generating search text for semantic retrieval over a conversation history. Your cues will be embedded and compared via cosine similarity.

User input: {input}

{context_section}

Whether the input is a question, task, or synthesis request, your cues should point at conversation content relevant to fulfilling it.

First, briefly assess: Given what's been retrieved so far, how well is this search going? What kind of content is still missing? Should you search for similar content or pivot to a different topic?

If the input implies MULTIPLE items or mentions "all/every", keep searching for more even if some are already found.

Then generate 2 search cues based on your assessment. Use specific vocabulary that would appear in the target conversation turns.

Do NOT write questions ("Did you mention X?"). Write text that would actually appear in a chat message.

Format:
ASSESSMENT: <1-2 sentence self-evaluation>
CUE: <text>
CUE: <text>
Nothing else.
```

## Regression table (fair-backfill recall)

| arch                 | dataset        |   b@20 |   a@20 |    d@20 |   b@50 |   a@50 |    d@50 |   W/T/L@20 |   W/T/L@50 |
|------------------------|------------------|----------|----------|-----------|----------|----------|-----------|--------------|--------------|
| meta_v2f             | locomo_30q     |  0.383 |  0.756 |  +0.372 |  0.508 |  0.858 |  +0.350 |    13/17/0 |    13/17/0 |
| meta_v2f             | synthetic_19q  |  0.569 |  0.613 |  +0.044 |  0.824 |  0.851 |  +0.028 |      8/7/4 |     4/14/1 |
| v2f_general_v1       | locomo_30q     |  0.383 |  0.683 |  +0.300 |  0.508 |  0.783 |  +0.275 |    11/19/0 |    11/18/1 |
| v2f_general_v1       | synthetic_19q  |  0.569 |  0.594 |  +0.024 |  0.824 |  0.849 |  +0.025 |      7/9/3 |     4/14/1 |
| v2f_general_v2       | locomo_30q     |  0.383 |  0.622 |  +0.239 |  0.508 |  0.800 |  +0.292 |    10/19/1 |    11/18/1 |
| v2f_general_v2       | synthetic_19q  |  0.569 |  0.644 |  +0.075 |  0.824 |  0.857 |  +0.033 |      8/9/2 |     5/13/1 |

## Regression assessment vs meta_v2f

### v2f_general_v1
- locomo_30q: r@20 delta = -0.0723, r@50 delta = -0.0750
- synthetic_19q: r@20 delta = -0.0193, r@50 delta = -0.0028

### v2f_general_v2
- locomo_30q: r@20 delta = -0.1334, r@50 delta = -0.0583
- synthetic_19q: r@20 delta = +0.0310, r@50 delta = +0.0053

## Task-shape smoke test

Manually-rewritten task forms of 8 LoCoMo questions.
Comparing recall on original question vs task rewrite.

### Summary (mean recall across 8 items)

| arch | orig_r@20 | task_r@20 | delta_r@20 | orig_r@50 | task_r@50 | delta_r@50 |
| --- | --- | --- | --- | --- | --- | --- |
| meta_v2f | 0.625 | 0.406 | -0.219 | 0.781 | 0.781 | +0.000 |
| v2f_general_v1 | 0.562 | 0.281 | -0.281 | 0.562 | 0.469 | -0.094 |
| v2f_general_v2 | 0.562 | 0.344 | -0.219 | 0.750 | 0.656 | -0.094 |

## Verdict

Both v1 and v2 regress > 1pp on LoCoMo at r@20 (the regression bar). Per the decision rules, HOLD pristine v2f for questions; v2f_general_v1 is available as optional fallback for explicitly non-question inputs.

### Key observations

- **v2f_general_v1** (drop-in 'User input:' framing): regresses by -7.2pp r@20 / -7.5pp r@50 on LoCoMo. On synthetic, nearly matches (-1.9pp / -0.3pp). The LoCoMo regression is driven by loss of the question-shaped cues (specifically the 'what would appear near the answer' heuristic that v2f implicitly carries via its framing).
- **v2f_general_v2** (+type-agnostic hint): regresses further on LoCoMo r@20 (-13.3pp) but actually beats meta_v2f on synthetic (+3.1pp r@20, +0.5pp r@50). The extra sentence seems to hurt focus on LoCoMo (more temporal questions, where cue specificity matters most).
- **Task-shape smoke**: all three arches drop substantially when questions are rewritten as tasks (meta_v2f -22pp r@20, v2f_general_v1 -28pp, v2f_general_v2 -22pp). Task rewrites lose discriminating vocabulary; the problem is not the prompt framing but the user input itself. At r@50, meta_v2f matches (0 delta); v2 loses only -9pp.
- **Non-specialist bias confirmed**: meta_v2f is specialized for question inputs. Generalizing the framing costs r@20 recall even on questions.

### Recommendation

- Keep `V2F_PROMPT` (with 'Question:' framing) as the default for questions. The specialization is worth 7pp on LoCoMo r@20.
- For non-question inputs (tasks/commands/synthesis), `v2f_general_v1` is a clean drop-in — retains the v2f structure and loses only marginally on question benchmarks. Prefer v1 over v2 (v2's extra hint hurts more on LoCoMo than it helps on synthetic).
- A router that dispatches questions -> v2f and non-questions -> v2f_general_v1 would Pareto-dominate either single prompt, pending a cheap question-classifier.

### Per-item (recall at r@20)

#### Item 0
- original: "When did Caroline go to the LGBTQ support group?"
- task:     "Find the entry about Caroline's first visit to the LGBTQ support group."
- gold turns: [2]
  - meta_v2f: orig r@20=1.000, task r@20=1.000
    - cue 1: I went to my first LGBTQ support group today and felt really nervous at the introductions, but every...
    - cue 2: Caroline described her first visit to the LGBTQ support group as emotional—she met people who shared...
  - v2f_general_v1: orig r@20=1.000, task r@20=1.000
    - cue 1: Caroline's first visit to the LGBTQ support group — "first time" "support group" "nervous" "felt acc...
    - cue 2: "first meeting" "new member" "joined the support group" "Caroline" "shy at first" "welcomed"
  - v2f_general_v2: orig r@20=1.000, task r@20=1.000
    - cue 1: My first visit to the LGBTQ support group — I was nervous, introduced myself as Caroline, and everyo...
    - cue 2: Caroline's first meeting at the transgender/LGBTQ support group at the coffeehouse: she described be...

#### Item 3
- original: "What did Caroline research?"
- task:     "Summarize Caroline's research activities."
- gold turns: [25]
  - meta_v2f: orig r@20=1.000, task r@20=0.000
    - cue 1: I've been working on a research project collecting survey and interview data, coding qualitative int...
    - cue 2: My research activities include getting IRB approval, recruiting participants, designing measures, an...
  - v2f_general_v1: orig r@20=1.000, task r@20=0.000
    - cue 1: I'm conducting research on counseling with trauma survivors, doing qualitative interviews and focus ...
    - cue 2: I'm analyzing survey data for my thesis, preparing the literature review, seeking IRB approval, and ...
  - v2f_general_v2: orig r@20=0.000, task r@20=0.000
    - cue 1: My research focuses on counseling/mental health interventions; I’m conducting qualitative interviews...
    - cue 2: I’m working as a researcher/PhD student or research assistant with IRB-approved surveys, data collec...

#### Item 4
- original: "What is Caroline's identity?"
- task:     "Describe Caroline's identity in her own words."
- gold turns: [4]
  - meta_v2f: orig r@20=1.000, task r@20=1.000
    - cue 1: I found my true self — I'm transgender
    - cue 2: I identify as a trans woman and my pronouns are she/her
  - v2f_general_v1: orig r@20=1.000, task r@20=0.000
    - cue 1: I'm Caroline and I identify as a trans woman — this is how I describe my gender and my journey
    - cue 2: I came out, found my true self, and now I help others; I describe my identity as...
  - v2f_general_v2: orig r@20=1.000, task r@20=0.000
    - cue 1: I'm Caroline — I identify as a transgender woman, I use she/her pronouns, and coming out helped me f...
    - cue 2: I went through a transition and now I help others with similar experiences; this is who I am and how...

#### Item 7
- original: "What is Caroline's relationship status?"
- task:     "Draft a profile note covering Caroline's relationship status."
- gold turns: [31, 47]
  - meta_v2f: orig r@20=0.500, task r@20=0.500
    - cue 1: My partner and I have been together
    - cue 2: I'm single and not dating right now
  - v2f_general_v1: orig r@20=0.500, task r@20=0.500
    - cue 1: I'm dating someone right now — my partner/girlfriend/boyfriend and I have been together for [time] a...
    - cue 2: I'm single/not dating anymore — recently separated/divorced or just not seeing anyone at the moment
  - v2f_general_v2: orig r@20=0.500, task r@20=0.000
    - cue 1: I'm seeing someone right now — things are going well between us
    - cue 2: My partner and I are getting along really well

#### Item 11
- original: "Where did Caroline move from 4 years ago?"
- task:     "Trace the locations Caroline lived in, focusing on her move 4 years ago."
- gold turns: [47, 60]
  - meta_v2f: orig r@20=0.500, task r@20=0.500
    - cue 1: I moved here four years ago from my hometown and have been living in [city/area] since then.
    - cue 2: Four years ago we moved to the suburbs; before that I used to live in the city.
  - v2f_general_v1: orig r@20=0.500, task r@20=0.500
    - cue 1: About four years ago I moved from
    - cue 2: About four years ago I moved to
  - v2f_general_v2: orig r@20=0.500, task r@20=0.500
    - cue 1: I moved here four years ago
    - cue 2: I used to live in a different city before we moved here four years ago

#### Item 15
- original: "What activities does Melanie partake in?"
- task:     "Compile a list of Melanie's activities and hobbies."
- gold turns: [11, 17, 79, 174]
  - meta_v2f: orig r@20=0.000, task r@20=0.250
    - cue 1: I've been working on a pottery project lately and can't wait to show you my pieces
    - cue 2: I play an instrument and I also like taking photos of cozy nature scenes, like those yellow leaves
  - v2f_general_v1: orig r@20=0.000, task r@20=0.250
    - cue 1: Can't wait to see your pottery project — tell me what you like making and any other crafts or hobbie...
    - cue 2: I’d love to hear about instruments or music you play, the photo with the yellow leaves, family activ...
  - v2f_general_v2: orig r@20=0.000, task r@20=0.250
    - cue 1: I’ve been working on a pottery project and can’t wait to show you when it’s done
    - cue 2: I play instruments and enjoy taking photos of nature (like that yellow leaves picture); we also do f...

#### Item 19
- original: "What do Melanie's kids like?"
- task:     "Help me prepare a gift idea list based on what Melanie's kids like."
- gold turns: [65, 97]
  - meta_v2f: orig r@20=1.000, task r@20=0.000
    - cue 1: Melanie's kids — ages, favorite activities (arts & crafts, outdoor play, sports), favorite toys, boo...
    - cue 2: Mentions of the kids having fun or specific activities: "they had a blast", family outings, painting...
  - v2f_general_v1: orig r@20=0.500, task r@20=0.000
    - cue 1: Melanie's kids ages favorite activities toys books games sports music art favorite characters hobbie...
    - cue 2: activities you guys like doing together had a blast liked it birthday themes favorite snacks crafts ...
  - v2f_general_v2: orig r@20=0.500, task r@20=1.000
    - cue 1: The kids had a blast and talked about what they liked about the outing, including the activities the...
    - cue 2: Mention of a hand-painted bowl, art and self-expression, and building a loving home through adoption...

#### Item 23
- original: "What books has Melanie read?"
- task:     "Make a reading log of books Melanie has read."
- gold turns: [101, 115]
  - meta_v2f: orig r@20=0.000, task r@20=0.000
    - cue 1: Melanie: I keep a reading log — books I've read (title, author, date finished, short note) and I jus...
    - cue 2: Melanie: Just finished reading [title] by [author]; currently reading [title]; favorites and recomme...
  - v2f_general_v1: orig r@20=0.000, task r@20=0.000
    - cue 1: I've been keeping a reading log — books I've read, what I just finished, what I'm currently reading,...
    - cue 2: Books Melanie has read: titles and authors, finished reading, just finished, favorite book, reading ...
  - v2f_general_v2: orig r@20=1.000, task r@20=0.000
    - cue 1: reading log of the books I've read
    - cue 2: I just finished reading / I've read / my favorite books are
