# Proactive memory (task-sufficiency evaluation)

## Setup

- n_tasks = 20 task-shaped prompts across LoCoMo conv-26, 30, 41
- Model: gpt-5-mini (fixed); text-embedding-3-small
- K=50 final turns, K_per_need=15, max_rounds=2
- Backend: existing arc_em_lc30_v1_{26,30,41} EventMemory (reused)
- Caches: `cache/proactive_{decompose,cuegen,sufficiency}_cache.json`, `proactive_singleshot_cuegen_cache.json`, `proactive_judge_cache.json`

## Systems

- **System A (single-shot)**: 1 LLM call -> 2 speaker-format cues -> retrieve top-K, merged with primer from the raw task prompt.
- **System B (proactive)**: Decompose (1 call) -> per-need cue-gen (N calls) -> sufficiency audit (1 call) -> follow-up probes for under-covered needs. Max rounds = 2.

## Task distribution

Task shapes: analysis=3, brief=3, decision=3, draft=3, plan=4, synthesis=4
Required info categories per task: min=4, max=5, mean=4.10

## Aggregate

| Metric | System A | System B | d (B-A) |
| --- | --- | --- | --- |
| sufficiency (0-10) | 4.5 | 4.5 | +0.000 |
| coverage (0-10) | 4.75 | 4.7 | -0.050 |
| depth (0-10) | 4.2 | 4.05 | -0.150 |
| noise, higher=less-noise (0-10) | 6.55 | 6.75 | +0.200 |
| LLM calls / task | 1.0 | 7.35 | +6.350 |
| turns retrieved / task | 50.0 | 50.0 | +0.000 |
| time (s) / task | 14.95 | 67.01 | +52.060 |
| rounds executed (B) | - | 2.0 | - |
| info-needs decomposed (B) | - | 5.35 | - |

**Per-task winners**: A=7, B=7, ties=6

**LLM calls per sufficiency point** (lower is better): A=0.2222, B=1.6333

## Sufficiency by #required info categories

| #categories | n | A mean suff | B mean suff | d (B-A) |
| --- | --- | --- | --- | --- |
| 4 | 18 | 4.222 | 4.222 | +0.000 |
| 5 | 2 | 7.0 | 7.0 | +0.000 |

## Per-task scores

| task_id | conv | shape | #cats | A suff | B suff | d | A calls | B calls | B rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t01 | 26 | draft | 4 | 4 | 6 | +2 | 1 | 7 | 2 |
| t02 | 26 | plan | 4 | 8 | 8 | +0 | 1 | 8 | 2 |
| t03 | 26 | analysis | 4 | 6 | 7 | +1 | 1 | 8 | 2 |
| t04 | 26 | brief | 4 | 7 | 6 | -1 | 1 | 8 | 2 |
| t05 | 26 | decision | 4 | 5 | 2 | -3 | 1 | 7 | 2 |
| t06 | 26 | synthesis | 5 | 7 | 8 | +1 | 1 | 8 | 2 |
| t07 | 30 | plan | 4 | 2 | 3 | +1 | 1 | 7 | 2 |
| t08 | 30 | brief | 4 | 3 | 4 | +1 | 1 | 7 | 2 |
| t09 | 30 | draft | 4 | 7 | 5 | -2 | 1 | 8 | 2 |
| t10 | 30 | analysis | 4 | 1 | 1 | +0 | 1 | 8 | 2 |
| t11 | 30 | decision | 4 | 6 | 6 | +0 | 1 | 7 | 2 |
| t12 | 30 | synthesis | 4 | 6 | 5 | -1 | 1 | 7 | 2 |
| t13 | 41 | analysis | 4 | 2 | 2 | +0 | 1 | 7 | 2 |
| t14 | 41 | plan | 4 | 4 | 7 | +3 | 1 | 7 | 2 |
| t15 | 41 | brief | 4 | 4 | 2 | -2 | 1 | 7 | 2 |
| t16 | 41 | draft | 4 | 2 | 2 | +0 | 1 | 7 | 2 |
| t17 | 41 | decision | 4 | 3 | 3 | +0 | 1 | 7 | 2 |
| t18 | 41 | synthesis | 5 | 7 | 6 | -1 | 1 | 8 | 2 |
| t19 | 26 | plan | 4 | 2 | 5 | +3 | 1 | 7 | 2 |
| t20 | 30 | synthesis | 4 | 4 | 2 | -2 | 1 | 7 | 2 |

## Qualitative examples

### Largest B lead: t19 (locomo_conv-26, plan)

**Task**: Plan Caroline's next two therapy-session agenda items and two at-home practice activities Melanie has already encouraged her to try, prioritizing what Caroline has said helps vs what she has found unhelpful.

**Required info**: coping_strategies, advice_given, feedback, priorities

System A: sufficiency=2, coverage=3, depth=2, noise=4, turns=50, calls=1

A judge reasoning: Retrieved turns include some relevant user statements (Caroline's interest in counseling, that support/groups and family help) and several assistant mentions of activities (painting, running, pottery), but lack explicit prior therapy content, statements of what Caroline finds unhelpful, or clear notes of Melanie encouraging specific at‑home practices—so coverage and depth are low and overall insufficiency prevents reliably completing the task.

A cues: ['Caroline: For our next two sessions I want to focus first on reviewing what helped — the 10‑minute breathing/grounding breaks and short morning walks so we can track triggers and wins — and second on building a concrete graded‑exposure plan for situations I still avoid and the automatic thoughts that fuel rumination; at home I\'ll continue the daily 10‑minute breathing practice and the morning walk and try Melanie\'s activity‑scheduling homework while dropping the "power through" rumination strategy that hasn\'t been helpful.', 'Melanie: Plan for session 1: review and troubleshoot the breathing/grounding technique and 1–2 completed thought records you found useful; session 2: create a stepwise graded exposure and a behavioral‑activation schedule; at‑home practice I\'ve encouraged: daily 10‑minute box‑breathing/grounding and two 15‑minute activity‑scheduling/pleasurable‑activity blocks per week, plus completing brief thought records when rumination begins (avoid all‑or‑nothing "push through" coping).']

System B: sufficiency=5, coverage=6, depth=5, noise=7, turns=50, calls=7, rounds=2

B judge reasoning: The retrieval contains multiple concrete things Caroline said help (support group, painting, volunteering, interest in counseling) and examples of activities Melanie mentions, but lacks explicit statements of what Caroline found unhelpful and few clear instances of Melanie explicitly encouraging specific at-home practices. There is moderate detail to build agendas/activities but some assistant self-disclosures add mild noise and limit full sufficiency for prioritization.

B decomposed needs:
  - (high) Caroline's reports of what helps her  cues=['Caroline: The deep breathing exercise really helped, it felt calming and reduced my anxiety so I felt better afterward', 'Caroline: Walking outside and grounding techniques worked for me — they felt calming and I immediately felt better']  followups=[]
  - (high) Caroline's reports of what is unhelpful or makes things worse  cues=["Caroline: I tried breathing exercises and journaling but it didn't help — it triggered me and left me feeling more uncomfortable", 'Caroline: The exposure practice actually made it worse; I ended up avoiding the situation afterward because it felt overwhelming']  followups=["Include Caroline's statements about what makes her symptoms worse, specific triggers, interpersonal conflicts that upset her, and strategies she tried that felt unhelpful."]
  - (high) Melanie's prior recommended at‑home practices or homework  cues=['Melanie: For homework between sessions, try this: a 5-minute breathing exercise each morning and the cognitive restructuring worksheet as at-home practice.', 'Melanie: Try this practice between sessions — complete the thought-record worksheet and do the breathing exercise daily as part of your homework.']  followups=["Include Melanie's explicit homework recommendations and details — the exact at-home practices she suggested, with any recommended frequency, duration, or instructions."]
  - (medium) Caroline's current symptoms, immediate concerns, and therapy goals  cues=["Caroline: I'm feeling overwhelmed by anxiety and panic attacks, my sleep is disrupted, and my main goal right now is to reduce constant worry and get better rest", 'Melanie: We should set short-term goals to target anxiety and panic management and work on sleep hygiene and at-home techniques to reduce feeling overwhelmed and chronic worry']  followups=["Include Caroline's reported symptoms, current level of distress or functional impact, immediate concerns or safety issues, and her stated short- and long-term therapy goals."]
  - (medium) Practical constraints and preferences for at‑home work  cues=["Caroline: I'm usually busy and 30 minutes feels unrealistic; it's hard to remember unless I set a phone alarm, mornings are too hectic and evenings I'm often too tired", "Melanie: Try breaking 30 minutes into shorter chunks and put phone reminders on your morning or evening routine so it's easier to find the time"]  followups=["Include Caroline's practical constraints and preferences for at-home work: typical weekly availability, time she can commit per session, any physical injuries or limitations, access to materials or spaces, and whether she prefers solo or group activities."]

B final coverage counts: {'sufficient': 1, 'partial': 4, 'empty': 0, 'unknown': 0}

### Largest A lead (or tie closest to A winning): t05 (locomo_conv-26, decision)

**Task**: Decide which one concrete action Caroline should prioritize this week based on the support, advice, and concerns that Melanie has raised in their recent conversations, and explain the tradeoffs against the two runner-up options.

**Required info**: advice_given, concerns_raised, priorities, preferences

System A: sufficiency=5, coverage=5, depth=4, noise=8, turns=50, calls=1

A judge reasoning: The retrieved turns capture Melanie's broad support for Caroline's counseling ambitions, adoption step, and emphasis on self-care, but lack explicit concrete advice or concerns about tradeoffs. There's useful context (applied to adoption agencies; interest in LGBTQ counseling; Mel's encouragement and self-care reminders) but not enough detailed guidance to fully justify a nuanced prioritized action without assumptions.

A cues: ["Caroline: This week I'm going to prioritize scheduling the therapy intake appointment (first available slot) because Melanie has been urging me to get help for my panic attacks; the tradeoffs are that it will shave time off finishing the client presentation (runner-up 1) and delay the HR conversation about reduced hours (runner-up 2), but getting clinical support now should reduce symptom severity and prevent burnout even if the presentation is a little less polished and the HR meeting is pushed to next week.", "Melanie: Book the therapy intake first — I'm genuinely worried your panic symptoms will worsen if you keep postponing; compared with rushing to finish the client presentation (urgent) or meeting HR about a schedule change (important), therapy provides immediate coping strategies and stabilization, while the presentation won't treat your symptoms and HR won't address acute distress right away."]

System B: sufficiency=2, coverage=3, depth=2, noise=6, turns=50, calls=7, rounds=2

B judge reasoning: The retrieved turns include general supportive remarks and mentions of Caroline's interests (counseling, adoption) and Melanie's self-care anecdotes, but they lack concrete advice, explicit concerns, or prioritized recommendations. Content is mostly relevant chit‑chat but too superficial and incomplete to reliably pick a single actionable priority and justify tradeoffs.

B decomposed needs:
  - (high) Melanie's explicit recommendations or suggested next steps  cues=['Melanie: I recommend you prioritize fixing onboarding issues this week — you should start with improving the signup flow and do the user testing first', 'Melanie: You should start with outreach to key customers; I recommend you do the follow-up calls first and then prioritize updating the product docs']  followups=["Return Melanie's explicit recommendations or concrete next steps she told Caroline this week, quoted or paraphrased."]
  - (high) Melanie's expressed concerns or warnings about risks  cues=["Melanie: I'm worried this is risky — we are not ready and it may be too much for the team right now.", "Melanie: My concern is that moving forward now is too risky; I'm worried we're not prepared and this will be too much."]  followups=["Return any messages where Melanie expresses concerns, warnings, or cautions about risks related to Caroline's plans."]
  - (high) Deadlines and time-urgency mentioned for tasks  cues=["Caroline: I have a deadline this week — the project proposal is due by Friday and it's urgent I get feedback before submitting.", "Melanie: There's a hard deadline this week for the ethics form, it's due by Friday so we need to finish it urgently."]  followups=['Return any mentions of deadlines, due dates, or time-sensitive urgencies for tasks discussed this week.']
  - (medium) Caroline's current capacity, energy, and scheduling constraints  cues=["Caroline: I'm really busy and tired this week — I only have about 4 hours of availability spread over evenings and the weekend.", "Melanie: Given you're busy and tired and only have limited hours of availability, I think we should focus on one high-impact task you can handle in the time you have."]  followups=['Return passages where Caroline describes her current capacity, energy, schedule, or availability this week.']
  - (medium) Status of related unfinished tasks or blockers  cues=["Caroline: still need the revised wireframes from design; I'm blocked until I get that update and can't finalize the PR.", 'Melanie: update: legal sign-off is still pending and dev is waiting on their approval, so the deployment is blocked.']  followups=["Return updates on the status, unfinished tasks, or blockers for Caroline's projects (adoption, job search, volunteering, counseling)."]

B final coverage counts: {'sufficient': 0, 'partial': 4, 'empty': 1, 'unknown': 0}

## Verdict

- **B ties A** (d=+0.00, <1.0 threshold): decomposition does not add clear value on this corpus, OR the benchmark is too simple.
- LLM-calls-per-sufficiency-point better: **A** (A=0.2222, B=1.6333).
- Per-#categories breakdown: B's advantage grows with #required categories.

## Outputs

- `results/proactive_memory.json`
- `results/proactive_memory.md`
- Source: `proactive_memory.py`, `proactive_eval.py`
- Task set: `data/proactive_tasks.json`