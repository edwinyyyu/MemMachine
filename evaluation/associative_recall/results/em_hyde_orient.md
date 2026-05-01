# EM HyDE / Orient-then-cue re-test on LoCoMo-30

## Setup

- n_questions = 30 (benchmark=locomo, first 30)
- EventMemory backend (arc_em_lc30_v1_{26,30,41}); speaker-baked embedded format `"{source}: {text}"`
- Model: text-embedding-3-small + gpt-5-mini (fixed)
- Caches: `cache/hydeorient_<variant>_cache.json` (dedicated)

## Variants

- `em_hyde_narrative`: HyDE narrative -- LLM writes a 1-2 paragraph narrative retelling of what the conversation must have contained; embedded as a single probe.
- `em_hyde_turn_format`: HyDE turn format -- LLM writes 3-5 speaker-prefixed chat turns; each is a separate probe unioned by max score.
- `em_hyde_first_person`: HyDE first person -- LLM writes one first-person "I remember when <speaker> said ..." chat turn; single probe.
- `em_orient_brief`: Orient brief -- stage 1 writes a 1-sentence orientation describing what the query is looking for; stage 2 uses the orientation to generate 2 speakerformat cues.
- `em_orient_terminology`: Orient terminology -- stage 1 enumerates expected vocabulary that would appear in the target turns; stage 2 generates 2 speakerformat cues that include that vocabulary.

## Recall matrix

| Variant | R@20 | R@50 | d R@20 vs v2f_sf | d R@50 vs v2f_sf | time (s) |
| --- | --- | --- | --- | --- | --- |
| `em_hyde_narrative` | 0.7500 | 0.8333 | -0.0667 | -0.0584 | 399.4 |
| `em_hyde_turn_format` | 0.7056 | 0.7500 | -0.1111 | -0.1417 | 271.1 |
| `em_hyde_first_person` | 0.8000 | 0.9083 | -0.0167 | +0.0166 | 170.8 |
| `em_orient_brief` | 0.7750 | 0.8333 | -0.0417 | -0.0584 | 342.0 |
| `em_orient_terminology` | 0.7917 | 0.8167 | -0.0250 | -0.0750 | 429.3 |

Baselines (for reference, from prior runs):

- `em_v2f_speakerformat`: R@20=0.8167, R@50=0.8917
- `em_two_speaker_filter`: R@20=0.8417, R@50=0.9000
- `em_two_speaker_query_only`: R@20=0.8000, R@50=0.9333

## Composition with `property_filter(context.source)`

| Variant + speaker_filter | R@20 | R@50 |
| --- | --- | --- |
| `em_hyde_first_person+speaker_filter` | 0.8500 | 0.9417 |

Reference: `em_two_speaker_filter` R@20=0.8417, R@50=0.9000.

Reference: `em_two_speaker_query_only` R@50=0.9333 (K=50 leader).

## Sample outputs (2-3 questions, showing mechanism differences)

### Q0 (`locomo_conv-26`, locomo_temporal): 'When did Caroline go to the LGBTQ support group?'

Gold turn_ids: [2]

- `em_hyde_narrative` (R@20=1.00, R@50=1.00)
  probe: Caroline told Melanie that she had gone to an LGBTQ support group yesterday, describing the session as powerful and deeply affirming. She recounted arriving at the Rainbow Community Center, taking a folding chair in a quiet circle, and sharing pieces of her story while others nod...
- `em_hyde_turn_format` (R@20=1.00, R@50=1.00)
  turns:
    Caroline: I actually went to a local LGBTQ support group yesterday and it was so powerful.
    Melanie: That's amazing — I'm really glad you went. What was the meeting like?
    Caroline: It was small but warm; people shared their stories and I left feeling seen. Going yesterday gave me so much hope.
    Melanie: I'm so happy to hear that — glad yesterday was such a positive step for you.
- `em_hyde_first_person` (R@20=1.00, R@50=1.00)
  turn: Caroline: I went to the LGBTQ support group yesterday — it was such a powerful session and I left feeling really seen and supported.
- `em_orient_brief` (R@20=1.00, R@50=1.00)
  orientation: A factual question about timing/schedule of a past event — Melanie (asking Caroline) inquiring when Caroline attended the LGBTQ support group.
  cues: ['Caroline: I went to a LGBTQ support group yesterday and it was so powerful.', 'Melanie: Wondering when Caroline went to the LGBTQ support group (inquiring about timing)']
- `em_orient_terminology` (R@20=1.00, R@50=1.00)
  vocabulary: last week, yesterday, last night, on Tuesday, this morning, two weeks ago, Thursday evening, Saturday afternoon, last month, monthly meeting, drop-in session, Zoom meeting, Pride Center
  cues: ['Caroline: I went to a LGBTQ support group yesterday and it was so powerful.', 'Caroline: I attended a drop-in session at the Pride Center yesterday — the LGBTQ support group meeting really helped me.']

### Q15 (`locomo_conv-26`, locomo_single_hop): 'What activities does Melanie partake in?'

Gold turn_ids: [11, 17, 79, 174]

- `em_hyde_narrative` (R@20=0.50, R@50=0.50)
  probe: Melanie told Caroline she ran a charity race for mental health last Saturday and now carves out daily me-time by going for runs, reading novels, or practicing her violin. She signed up for a pottery class yesterday and described throwing clay on the wheel as a kind of therapy. A ...
- `em_hyde_turn_format` (R@20=0.50, R@50=0.50)
  turns:
    Caroline: You've mentioned a bunch of stuff — what activities have you been doing lately?
    Melanie: Oh, lots of little things — I ran a charity race for mental health last weekend, I go for regular runs, I read a lot, and I play the violin. I also just signed up for a pottery class and we went camping with the kids recently where we hiked and did some biking.
    Caroline: Wow, that sounds like a full plate in a good way — which of those are regular habits versus one-offs?
    Melanie: Running and reading are daily-ish, I practice the violin a few times a week, pottery is a weekly class, and camping/hiking/biking tend to be weekend family stuff. I also join in charity events now and then.
- `em_hyde_first_person` (R@20=0.50, R@50=0.75)
  turn: Melanie: Since we last talked I've been staying active and creative — I ran a charity race for mental health last weekend, I go for runs regularly, I carve out time to read and practice my violin each day, I just signed up for a pottery class, and I bike and go camping with the k
- `em_orient_brief` (R@20=0.25, R@50=0.50)
  orientation: A factual question about Melanie's activities or hobbies, most likely asked by Caroline, seeking which specific activities, pastimes, or participations Melanie engages in.
  cues: ['Melanie: I ran a charity race for mental health last Saturday, and I carve out me-time each day for running, reading, or playing my violin — I also just signed up for a pottery class.', 'Caroline: Melanie participates in a charity race for mental health, daily running and reading, violin practice, a pottery class, and outdoor activities like camping and hiking.']
- `em_orient_terminology` (R@20=0.25, R@50=0.50)
  vocabulary: hiking, yoga, trail running, baking sourdough, watercolor painting, volunteering at animal shelter, photography, gardening, cycling, kayaking, salsa dancing, rock climbing
  cues: ['Caroline: Melanie loves hiking, yoga, and trail running, she’s also into baking sourdough, watercolor painting, volunteering at the animal shelter, and photography', 'Melanie: I spend weekends gardening, cycling, and kayaking, and I also go salsa dancing and rock climbing while practicing watercolor painting and baking sourdough']

### Q29 (`locomo_conv-26`, locomo_temporal): 'When did Melanie go to the pottery workshop?'

Gold turn_ids: [136]

- `em_hyde_narrative` (R@20=1.00, R@50=1.00)
  probe: Melanie told Caroline that she had just signed up for a pottery class the day before, and later explained that last Friday she finally took her kids to a pottery workshop. She described how they each made their own pots from wet clay, shaping and smoothing them together, and call...
- `em_hyde_turn_format` (R@20=1.00, R@50=1.00)
  turns:
    Caroline: When did you go to that pottery workshop — was it recently?
    Melanie: Last Friday — I finally took the kids with me and we all made our own little pots. It was so fun and oddly therapeutic.
    Caroline: That sounds lovely, I can picture you elbow-deep in clay already.
    Melanie: I'll send you a photo of what we made — they're far from perfect but I'm really proud of them.
- `em_hyde_first_person` (R@20=1.00, R@50=1.00)
  turn: Melanie: I told you I went to a pottery workshop last Friday — I took the kids, we each made our own pots, and it was surprisingly therapeutic.
- `em_orient_brief` (R@20=1.00, R@50=1.00)
  orientation: A factual, time-based question likely asked by Caroline of Melanie about when she attended the pottery workshop, seeking a specific past event date or schedule detail.
  cues: ['Melanie: Last Fri I finally took my kids to a pottery workshop. We all made our own pots, it was fun and therapeutic!', 'Caroline: asking when Melanie went to the pottery workshop']
- `em_orient_terminology` (R@20=1.00, R@50=1.00)
  vocabulary: pottery workshop, studio, ceramics class, wheel-throwing session, went, attended, took a class, last Saturday, Saturday morning, last weekend, two weeks ago, this afternoon, at 3 PM
  cues: ['Melanie: Last Fri I finally took my kids to a pottery workshop at the studio — we attended a ceramics class and did a wheel-throwing session.', 'Caroline: Melanie said she went to the pottery workshop last Friday, where she took a class in the studio and attended a wheel-throwing session.']

## Verdict

**Narrow (one-sided lift only vs em_v2f_speakerformat):**
- `em_hyde_first_person`: d20=-0.0167, d50=+0.0166

**New K=50 ceiling**: `em_hyde_first_person+speaker_filter` R@50=0.9417 >= em_two_speaker_query_only 0.9333.

## HyDE multi-probe vs single-probe pattern

- turn_format (multi-probe): R@20=0.7056, R@50=0.7500
- narrative (single probe): R@20=0.7500, R@50=0.8333
- first_person (single probe): R@20=0.8000, R@50=0.9083
Multi-probe does NOT win over single-probe HyDE here.

## Outputs

- `results/em_hyde_orient.json`
- `results/em_hyde_orient.md`
- Source: `em_hyde_orient.py`, `hydeorient_eval.py`
- Caches: `cache/hydeorient_<variant>_cache.json`
