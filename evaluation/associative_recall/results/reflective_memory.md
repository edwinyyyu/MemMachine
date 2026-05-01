# Reflective LLM writes-to-memory (LoCoMo-30)

## Setup

- n_questions = 30 (benchmark=locomo, first 30)
- EventMemory backend (arc_em_lc30_v1_{26,30,41}); speaker-baked embedded format `"{source}: {text}"`
- Model: text-embedding-3-small + gpt-5-mini (fixed)
- Scratch memory: in-memory numpy cosine; per-query, not persisted
- Caches: `cache/reflmem_{cuegen_r1,reflect,cuegen_r2}_cache.json` (dedicated)

## Variants

- `reflmem_1round`: Single round: LLM cue-gen -> retrieve -> reflect (learned / still_need) -> write to per-query scratch memory -> top-N scratch entries re-probe the corpus -> merge.
- `reflmem_2round`: Two full rounds. Round 1 cue-gen -> retrieve -> reflect -> write scratch. Round 2 cue-gen is informed by scratch; its cues + scratch re-probes augment the corpus hits.

## Recall matrix

| Variant | R@20 | R@50 | d R@20 vs v2f_sf | d R@50 vs HyDE+sf | avg_rounds | avg_scratch | time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reflmem_1round` | 0.7583 | 0.8500 | -0.0584 | -0.0917 | 1.00 | 6.07 | 724.2 |
| `reflmem_2round` | 0.7583 | 0.8500 | -0.0584 | -0.0917 | 2.00 | 6.07 | 533.5 |

Baselines (for reference):

- `em_v2f_speakerformat`: R@20=0.8167, R@50=0.8917
- `em_hyde_first_person+speaker_filter`: R@20=0.8500, R@50=0.9417
- `em_hypothesis_driven+speaker_filter`: R@20=0.8080, R@50=0.9330
- `em_two_speaker_query_only`: R@20=0.8000, R@50=0.9333

## Composition with `property_filter(context.source)`

| Variant + speaker_filter | R@20 | R@50 | d R@50 vs HyDE+sf |
| --- | --- | --- | --- |
| `reflmem_2round_filter` | 0.7917 | 0.9417 | +0.0000 |

Current ceiling: `em_hyde_first_person+speaker_filter` R@20=0.8500, R@50=0.9417.

## Round 2 gold novelty

- Queries where round 2 added NEW gold over round-1-only: **1/30** (frac=0.033)

Queries with novel gold from round 2:

- (locomo_conv-26) 'Where did Caroline move from 4 years ago?' +1 new gold

## Sample round-trips (reflmem_2round)

### Q0 (`locomo_conv-26`, locomo_temporal): 'When did Caroline go to the LGBTQ support group?'

Gold turn_ids: [2]

Round-1 cues: ['Caroline: I went to a LGBTQ support group yesterday and it was so powerful.', 'Caroline: Last Friday I went to an LGBTQ+ counseling workshop and it was really enlightening.']

Scratch state after round 1:
  - [learned] Turn 2: user said, 'I went to a LGBTQ support group yesterday,' reporting attendance yesterday at the LGBTQ support group.
  - [learned] Turn 6: 'The support group has made me feel accepted and given me courage,' showing emotional impact but no date.
  - [learned] Turn 70 fragment: 'Last Friday, I went to an LGBT...' indicates a 'Last Friday' time marker for an LGBT event, text truncated.
  - [still_need] Confirm whether the 'I' speaker in Turn 2 is Caroline, i.e., that Caroline said she went yesterday.
  - [still_need] Clarify whether 'yesterday' in Turn 2 refers specifically to the LGBTQ support group versus other LGBTQ events mentioned.
  - [still_need] Resolve conflicting time markers like 'yesterday', 'Last Friday', 'last week', and 'two days ago' to identify exact date.

Round-2 cues: ['Caroline: I attended the queer peer-support meeting last Friday.', 'Melanie: You mentioned going to the community support gathering two days ago.']
Scratch reprobe texts (top-3 by cosine): ["Turn 2: user said, 'I went to a LGBTQ support group yesterday,' reporting attendance yesterday at the LGBTQ support group.", "Clarify whether 'yesterday' in Turn 2 refers specifically to the LGBTQ support group versus other LGBTQ events mentioned.", "Turn 6: 'The support group has made me feel accepted and given me courage,' showing emotional impact but no date."]
Scratch reprobe cosine scores: [0.5934, 0.5227, 0.4864]
R@20=1.00, R@50=1.00, novel_turn_ids_round2=58

### Q15 (`locomo_conv-26`, locomo_single_hop): 'What activities does Melanie partake in?'

Gold turn_ids: [11, 17, 79, 174]

Round-1 cues: ['Melanie: I ran a charity race for mental health last Saturday – it was really rewarding.', "Melanie: I'm carving out some me-time each day - running, reading, or playing my violin"]

Scratch state after round 1:
  - [learned] Melanie ran a charity race for mental health last Saturday and has been running longer and farther to de-stress.
  - [learned] She carves out me-time each day with running, reading, or playing her violin to refresh and stay present for family.
  - [learned] She signed up for a pottery class yesterday, calling it like therapy and a way to express herself creatively.
  - [learned] Melanie plays clarinet; she started when she was young and finds it relaxing and a way to express herself.
  - [still_need] Confirm whether Melanie currently plays both violin and clarinet, or if one reference referred to a past activity.
  - [still_need] Clarify whether the pottery class she signed up for yesterday is an ongoing class, a short series, or a one-time workshop.
  - [still_need] Specify running details: is running a daily me-time habit in addition to the charity race, and how often/distance has she increased for de-stressing?

Round-2 cues: ['Caroline: Record if Melanie currently practices both a string instrument and a woodwind instrument, or if one belonged to an earlier period of her life.', 'Caroline: Specify the pottery enrollment format and length — ongoing multi-session course, brief series, or one-off workshop — and note whether running functions as a daily self-care ritual, including typical cadence and how much farther she’s been covering since the charity event.']
Scratch reprobe texts (top-3 by cosine): ['Melanie plays clarinet; she started when she was young and finds it relaxing and a way to express herself.', 'Confirm whether Melanie currently plays both violin and clarinet, or if one reference referred to a past activity.', 'Melanie ran a charity race for mental health last Saturday and has been running longer and farther to de-stress.']
Scratch reprobe cosine scores: [0.6147, 0.5833, 0.5492]
R@20=0.25, R@50=0.50, novel_turn_ids_round2=104

## Verdict

- **Ties HyDE+speaker_filter**: `reflmem_2round_filter` R@50=0.9417 = 0.9417. Substrate-ceiling hypothesis confirmed: iteration cannot lift beyond what the corpus embedding geometry allows.
- Round-2 novel-gold rate = 0.033 (<10%): reflection-as-retrieval-entry adds little incremental value at the query level.

## Outputs

- `results/reflective_memory.json`
- `results/reflective_memory.md`
- Source: `reflective_memory.py`, `reflmem_eval.py`
- Caches: `cache/reflmem_{cuegen_r1,reflect,cuegen_r2}_cache.json`
