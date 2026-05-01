# EventMemory Cue-Gen Task-Shape Robustness (LoCoMo-30)

## Setup

- n = 30 LoCoMo questions per shape (120 total queries: 30x4 shapes)
- Shapes: ORIGINAL, CMD ("Find ..."), DRAFT ("Summarize/Draft ..."),
  META ("What do we know about ..." / "Tell me about ...")
- Backend: EventMemory (Qdrant + SQLite), speaker-baked embeddings
- Embedder: `text-embedding-3-small`, cue LLM: `gpt-5-mini`
- Dedicated caches: `cache/emts_<arch>_cache.json`

## Recall matrix (mean R@K across 30 questions per shape)

### R@20

| Architecture | ORIGINAL | CMD | DRAFT | META |
| --- | --- | --- | --- | --- |
| `em_cosine_baseline` | 0.7333 | 0.7667 | 0.7667 | 0.7667 |
| `em_v2f` | 0.7417 | 0.7722 | 0.7417 | 0.7750 |
| `em_v2f_speakerformat` | 0.8167 | 0.8333 | 0.7750 | 0.7417 |
| `em_hyde_first_person` | 0.8000 | 0.7833 | 0.7556 | 0.7306 |

### R@50

| Architecture | ORIGINAL | CMD | DRAFT | META |
| --- | --- | --- | --- | --- |
| `em_cosine_baseline` | 0.8833 | 0.8333 | 0.8500 | 0.8667 |
| `em_v2f` | 0.8833 | 0.8500 | 0.8417 | 0.8667 |
| `em_v2f_speakerformat` | 0.8917 | 0.8333 | 0.8917 | 0.8583 |
| `em_hyde_first_person` | 0.9083 | 0.8500 | 0.8833 | 0.8333 |

## Cue-gen lift vs em_cosine_baseline (same shape)

Positive = cue gen helps that shape; negative = cue gen hurts.

### R@20

| Architecture | ORIGINAL | CMD | DRAFT | META |
| --- | --- | --- | --- | --- |
| `em_v2f` | +0.0084 | +0.0055 | -0.0250 | +0.0083 |
| `em_v2f_speakerformat` | +0.0834 | +0.0666 | +0.0083 | -0.0250 |
| `em_hyde_first_person` | +0.0667 | +0.0166 | -0.0111 | -0.0361 |

### R@50

| Architecture | ORIGINAL | CMD | DRAFT | META |
| --- | --- | --- | --- | --- |
| `em_v2f` | +0.0000 | +0.0167 | -0.0083 | +0.0000 |
| `em_v2f_speakerformat` | +0.0084 | +0.0000 | +0.0417 | -0.0084 |
| `em_hyde_first_person` | +0.0250 | +0.0167 | +0.0333 | -0.0334 |

## Shape-sensitivity per architecture

Drop from ORIGINAL shape to worst-shape (larger = more shape-sensitive).

| Architecture | R@20 drop | Worst shape@20 | R@50 drop | Worst shape@50 |
| --- | --- | --- | --- | --- |
| `em_cosine_baseline` | -0.0334 | CMD | +0.0500 | CMD |
| `em_v2f` | +0.0000 | DRAFT | +0.0416 | DRAFT |
| `em_v2f_speakerformat` | +0.0750 | META | +0.0584 | CMD |
| `em_hyde_first_person` | +0.0694 | META | +0.0750 | META |

## Per-shape ceiling (best architecture per shape)

| Shape | Best @R@20 | Best @R@50 |
| --- | --- | --- |
| ORIGINAL | `em_v2f_speakerformat` (0.8167) | `em_hyde_first_person` (0.9083) |
| CMD | `em_v2f_speakerformat` (0.8333) | `em_v2f` (0.8500) |
| DRAFT | `em_v2f_speakerformat` (0.7750) | `em_v2f_speakerformat` (0.8917) |
| META | `em_v2f` (0.7750) | `em_cosine_baseline` (0.8667) |

## Verdict

- em_v2f_speakerformat ORIGINAL K=50 lift vs cosine: +0.0084
- em_v2f_speakerformat K=50 lifts on task shapes: CMD +0.0000, DRAFT +0.0417, META -0.0084 (max +0.0417)
- Active-harm cells (drop >= 1pp vs cosine): 5
  - `em_v2f` on DRAFT at K=20: -0.0250
  - `em_v2f_speakerformat` on META at K=20: -0.0250
  - `em_hyde_first_person` on DRAFT at K=20: -0.0111
  - `em_hyde_first_person` on META at K=20: -0.0361
  - `em_hyde_first_person` on META at K=50: -0.0334

**Verdict: inconclusive / mixed.** See table above for per-shape behavior.

## HyDE first-person collapse check

Hypothesis: em_hyde_first_person's "I remember ..." framing hurts DRAFT/META more than v2f's flexible chat framing.

- `em_hyde_first_person` CMD at K=20: drop from ORIGINAL = +0.0167
- `em_hyde_first_person` CMD at K=50: drop from ORIGINAL = +0.0583
- `em_hyde_first_person` DRAFT at K=20: drop from ORIGINAL = +0.0444
- `em_hyde_first_person` DRAFT at K=50: drop from ORIGINAL = +0.0250
- `em_hyde_first_person` META at K=20: drop from ORIGINAL = +0.0694
- `em_hyde_first_person` META at K=50: drop from ORIGINAL = +0.0750

Worst task-shape R@50: v2f_sf=0.8333 vs hyde_fp=0.8333

## Sample cues (3 questions x 4 shapes)

### orig_row_index=0 (locomo_conv-26, locomo_temporal)

- **ORIGINAL**: 'When did Caroline go to the LGBTQ support group?'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Caroline: I went to an LGBTQ support group yesterday and it was so powerful.`
    - `Caroline: The support group I went to yesterday really made me feel accepted and gave me courage.`
- **CMD**: 'Find when Caroline went to the LGBTQ support group.'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Caroline: I went to an LGBTQ support group yesterday and it was so powerful.`
    - `Caroline: Yesterday I went to an LGBTQ support group; it really made me feel accepted.`
- **DRAFT**: 'Summarize when Caroline went to the LGBTQ support group.'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Caroline: I went to a LGBTQ support group yesterday and it was so powerful.`
    - `Caroline: Last Friday I went to an LGBTQ+ counseling workshop, and recently I also attended an LGBTQ support group that felt really supportive.`
- **META**: 'What do we know about when Caroline went to the LGBTQ support group?'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Caroline: I went to a LGBTQ support group yesterday and it was so powerful.`
    - `Caroline: The support group has made me feel accepted and given me courage to embrace myself.`

### orig_row_index=15 (locomo_conv-26, locomo_single_hop)

- **ORIGINAL**: 'What activities does Melanie partake in?'
  - `em_v2f_speakerformat` cues (R@20=0.50, R@50=0.75):
    - `Melanie: I ran a charity race for mental health last Saturday and it felt really rewarding, like taking care of our minds.`
    - `Melanie: I'm carving out me-time these days — running, reading, playing my violin, signed up for a pottery class, and we went camping and hiking with the kids (also been biking).`
- **CMD**: 'List the activities Melanie partakes in.'
  - `em_v2f_speakerformat` cues (R@20=0.50, R@50=0.50):
    - `Melanie: I ran a charity race for mental health last Saturday, and lately I've been carving out me-time—running, reading, and playing my violin.`
    - `Melanie: I just signed up for a pottery class, and we went camping with the kids a few weeks ago where we hiked, biked, played games, ate good food, and unplugged together.`
- **DRAFT**: 'Summarize the activities Melanie partakes in.'
  - `em_v2f_speakerformat` cues (R@20=0.25, R@50=0.25):
    - `Melanie: I ran a charity race for mental health last Saturday; it was really rewarding and made me think about taking care of our minds.`
    - `Melanie: I'm carving out me-time each day — running, reading, playing my violin — and I just signed up for a pottery class yesterday.`
- **META**: 'Tell me about the activities Melanie partakes in.'
  - `em_v2f_speakerformat` cues (R@20=0.25, R@50=0.25):
    - `Melanie: I ran a charity race for mental health last Saturday — it was really rewarding and made me think more about taking care of our minds.`
    - `Melanie: I just signed up for a pottery class yesterday; it's like therapy for me, letting me express myself and get creative.`

### orig_row_index=29 (locomo_conv-26, locomo_temporal)

- **ORIGINAL**: 'When did Melanie go to the pottery workshop?'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Melanie: Last Fri I finally took my kids to a pottery workshop.`
    - `Melanie: I finally took my kids to a pottery workshop last Friday; we all made our own pots.`
- **CMD**: 'Find when Melanie went to the pottery workshop.'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Melanie: Last Fri I finally took my kids to a pottery workshop; we all made our own pots, it was fun and therapeutic!`
    - `Melanie: I made it in pottery class yesterday. I love it! Pottery's so relaxing and creative.`
- **DRAFT**: 'Summarize when Melanie went to the pottery workshop.'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Melanie: Last Fri I finally took my kids to a pottery workshop. We all made our own pots, it was fun and therapeutic!`
    - `Melanie: Yeah, I made it in pottery class yesterday. Pottery's so relaxing and creative.`
- **META**: 'What do we know about when Melanie went to the pottery workshop?'
  - `em_v2f_speakerformat` cues (R@20=1.00, R@50=1.00):
    - `Melanie: Last Fri I finally took my kids to a pottery workshop.`
    - `Melanie: Yeah, I made it in pottery class yesterday.`


## Outputs

- `results/em_task_shape.json`
- `results/em_task_shape.md`
- Source: `em_task_shape_eval.py`
- Caches: `cache/emts_<arch>_cache.json` (dedicated)
