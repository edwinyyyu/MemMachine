# EventMemory Deeper Cue-Gen Tuning (LoCoMo-30)

## Setup

- n_questions = 30 (benchmark=locomo, first 30)
- EventMemory backend, `text-embedding-3-small`, `gpt-5-mini`
- Speaker-baked embedded format: `"{source}: {text}"`
- Retrieval at max_K=50, sliced to K=20 / K=50 (fair backfill)
- Caches: `cache/emtune_<variant>_cache.json` (dedicated)

## Recall table

| Variant | R@20 | R@50 | d vs SF R@20 | d vs SF R@50 | time (s) |
| --- | --- | --- | --- | --- | --- |
| `em_v2f_speakerformat` | 0.8167 | 0.8917 | +0.0000 | +0.0000 | 37.3 |
| `v2f_speakerformat_short` | 0.7500 | 0.8667 | -0.0667 | -0.0250 | 432.5 |
| `v2f_speakerformat_5cues` | 0.7833 | 0.8167 | -0.0334 | -0.0750 | 578.0 |
| `v2f_speakerformat_natural_turn` | 0.8056 | 0.8667 | -0.0111 | -0.0250 | 238.6 |
| `chain_with_scratchpad_speakerformat` | 0.7556 | 0.8500 | -0.0611 | -0.0417 | 298.0 |
| `type_enumerated_em_retuned` | 0.8000 | 0.8833 | -0.0167 | -0.0084 | 389.1 |
| `alias_expand_speakerformat` | 0.8333 | 0.9000 | +0.0166 | +0.0083 | 512.9 |
| `two_speaker_filter_sf_cues` | 0.8333 | 0.9250 | +0.0166 | +0.0333 | 40.8 |
| `em_two_speaker_query_only` | 0.8000 | 0.9333 | -0.0167 | +0.0416 | 14.6 |

Anchors from prior runs:
- em_v2f_speakerformat prior: {'mean_r@20': 0.817, 'mean_r@50': 0.892}
- em_two_speaker_filter (v2f cues) prior: {'mean_r@20': 0.842, 'mean_r@50': 0.9}
- em_two_speaker_query_only prior: {'mean_r@20': 0.8, 'mean_r@50': 0.933}
- em_alias_expand_v2f prior: {'mean_r@20': 0.825, 'mean_r@50': 0.883}

## W/T/L per variant vs em_v2f_speakerformat

| Variant | K=20 W/T/L | K=50 W/T/L |
| --- | --- | --- |
| `v2f_speakerformat_short` | 1/25/4 | 1/26/3 |
| `v2f_speakerformat_5cues` | 0/29/1 | 0/27/3 |
| `v2f_speakerformat_natural_turn` | 1/27/2 | 1/27/2 |
| `chain_with_scratchpad_speakerformat` | 0/27/3 | 0/27/3 |
| `type_enumerated_em_retuned` | 1/27/2 | 1/27/2 |
| `alias_expand_speakerformat` | 1/29/0 | 1/28/1 |
| `two_speaker_filter_sf_cues` | 1/29/0 | 2/28/0 |
| `em_two_speaker_query_only` | 2/26/2 | 3/26/1 |

## Top-2 cue variants by R@20 lift over SF baseline

- `v2f_speakerformat_natural_turn` : R@20=0.8056 (d=-0.0111), R@50=0.8667
- `type_enumerated_em_retuned` : R@20=0.8000 (d=-0.0167), R@50=0.8833

## Sample cues: `em_v2f_speakerformat` (old) vs `v2f_speakerformat_natural_turn` (best new)

### Q0: `locomo_conv-26` -- 'When did Caroline go to the LGBTQ support group?'
- OLD `em_v2f_speakerformat` (R@20=1.00, R@50=1.00):
  - `Caroline: I went to an LGBTQ support group yesterday and it was so powerful.`
  - `Caroline: The support group I went to yesterday really made me feel accepted and gave me courage.`
- NEW `v2f_speakerformat_natural_turn` (R@20=1.00, R@50=1.00):
  - `Caroline: I went to an LGBTQ support group yesterday and it was so powerful.`
  - `Caroline: I keep thinking about the LGBTQ support group I went to yesterday — everyone was so supportive and I felt really seen.`

### Q15: `locomo_conv-26` -- 'What activities does Melanie partake in?'
- OLD `em_v2f_speakerformat` (R@20=0.50, R@50=0.75):
  - `Melanie: I ran a charity race for mental health last Saturday and it felt really rewarding, like taking care of our minds.`
  - `Melanie: I'm carving out me-time these days — running, reading, playing my violin, signed up for a pottery class, and we went camping and hiking with the kids (also been biking).`
- NEW `v2f_speakerformat_natural_turn` (R@20=0.50, R@50=0.50):
  - `Melanie: I ran a charity race for mental health last Saturday — it was really rewarding, and I carve out me-time each day for running, reading, or practicing my violin.`
  - `Melanie: I just signed up for a pottery class, and a few weeks ago we went camping with the kids — we hiked, biked, and explored the forest together.`

### Q29: `locomo_conv-26` -- 'When did Melanie go to the pottery workshop?'
- OLD `em_v2f_speakerformat` (R@20=1.00, R@50=1.00):
  - `Melanie: Last Fri I finally took my kids to a pottery workshop.`
  - `Melanie: I finally took my kids to a pottery workshop last Friday; we all made our own pots.`
- NEW `v2f_speakerformat_natural_turn` (R@20=1.00, R@50=1.00):
  - `Melanie: Last Friday I finally took my kids to a pottery workshop and we all made our own little pots.`
  - `Caroline: Last Friday — that's great to hear, I'm so glad you and the kids had a fun, creative day.`

## Verdict per variant (vs em_v2f_speakerformat)

| Variant | R@20 d | R@50 d | Decision |
| --- | --- | --- | --- |
| `em_v2f_speakerformat` | +0.0000 | +0.0000 | baseline |
| `v2f_speakerformat_short` | -0.0667 | -0.0250 | abandon |
| `v2f_speakerformat_5cues` | -0.0334 | -0.0750 | abandon |
| `v2f_speakerformat_natural_turn` | -0.0111 | -0.0250 | abandon |
| `chain_with_scratchpad_speakerformat` | -0.0611 | -0.0417 | abandon |
| `type_enumerated_em_retuned` | -0.0167 | -0.0084 | abandon |
| `alias_expand_speakerformat` | +0.0166 | +0.0083 | narrow: K=20 only |
| `two_speaker_filter_sf_cues` | +0.0166 | +0.0333 | SHIP (both K) |
| `em_two_speaker_query_only` | -0.0167 | +0.0416 | narrow: K=50 only |

## Composition / regime ceilings

- `two_speaker_filter_sf_cues` K=20 = 0.8333 (prior two_speaker_filter K=20 ceiling 0.842, d=-0.0087)
- `em_two_speaker_query_only` K=50 = 0.9333 (prior K=50 ceiling 0.933, d=+0.0003)

## Updated production recipe (per K regime)

- K=20 winner: `alias_expand_speakerformat` @ 0.8333
- K=50 winner: `em_two_speaker_query_only` @ 0.9333

**Ship cue-gen default**: `two_speaker_filter_sf_cues` beats SF at BOTH K budgets by >=1pp.

## Outputs

- `results/em_deeper_tune.json`
- `results/em_deeper_tune.md`
- Source: `em_deeper_tune.py`, `em_deeper_eval.py`
- Caches: `cache/emtune_<variant>_cache.json`
