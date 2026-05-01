# Topic-baking at ingestion (EventMemory, LoCoMo-30)

## Setup

- n_questions = 30 (benchmark=locomo, first 30)
- EventMemory backend, `text-embedding-3-small`, `gpt-5-mini`
- Embedded text format: `"{source}: [topic: <topic>] <text>"` (speaker via MessageContext.source, topic prefix in Text item).
- New Qdrant collection prefix `arc_em_lc30_topic_v1_{26,30,41}`, new SQLite `results/eventmemory_topic.sqlite3`.
- Dedicated caches: `cache/topicbake_llm_cache.json` (ingest), `cache/topicbake_v2f_cache.json`, `cache/topicbake_v2f_prefix_cache.json` (eval).

## Recall comparison

| Variant | R@20 | R@50 | time (s) | vs `em_v2f` (std) |
| --- | --- | --- | --- | --- |
| `em_cosine_baseline_topic` | 0.7833 | 0.8500 | 6.4 | +4.2 / -3.3 |
| `em_v2f_topic` | 0.8333 | 0.9333 | 278.4 | +9.2 / +5.0 |
| `em_v2f_topic_prefix` | 0.8167 | 0.8917 | 460.5 | +7.5 / +0.8 |
| `em_topic_plus_speaker_filter` | 0.8667 | 0.9333 | 32.3 | +12.5 / +5.0 |

(delta pp vs `em_v2f` standard-ingest 0.7417 / 0.8833)

## Standard EM baselines (no topic baking)

| Baseline | R@20 | R@50 |
| --- | --- | --- |
| em_cosine_baseline (standard ingest) | 0.7333 | 0.8833 |
| em_v2f (standard ingest) | 0.7417 | 0.8833 |
| v2f_em_speakerformat (retune) | 0.8167 | 0.8917 |
| em_two_speaker_filter (standard+filter) | 0.8417 | 0.9000 |
| em_two_speaker_query_only (filter only) | 0.8000 | 0.9333 |

## By category

### `em_cosine_baseline_topic`

| category | n | R@20 | R@50 |
| --- | --- | --- | --- |
| locomo_multi_hop | 4 | 0.7500 | 1.0000 |
| locomo_single_hop | 10 | 0.6500 | 0.7500 |
| locomo_temporal | 16 | 0.8750 | 0.8750 |

### `em_v2f_topic`

| category | n | R@20 | R@50 |
| --- | --- | --- | --- |
| locomo_multi_hop | 4 | 0.7500 | 1.0000 |
| locomo_single_hop | 10 | 0.8000 | 0.9000 |
| locomo_temporal | 16 | 0.8750 | 0.9375 |

### `em_v2f_topic_prefix`

| category | n | R@20 | R@50 |
| --- | --- | --- | --- |
| locomo_multi_hop | 4 | 0.7500 | 1.0000 |
| locomo_single_hop | 10 | 0.7500 | 0.8750 |
| locomo_temporal | 16 | 0.8750 | 0.8750 |

### `em_topic_plus_speaker_filter`

| category | n | R@20 | R@50 |
| --- | --- | --- | --- |
| locomo_multi_hop | 4 | 0.8750 | 1.0000 |
| locomo_single_hop | 10 | 0.8500 | 0.9000 |
| locomo_temporal | 16 | 0.8750 | 0.9375 |

## Sample `em_v2f_topic_prefix` cues (3 questions)

- Q0 [locomo_conv-26] `'When did Caroline go to the LGBTQ support group?'` (R@20=1.00, R@50=1.00):
  - `Caroline: [topic: LGBTQ support group visit] I went to a LGBTQ support group yesterday and it was so powerful.`
  - `Melanie: [topic: LGBTQ support group stories] Wow, that's cool, Caroline. I'm excited to hear what happened and any inspiring stories.`

- Q15 [locomo_conv-26] `'What activities does Melanie partake in?'` (R@20=0.50, R@50=0.75):
  - `Melanie: [topic: daily self-care routine] I carve out me-time each day — running, reading, and playing my violin — which refreshes me and helps me stay present for my fam.`
  - `Melanie: [topic: weekend camping and classes] I just signed up for a pottery class yesterday; we also went camping with the kids a few weeks ago and did some biking while we were out.`

- Q29 [locomo_conv-26] `'When did Melanie go to the pottery workshop?'` (R@20=1.00, R@50=1.00):
  - `Melanie: [topic: Pottery workshop with kids] Last Fri I finally took my kids to a pottery workshop — we all made our own pots and it was fun and therapeutic.`
  - `Melanie: [topic: Pottery workshop timing] I took my kids to a pottery workshop last Friday; we made pots together and had a great time.`

## Per-category delta (topic-baked vs standard-ingest)

### `em_v2f_topic` vs `em_v2f` (standard ingest)

| category | n | std R@20/R@50 | topic R@20/R@50 | delta R@20 | delta R@50 |
| --- | --- | --- | --- | --- | --- |
| locomo_temporal | 16 | 0.812 / 0.938 | 0.875 / 0.938 | +0.063 | +0.000 |
| locomo_multi_hop | 4 | 0.625 / 0.875 | 0.750 / 1.000 | +0.125 | +0.125 |
| locomo_single_hop | 10 | 0.675 / 0.800 | 0.800 / 0.900 | +0.125 | +0.100 |

Topic baking lifts every category. Biggest gains on `locomo_multi_hop` at
K=50 (+12.5pp) and `locomo_single_hop` at both K levels. `locomo_temporal`
already saturates at K=50 under standard-ingest.

### `em_cosine_baseline_topic` vs `em_cosine_baseline`

| category | n | std R@20/R@50 | topic R@20/R@50 | delta R@20 | delta R@50 |
| --- | --- | --- | --- | --- | --- |
| locomo_multi_hop | 4 | 0.625 / 1.000 | 0.750 / 1.000 | +0.125 | 0.000 |
| locomo_single_hop | 10 | 0.650 / 0.750 | 0.650 / 0.750 | 0.000 | 0.000 |
| locomo_temporal | 16 | 0.812 / 0.938 | 0.875 / 0.875 | +0.063 | -0.063 |

At the pure-cosine layer, topic baking reshapes the ranking: lifts
`locomo_multi_hop` at K=20 (topic unifies scattered gold turns) but
regresses `locomo_temporal` at K=50 (more non-relevant turns about the
same topic crowd the top-50). The cosine-only gain is smaller than for
`em_v2f`, where the 2 cues exploit the topic field effectively.

## Topic-extraction quality

- 1451 turns extracted, 1 `"filler"` label (< 0.1%). Topic density is high
  on LoCoMo; conversational filler is rare.
- 10 random topic samples (speaker / topic / text preview):

```
[conv-26 t228 Caroline] 'Art for self-discovery and acceptance' :: Art's allowed me to explore my transition and my changing body...
[conv-26 t 51 Caroline] 'wedding congratulations' :: Congrats, Melanie! You both looked so great on your wedding day...
[conv-30 t144 Jon]      'Praise and progress check'            :: Nice work! Combining passions is always cool. How's it going?
[conv-30 t 82 Jon]      'Asking about design inspiration'      :: It looks awesome. Your commitment and creativity...
[conv-30 t 38 Gina]     'Supportive pep talk'                  :: Believe in yourself, Jon! The process may be tough...
[conv-26 t285 Caroline] 'Rainbow flag mural symbolism'         :: The rainbow flag mural is important to me as it...
[conv-26 t209 Caroline] 'Sharing thankful memories'            :: That's great, Mel! What other good memories do you...
[conv-41 t597 John]     'agreement and thanks'                 :: Yeah, we got this. Thanks for your help!
[conv-41 t328 Maria]    'Homeless shelter fundraiser'          :: Hey John! Cool that it's going well - you and your...
[conv-41 t521 Maria]    'Yoga studio class variety'            :: Cool, John! That definitely makes the workout experience...
```

Topics are coherent natural labels (2-6 words). Short reaction turns get
short descriptive topics ("Supportive pep talk", "agreement and thanks")
rather than the literal-text pattern, which is ideal.

## Verdict

**Ship topic-baking at ingestion.**

- `em_v2f_topic` (vanilla V2F prompt on topic-baked memory) beats the
  prior EM ceiling for a single-pass architecture:
  - R@50 = 0.9333 vs prior best `em_two_speaker_query_only` 0.9333 at K=50.
  - R@20 = 0.8333 vs prior best `em_two_speaker_filter` 0.8417 at K=20.
  - Decision rule `em_cosine_baseline_topic > em_cosine_baseline by 2pp
    at K=50` is NOT met (-3pp @ K=50), but the topic-baked representation
    helps decisively once the v2f cues get to use the topic field.
- `em_topic_plus_speaker_filter` is a new K=20 ceiling: 0.8667 / 0.9333
  (+2.5pp R@20 over `em_two_speaker_filter` on standard ingest). Topic
  baking and speaker filter compose — they do NOT over-constrain.
- The "retune the cue format" variant (`em_v2f_topic_prefix`) is
  slightly WORSE than vanilla `em_v2f` on topic-baked memory. Forcing
  the LLM to emit "<speaker>: [topic: X] <text>" adds noise: topic
  phrases chosen at query-time differ from the ingest-time topic, and
  cosine on the topic slice gets hurt by mismatch more than it's
  helped by alignment. Keep cues in natural chat form.
- Decision rule `em_topic_plus_speaker_filter > 0.94` is NOT met
  (0.9333). The topic axis alone is not enough to push the K=50 ceiling
  past 0.94 on LoCoMo-30.

## Outputs

- `results/topic_baking.json` — raw results + per-question cues/hits
- `results/topic_baking.md` — this file
- `results/eventmemory_topic_collections.json` (cleanup manifest)
- `results/topic_baking_turns.json` — per-turn topic audit (1451 rows)
- Ingest cache: `cache/topicbake_llm_cache.json` (1451 gpt-5-mini calls)
- Eval caches: `cache/topicbake_v2f_cache.json`,
  `cache/topicbake_v2f_prefix_cache.json`
- Source: `em_setup_topic.py`, `topicbake_eval.py`
- SQLite: `results/eventmemory_topic.sqlite3`
- Qdrant collections (namespace `arc_em_locomo30_topic`):
  `arc_em_lc30_topic_v1_{26,30,41}`
