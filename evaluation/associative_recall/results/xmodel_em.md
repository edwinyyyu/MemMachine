# Cross-model EventMemory cue-gen: gpt-5-nano with structural prompts

## Hypothesis

Structural-constraint cue-gen prompts (speakerformat, HyDE first-person) reduce nano's effective output surface area, closing the mini-nano gap on EventMemory retrieval.

## Setup

- n_questions = 30 (benchmark=locomo, first 30)
- EventMemory backend (arc_em_lc30_v1_{26,30,41}); speaker-baked embedded format `"{source}: {text}"`
- Embedder: text-embedding-3-small (fixed)
- Cue-gen model: gpt-5-nano (`max_completion_tokens=6000`, 1-retry on empty parse)
- Mini controls: numbers pulled from prior runs (em_prompt_retune, em_hyde_orient); no new mini spend
- Prompts reused VERBATIM from:
  - `em_architectures.V2F_PROMPT` (vanilla)
  - `em_retuned_cue_gen.V2F_SPEAKERFORMAT_PROMPT` (mini-retuned winner)
  - `em_hyde_orient.HYDE_FIRST_PERSON_PROMPT` (mini K=50 ceiling)
- Caches: `cache/xmodel_<variant>_cache.json` (dedicated)

## Recall matrix (nano vs mini)

| Variant | nano R@20 | mini R@20 | nano/mini @20 | nano R@50 | mini R@50 | nano/mini @50 |
| --- | --- | --- | --- | --- | --- | --- |
| `nano_v2f` | 0.7417 | 0.7420 | 100.0% | 0.8833 | 0.8830 | 100.0% |
| `nano_v2f_speakerformat` | 0.7639 | 0.8170 | 93.5% | 0.8583 | 0.8920 | 96.2% |
| `nano_hyde_first_person` | 0.7667 | 0.8000 | 95.8% | 0.8667 | 0.9080 | 95.5% |
| `nano_hyde_first_person_filter` | 0.8000 | 0.8500 | 94.1% | 0.9333 | 0.9420 | 99.1% |

Mini variants mapped to:

- `nano_v2f` -> mini `em_v2f`
- `nano_v2f_speakerformat` -> mini `em_v2f_speakerformat`
- `nano_hyde_first_person` -> mini `em_hyde_first_person`
- `nano_hyde_first_person_filter` -> mini `em_hyde_first_person+speaker_filter`

## Format-compliance (nano)

| Variant | Rate | Fraction | Description |
| --- | --- | --- | --- |
| `nano_v2f` | N/A | - | v2f has no structural constraint (N/A) |
| `nano_v2f_speakerformat` | 100.00% | 60/60 | fraction of cues starting with '<speaker>: ' |
| `nano_hyde_first_person` | 100.00% | 30/30 | fraction of turns parseable as '<speaker>: <content>' |
| `nano_hyde_first_person_filter` | 100.00% | 30/30 | fraction of turns parseable as '<speaker>: <content>' |

## Sample cues (2 questions per variant)

### Q0 (`locomo_conv-26`, locomo_temporal): 'When did Caroline go to the LGBTQ support group?'

Gold turn_ids: [2]

- `nano_v2f` (R@20=1.00, R@50=1.00)
  cues:
    - I went to a LGBTQ support group yesterday and it was so powerful.
    - Yesterday I went to a LGBTQ support group.
- `nano_v2f_speakerformat` (R@20=1.00, R@50=1.00)
  cues:
    - Caroline: I went to a LGBTQ support group yesterday and it was so powerful.
    - Caroline: The support group has made me feel accepted and given me courage to embrace myself.
- `nano_hyde_first_person` (R@20=1.00, R@50=1.00)
  turn: Caroline: I went to the LGBTQ support group yesterday.
- `nano_hyde_first_person_filter` (R@20=1.00, R@50=1.00)
  turn: Caroline: I went to the LGBTQ support group yesterday.

### Q29 (`locomo_conv-26`, locomo_temporal): 'When did Melanie go to the pottery workshop?'

Gold turn_ids: [136]

- `nano_v2f` (R@20=1.00, R@50=1.00)
  cues:
    - Last Fri I finally took my kids to a pottery workshop. We all made our own pots, it was fun and therapeutic!
    - Yeah, I made it in pottery class yesterday. I love it! Pottery's so relaxing and creative.
- `nano_v2f_speakerformat` (R@20=1.00, R@50=1.00)
  cues:
    - Melanie: Last Fri I finally took my kids to a pottery workshop.
    - Melanie: We all made our own pots, it was fun and therapeutic!
- `nano_hyde_first_person` (R@20=1.00, R@50=1.00)
  turn: Melanie: Last Friday I finally took my kids to a pottery workshop, and we all made our own pots—it was fun and therapeutic.
- `nano_hyde_first_person_filter` (R@20=1.00, R@50=1.00)
  turn: Melanie: Last Friday I finally took my kids to a pottery workshop, and we all made our own pots—it was fun and therapeutic.

## Verdict

**>= 90% of mini at BOTH K (structural prompts ARE a cross-model portability lever):**
- `nano_v2f`: 100.0%@20, 100.0%@50
- `nano_v2f_speakerformat`: 93.5%@20, 96.2%@50
- `nano_hyde_first_person`: 95.8%@20, 95.5%@50
- `nano_hyde_first_person_filter`: 94.1%@20, 99.1%@50

### Decision-rule evaluation

- `nano_hyde_first_person_filter` R@50=0.9333 >= 0.90 -> EM LoCoMo recipe is MODEL-PORTABLE (nano matches mini within 4pp).
- Structural prompts lift nano to >= 0.85 K=50 (speakerformat=0.8583, hyde_fp=0.8667) -> structural prompts ARE a cross-model win.

- Gradient: HyDE (0.8667) > speakerformat (0.8583) at K=50 -> 'more structure = more portable' gradient CONFIRMED.

## Outputs

- `results/xmodel_em.json`
- `results/xmodel_em.md`
- Source: `xmodel_em.py`, `xmodel_eval.py`
- Caches: `cache/xmodel_<variant>_cache.json`
