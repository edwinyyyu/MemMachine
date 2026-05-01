# Contrastive Retrieval Probe

Score = cos(answer_probe, turn) - α · cos(distractor_probe, turn). Turns matching BOTH probes (paraphrase-style matches) get penalized; turns matching only the answer-probe get surfaced.

## Setup
- `text-embedding-3-small`, `gpt-5-mini` (fixed).
- Fair-backfill eval at K=20 and K=50.
- Datasets: locomo_30q, synthetic_19q.
- Variants: `contrast_only_a{0.2,0.5,1.0}` (raw query answer-probe, re-rank whole conversation) and `contrast_v2f_a{0.2,0.5,1.0}` (v2f retrieval + contrastive re-rank of candidate pool).

## Recall table (arch_r@K, fair-backfill)
| Dataset | K | v2f | cosine_baseline | contrast_only_a0.2 | contrast_only_a0.5 | contrast_only_a1 | contrast_v2f_a0.2 | contrast_v2f_a0.5 | contrast_v2f_a1 |
|---------|---|------:|------:|------:|------:|------:|------:|------:|------:|
| locomo_30q | 20 | 0.756 | 0.383 | 0.350 | 0.317 | 0.100 | 0.708 | 0.697 | 0.547 |
| locomo_30q | 50 | 0.858 | 0.508 | 0.508 | 0.442 | 0.225 | 0.858 | 0.858 | 0.858 |
| synthetic_19q | 20 | 0.613 | 0.569 | 0.572 | 0.589 | 0.330 | 0.600 | 0.594 | 0.525 |
| synthetic_19q | 50 | 0.851 | 0.824 | 0.824 | 0.838 | 0.659 | 0.851 | 0.851 | 0.851 |

## Deltas vs `v2f` (arch_r@K)
| Dataset | K | cosine_baseline | contrast_only_a0.2 | contrast_only_a0.5 | contrast_only_a1 | contrast_v2f_a0.2 | contrast_v2f_a0.5 | contrast_v2f_a1 |
|---------|---|------:|------:|------:|------:|------:|------:|------:|
| locomo_30q | 20 | -0.372 | -0.406 | -0.439 | -0.656 | -0.047 | -0.058 | -0.208 |
| locomo_30q | 50 | -0.350 | -0.350 | -0.417 | -0.633 | +0.000 | +0.000 | +0.000 |
| synthetic_19q | 20 | -0.044 | -0.041 | -0.024 | -0.283 | -0.013 | -0.019 | -0.088 |
| synthetic_19q | 50 | -0.027 | -0.027 | -0.013 | -0.192 | +0.000 | +0.000 | +0.000 |

## W/T/L vs cosine baseline (delta_r@K)
| Dataset | K | v2f | cosine_baseline | contrast_only_a0.2 | contrast_only_a0.5 | contrast_only_a1 | contrast_v2f_a0.2 | contrast_v2f_a0.5 | contrast_v2f_a1 |
|---------|---|------:|------:|------:|------:|------:|------:|------:|------:|
| locomo_30q | 20 | 13/17/0 | 0/30/0 | 1/26/3 | 2/23/5 | 2/16/12 | 12/18/0 | 11/19/0 | 13/12/5 |
| locomo_30q | 50 | 13/17/0 | 0/30/0 | 0/30/0 | 0/28/2 | 2/17/11 | 13/17/0 | 13/17/0 | 13/17/0 |
| synthetic_19q | 20 | 8/7/4 | 0/19/0 | 2/15/2 | 4/13/2 | 5/3/11 | 5/12/2 | 5/11/3 | 8/5/6 |
| synthetic_19q | 50 | 4/14/1 | 0/19/0 | 0/19/0 | 3/14/2 | 3/3/13 | 4/14/1 | 4/14/1 | 4/14/1 |

## Best α per family (on locomo_30q, K=50)
- `contrast_only` best: `contrast_only_a0.2` at r@50=0.508
- `contrast_v2f` best: `contrast_v2f_a0.2` at r@50=0.858

## Per-category deltas at K=20 (where re-rank matters)
K=50 ties exactly on every category (pool is identical), so K=20 is where
re-ranking can show effects. Categories sorted by delta vs `v2f`; negative = loss.

**`contrast_v2f_a0.2` (best contrastive variant) vs `v2f`:**

- locomo_30q:
    - locomo_multi_hop (n=4): Δ=+0.000
    - locomo_temporal (n=16): Δ=+0.000
    - locomo_single_hop (n=10): Δ=-0.142 (largest loser)
- synthetic_19q:
    - inference (n=3): Δ=+0.065 (only gainer, noisy)
    - control (n=3): Δ=+0.000
    - procedural (n=2): Δ=-0.004
    - completeness (n=4): Δ=-0.035
    - proactive (n=4): Δ=-0.037
    - conjunction (n=3): Δ=-0.048

**Top 2 gaining / losing (any contrastive variant, any dataset, K=20):**
- Gainers: `synthetic:inference` (+0.065 under α=0.2/0.5, n=3),
  `synthetic:completeness` (+0.099 under α=1, n=4; but α=1 loses overall)
- Losers: `locomo:single_hop` (-0.142 under α=0.2),
  `locomo:temporal` (-0.250 under α=1)

## Paraphrase-penalty inspection (samples)
For the first 3 locomo_30q questions: show answer_probe, distractor_probe, and the top-5 ranked turns for v2f vs `contrast_v2f_a0.5`. `cos_a` = cosine with answer-probe (query); `cos_d` = cosine with distractor-probe; `score` = cos_a − α·cos_d.

### Question: _When did Caroline go to the LGBTQ support group?_
- category: `locomo_temporal`
- distractor_probe: _When did Caroline first attend the LGBTQ support group and how often did she go afterward?_
- gold turn_ids: [2]

**v2f top 5 (retrieval order):**
- CUE 0: I went to a LGBTQ support group yesterday and it was so powerful.
- CUE 1: I went to a support group yesterday
- fair-backfill r@20=1.000 r@50=1.000

**contrast_v2f_a0.5 re-rank top 5:**
- [   ] turn 184 cos_a=+0.603 cos_d=+0.564 score=+0.321  assistant: Wow, Caroline! They must have felt so appreciated. It's awesome to see the difference we can make in each other's lives. Any other exciting LGBTQ advocacy stuff
- [   ] turn  77 cos_a=+0.595 cos_d=+0.565 score=+0.313  assistant: Wow, Caroline, sounds like the parade was an awesome experience! It's great to see the love and support for the LGBTQ+ community. Congrats! Has this experience 
- [   ] turn  36 cos_a=+0.559 cos_d=+0.525 score=+0.297  assistant: Hey Caroline! Great to hear from you. Sounds like your event was amazing! I'm so proud of you for spreading awareness and getting others involved in the LGBTQ c
- [   ] turn 304 cos_a=+0.544 cos_d=+0.501 score=+0.294  assistant: Wow, Caroline, that's awesome! Can't wait to see your show - the LGBTQ community needs more platforms like this!
- [   ] turn 156 cos_a=+0.509 cos_d=+0.460 score=+0.278  assistant: Wow, Caroline! That's huge! How did it feel to be around so much love and acceptance?
- fair-backfill r@20=1.000 r@50=1.000

### Question: _When did Melanie paint a sunrise?_
- category: `locomo_temporal`
- distractor_probe: _Can you recall when Melanie painted a sunrise and what inspired her choice of colors?_
- gold turn_ids: [11]

**v2f top 5 (retrieval order):**
- CUE 0: Is this your own painting?
- CUE 1: Yeah, I painted that lake sunrise last year! It's special to me.
- fair-backfill r@20=0.000 r@50=0.000

**contrast_v2f_a0.5 re-rank top 5:**
- [   ] turn 277 cos_a=+0.616 cos_d=+0.643 score=+0.294  user: Thanks, Melanie! I painted it after I visited the beach last week. Just seeing the sun dip below the horizon, all the amazing colors - it was amazing and calmin
- [   ] turn  12 cos_a=+0.542 cos_d=+0.542 score=+0.271  user: Thanks, Melanie! That's really sweet. Is this your own painting?
- [   ] turn 301 cos_a=+0.529 cos_d=+0.520 score=+0.269  user: Wow, Mel! Any more paintings coming up?
- [   ] turn  13 cos_a=+0.504 cos_d=+0.507 score=+0.250  assistant: Yeah, I painted that lake sunrise last year! It's special to me.
- [   ] turn 366 cos_a=+0.551 cos_d=+0.610 score=+0.246  user: Wow Mel, that's stunning! Love the colors and the chilled-out sunset vibe. What made you paint it? I've been trying out abstract stuff recently. It's kinda free
- fair-backfill r@20=0.000 r@50=0.000

### Question: _What fields would Caroline be likely to pursue in her educaton?_
- category: `locomo_multi_hop`
- distractor_probe: _In which fields might Caroline choose to pursue further education, given her background and interests?_
- gold turn_ids: [8, 10]

**v2f top 5 (retrieval order):**
- CUE 0: I'm studying counseling and thinking about a master's in social work, counseling psychology, or becoming a therapist
- CUE 1: I'm taking pottery classes and considering art therapy, studio art programs, or teaching community art classes
- fair-backfill r@20=1.000 r@50=1.000

**contrast_v2f_a0.5 re-rank top 5:**
- [   ] turn   9 cos_a=+0.591 cos_d=+0.545 score=+0.319  assistant: Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?
- [   ] turn  89 cos_a=+0.438 cos_d=+0.401 score=+0.238  assistant: Sounds awesome, Caroline! Have a great time and learn a lot. Have fun!
- [   ] turn 251 cos_a=+0.433 cos_d=+0.403 score=+0.232  assistant: Yeah, Caroline! I'll start thinking about what we can do.
- [   ] turn  75 cos_a=+0.471 cos_d=+0.480 score=+0.231  assistant: Congrats Caroline! Good on you for going after what you really care about.
- [   ] turn  93 cos_a=+0.410 cos_d=+0.368 score=+0.226  assistant: Hey Caroline! Missed you. Anything new? Spill the beans!
- fair-backfill r@20=0.500 r@50=1.000

## Verdict: ABANDON

**No contrastive variant ever beats v2f.** Summary:

- **`contrast_v2f_*` (re-rank v2f pool):** At K=50, all three α values tie v2f exactly
  on both datasets (pool is the same; re-ranking only changes order inside K≥pool_size).
  At K=20, re-ranking hurts: locomo_30q loses 5-21pp, synthetic_19q loses 1-9pp.
  Even α=0.2 loses 5pp at K=20 on locomo (rule 4 trigger: not "broken" but clearly
  misaligned — v2f's retrieval ordering is already better than a contrastive re-score).
- **`contrast_only_*` (no v2f, raw query vs LLM distractor):** Loses everywhere,
  catastrophically at α=1.0 (locomo_30q r@50 drops from 0.508→0.225). α=0.2 barely
  matches the cosine baseline; α=0.5 is already below it on locomo.
- **Which rule fires:** Rule 3 — distractor-probe is well-formed (distractors are
  plausible paraphrases; see samples) but paraphrase matches are **already filtered
  by v2f's cue-merging structure**. v2f's cues move the search away from query-
  paraphrase space; contrastive re-scoring adds no extra signal.

**Diagnostic from samples:** When v2f already hits gold (q1), the re-rank shuffles
top-K without changing K=50 coverage. When v2f misses gold (q2 "Melanie paint a
sunrise", gold turn 11 not in any top-30), the distractor is so close to the query
(cos_a=cos_d≈0.5) that the contrastive score barely re-orders anything — the
"paraphrase-match penalty" signal is drowned in noise because both probes share
vocabulary by construction.

**Per-category at K=20 (where re-rank matters):** locomo_single_hop is hit
hardest by contrast_v2f_a0.2 (-14pp). No category gains. On synthetic_19q,
`inference` gains +6.5pp under α=0.2 and 0.5 (3 questions, noisy).

**Do we ship, supplement, abandon?** **Abandon.** The contrast_v2f variants never
beat v2f; the only equal-score cells (K=50) are artifacts of pool identity.
contrast_only is uniformly worse than the cosine baseline. The hypothesized
"paraphrase-matches retrieved by v2f" are not actually present in the v2f output
at retrieval-time — they have already been pushed below the fold by cue merging.

## Outputs
- `results/contrastive_retrieval.md` (this file)
- `results/contrastive_retrieval.json` (consolidated summaries + category breakdowns)
- Per-(arch, dataset) raw results at `results/contrast_{arch}_{dataset}.json`
- Source: `contrastive_retrieval.py`, `contrastive_eval.py`
- Dedicated caches: `cache/contrast_embedding_cache.json`, `cache/contrast_llm_cache.json`