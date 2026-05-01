# Query-Direction Embedding Projection

Zero-LLM architectural variant. Learn a unit direction separating question-form from statement-form prose, then project queries away from that direction before cosine retrieval.

- `q_dir = normalize(mean(question_embs) - mean(statement_embs))`
- `q_emb' = normalize(q_emb - alpha * (q_emb . q_dir) * q_dir)`

**Direction stats**: n_q=100, n_s=100, diff_norm=0.2263, q_mean_norm=0.4832, s_mean_norm=0.4595.

## Sanity check — held-out samples cosine with q_dir

- Question samples (n=10): mean=+0.2126 (min=+0.1283, max=+0.2743)
- Statement samples (n=10): mean=-0.1369 (min=-0.1777, max=-0.0977)
- Separation (Q-S): +0.3495


## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cosine_baseline | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.508 | +0.000 | 0.0 |
| cosine_baseline | synthetic_19q | 0.569 | 0.569 | +0.000 | 0.824 | 0.824 | +0.000 | 0.0 |
| qproj_0.5 | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.508 | +0.000 | 0.0 |
| qproj_0.5 | synthetic_19q | 0.569 | 0.569 | +0.000 | 0.824 | 0.834 | +0.010 | 0.0 |
| qproj_1.0 | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.475 | -0.033 | 0.0 |
| qproj_1.0 | synthetic_19q | 0.569 | 0.569 | +0.000 | 0.824 | 0.829 | +0.005 | 0.0 |
| qproj_1.5 | locomo_30q | 0.383 | 0.350 | -0.033 | 0.508 | 0.486 | -0.022 | 0.0 |
| qproj_1.5 | synthetic_19q | 0.569 | 0.572 | +0.002 | 0.824 | 0.816 | -0.007 | 0.0 |
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| qproj_0.5_v2f | locomo_30q | 0.383 | 0.672 | +0.289 | 0.508 | 0.792 | +0.283 | 1.0 |
| qproj_0.5_v2f | synthetic_19q | 0.569 | 0.608 | +0.039 | 0.824 | 0.856 | +0.032 | 1.0 |

Note: `base@K` in all rows is raw cosine top-K (same reference).


## Best alpha (sum of delta_r@50 vs cosine across datasets)

- **Best alpha**: 0.5

- **Score**: +0.0100


## Top categories by Δr@50 for qproj_0.5 on LoCoMo-30

Gaining:
Losing:

## Projection-effect visualization

For each sample query: cosine-with-direction, raw top-5 vs projected top-5 retrieved turn_ids and cosine scores.


### Q: When did Caroline go to the LGBTQ support group?
- cos(q_emb, q_dir) = +0.1484

**Raw top-5:**

  - tid=184 score=+0.603 :: "Wow, Caroline! They must have felt so appreciated. It's awesome to see the diffe"
  - tid=77 score=+0.595 :: "Wow, Caroline, sounds like the parade was an awesome experience! It's great to s"
  - tid=36 score=+0.559 :: "Hey Caroline! Great to hear from you. Sounds like your event was amazing! I'm so"
  - tid=304 score=+0.544 :: "Wow, Caroline, that's awesome! Can't wait to see your show - the LGBTQ community"
  - tid=2 score=+0.538 :: 'I went to a LGBTQ support group yesterday and it was so powerful.'

**Projected top-5:**

  - tid=184 score=+0.601 :: "Wow, Caroline! They must have felt so appreciated. It's awesome to see the diffe"
  - tid=77 score=+0.591 :: "Wow, Caroline, sounds like the parade was an awesome experience! It's great to s"
  - tid=36 score=+0.558 :: "Hey Caroline! Great to hear from you. Sounds like your event was amazing! I'm so"
  - tid=304 score=+0.546 :: "Wow, Caroline, that's awesome! Can't wait to see your show - the LGBTQ community"
  - tid=2 score=+0.539 :: 'I went to a LGBTQ support group yesterday and it was so powerful.'

### Q: When did Melanie paint a sunrise?
- cos(q_emb, q_dir) = +0.0903

**Raw top-5:**

  - tid=277 score=+0.616 :: 'Thanks, Melanie! I painted it after I visited the beach last week. Just seeing t'
  - tid=366 score=+0.551 :: "Wow Mel, that's stunning! Love the colors and the chilled-out sunset vibe. What "
  - tid=276 score=+0.545 :: 'Wow Caroline, that looks amazing! Those colors are so vivid, it really looks lik'
  - tid=12 score=+0.542 :: "Thanks, Melanie! That's really sweet. Is this your own painting?"
  - tid=301 score=+0.528 :: 'Wow, Mel! Any more paintings coming up?'

**Projected top-5:**

  - tid=277 score=+0.615 :: 'Thanks, Melanie! I painted it after I visited the beach last week. Just seeing t'
  - tid=366 score=+0.548 :: "Wow Mel, that's stunning! Love the colors and the chilled-out sunset vibe. What "
  - tid=276 score=+0.543 :: 'Wow Caroline, that looks amazing! Those colors are so vivid, it really looks lik'
  - tid=12 score=+0.542 :: "Thanks, Melanie! That's really sweet. Is this your own painting?"
  - tid=301 score=+0.526 :: 'Wow, Mel! Any more paintings coming up?'

### Q: What fields would Caroline be likely to pursue in her educaton?
- cos(q_emb, q_dir) = +0.1338

**Raw top-5:**

  - tid=9 score=+0.591 :: "Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?"
  - tid=75 score=+0.471 :: 'Congrats Caroline! Good on you for going after what you really care about.'
  - tid=89 score=+0.438 :: 'Sounds awesome, Caroline! Have a great time and learn a lot. Have fun!'
  - tid=251 score=+0.433 :: "Yeah, Caroline! I'll start thinking about what we can do."
  - tid=315 score=+0.423 :: "That's great news, Caroline! Love seeing your dedication to helping others. Any "

**Projected top-5:**

  - tid=9 score=+0.585 :: "Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?"
  - tid=75 score=+0.471 :: 'Congrats Caroline! Good on you for going after what you really care about.'
  - tid=89 score=+0.440 :: 'Sounds awesome, Caroline! Have a great time and learn a lot. Have fun!'
  - tid=251 score=+0.430 :: "Yeah, Caroline! I'll start thinking about what we can do."
  - tid=315 score=+0.422 :: "That's great news, Caroline! Love seeing your dedication to helping others. Any "

### Q: What did Caroline research?
- cos(q_emb, q_dir) = +0.1172

**Raw top-5:**

  - tid=9 score=+0.490 :: "Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?"
  - tid=93 score=+0.474 :: 'Hey Caroline! Missed you. Anything new? Spill the beans!'
  - tid=288 score=+0.473 :: 'Wow, Caroline, that looks amazing! What inspired it?'
  - tid=180 score=+0.471 :: 'Caroline, awesome news that you two are getting along! What was it like for you '
  - tid=75 score=+0.463 :: 'Congrats Caroline! Good on you for going after what you really care about.'

**Projected top-5:**

  - tid=9 score=+0.483 :: "Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?"
  - tid=288 score=+0.470 :: 'Wow, Caroline, that looks amazing! What inspired it?'
  - tid=93 score=+0.469 :: 'Hey Caroline! Missed you. Anything new? Spill the beans!'
  - tid=180 score=+0.463 :: 'Caroline, awesome news that you two are getting along! What was it like for you '
  - tid=75 score=+0.463 :: 'Congrats Caroline! Good on you for going after what you really care about.'

### Q: What is Caroline's identity?
- cos(q_emb, q_dir) = +0.1486

**Raw top-5:**

  - tid=93 score=+0.505 :: 'Hey Caroline! Missed you. Anything new? Spill the beans!'
  - tid=59 score=+0.492 :: 'Hey, Caroline! Nice to hear from you! Love the necklace, any special meaning to '
  - tid=270 score=+0.492 :: "Bye Caroline. I'm here for you. Take care of yourself."
  - tid=288 score=+0.487 :: 'Wow, Caroline, that looks amazing! What inspired it?'
  - tid=9 score=+0.487 :: "Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?"

**Projected top-5:**

  - tid=93 score=+0.499 :: 'Hey Caroline! Missed you. Anything new? Spill the beans!'
  - tid=270 score=+0.491 :: "Bye Caroline. I'm here for you. Take care of yourself."
  - tid=59 score=+0.487 :: 'Hey, Caroline! Nice to hear from you! Love the necklace, any special meaning to '
  - tid=288 score=+0.484 :: 'Wow, Caroline, that looks amazing! What inspired it?'
  - tid=9 score=+0.479 :: "Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?"

## Verdict

LoCoMo-30 @K=50:
  cosine        = 0.508
  qproj(α=0.5) = 0.508 (Δ vs cosine = +0.000)
  meta_v2f      = 0.858
  qproj_v2f(α=0.5) = 0.792 (Δ vs v2f = -0.067)


**ABANDON**: no qproj_only alpha beats cosine baseline on LoCoMo @K=50.
