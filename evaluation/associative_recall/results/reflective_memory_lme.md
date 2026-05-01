# LME-hard Reflective Memory (scratch-memory write during query)

All variants use `expand_context=3` + `User: ` prefix. Round-1 cue prompt is `em_v2f_lme_mixed_7030` (cache-reused). Reflection writes `learned` + `still_need` scratch entries; retrieval uses corpus + scratch entries as additional anchors, merged by max-per-turn.

## References (LME-hard R@50 overall)

- `em_v2f_expand_3` baseline: 0.832
- `em_ens_2_lme_retuned`: 0.850
- `em_v2f_lme_mixed_7030` ceiling: 0.863

## Per-variant summary

| Variant | R@20 | R@50 | Δ vs mixed_7030 (0.8631) | avg LLM calls | avg rounds | time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `reflmemlme_1round` | 0.6365 | 0.8631 | +0.0000 | 0.98 | 1.00 | 177.3 |
| `reflmemlme_2round` | 0.6415 | 0.8730 | +0.0099 | 1.99 | 2.00 | 427.5 |
| `reflmemlme_3round` | 0.6373 | 0.8764 | +0.0133 | 2.06 | 3.00 | 599.7 |

## Recall matrix (R@20)

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `reflmemlme_1round` | 0.5785 | 0.8239 | 0.5073 |
| `reflmemlme_2round` | 0.5746 | 0.8365 | 0.5133 |
| `reflmemlme_3round` | 0.5686 | 0.8259 | 0.5176 |

## Recall matrix (R@50)

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `reflmemlme_1round` | 0.8484 | 0.9449 | 0.7961 |
| `reflmemlme_2round` | 0.8564 | 0.9614 | 0.8010 |
| `reflmemlme_3round` | 0.8606 | 0.9614 | 0.8073 |

### Reference: `em_v2f_lme_mixed_7030` per-category @ K=50

- multi-session: 0.8484
- single-session-preference: 0.9449
- temporal-reasoning: 0.7961

## Round-novelty analysis

### `reflmemlme_1round`
- K=20:
  - Round 1: frac_queries_with_novel_gold=0.978, frac_queries_with_recall_gain=0.978, mean_recall_delta=+0.6365
- K=50:
  - Round 1: frac_queries_with_novel_gold=0.989, frac_queries_with_recall_gain=0.989, mean_recall_delta=+0.8631

### `reflmemlme_2round`
- K=20:
  - Round 1: frac_queries_with_novel_gold=0.978, frac_queries_with_recall_gain=0.978, mean_recall_delta=+0.6357
  - Round 2: frac_queries_with_novel_gold=0.389, frac_queries_with_recall_gain=0.244, mean_recall_delta=+0.0058
- K=50:
  - Round 1: frac_queries_with_novel_gold=0.989, frac_queries_with_recall_gain=0.989, mean_recall_delta=+0.8631
  - Round 2: frac_queries_with_novel_gold=0.222, frac_queries_with_recall_gain=0.178, mean_recall_delta=+0.0098

### `reflmemlme_3round`
- K=20:
  - Round 1: frac_queries_with_novel_gold=0.978, frac_queries_with_recall_gain=0.978, mean_recall_delta=+0.6365
  - Round 2: frac_queries_with_novel_gold=0.389, frac_queries_with_recall_gain=0.244, mean_recall_delta=+0.0042
  - Round 3: frac_queries_with_novel_gold=0.089, frac_queries_with_recall_gain=0.033, mean_recall_delta=-0.0034
- K=50:
  - Round 1: frac_queries_with_novel_gold=0.989, frac_queries_with_recall_gain=0.989, mean_recall_delta=+0.8631
  - Round 2: frac_queries_with_novel_gold=0.222, frac_queries_with_recall_gain=0.178, mean_recall_delta=+0.0103
  - Round 3: frac_queries_with_novel_gold=0.089, frac_queries_with_recall_gain=0.078, mean_recall_delta=+0.0030

## Sample scratch state + round-2 gains (from `reflmemlme_2round`)

### Q `6c49646a` (multi-session)
- Question: What is the total distance I covered in my four road trips?
- Gold turns: 24
- Round 1 cues:
  - `User: I'm glad I could fit in Maroon Lake. Since I've covered a total of 1,800 miles on my recent three road trips, including a solo trip to Durango, a weekend trip to Breckenridge, and a family trip to Santa Fe, I'm comfortable with the drive and explorin`
  - `User: I'm glad I could get some helpful information about the shuttle service. By the way, I was thinking about our Yellowstone trip last month, and I realized that we drove around 300 miles on the first day to reach Jackson, Wyoming. That was a long drive`
  - `Assistant: That sounds like an amazing trip to Yellowstone! 1,200 miles in 4 days is quite an adventure! I'm sure you must have seen some incredible sights.`
- Round 1 LEARNED:
  - `I covered a total of 1,800 miles on my recent three road trips (a solo trip to Durango, a weekend trip to Breckenridge, and a family trip to Santa Fe).`
  - `I just got back from a 4-day family trip to Yellowstone National Park last month.`
  - `On the first day of the Yellowstone trip I drove around 300 miles to reach Jackson, Wyoming.`
  - `I'm planning another road trip and asked for the best route from Denver to Mount Rushmore.`
- Round 1 STILL_NEED:
  - `My Yellowstone trip totaled 1,200 miles.`
  - `The four road trips to sum are Durango, Breckenridge, Santa Fe, and Yellowstone.`
  - `The 1,800 miles for my three trips does not include the Yellowstone trip mileage.`
- Round 2 cues:
  - `User: Just to be clear, my Yellowstone family trip last month totaled about 1,200 miles over the four days.`
  - `User: And the 1,800 miles I mentioned for my three recent trips (Durango, Breckenridge, and Santa Fe) does not include that Yellowstone mileage.`
  - `Assistant: Got it — add Yellowstone's 1,200 miles to the 1,800 from the other three trips, which comes to 3,000 miles total for all four road trips.`
- Round 2 LEARNED:
  - `I covered a total of 1,800 miles on my recent three road trips, including a solo trip to Durango, a weekend trip to Breckenridge, and a family trip to Santa Fe.`
  - `I just got back from an amazing 4-day family trip to Yellowstone National Park last month.`
  - `My Yellowstone trip totaled 1,200 miles.`
  - `On the first day of the Yellowstone trip I drove around 300 miles to reach Jackson, Wyoming.`
  - `I'm planning another road trip and asked for the best route from Denver to Mount Rushmore.`
- Round 2 STILL_NEED:
  - `The 1,800 miles for my three trips excludes the Yellowstone trip mileage.`
  - `The four road trips to sum are Durango, Breckenridge, Santa Fe, and Yellowstone.`
- Per-round recall:
  - Round 1: r@20=0.750 r@50=0.958
  - Round 2: r@20=0.750 r@50=1.000

### Q `e56a43b9` (multi-session)
- Question: How much discount will I get on my next purchase at FreshMart?
- Gold turns: 24
- Round 1 cues:
  - `User: I meant FreshMart, it's my local grocery store. I have some points there and every 100 points translate to a $1 discount on my next purchase. Thanks for the list, but I was thinking of something more personalized. Can you help me make a list based on`
  - `User: I usually cook for 2 people, and we eat a bit of everything. I'm not too particular about brands, but I do like to stock up on staples like rice, pasta, and canned goods. Also, I've been trying to shop at FreshMart more often since, you know, 500 poi`
  - `Assistant: Congratulations on reaching the minimum points for a discount at FreshMart! That's always exciting. Now, let's get cracking (pun intended) on some delicious recipe ideas that feature eggs and spinach as main ingredients. Here are a few ideas to get`
- Round 1 LEARNED:
  - `I reached 500 points, the minimum points required for a discount at FreshMart.`
  - `Every 100 points translates to a $1 discount on my next purchase at FreshMart.`
  - `I'm planning a big grocery shopping trip to FreshMart this weekend.`
  - `I usually cook for 2 people and stock staples like rice, pasta, and canned goods.`
- Round 1 STILL_NEED:
  - `I currently have <number> points at FreshMart.`
  - `FreshMart has no maximum dollar cap on the number of points I can redeem per transaction.`
  - `I will use the FreshMart account that holds the points at checkout for my next purchase.`
- Round 2 cues:
  - `User: What's my current FreshMart points balance right now — I need the exact number so we can calculate the discount`
  - `User: I'll be using the FreshMart account that holds the loyalty points at checkout for my big grocery trip this weekend, just confirming that`
  - `Assistant: I'll check FreshMart's rewards/terms to confirm whether they impose any maximum dollar cap on redeeming points per transaction`
- Round 2 LEARNED:
  - `I reached 500 points at FreshMart.`
  - `500 points is the minimum points required for a discount at FreshMart.`
  - `Every 100 points translates to a $1 discount on my next purchase at FreshMart.`
  - `I'm planning a big grocery shopping trip to FreshMart this weekend.`
  - `I usually cook for 2 people and stock staples like rice, pasta, and canned goods.`
- Round 2 STILL_NEED:
  - `My current FreshMart points balance is <number>.`
  - `FreshMart has a maximum dollar cap on points redemption per transaction of <number>.`
  - `I will redeem my FreshMart points on my next purchase (I will use the account that holds the points at checkout).`
- Per-round recall:
  - Round 1: r@20=0.583 r@50=0.708
  - Round 2: r@20=0.625 r@50=0.792

## Verdict

- Top variant: `reflmemlme_3round` R@50 = 0.8764 (Δ vs em_v2f_lme_mixed_7030 0.8631 = +0.0133)

Decision rule check:
- If best R@50 > 0.863 overall: reflective memory lifts the mixed_7030 ceiling. (best = 0.8764, LIFT)
- Temporal-reasoning (best variant @ K=50): 0.8073 (ref 0.7961; LIFT)

## Outputs

- JSON: `results/reflective_memory_lme.json`
- Sources: `reflective_memory_lme.py`, `reflmemlme_eval.py`
- Caches: `cache/reflmemlme_cue_round1_cache.json` (reads from `cache/lmetune_v2f_mixed7030_cache.json`), `cache/reflmemlme_cue_roundn_cache.json`, `cache/reflmemlme_reflect_cache.json`