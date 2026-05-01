# Active Embedding Steering

Probe = normalize(initial_embedding + alpha*sum(add_phrase_embs) - beta*sum(sub_phrase_embs)) applied cumulatively over LLM-generated add/sub phrases.

Fixed: alpha=beta=0.1, text-embedding-3-small, gpt-5-mini, LoCoMo-30 and LME-hard-30 POC subset.

## Recall matrix -- locomo

| Variant | n | R@20 | R@50 | time (s) |
| --- | --- | --- | --- | --- |
| `steer_v2f_1round` | 30 | 0.7667 | 0.8750 | 59.5 |
| `steer_v2f_2round` | 30 | 0.7833 | 0.8833 | 62.5 |
| `steer_v2f_3round` | 30 | 0.8000 | 0.8667 | 55.5 |
| `steer_v2f_modelweight` | 30 | 0.7917 | 0.9000 | 153.9 |
| `steer_v2f_addonly` | 30 | 0.7500 | 0.9000 | 35.1 |
| `steer_v2f_subonly` | 30 | 0.7667 | 0.8833 | 45.7 |
| `steer_query_direct` | 30 | 0.8500 | 0.9000 | 118.4 |
| `baseline_query_direct` | 30 | 0.7333 | 0.8833 | 0.2 |
| `baseline_v2f_direct` | 30 | 0.7500 | 0.9167 | 1.0 |

## Recall matrix -- lme

| Variant | n | R@20 | R@50 | time (s) |
| --- | --- | --- | --- | --- |
| `steer_v2f_1round` | 30 | 0.6531 | 0.8004 | 80.8 |
| `steer_v2f_2round` | 30 | 0.6358 | 0.7994 | 26.6 |
| `steer_v2f_3round` | 30 | 0.6427 | 0.7934 | 20.1 |
| `steer_v2f_modelweight` | 30 | 0.6407 | 0.7507 | 83.3 |
| `steer_v2f_addonly` | 30 | 0.6583 | 0.8085 | 14.5 |
| `steer_v2f_subonly` | 30 | 0.6339 | 0.8128 | 18.4 |
| `steer_query_direct` | 30 | 0.5759 | 0.7884 | 84.3 |
| `baseline_query_direct` | 30 | 0.6173 | 0.8222 | 1.0 |
| `baseline_v2f_direct` | 30 | 0.6303 | 0.8169 | 2.0 |

## LME category recall (R@20)

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `steer_v2f_1round` | 0.5485 | 0.8604 | 0.5503 |
| `steer_v2f_2round` | 0.5279 | 0.8292 | 0.5503 |
| `steer_v2f_3round` | 0.5237 | 0.8542 | 0.5503 |
| `steer_v2f_modelweight` | 0.5352 | 0.8542 | 0.5328 |
| `steer_v2f_addonly` | 0.5633 | 0.8521 | 0.5595 |
| `steer_v2f_subonly` | 0.5167 | 0.8396 | 0.5453 |
| `steer_query_direct` | 0.5087 | 0.7500 | 0.4689 |
| `baseline_query_direct` | 0.5457 | 0.7458 | 0.5605 |
| `baseline_v2f_direct` | 0.5452 | 0.8104 | 0.5353 |

## LME category recall (R@50)

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `steer_v2f_1round` | 0.7065 | 0.9062 | 0.7886 |
| `steer_v2f_2round` | 0.6999 | 0.9104 | 0.7880 |
| `steer_v2f_3round` | 0.6859 | 0.9104 | 0.7838 |
| `steer_v2f_modelweight` | 0.6431 | 0.8938 | 0.7151 |
| `steer_v2f_addonly` | 0.7181 | 0.9062 | 0.8012 |
| `steer_v2f_subonly` | 0.7223 | 0.9167 | 0.7994 |
| `steer_query_direct` | 0.7260 | 0.8896 | 0.7496 |
| `baseline_query_direct` | 0.7368 | 0.8854 | 0.8444 |
| `baseline_v2f_direct` | 0.7398 | 0.9021 | 0.8089 |

## Round-by-round recall trajectory (mean R@50)

Rows: variants. Columns: round 0 (initial cue) .. final round.

### locomo

| Variant | rd 0 | rd 1 | rd 2 | rd 3 | drift@final |
| --- | --- | --- | --- | --- | --- |
| `steer_v2f_1round` | 0.9167 | 0.8750 | -- | -- | 0.9800 |
| `steer_v2f_2round` | 0.9167 | 0.8750 | 0.8833 | -- | 0.9480 |
| `steer_v2f_3round` | 0.9167 | 0.8750 | 0.8833 | 0.8667 | 0.9098 |
| `steer_v2f_modelweight` | 0.9167 | 0.8667 | 0.9000 | -- | 0.9313 |
| `steer_v2f_addonly` | 0.9167 | 0.9000 | 0.9000 | -- | 0.9644 |
| `steer_v2f_subonly` | 0.9167 | 0.8750 | 0.8833 | -- | 0.9455 |
| `steer_query_direct` | 0.8833 | 0.8833 | 0.9000 | -- | 0.9436 |
| `baseline_query_direct` | 0.8833 | -- | -- | -- | 1.0000 |
| `baseline_v2f_direct` | 0.9167 | -- | -- | -- | 1.0000 |

### lme

| Variant | rd 0 | rd 1 | rd 2 | rd 3 | drift@final |
| --- | --- | --- | --- | --- | --- |
| `steer_v2f_1round` | 0.8169 | 0.8004 | -- | -- | 0.9787 |
| `steer_v2f_2round` | 0.8169 | 0.8004 | 0.7994 | -- | 0.9301 |
| `steer_v2f_3round` | 0.8169 | 0.8004 | 0.7994 | 0.7934 | 0.8690 |
| `steer_v2f_modelweight` | 0.8169 | 0.7890 | 0.7507 | -- | 0.8903 |
| `steer_v2f_addonly` | 0.8169 | 0.8072 | 0.8085 | -- | 0.9543 |
| `steer_v2f_subonly` | 0.8169 | 0.8317 | 0.8128 | -- | 0.9470 |
| `steer_query_direct` | 0.8222 | 0.7987 | 0.7884 | -- | 0.9371 |
| `baseline_query_direct` | 0.8222 | -- | -- | -- | 1.0000 |
| `baseline_v2f_direct` | 0.8169 | -- | -- | -- | 1.0000 |

## SUBTRACT primitive usage

Fraction of steering rounds where the LLM emitted a non-empty SUBTRACT list.

| Variant | corpus | sub_nonempty_fraction |
| --- | --- | --- |
| `steer_v2f_1round` | locomo | 1.000 |
| `steer_v2f_1round` | lme | 1.000 |
| `steer_v2f_2round` | locomo | 1.000 |
| `steer_v2f_2round` | lme | 1.000 |
| `steer_v2f_3round` | locomo | 1.000 |
| `steer_v2f_3round` | lme | 1.000 |
| `steer_v2f_modelweight` | locomo | 1.000 |
| `steer_v2f_modelweight` | lme | 1.000 |
| `steer_v2f_addonly` | locomo | 0.000 |
| `steer_v2f_addonly` | lme | 0.000 |
| `steer_v2f_subonly` | locomo | 1.000 |
| `steer_v2f_subonly` | lme | 1.000 |
| `steer_query_direct` | locomo | 1.000 |
| `steer_query_direct` | lme | 1.000 |
| `baseline_query_direct` | locomo | 0.000 |
| `baseline_query_direct` | lme | 0.000 |
| `baseline_v2f_direct` | locomo | 0.000 |
| `baseline_v2f_direct` | lme | 0.000 |

## Update magnitudes (alpha*||add|| and beta*||sub||)

### locomo

| Variant | rd1 add_mag | rd1 sub_mag | rd2 add_mag | rd2 sub_mag |
| --- | --- | --- | --- | --- |
| `steer_v2f_1round` | 0.2334 | 0.1904 | 0.0000 | 0.0000 |
| `steer_v2f_2round` | 0.2334 | 0.1904 | 0.2256 | 0.1789 |
| `steer_v2f_3round` | 0.2334 | 0.1904 | 0.2256 | 0.1789 |
| `steer_v2f_modelweight` | 0.3006 | 0.2435 | 0.3152 | 0.2553 |
| `steer_v2f_addonly` | 0.2334 | 0.0000 | 0.2347 | 0.0000 |
| `steer_v2f_subonly` | 0.0000 | 0.1904 | 0.0000 | 0.1812 |
| `steer_query_direct` | 0.2450 | 0.1839 | 0.2460 | 0.1873 |
| `baseline_query_direct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `baseline_v2f_direct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### lme

| Variant | rd1 add_mag | rd1 sub_mag | rd2 add_mag | rd2 sub_mag |
| --- | --- | --- | --- | --- |
| `steer_v2f_1round` | 0.2353 | 0.1734 | 0.0000 | 0.0000 |
| `steer_v2f_2round` | 0.2353 | 0.1734 | 0.2346 | 0.1698 |
| `steer_v2f_3round` | 0.2353 | 0.1734 | 0.2346 | 0.1698 |
| `steer_v2f_modelweight` | 0.3131 | 0.2531 | 0.3121 | 0.2466 |
| `steer_v2f_addonly` | 0.2353 | 0.0000 | 0.2346 | 0.0000 |
| `steer_v2f_subonly` | 0.0000 | 0.1734 | 0.0000 | 0.1669 |
| `steer_query_direct` | 0.2380 | 0.1816 | 0.2355 | 0.1742 |
| `baseline_query_direct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `baseline_v2f_direct` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Sample add/sub phrases (LME)

### `steer_v2f_2round`

- Q `ba358f49` (`multi-session`): How many years will I be when my friend Rachel gets married?
  - initial cue: `User: I'm looking for some advice on skincare routines for my age group. I've been noticing some fine lines and wrinkles lately, and I want to start taking better care of my skin. By the way, my friend Rachel's getting married next year, and it's got me th`
  - round 1: drift=0.978, R@50=0.625
    - ADD: I’d wear SPF 30+ broad‑spectrum sunscreen every morning (w=1.0); I can introduce retinol 0.25–1% at night 2–3 times weekly (w=1.0); I’m planning a 6‑month skincare timeline before Rachel’s wedding (w=1.0)
    - SUB: Memory Palace imagery and prompts (w=1.0); lists of video game titles and demos (w=1.0)
  - round 2: drift=0.925, R@50=0.5833
    - ADD: I’d wear SPF 30+ broad‑spectrum sunscreen every morning (w=1.0); I can introduce retinol 0.25–1% at night 2–3 times weekly (w=1.0); I’m planning a 6‑month skincare timeline before Rachel’s wedding (w=1.0)
    - SUB: Memory Palace imagery and prompts (w=1.0); lists of video game titles and demos (w=1.0)
- Q `5a7937c8` (`multi-session`): How many days did I spend participating in faith-related activities in December?
  - initial cue: `User: I'm looking for some volunteer opportunities in my community, preferably something related to food banks or pantries. I actually helped out at the church's annual holiday food drive on December 10th, sorting donations and packing boxes for families i`
  - round 1: drift=0.982, R@50=0.5312
    - ADD: I can commit to recurring Saturday morning shifts at the community food pantry (w=1.0); I'm interested in family programs like kids' meal packs and school-supply distributions (w=1.0); High-need donations: canned protein, baby formula, diapers, infant food (w=1.0)
    - SUB: comments about an adoption journey (w=1.0); generic praise about volunteering benefits (w=1.0)
  - round 2: drift=0.936, R@50=0.5312
    - ADD: I can commit to recurring Saturday morning shifts at the community food pantry (w=1.0); I'm interested in family programs like kids' meal packs and school-supply distributions (w=1.0); High-need donations: canned protein, baby formula, diapers, infant food (w=1.0)
    - SUB: comments about an adoption journey (w=1.0); generic praise about volunteering benefits (w=1.0)
- Q `6c49646a` (`multi-session`): What is the total distance I covered in my four road trips?
  - initial cue: `User: I'm glad I could fit in Maroon Lake. Since I've covered a total of 1,800 miles on my recent three road trips, including a solo trip to Durango, a weekend trip to Breckenridge, and a family trip to Santa Fe, I'm comfortable with the drive and explorin`
  - round 1: drift=0.982, R@50=0.875
    - ADD: I drove from Aspen to Maroon Lake in about 25 minutes (w=1.0); On my 1,800-mile road trips I scheduled two-hour driving blocks and overnight breaks (w=1.0); I paired Maroon Lake with a short 1-mile hike and an early-morning visit to beat crowds (w=1.0)
    - SUB: parking tips and photo advice (w=1.0); generic encouragement about enjoying the trip (w=1.0)
  - round 2: drift=0.938, R@50=0.875
    - ADD: I drove from Aspen to Maroon Lake in about 25 minutes (w=1.0); On my 1,800-mile road trips I scheduled two-hour driving blocks and overnight breaks (w=1.0); I paired Maroon Lake with a short 1-mile hike and an early-morning visit to beat crowds (w=1.0)
    - SUB: parking tips and photo advice (w=1.0); generic encouragement about enjoying the trip (w=1.0)

## Verdict

### locomo

- baseline (v2f cue direct): R@50 = 0.9167
- baseline (query direct): R@50 = 0.8833
- steer_v2f_2round: R@50 = 0.8833 (Δ vs baseline = -0.0334)
- steer_v2f_addonly: R@50 = 0.9000
- steer_v2f_subonly: R@50 = 0.8833

### lme

- baseline (v2f cue direct): R@50 = 0.8169
- baseline (query direct): R@50 = 0.8222
- steer_v2f_2round: R@50 = 0.7994 (Δ vs baseline = -0.0175)
- steer_v2f_addonly: R@50 = 0.8085
- steer_v2f_subonly: R@50 = 0.8128

## Outputs

- JSON: `results/active_steering.json`
- Sources: `active_steering.py`, `steer_eval.py`
- Caches: `cache/steer_llm_cache.json`, `cache/steer_embedding_cache.json`
