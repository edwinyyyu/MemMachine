# two_speaker_filter base + narrow supplements composition

Swaps the single-sided `speaker_user_filter` base of `composition_v2_all` for the two-sided `two_speaker_filter` and measures whether the full composition breaks past the prior LoCoMo K=50 = 0.917 ceiling.


Elapsed: 97s.


## Recall matrix (r@20)

| Variant | locomo_30q | synthetic_19q | overall |
|---|---|---|---|
| v2f | 0.7556 | 0.6130 | **0.7003** |
| two_speaker_alone | 0.8917 | 0.6130 | **0.7836** |
| two_speaker_plus_ens2 | 0.8917 | 0.6130 | **0.7836** |
| two_speaker_plus_critical | 0.8917 | 0.6374 | **0.7931** |
| two_speaker_plus_alias | 0.8917 | 0.6130 | **0.7836** |
| two_speaker_plus_context | 0.8917 | 0.6130 | **0.7836** |
| two_speaker_all_supplements | 0.8917 | 0.6374 | **0.7931** |

## Recall matrix (r@50)

| Variant | locomo_30q | synthetic_19q | overall |
|---|---|---|---|
| v2f | 0.8583 | 0.8513 | **0.8556** |
| two_speaker_alone | 0.8917 | 0.8513 | **0.8760** |
| two_speaker_plus_ens2 | 0.9083 | 0.8606 | **0.8898** |
| two_speaker_plus_critical | 0.8917 | 0.8554 | **0.8776** |
| two_speaker_plus_alias | 0.9083 | 0.8533 | **0.8870** |
| two_speaker_plus_context | 0.9000 | 0.8632 | **0.8857** |
| two_speaker_all_supplements | 0.9083 | 0.8753 | **0.8955** |

## LoCoMo K=50 ablation vs two_speaker_all_supplements

| Variant | r@50 | Δ vs all_supplements |
|---|---:|---:|
| v2f | 0.8583 | -0.0500 |
| two_speaker_alone | 0.8917 | -0.0166 |
| two_speaker_plus_ens2 | 0.9083 | +0.0000 |
| two_speaker_plus_critical | 0.8917 | -0.0166 |
| two_speaker_plus_alias | 0.9083 | +0.0000 |
| two_speaker_plus_context | 0.9000 | -0.0083 |
| two_speaker_all_supplements | 0.9083 | +0.0000 |

## LoCoMo K=20 ablation vs two_speaker_all_supplements

| Variant | r@20 | Δ vs all_supplements |
|---|---:|---:|
| v2f | 0.7556 | -0.1361 |
| two_speaker_alone | 0.8917 | +0.0000 |
| two_speaker_plus_ens2 | 0.8917 | +0.0000 |
| two_speaker_plus_critical | 0.8917 | +0.0000 |
| two_speaker_plus_alias | 0.8917 | +0.0000 |
| two_speaker_plus_context | 0.8917 | +0.0000 |
| two_speaker_all_supplements | 0.8917 | +0.0000 |

## LoCoMo subset analysis (two_speaker_all_supplements)

| Subset | n | v2f@50 | two_speaker_alone@50 | two_speaker_all_supplements@50 | Δ(all vs v2f) |
|---|---:|---:|---:|---:|---:|
| matched_user | 18 | 0.8889 | 0.9444 | 0.9444 | +0.0555 |
| matched_assistant | 12 | 0.8125 | 0.8125 | 0.8542 | +0.0417 |
| matched_both | 0 | - | - | - | - |
| matched_none | 0 | - | - | - | - |
| speaker_fires | 30 | 0.8583 | 0.8917 | 0.9083 | +0.0500 |

## Decision

- LoCoMo K=20 v2f = 0.7556
- LoCoMo K=20 two_speaker_alone = 0.8917
- LoCoMo K=20 two_speaker_all_supplements = 0.8917
- LoCoMo K=50 v2f = 0.8583
- LoCoMo K=50 two_speaker_alone = 0.8917
- LoCoMo K=50 two_speaker_all_supplements = 0.9083
- Prior ceiling (composition_v2_all) = 0.9170
- Δ vs 0.9170: -0.0087
  => two_speaker_filter base interacts negatively with supplements (Δ=-0.0087). Interesting but composition_v2_all's speaker_user_filter base is better.

- K=20: two_speaker_alone ≈ all_supplements (0.8917 vs 0.8917).
