# MMR Cue-Diversity Selection Study

Motivation: v2f generates 2-3 cues per call — are they redundant variants probing the same region of embedding space? MMR selection generates 8 candidates per call, selects 3 (or 4) maximizing mutual distance balanced against query relevance. Tests whether cue redundancy is leaving gold unretrieved.

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | d@20 | base@50 | arch@50 | d@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| mmr_lam0.5_k3 | locomo_30q | 0.383 | 0.628 | +0.244 | 0.508 | 0.678 | +0.169 | 1.0 |
| mmr_lam0.5_k3 | synthetic_19q | 0.569 | 0.569 | -0.001 | 0.824 | 0.843 | +0.019 | 1.0 |
| mmr_lam0.3_k3 | locomo_30q | 0.383 | 0.628 | +0.244 | 0.508 | 0.756 | +0.247 | 1.0 |
| mmr_lam0.3_k3 | synthetic_19q | 0.569 | 0.611 | +0.042 | 0.824 | 0.840 | +0.016 | 1.0 |
| mmr_lam0.7_k3 | locomo_30q | 0.383 | 0.644 | +0.261 | 0.508 | 0.711 | +0.203 | 1.0 |
| mmr_lam0.7_k3 | synthetic_19q | 0.569 | 0.593 | +0.024 | 0.824 | 0.853 | +0.029 | 1.0 |
| mmr_lam0.5_k4 | locomo_30q | 0.383 | 0.628 | +0.244 | 0.508 | 0.678 | +0.169 | 1.0 |
| mmr_lam0.5_k4 | synthetic_19q | 0.569 | 0.598 | +0.028 | 0.824 | 0.843 | +0.019 | 1.0 |
| v2f_3cues | locomo_30q | 0.383 | 0.656 | +0.272 | 0.508 | 0.850 | +0.342 | 1.0 |
| v2f_3cues | synthetic_19q | 0.569 | 0.624 | +0.055 | 0.824 | 0.835 | +0.011 | 1.0 |

## Cue pairwise cosine (diversity metric)

Lower pairwise cosine = more diverse cues (cover distinct regions). Compare v2f natural (2 cues) vs MMR selected (3 or 4).

| Dataset | Variant | mean pairwise cos | n_q w/ >=2 cues |
|---|---|---:|---:|
| locomo_30q | v2f_natural | 0.636 | 30 |
| locomo_30q | mmr_lam0.5_k3 (selected) | 0.527 | 30 |
| locomo_30q | mmr_lam0.3_k3 (selected) | 0.490 | 30 |
| locomo_30q | mmr_lam0.7_k3 (selected) | 0.614 | 30 |
| locomo_30q | mmr_lam0.5_k4 (selected) | 0.538 | 30 |
| locomo_30q | v2f_3cues (selected) | 0.608 | 30 |
| synthetic_19q | v2f_natural | 0.557 | 19 |
| synthetic_19q | mmr_lam0.5_k3 (selected) | 0.431 | 19 |
| synthetic_19q | mmr_lam0.3_k3 (selected) | 0.384 | 19 |
| synthetic_19q | mmr_lam0.7_k3 (selected) | 0.498 | 19 |
| synthetic_19q | mmr_lam0.5_k4 (selected) | 0.436 | 19 |
| synthetic_19q | v2f_3cues (selected) | 0.466 | 19 |

## Sample: 8 candidates, which 3 MMR selected

**Question** (conv=locomo_conv-26, idx=0, category=locomo_temporal): When did Caroline go to the LGBTQ support group?

baseline_r@50=1.0, arch_r@50=1.0 | selected-pairwise-cos=0.6317, all-candidates-pairwise-cos=0.6988


**Candidates:**

1.            I went to a LGBTQ support group yesterday and it was so powerful.

2.            I attended an LGBTQ+ support group yesterday

3.            Went to the LGBTQ support meeting last night and felt really supported

4. [SELECTED] Caroline mentioned she went to a support group yesterday

5.            I went to a LGBTQ support group this weekend — it was so powerful

6. [SELECTED] I attended the LGBTQ support session earlier today

7. [SELECTED] I went to the LGBTQ support group last week

8.            I went to the support group yesterday morning and it felt powerful


## Top categories by d_r@50 (mmr_lam0.5_k3 on LoCoMo-30)

Gaining:
  - locomo_single_hop (n=10): delta=+0.358 W/T/L=5/4/1
  - locomo_multi_hop (n=4): delta=+0.125 W/T/L=1/3/0
Losing:

## Verdict

**ABANDON**: mmr_lam0.5_k3 does NOT beat v2f on LoCoMo-30 @K=50 (v2f=0.858, mmr=0.678).
