# Answer-Type Aware Reranking — Empirical Recall Test

Classify each query's expected answer type (DATE, PERSON, NUMBER, LOCATION, REASON, DESCRIPTION) via rules (gpt-5-mini fallback for ambiguous what/which). Rerank v2f's already-retrieved candidates by adding an alpha bonus to turns containing answer-type tokens, or (in hard-filter mode) promoting matches to the front. DESCRIPTION queries are left un-reranked (no informative filter).

## 1. Answer-type distribution

| dataset | DATE | PERSON | NUMBER | LOCATION | REASON | DESCRIPTION |
|---|---:|---:|---:|---:|---:|---:|
| locomo_30q | 14 (47%) | 0 (0%) | 2 (7%) | 2 (7%) | 0 (0%) | 12 (40%) |
| synthetic_19q | 5 (26%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 14 (74%) |

## 2. Recall (fair-backfill)

| dataset | K | baseline v2f | atr_bonus_0.05 | Δ | W/T/L | atr_bonus_0.1 | Δ | W/T/L | atr_bonus_0.2 | Δ | W/T/L | atr_hard_filter | Δ | W/T/L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| locomo_30q | 20 | 0.7556 | 0.7667 | +0.0111 | 1/29/0 | 0.7667 | +0.0111 | 1/29/0 | 0.7667 | +0.0111 | 1/29/0 | 0.7667 | +0.0111 | 1/29/0 |
| locomo_30q | 50 | 0.8583 | 0.8583 | +0.0000 | 0/30/0 | 0.8583 | +0.0000 | 0/30/0 | 0.8583 | +0.0000 | 0/30/0 | 0.8583 | +0.0000 | 0/30/0 |
| synthetic_19q | 20 | 0.6130 | 0.6130 | +0.0000 | 0/19/0 | 0.6130 | +0.0000 | 0/19/0 | 0.6130 | +0.0000 | 0/19/0 | 0.5516 | -0.0614 | 0/16/3 |
| synthetic_19q | 50 | 0.8513 | 0.8513 | +0.0000 | 0/19/0 | 0.8513 | +0.0000 | 0/19/0 | 0.8513 | +0.0000 | 0/19/0 | 0.8513 | +0.0000 | 0/19/0 |

## 3. Per-category — locomo_30q

| category | n | base@20 | atr_bonus_0.05@20 | atr_bonus_0.1@20 | atr_bonus_0.2@20 | atr_hard_filter@20 | base@50 | atr_bonus_0.05@50 | atr_bonus_0.1@50 | atr_bonus_0.2@50 | atr_hard_filter@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| locomo_multi_hop | 4 | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| locomo_single_hop | 10 | 0.617 | 0.650 | 0.650 | 0.650 | 0.650 | 0.825 | 0.825 | 0.825 | 0.825 | 0.825 |
| locomo_temporal | 16 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |

## 3. Per-category — synthetic_19q

| category | n | base@20 | atr_bonus_0.05@20 | atr_bonus_0.1@20 | atr_bonus_0.2@20 | atr_hard_filter@20 | base@50 | atr_bonus_0.05@50 | atr_bonus_0.1@50 | atr_bonus_0.2@50 | atr_hard_filter@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| completeness | 4 | 0.455 | 0.455 | 0.455 | 0.455 | 0.413 | 0.865 | 0.865 | 0.865 | 0.865 | 0.865 |
| conjunction | 3 | 0.809 | 0.809 | 0.809 | 0.809 | 0.643 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| control | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| inference | 3 | 0.766 | 0.766 | 0.766 | 0.766 | 0.600 | 0.939 | 0.939 | 0.939 | 0.939 | 0.939 |
| proactive | 4 | 0.351 | 0.351 | 0.351 | 0.351 | 0.351 | 0.643 | 0.643 | 0.643 | 0.643 | 0.643 |
| procedural | 2 | 0.347 | 0.347 | 0.347 | 0.347 | 0.347 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 |

## 4. Sample rerank effect — locomo_30q

**Question:** When did Caroline go to the LGBTQ support group?

**Answer type:** DATE  
**Source turn_ids:** [2]

### Before (v2f top-5)

| rank | turn_id | role | has_at_token | score | text |
|---:|---:|---|---:|---:|---|
| 1 | 184 | assistant | no | 10.000 | Wow, Caroline! They must have felt so appreciated. It's awesome to see the difference we can make in each other's lives. Any other exciting LGBTQ advocacy stuff |
| 2 | 77 | assistant | no | 9.999 | Wow, Caroline, sounds like the parade was an awesome experience! It's great to see the love and support for the LGBTQ+ community. Congrats! Has this experience  |
| 3 | 36 | assistant | no | 9.998 | Hey Caroline! Great to hear from you. Sounds like your event was amazing! I'm so proud of you for spreading awareness and getting others involved in the LGBTQ c |
| 4 | 304 | assistant | no | 9.997 | Wow, Caroline, that's awesome! Can't wait to see your show - the LGBTQ community needs more platforms like this! |
| 5 | 2 | user | yes | 9.996 | I went to a LGBTQ support group yesterday and it was so powerful. |

### After (atr_bonus_0.1 top-5)

| rank | turn_id | role | has_at_token | score | text |
|---:|---:|---|---:|---:|---|
| 1 | 2 | user | yes | 10.096 | I went to a LGBTQ support group yesterday and it was so powerful. |
| 2 | 76 | user | yes | 10.090 | Since we last spoke, some big things have happened. Last week I went to an LGBTQ+ pride parade. Everyone was so happy and it made me feel like I belonged. It sh |
| 3 | 108 | user | yes | 10.089 | Hey Mel, great to chat with you again! So much has happened since we last spoke - I went to an LGBTQ conference two days ago and it was really special. I got th |
| 4 | 70 | user | yes | 10.087 | I'm still figuring out the details, but I'm thinking of working with trans people, helping them accept themselves and supporting their mental health. Last Frida |
| 5 | 193 | user | yes | 10.086 | Hey Mel! A lot's happened since we last chatted - I just joined a new LGBTQ activist group last Tues. I'm meeting so many cool people who are as passionate as I |

## 5. Verdict

- locomo_30q: best variant = atr_bonus_0.05@K=20 (+0.0111). Ship/narrow-use.
- synthetic_19q: no variant beats baseline. Abandon on this dataset.
