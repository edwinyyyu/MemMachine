# Timestamp-aware temporal scoring — LongMemEval hard

Pure-metadata temporal scoring substrate: LLM parses a structured temporal constraint per query (gpt-5-mini), then a metadata-math compatibility scorer pairs turn dates (from `haystack_dates`) with the constraint. Applied as a confidence-gated displacement channel on top of v2f.

Elapsed: 446s. Questions: 90. text-embedding-3-small + gpt-5-mini.


## Temporal-constraint detection

Overall: **26/90 = 28.9%** of queries have a parsed temporal constraint.

| Category | fired / total | fire rate |
|---|---:|---:|
| multi-session | 7/30 | 23.3% |
| single-session-preference | 7/30 | 23.3% |
| temporal-reasoning | 12/30 | 40.0% |

Temporal-type counts (among fired):
- `during`: 8
- `relative-past`: 17
- `relative-future`: 1

## Overall recall matrix (3 hard categories combined)

| Architecture | r@20 | r@50 |
|---|---:|---:|
| baseline_cosine | 0.5487 | 0.7544 |
| baseline_v2f | 0.6045 | 0.8170 |
| tsscore_v2f | 0.5951 | 0.7850 |
| tsscore_strict | 0.4931 | 0.6783 |
| tsscore_soft_boost | 0.5934 | 0.7965 |

## Per-category recall @K=20

| Architecture | multi-session | single-session-preference | temporal-reasoning |
|---|---:|---:|---:|
| baseline_cosine | 0.5210 | 0.6360 | 0.4893 |
| baseline_v2f | 0.5519 | 0.7546 | 0.5069 |
| tsscore_v2f | 0.5492 | 0.7257 | 0.5104 |
| tsscore_strict | 0.4662 | 0.5289 | 0.4841 |
| tsscore_soft_boost | 0.5475 | 0.7325 | 0.5002 |

## Per-category recall @K=50

| Architecture | multi-session | single-session-preference | temporal-reasoning |
|---|---:|---:|---:|
| baseline_cosine | 0.7596 | 0.7875 | 0.7160 |
| baseline_v2f | 0.8179 | 0.8678 | 0.7653 |
| tsscore_v2f | 0.7964 | 0.8276 | 0.7309 |
| tsscore_strict | 0.6935 | 0.6497 | 0.6917 |
| tsscore_soft_boost | 0.8025 | 0.8434 | 0.7435 |

## Δ vs baseline_v2f per category @K=20

| Architecture | Δ multi-session | Δ single-session-preference | Δ temporal-reasoning |
|---|---:|---:|---:|
| baseline_cosine | -0.0309 | -0.1186 | -0.0176 |
| tsscore_v2f | -0.0027 | -0.0289 | +0.0035 |
| tsscore_strict | -0.0857 | -0.2257 | -0.0228 |
| tsscore_soft_boost | -0.0044 | -0.0221 | -0.0067 |

## Δ vs baseline_v2f per category @K=50

| Architecture | Δ multi-session | Δ single-session-preference | Δ temporal-reasoning |
|---|---:|---:|---:|
| baseline_cosine | -0.0583 | -0.0803 | -0.0493 |
| tsscore_v2f | -0.0215 | -0.0402 | -0.0344 |
| tsscore_strict | -0.1244 | -0.2181 | -0.0736 |
| tsscore_soft_boost | -0.0154 | -0.0244 | -0.0218 |

## Sample temporal queries (parse + retrieval effect)

### gpt4_468eb064
- **question**: Who did I meet with during the lunch last Tuesday?
- **question_date**: 2023-04-18
- **parsed**: type=`during`  ref_date=`2023-04-11`  window=2  uses_qdate=False
- **recall@50**: cosine=0.250  v2f=0.750  ts_v2f=1.000  ts_strict=0.333  ts_boost=0.417

### gpt4_e072b769
- **question**: How many weeks ago did I start using the cashback app 'Ibotta'?
- **question_date**: 2023-05-06
- **parsed**: type=`relative-past`  ref_date=`None`  window=30  uses_qdate=True
- **recall@50**: cosine=1.000  v2f=1.000  ts_v2f=1.000  ts_strict=1.000  ts_boost=1.000


## Verdict

On **temporal-reasoning @K=50**: baseline_v2f=0.7653, best tsscore variant = **tsscore_soft_boost = 0.7435** (Δ=-0.0218).

**VERDICT: ABANDON (hurts)** — metadata channel regresses temporal-reasoning.

### Regression check on non-target categories @K=50

| Category | v2f | ts_v2f | Δ | ts_strict | Δ | ts_boost | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi-session | 0.818 | 0.796 | -0.021 | 0.694 | -0.124 | 0.802 | -0.015 |
| single-session-preference | 0.868 | 0.828 | -0.040 | 0.650 | -0.218 | 0.843 | -0.024 |
| temporal-reasoning | 0.765 | 0.731 | -0.034 | 0.692 | -0.074 | 0.744 | -0.022 |

### Fire-only analysis (only queries where constraint fired)

Recall@50 averaged over the subset of queries where the LLM parser emitted a temporal constraint.

| Category | n_fired | v2f | ts_v2f | Δ | ts_strict | Δ | ts_boost | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi-session | 7 | 0.803 | 0.711 | -0.092 | 0.441 | -0.362 | 0.737 | -0.066 |
| single-session-preference | 7 | 0.826 | 0.654 | -0.173 | 0.119 | -0.707 | 0.721 | -0.105 |
| temporal-reasoning | 12 | 0.673 | 0.587 | -0.086 | 0.529 | -0.144 | 0.619 | -0.054 |

### Why the channel underperforms

Inspection of fire-case per-question deltas reveals the core failure mode: **event occurrence date ≠ mention date**. LongMemEval's gold `source_ids` are derived from sessions labelled by the dataset as containing evidence for the answer — which often includes MULTIPLE past sessions where the user mentioned related events across a wide time range, not only the session(s) that literally match the temporal phrase. E.g. `gpt4_d6585ce9`: "Who did I go with to the music event last Saturday?" — the gold sessions span FIVE Saturdays (3/18, 3/25, 4/1, 4/8, 4/15); the temporal parser correctly narrows to 4/15 but v2f's broader retrieval hits more gold turns. The metadata channel is "too accurate" for the metric — it correctly identifies the primary session but the gold-recall target rewards breadth.

The three exceptions where tsscore_v2f lifts (gpt4_468eb064 "lunch last Tuesday": v2f=0.75 → ts_v2f=1.00; 4dfccbf8 "Wednesday two months ago": ts_strict=0.33 vs v2f=0.17; a few wins on relative-past with window=30) are genuine, but the larger hits from narrowing on `during` with tight windows dominate the aggregate.
