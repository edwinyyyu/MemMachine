# Per-Doc Cue Salience: Eval Report

_Total wall: 59s_


## 1. Salience extractor sanity check

Sample inputs and resulting salience vectors (S/T/L/E):

| Text | S | T | L | E |
|---|---|---|---|---|
| On March 15, 2024, I had dinner with Sarah at Gusto. | 0.40 | 0.40 | 0.15 | 0.05 |
| I love hiking in the mountains. | 0.85 | 0.05 | 0.05 | 0.05 |
| Every Thursday I do tennis lessons at 6pm. | 0.20 | 0.10 | 0.65 | 0.05 |
| Back in the 90s we used to spend summers in Maine. | 0.20 | 0.05 | 0.05 | 0.70 |
| Yesterday I picked up the package from the post office. | 0.50 | 0.40 | 0.05 | 0.05 |
| Vincent Ostrom works for Indiana University Bloomington from Jan, 1964 to Jan, 1990. | 0.45 | 0.45 | 0.10 | 0.00 |
| What if I had been born in 1980? How different things would be. | 0.95 | 0.00 | 0.00 | 0.05 |

## 2. Salience distribution per benchmark

Average per-channel salience over docs:

| Benchmark | n | S | T | L | E |
|---|---|---|---|---|---|
| mixed_cue | 200 | 0.49 | 0.15 | 0.21 | 0.15 |
| dense_cluster | 100 | 0.42 | 0.50 | 0.04 | 0.04 |
| tempreason_small | 139 | 0.45 | 0.45 | 0.07 | 0.02 |
| hard_bench | 600 | 0.43 | 0.45 | 0.06 | 0.06 |

## 3. Mixed-cue: per-cue-type salience averages

| Cue type | S | T | L | E |
|---|---|---|---|---|
| date | 0.41 | 0.49 | 0.05 | 0.05 |
| content | 0.91 | 0.02 | 0.04 | 0.02 |
| recurrence | 0.20 | 0.05 | 0.70 | 0.05 |
| era | 0.43 | 0.03 | 0.05 | 0.50 |

## 4. Per-benchmark variant metrics

### mixed_cue

| Variant | subset | n | R@1 | R@3 | R@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|---|
| SEMANTIC | all | 40 | 0.925 | 0.975 | 0.975 | 0.946 | 0.959 |
| SEMANTIC | date | 10 | 0.800 | 0.900 | 0.900 | 0.850 | 0.886 |
| SEMANTIC | content | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SEMANTIC | recurrence | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SEMANTIC | era | 10 | 0.900 | 1.000 | 1.000 | 0.933 | 0.950 |
| V7 | all | 40 | 0.925 | 0.975 | 0.975 | 0.946 | 0.959 |
| V7 | date | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 | content | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 | recurrence | 10 | 0.800 | 0.900 | 0.900 | 0.850 | 0.886 |
| V7 | era | 10 | 0.900 | 1.000 | 1.000 | 0.933 | 0.950 |
| V7+salience | all | 40 | 0.725 | 0.775 | 0.825 | 0.771 | 0.798 |
| V7+salience | date | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7+salience | content | 10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7+salience | recurrence | 10 | 0.200 | 0.400 | 0.600 | 0.361 | 0.459 |
| V7+salience | era | 10 | 0.700 | 0.700 | 0.700 | 0.721 | 0.733 |

### dense_cluster

| Variant | subset | n | R@1 | R@3 | R@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|---|
| SEMANTIC | all | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 | all | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7+salience | all | 30 | 0.967 | 1.000 | 1.000 | 0.983 | 0.988 |

### tempreason_small

| Variant | subset | n | R@1 | R@3 | R@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|---|
| SEMANTIC | all | 60 | 0.450 | 0.983 | 1.000 | 0.690 | 0.770 |
| V7 | all | 60 | 0.733 | 0.983 | 1.000 | 0.846 | 0.885 |
| V7+salience | all | 60 | 0.733 | 0.983 | 1.000 | 0.849 | 0.887 |

### hard_bench

| Variant | subset | n | R@1 | R@3 | R@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|---|
| SEMANTIC | all | 75 | 0.600 | 0.733 | 0.800 | 0.689 | 0.722 |
| SEMANTIC | easy | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SEMANTIC | medium | 30 | 0.467 | 0.767 | 0.900 | 0.641 | 0.719 |
| SEMANTIC | hard | 15 | 0.067 | 0.133 | 0.200 | 0.160 | 0.170 |
| V7 | all | 75 | 0.613 | 0.653 | 0.667 | 0.657 | 0.668 |
| V7 | easy | 30 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| V7 | medium | 30 | 0.433 | 0.433 | 0.467 | 0.473 | 0.480 |
| V7 | hard | 15 | 0.200 | 0.400 | 0.400 | 0.338 | 0.383 |
| V7+salience | all | 75 | 0.320 | 0.467 | 0.507 | 0.438 | 0.467 |
| V7+salience | easy | 30 | 0.733 | 0.900 | 0.933 | 0.832 | 0.863 |
| V7+salience | medium | 30 | 0.067 | 0.200 | 0.233 | 0.195 | 0.227 |
| V7+salience | hard | 15 | 0.000 | 0.133 | 0.200 | 0.133 | 0.152 |

## 5. Hard-medium failure analysis

V7 medium-tier queries that V7 ranked > 1 and whether salience helped:

| qid | query | rank_V7 | rank_V7+sal | delta |
|---|---|---|---|---|
| q_medium_011 | When did Marcus host a workshop in Q2 2022? | 5 | 22 | -17 |
| q_medium_001 | When did Priya lead the project kickoff in Q2 2023? | 6 | 30 | -24 |
| q_medium_000 | When did Kim deliver the quarterly review in Q4 2024? | 7 | 9 | -2 |
| q_medium_028 | When did Kim move to a new office in Q1 2023? | 8 | 6 | +2 |
| q_medium_002 | When did Patel complete onboarding in Q4 2024? | 14 | 4 | +10 |

## 6. Verdict

See report body — comparison V7 vs V7+salience per benchmark.


## 7. Cost

Salience extraction: gpt-5-mini reasoning_effort=minimal. ~1k docs total at ~$0.0015/doc = ~$1.50 one-time at ingest. Retrieval cost unchanged (pure local arithmetic).
