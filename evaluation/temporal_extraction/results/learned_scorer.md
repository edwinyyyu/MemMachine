# F1 — Learned Relevance Scorer

## Dataset
- Picked queries: 20
- Labelled pairs: 400
  (train 320 / test 80)
- Train positive rate: 0.284  |  Test positive rate: 0.125

## Model-level metrics

| Model | Test AUC | R² (reg) |
| --- | --- | --- |
| LogisticRegression | 0.729 | — |
| MLPClassifier(16)  | 0.647 | — |
| LinearRegression   | — | -1.939 |
| MLPRegressor(16)   | — | -2.290 |

## Held-out retrieval metrics

| System | R@5 | R@10 | MRR | NDCG@10 |
| --- | --- | --- | --- | --- |
| judge_oracle_plus_sem | 0.841 | 0.864 | 1.000 | 0.908 |
| judge_oracle | 0.841 | 0.864 | 1.000 | 0.909 |
| hand_crafted | 0.591 | 0.784 | 1.000 | 0.800 |
| semantic | 0.568 | 0.568 | 1.000 | 0.660 |
| logreg | 0.466 | 0.614 | 0.759 | 0.570 |
| mlp_cls | 0.341 | 0.489 | 0.759 | 0.505 |
| linreg | 0.670 | 0.693 | 1.000 | 0.710 |
| mlp_reg | 0.148 | 0.193 | 0.522 | 0.244 |

## Train (in-sample) retrieval metrics — sanity

| System | R@5 | R@10 | MRR | NDCG@10 |
| --- | --- | --- | --- | --- |
| judge_oracle_plus_sem | 0.609 | 0.686 | 0.865 | 0.688 |
| judge_oracle | 0.609 | 0.654 | 0.891 | 0.691 |
| hand_crafted | 0.542 | 0.876 | 0.650 | 0.684 |
| semantic | 0.432 | 0.526 | 0.704 | 0.485 |
| logreg | 0.676 | 0.710 | 0.836 | 0.710 |
| mlp_cls | 0.502 | 0.556 | 0.794 | 0.583 |
| linreg | 0.523 | 0.589 | 0.904 | 0.598 |
| mlp_reg | 0.317 | 0.394 | 0.519 | 0.384 |

## Feature importances (logistic regression, standardized coefs)

| Feature | Coef |
| --- | --- |
| best_proximity_log | -1.432 |
| max_pair_score_jaccard | +1.065 |
| num_q_exprs | -0.963 |
| jaccard_bracket | -0.838 |
| query_length_chars | -0.819 |
| granularity_gap | +0.488 |
| granularity_compat | -0.488 |
| num_d_exprs | -0.311 |
| has_recurrence_instance | +0.292 |
| max_pair_score_gaussian | +0.269 |
| best_proximity_sec | +0.219 |
| doc_length_chars | +0.195 |
| semantic_cosine | -0.103 |
| has_anchor_match | +0.008 |

## Decision

- Best learned scorer (held-out): **linreg** at R@5 = 0.670 vs hand-crafted 0.591 (Δ = +8.0 pp, 32% of the 25.0-pp gap to judge-oracle 0.841).
- Test set is only 4 queries / 80 labelled pairs — signal is noisy; treat directional.
- MLP variants underfit/overfit (tiny label set, 14 features).
- Dominant LR features: best_proximity_log (−), max_pair_score_jaccard (+), num_q_exprs (−), granularity_gap (+).
