# Embedding width: qwen3-8b-di · longmemeval_m · seg500 · 100 questions

- One full-width embedding (4,096 dimensions) per text; each width keeps the leading dimensions, re-normalized.
- Reduced haystacks: each question's evidence sessions plus 25 others. Δ: minus 4,096, with its 95% paired interval over questions.

| width | every@5 | every@10 | every@20 | every@50 | first@5 | first@10 | first@20 | first@50 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| 4,096 | 0.310 | 0.460 | 0.680 | 0.870 | 0.690 | 0.800 | 0.920 | 1.000 | 0.441 |
| 2,048 | 0.320 | 0.470 | 0.690 | 0.880 | 0.720 | 0.810 | 0.940 | 1.000 | 0.461 |
| 1,024 | 0.320 | 0.460 | 0.700 | 0.840 | 0.690 | 0.800 | 0.920 | 0.990 | 0.442 |

| Δ vs 4,096 | every@5 | every@10 | every@20 | every@50 | first@5 | first@10 | first@20 | first@50 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| 2,048 | +0.010 [-0.020, +0.050] | +0.010 [-0.030, +0.050] | +0.010 [-0.040, +0.070] | +0.010 [-0.020, +0.050] | +0.030 [-0.010, +0.070] | +0.010 [-0.020, +0.040] | +0.020 [+0.000, +0.050] | +0.000 [+0.000, +0.000] | +0.020 [-0.001, +0.047] |
| 1,024 | +0.010 [+0.000, +0.030] | +0.000 [-0.040, +0.040] | +0.020 [-0.030, +0.070] | -0.030 [-0.070, +0.000] | +0.000 [-0.070, +0.060] | +0.000 [-0.030, +0.030] | +0.000 [-0.030, +0.030] | -0.010 [-0.030, +0.000] | +0.001 [-0.027, +0.028] |

**Decision** (keep 1,024 unless clearly worse: the interval excludes 0 and the loss is at least 0.01): every@k averaged: +0.000 [-0.018, +0.020]; MRR: +0.001 [-0.027, +0.028]. **Keep 1,024.**

