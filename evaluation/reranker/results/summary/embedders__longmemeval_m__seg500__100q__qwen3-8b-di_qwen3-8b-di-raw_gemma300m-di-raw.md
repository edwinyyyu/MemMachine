# First stages: longmemeval_m · seg500 · 100 questions

Each question searches its evidence sessions plus 25 other sessions of its own history, embedded as EventMemory embeds them; absolute values are higher than over the whole history. Δ: minus qwen3-8b-di, 95% paired bootstrap over questions.

| embedder | every@5 | every@10 | every@20 | every@50 | first@5 | first@10 | first@20 | first@50 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b-di | 0.320 | 0.460 | 0.690 | 0.840 | 0.690 | 0.800 | 0.920 | 0.990 | 0.447 |
| qwen3-8b-di-raw | 0.290 | 0.420 | 0.620 | 0.780 | 0.580 | 0.790 | 0.900 | 0.950 | 0.433 |
| gemma300m-di-raw | 0.560 | 0.640 | 0.800 | 0.890 | 0.820 | 0.910 | 0.950 | 0.990 | 0.653 |

| Δ vs qwen3-8b-di | every@5 | every@10 | every@20 | every@50 | first@5 | first@10 | first@20 | first@50 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b-di-raw | -0.030 [-0.100, +0.040] | -0.040 [-0.120, +0.040] | -0.070 [-0.130, -0.010] | -0.060 [-0.110, -0.010] | -0.110 [-0.200, -0.020] | -0.010 [-0.090, +0.060] | -0.020 [-0.080, +0.030] | -0.040 [-0.080, -0.010] | -0.014 [-0.068, +0.039] |
| gemma300m-di-raw | +0.240 [+0.150, +0.340] | +0.180 [+0.090, +0.270] | +0.110 [+0.020, +0.200] | +0.050 [-0.010, +0.120] | +0.130 [+0.030, +0.220] | +0.110 [+0.030, +0.180] | +0.030 [-0.020, +0.080] | +0.000 [+0.000, +0.000] | +0.206 [+0.128, +0.284] |
