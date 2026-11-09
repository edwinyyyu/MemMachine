# Embedding prompt setups: longmemeval_m · seg500 · 200 questions · gemma300m

First stage only, no reranker. Each question searches its evidence sessions plus 25 other sessions of its own history (EventMemory segments, order, dates and embedding text), so absolute values are higher than over the whole history; the comparison between setups is the point. every@k: every gold turn complete within the top k segments; first@k: some gold turn; MRR: of the first gold segment. Δ: minus the model card's setup, 95% paired bootstrap over questions.

| setup | every@5 | every@10 | every@20 | every@50 | first@5 | first@10 | first@20 | first@50 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| both prompts (model card) | 0.455 | 0.640 | 0.775 | 0.900 | 0.810 | 0.885 | 0.955 | 0.985 | 0.617 |
| query prompt only (MemMachine sentence-transformers) | 0.435 | 0.615 | 0.780 | 0.895 | 0.800 | 0.880 | 0.960 | 0.985 | 0.610 |
| no prompts (MemMachine OpenAI-compatible) | 0.490 | 0.630 | 0.810 | 0.890 | 0.785 | 0.895 | 0.955 | 0.980 | 0.663 |
| document prompt only | 0.505 | 0.655 | 0.800 | 0.900 | 0.795 | 0.890 | 0.945 | 0.985 | 0.679 |

| Δ vs model card | every@5 | every@10 | every@20 | every@50 | first@5 | first@10 | first@20 | first@50 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| query prompt only (MemMachine sentence-transformers) | -0.020 [-0.055, +0.010] | -0.025 [-0.055, +0.005] | +0.005 [-0.010, +0.020] | -0.005 [-0.025, +0.015] | -0.010 [-0.040, +0.015] | -0.005 [-0.025, +0.010] | +0.005 [+0.000, +0.015] | +0.000 [+0.000, +0.000] | -0.007 [-0.029, +0.014] |
| no prompts (MemMachine OpenAI-compatible) | +0.035 [-0.005, +0.080] | -0.010 [-0.055, +0.035] | +0.035 [-0.010, +0.075] | -0.010 [-0.050, +0.030] | -0.025 [-0.075, +0.020] | +0.010 [-0.025, +0.040] | +0.000 [-0.020, +0.020] | -0.005 [-0.015, +0.000] | +0.046 [+0.002, +0.086] |
| document prompt only | +0.050 [+0.010, +0.095] | +0.015 [-0.025, +0.060] | +0.025 [-0.015, +0.070] | +0.000 [-0.035, +0.040] | -0.015 [-0.065, +0.030] | +0.005 [-0.025, +0.035] | -0.010 [-0.035, +0.015] | +0.000 [+0.000, +0.000] | +0.063 [+0.026, +0.100] |

Summary (every@k averaged over k = 5, 10, 20, 50): both prompts (model card) 0.693; query prompt only (MemMachine sentence-transformers) 0.681; no prompts (MemMachine OpenAI-compatible) 0.705; document prompt only 0.715.

**Each prompt's own effect** (the four setups are a 2 × 2 design, so each prompt's effect is the mean of the setups with it minus the mean of those without it, paired by question):

| measure | query prompt | document prompt | document prompt only − no prompts |
|---|---|---|---|
| every@k averaged | -0.023 [-0.045, -0.001] | +0.011 [-0.001, +0.022] | +0.010 [-0.003, +0.024] |
| every@5 | -0.052 [-0.092, -0.018] | +0.018 [-0.005, +0.043] | +0.015 [-0.015, +0.045] |
| MRR | -0.058 [-0.093, -0.022] | +0.012 [-0.006, +0.032] | +0.017 [-0.009, +0.043] |

A prompt whose intervals exclude 0 matters; one whose intervals include 0 cannot be told apart from none on this sample. Choosing among setups on the evaluated questions can only flatter the first stage, which makes a reranker's measured gain conservative.

