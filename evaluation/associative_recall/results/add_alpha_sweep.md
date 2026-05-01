# Add-only Embedding Steering: Alpha Sweep

ADD-only evidence-grounded steering on LME-hard-POC (30 questions). Prior v2 add-only at alpha=0.1 hit -1.3pp (0.8040 vs 0.8169). This sweep asks: does a smaller alpha find positive territory, and does score-merge beat embedding arithmetic?

Fixed: text-embedding-3-small, gpt-5-mini (via `em_v2f_lme_mixed_7030` + `expand_3`), 30 Qs, reuses steerv2 LLM/embedding caches.

## Recall by variant

| Variant | mode | alpha | rounds | R@20 | R@50 | Δ R@20 vs baseline | Δ R@50 vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `add_baseline` | baseline | -- | -- | 0.6303 | 0.8169 | +0.0000 | +0.0000 |
| `add_a0.01_1r` | arithmetic | 0.01 | 1 | 0.6386 | 0.8124 | +0.0083 | -0.0045 |
| `add_a0.03_1r` | arithmetic | 0.03 | 1 | 0.6485 | 0.8150 | +0.0182 | -0.0019 |
| `add_a0.05_1r` | arithmetic | 0.05 | 1 | 0.6468 | 0.8099 | +0.0165 | -0.0070 |
| `add_a0.1_1r` | arithmetic | 0.10 | 1 | 0.6454 | 0.8143 | +0.0151 | -0.0026 |
| `add_a0.2_1r` | arithmetic | 0.20 | 1 | 0.6538 | 0.8068 | +0.0235 | -0.0101 |
| `add_a0.05_3r` | arithmetic | 0.05 | 3 | 0.6519 | 0.8089 | +0.0216 | -0.0080 |
| `add_score_merge` | score_merge | -- | 1 | 0.6422 | 0.8349 | +0.0119 | +0.0180 |

## Round-by-round R@50 (arithmetic variants)

| Variant | rd 0 | rd 1 | rd 2 | rd 3 | drift@final |
| --- | --- | --- | --- | --- | --- |
| `add_a0.01_1r` | 0.8169 | 0.8124 | -- | -- | 0.9998 |
| `add_a0.03_1r` | 0.8169 | 0.8150 | -- | -- | 0.9985 |
| `add_a0.05_1r` | 0.8169 | 0.8099 | -- | -- | 0.9960 |
| `add_a0.1_1r` | 0.8169 | 0.8143 | -- | -- | 0.9858 |
| `add_a0.2_1r` | 0.8169 | 0.8068 | -- | -- | 0.9559 |
| `add_a0.05_3r` | 0.8169 | 0.8099 | 0.8143 | 0.8089 | 0.9694 |

## Category R@50

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `add_baseline` | 0.7398 | 0.9021 | 0.8089 |
| `add_a0.01_1r` | 0.7262 | 0.9021 | 0.8089 |
| `add_a0.03_1r` | 0.7382 | 0.9021 | 0.8047 |
| `add_a0.05_1r` | 0.7257 | 0.9021 | 0.8019 |
| `add_a0.1_1r` | 0.7223 | 0.9187 | 0.8019 |
| `add_a0.2_1r` | 0.7140 | 0.9187 | 0.7877 |
| `add_a0.05_3r` | 0.7203 | 0.9187 | 0.7877 |
| `add_score_merge` | 0.7963 | 0.9104 | 0.7979 |

## Score-merge vs arithmetic

- `add_score_merge` R@20 = 0.6422, R@50 = 0.8349
- best arithmetic variant: `add_a0.03_1r` R@50 = 0.8150
- Δ(score_merge - best_arith) R@50 = +0.0199

## Verdict

- baseline R@50: 0.8169
- best arithmetic: `add_a0.03_1r` (Δ R@50 = -0.0019)
- score-merge: `add_score_merge` (Δ R@50 = +0.0180)

**Additive arithmetic is substrate-incompatible, just less dramatically than subtractive** (best Δ R@50 = -0.19pp, nothing clears the +0.5pp bar).
Score-merge beats arithmetic by 1.99pp — confirms `additive = separate probe merged by score` is the right formulation, not embedding arithmetic. This is essentially what v2f already does.

## Outputs

- JSON: `results/add_alpha_sweep.json`
- Source: `add_alpha_sweep.py` (framework files untouched)
- Caches: `cache/addsweep_llm_cache.json`, `cache/addsweep_embedding_cache.json` (reads from `steerv2_*` caches for warm hits).
