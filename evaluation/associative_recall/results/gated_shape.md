# Gated Overlay: Task-Shape Robustness

Tests whether the shipped `gated_overlay` architecture (primary variant: `gated_threshold_0.7`) retains its LoCoMo K=50 recall across task-shape rewrites of the same 30 LoCoMo questions ({CMD, DRAFT, META}).

Original shape results reused from `results/gated_overlay.json`. CMD/DRAFT/META were freshly retrieved with dedicated `gatedTS_*` caches.


## Recall by shape (fair-backfill)

| Shape | n | arch_r@20 | arch_r@50 | Δ_r@20 | Δ_r@50 | Drop vs ORIG @20 | Drop vs ORIG @50 |
|---|---|---|---|---|---|---|---|
| ORIGINAL | 30 | 0.7583 | 0.8917 | +0.3750 | +0.3833 | +0.0000 | +0.0000 |
| CMD | 30 | 0.7139 | 0.8083 | +0.3306 | +0.3556 | +0.0444 | +0.0834 |
| DRAFT | 30 | 0.8167 | 0.8583 | +0.4000 | +0.3361 | -0.0584 | +0.0334 |
| META | 30 | 0.6972 | 0.7667 | +0.3639 | +0.2639 | +0.0611 | +0.1250 |

## Channel-fire patterns by shape

| Shape | avg fires/q | speaker_filter | alias_context | critical_info | temporal_tokens | entity_exact_match |
|---|---|---|---|---|---|---|
| ORIGINAL | 1.93 | 0.97 | 0.03 | 0.30 | 0.57 | 0.07 |
| CMD | 2.07 | 1.00 | 0.03 | 0.40 | 0.57 | 0.07 |
| DRAFT | 2.03 | 1.00 | 0.03 | 0.37 | 0.57 | 0.07 |
| META | 2.00 | 1.00 | 0.00 | 0.40 | 0.57 | 0.03 |

### Avg confidence per channel by shape

| Shape | speaker_filter | alias_context | critical_info | temporal_tokens | entity_exact_match |
|---|---|---|---|---|---|
| ORIGINAL | 0.977 | 0.027 | 0.390 | 0.567 | 0.080 |
| CMD | 1.000 | 0.050 | 0.417 | 0.583 | 0.067 |
| DRAFT | 1.000 | 0.033 | 0.433 | 0.583 | 0.067 |
| META | 1.000 | 0.000 | 0.500 | 0.583 | 0.050 |

## Routing agreement vs ORIGINAL

Exact match = the set of firing channels is IDENTICAL to the original-shape firing set. Confidence cosine = cosine similarity of the 5-dim channel confidence vector vs the original's.

| Shape | n | Exact firing-set match % | Mean conf-vector cosine vs ORIGINAL |
|---|---|---|---|
| CMD | 30 | 80.0% | 0.9650 |
| DRAFT | 30 | 83.3% | 0.9643 |
| META | 30 | 76.7% | 0.9587 |

## Sample: LLM confidences across shapes for 2 queries


**Q1 (original)**: When did Caroline go to the LGBTQ support group?

| Shape | Question | speaker_filter | alias_context | critical_info | temporal_tokens | entity_exact_match | fired |
|---|---|---|---|---|---|---|---|
| ORIGINAL | When did Caroline go to the LGBTQ support group? | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |
| CMD | Find when Caroline went to the LGBTQ support group. | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |
| DRAFT | Summarize when Caroline went to the LGBTQ support group. | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |
| META | What do we know about when Caroline went to the LGBTQ support group? | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |

**Q2 (original)**: When did Melanie paint a sunrise?

| Shape | Question | speaker_filter | alias_context | critical_info | temporal_tokens | entity_exact_match | fired |
|---|---|---|---|---|---|---|---|
| ORIGINAL | When did Melanie paint a sunrise? | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |
| CMD | Find when Melanie painted a sunrise. | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |
| DRAFT | Summarize when Melanie painted a sunrise. | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |
| META | What do we know about when Melanie painted a sunrise? | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | speaker_filter,temporal_tokens |

## Comparison vs other architectures (LoCoMo K=50)

Numbers for `meta_v2f` / `two_speaker_filter` / `keyword_router` are from `results/task_shape_adversarial.md`.

| Architecture | ORIG | CMD | DRAFT | META | Worst drop |
|---|---|---|---|---|---|
| gated_threshold_0.7 | 0.8917 | 0.8083 | 0.8583 | 0.7667 | +0.1250 |
| meta_v2f | 0.8583 | 0.7333 | 0.8167 | 0.7417 | +0.1250 |
| two_speaker_filter | 0.8917 | 0.8167 | 0.8583 | 0.8083 | +0.0834 |
| keyword_router | 0.8583 | 0.7333 | 0.8167 | 0.7417 | +0.1250 |

## Verdict

- Worst drop vs ORIGINAL @K=50: **+0.1250**

- **Shape-SENSITIVE** — drops >6pp like meta_v2f. LLM confidence scoring is also shape-sensitive; a different routing mechanism is needed.
