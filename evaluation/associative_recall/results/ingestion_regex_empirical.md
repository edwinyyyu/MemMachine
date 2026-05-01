# Ingestion-Regex Alt-Key — Empirical Recall Test

This report tests whether applying the 7 cheap regex heuristics from `ingestion_predictability.md` §7 at INGESTION time — generating alt-keys that are embedded alongside the original segment — actually lifts retrieval recall, or whether the extra index mass washes out the signal.

Benchmark: **LoCoMo-30** (30 questions, 1451 segments in the LoCoMo-retrievable corpus).

## 1. Index bloat

- Original segments in LoCoMo corpus: **1451**
- Alt-keys generated (deduped by text): **1178**
- Bloat factor (alt / original): **0.81x**
- Fraction of segments that fire at least one heuristic: **69.2%**

### Per-heuristic fire counts on the full LoCoMo corpus

| heuristic | fires | % of segments |
|---|---:|---:|
| anaphoric | 53 | 3.7% |
| short_response | 252 | 17.4% |
| update_marker | 3 | 0.2% |
| known_unknown | 8 | 0.6% |
| alias_evolution | 0 | 0.0% |
| structured_fact | 2 | 0.1% |
| rare_entity | 861 | 59.3% |

## 2. Overall recall

Fair-backfill recall on LoCoMo-30 at K=20 and K=50. For v2f conditions, any budget unused by v2f-found segments is backfilled by cosine on the same index, so all sides spend exactly K segments.

| condition | mean r@20 | mean r@50 |
|---|---:|---:|
| cosine_no_altkeys | 0.3833 | 0.5083 |
| cosine_with_altkeys | 0.3000 | 0.4111 |
| v2f_no_altkeys | 0.7556 | 0.8583 |
| v2f_with_altkeys | 0.6667 | 0.7833 |

## 3. Per-category recall

LoCoMo-30's question-categories are just the three native LoCoMo categories (`locomo_single_hop`, `locomo_multi_hop`, `locomo_temporal`). The 22-category breakdown referenced in `ingestion_predictability.md` §6 is from the advanced_23q benchmark, not LoCoMo; this test uses the dataset the task specifies.

| category | n | cos_no @20 | cos_with @20 | v2f_no @20 | v2f_with @20 | cos_no @50 | cos_with @50 | v2f_no @50 | v2f_with @50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| locomo_multi_hop | 4 | 0.500 | 0.125 | 0.625 | 0.500 | 0.500 | 0.375 | 0.875 | 0.500 |
| locomo_single_hop | 10 | 0.050 | 0.050 | 0.617 | 0.500 | 0.125 | 0.083 | 0.825 | 0.750 |
| locomo_temporal | 16 | 0.562 | 0.500 | 0.875 | 0.812 | 0.750 | 0.625 | 0.875 | 0.875 |

## 4. Verdict

- v2f lift from alt-keys @ K=20: **-0.0889**
- v2f lift from alt-keys @ K=50: **-0.0750**
- cosine lift from alt-keys @ K=20: **-0.0833**
- cosine lift from alt-keys @ K=50: **-0.0972**

Per-category alt-key lift on v2f @ K=20 (sorted):

| category | Δr@20 | sign |
|---|---:|:---:|
| locomo_temporal | -0.0625 | loss |
| locomo_single_hop | -0.1167 | loss |
| locomo_multi_hop | -0.1250 | loss |

Summary: 0 category gained, 3 categories lost, 0 tied.

**One-line verdict: pure-regex ingestion is not worth keeping**

## 5. False-positive / precision notes

The §7 heuristics have high recall on known missed turns but unknown precision on the full corpus. Two heuristics are dominant bloat drivers; their fire-on-likely-irrelevant-turn rate is estimated below by sampling turns that fired them and are NOT in any gold source set.

| heuristic | fires | turns NOT in any gold | est. FP share |
|---|---:|---:|---:|
| anaphoric | 53 | 49 | 92.5% |
| short_response | 252 | 248 | 98.4% |
| update_marker | 3 | 3 | 100.0% |
| known_unknown | 8 | 7 | 87.5% |
| alias_evolution | 0 | 0 | 0.0% |
| structured_fact | 2 | 2 | 100.0% |
| rare_entity | 861 | 835 | 97.0% |

## 6. Caveats

- Only 30 questions, from one LoCoMo conversation. Deltas below ~0.01 are within noise.
- Regex `by (day)` is interpreted as "by <weekday>" (case-insensitive, word-boundary); other interpretations are plausible.
- The `rare_entity` heuristic as specified emits every capitalized-not-sentence-initial token plus number/version tokens. True corpus-rare filtering is not possible at streaming ingest time and is NOT applied here; this means rare_entity is intentionally noisy, matching the analysis's §9 caveat.
- The anaphoric heuristic fires only on the pronoun set given in §7. A handful of `ingestion_predictability.md` examples (e.g. "Yeah, 16 weeks...") were labeled anaphoric in that report but do NOT match the literal first-token pronoun spec; our implementation follows the spec.
- Alt-key scoring is per-parent-max over original + alt-key embeddings. This is strictly non-decreasing in cosine for any single cosine query on any segment — so cosine_with_altkeys cannot DROP pure recall, only raise it or tie. Losses at fixed K come from non-gold segments being boosted past gold ones.
- v2f uses MetaV2f (gpt-5-mini) via the shared best-shot LLM cache. No new LLM calls are required for previously-cached questions.
