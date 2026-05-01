# Ingestion-LLM Alt-Key — Empirical Recall Test

Tests whether an LLM at ingestion time, asked per-turn to decide whether to emit alt-keys, can avoid the precision collapse seen in the pure-regex alt-key experiment (see `ingestion_regex_empirical.md`).

Benchmark: **LoCoMo-30** (30 questions, 1451 segments in the LoCoMo corpus). Model: **gpt-5-mini**, prompt **v3**.

## 1. Prompt tuning (Phase 1)

- **v1**: Initial spec prompt (verbose), over-generates alt-keys for questions and acknowledgements. — SKIP rate on 30-turn sample: 60%; alt-keys emitted: 22.
- **v2**: Tightened with explicit DO-NOT list and default-SKIP; slightly under-generates (misses some facts). — SKIP rate on 30-turn sample: 87%; alt-keys emitted: 4.
- **v3**: Four named cases (anaphora / correction / alias / personal fact) with strict 5-20 word output format; chosen as the ingestion prompt. — SKIP rate on 30-turn sample: 83%; alt-keys emitted: 5.

Selected prompt: **v3** (tightest SKIP discipline with specific third-person fact restatements).

## 2. Ingestion statistics (Phase 2)

- Total turns ingested: **1451**
- SKIP turns: **738** (50.9%)
- Turns with alt-keys: **713** (49.1%)
- Total alt-keys emitted (pre-dedup): **726**
- Alt-keys after dedup by text: **722**
- Bloat factor (alt / original): **0.50x**
- Mean alt-keys per non-SKIP turn: **1.02**

## 3. Overall recall (Phase 3)

Fair-backfill recall on LoCoMo-30 at K=20 and K=50. v2f conditions backfill with cosine on the same index so every side spends exactly K segments.

| condition | mean r@20 | mean r@50 |
|---|---:|---:|
| cosine_no_altkeys | 0.3833 | 0.5083 |
| cosine_llm_altkeys | 0.5111 | 0.6611 |
| v2f_no_altkeys | 0.7556 | 0.8583 |
| v2f_llm_altkeys | 0.6861 | 0.7806 |

## 4. Per-category recall

| category | n | cos_no @20 | cos_llm @20 | v2f_no @20 | v2f_llm @20 | cos_no @50 | cos_llm @50 | v2f_no @50 | v2f_llm @50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| locomo_multi_hop | 4 | 0.500 | 0.375 | 0.625 | 0.750 | 0.500 | 0.625 | 0.875 | 0.750 |
| locomo_single_hop | 10 | 0.050 | 0.183 | 0.617 | 0.358 | 0.125 | 0.333 | 0.825 | 0.642 |
| locomo_temporal | 16 | 0.562 | 0.750 | 0.875 | 0.875 | 0.750 | 0.875 | 0.875 | 0.875 |

## 5. Verdict

- v2f Δr@20 (llm − no): **-0.0695**
- v2f Δr@50 (llm − no): **-0.0777**
- cosine Δr@20: **+0.1278**
- cosine Δr@50: **+0.1528**

Per-category lift on v2f @ K=20 (sorted):

| category | Δr@20 | sign |
|---|---:|:---:|
| locomo_multi_hop | +0.1250 | gain |
| locomo_temporal | +0.0000 | tie |
| locomo_single_hop | -0.2584 | loss |

**One-line verdict: LLM alt-keys are not worth keeping**

## 6. Precision of alt-key hits (top-50 on LoCoMo-30)

- Total top-50 hits where an alt-key out-scored the original embedding: **953** across 30 questions
- Of those, fraction whose parent turn is gold: **19.5%**
- Hits where the ORIGINAL embedding won: **547**, gold share 5.5%

## 7. Comparison to pure-regex run

| metric | regex | llm |
|---|---:|---:|
| bloat factor | 0.81x | 0.50x |
| v2f r@20 | 0.6667 | 0.6861 |
| v2f r@50 | 0.7833 | 0.7806 |
| Δ vs no-altkeys @20 | -0.0889 | -0.0695 |
| Δ vs no-altkeys @50 | -0.0750 | -0.0777 |

## 8. Cost

- Model: gpt-5-mini (prompt v3)
- Uncached LLM calls: **3**
- Cached LLM calls: **1448**
- Input tokens: **1459**
- Output tokens: **1137**
- Est. LLM cost (gpt-5-mini @ $0.25/1M in, $2/1M out): **$0.00**
- Alt-key embedding calls: **2** new embeddings (~$0.00 at $0.02/1M @ ada-002 size)

## 9. Caveats

- LoCoMo-30 only (30 questions, 3 conversations). Deltas below ~0.01 are within noise.
- Prompt v3 was selected after 3 iterations. No further prompt tuning was attempted.
- Alt-key scoring is per-parent-max over original + alt-key embeddings, identical to the regex experiment for fair comparison.
- Precision audit counts top-K hits where max came from an alt-key vs original embedding — not a strict 'alt-key caused the segment to enter top-K' test; a segment already in original top-K but whose alt-key also beats original similarity is counted under alt_hits.
- Costs are estimated from token usage at gpt-5-mini posted rates and may vary slightly from actual billing.
