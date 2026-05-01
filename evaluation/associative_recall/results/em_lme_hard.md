# EventMemory on LongMemEval-hard (90 questions)

## Setup

- n_questions = 90 (30 multi-session + 30 single-session-preference + 30 temporal-reasoning)
- n_events_total = 41952
- ingest time = 162.0s (concurrency=3)
- segment store = `sqlite+aiosqlite:////Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/eventmemory_lme.sqlite3`
- namespace = `arc_em_lmehard`, collection prefix = `arc_em_lmehard_v1_<question_id>`
- speaker baking: `User` / `Assistant` via MessageContext.source
- timestamps: haystack_dates (per session) + monotonic +1s per turn
- queries/cues: prepended with `User: ` before embedding
- embedder = `text-embedding-3-small`, reranker=None, derive_sentences=False, max_text_chunk_length=500

## Per-architecture summary

| Architecture | R@20 | R@50 | time (s) |
| --- | --- | --- | --- |
| `em_cosine_baseline_userprefix` | 0.5393 | 0.6509 | 3.2 |
| `em_v2f_userprefix` | 0.6094 | 0.7797 | 190.6 |
| `em_v2f_expand_3` | 0.6154 | 0.8317 | 18.9 |
| `em_v2f_expand_6` | 0.6113 | 0.8313 | 15.8 |
| `em_ens_2_userprefix` | 0.5258 | 0.7349 | 260.0 |

## Recall matrix (R@20)

| Architecture | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `em_cosine_baseline_userprefix` | 0.5094 | 0.6210 | 0.4875 |
| `em_v2f_userprefix` | 0.5951 | 0.7064 | 0.5267 |
| `em_v2f_expand_3` | 0.5777 | 0.7664 | 0.5022 |
| `em_v2f_expand_6` | 0.5727 | 0.7713 | 0.4898 |
| `em_ens_2_userprefix` | 0.4734 | 0.6579 | 0.4462 |

## Recall matrix (R@50)

| Architecture | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `em_cosine_baseline_userprefix` | 0.6261 | 0.7343 | 0.5923 |
| `em_v2f_userprefix` | 0.7807 | 0.8481 | 0.7102 |
| `em_v2f_expand_3` | 0.8415 | 0.8852 | 0.7684 |
| `em_v2f_expand_6` | 0.8427 | 0.8804 | 0.7708 |
| `em_ens_2_userprefix` | 0.7255 | 0.8068 | 0.6725 |

## Side-by-side with prior SegmentStore v2f (LME-hard K=50)

SS v2f baseline (reference): overall=0.817, multi-session=0.818, single-session-preference=0.868, temporal-reasoning=0.765

EM v2f_userprefix K=50 overall Δ = -0.037, temporal-reasoning Δ = -0.055
EM v2f_expand_3 temporal-reasoning K=50 = 0.768 (SS baseline 0.765, Δ=+0.003)

## Findings

### Expand-context is the decisive lever (K=50)

- `em_v2f_userprefix` (expand=0): overall 0.780; temporal-reasoning 0.710
- `em_v2f_expand_3`: overall 0.832 (Δ vs expand=0: +0.052); temporal-reasoning 0.768
- `em_v2f_expand_6`: overall 0.831 (Δ vs expand=0: +0.052)

Expand_context gives a clean +5pp at K=50 across all categories; expand=6 saturates with expand=3 (no additional headroom past 3 neighbors per seed).  Opposite finding from LoCoMo, where expand_context REGRESSED recall — LME's long multi-session haystacks (~470 turns/q) hide gold across many adjacent turns that can be picked up by timestamp walk; LoCoMo's dense two-speaker conversations do not.

### User-prefix at expand=0 is weaker than SS v2f

- SS v2f on LME-hard K=50 (reference): overall=0.817
- EM `em_v2f_userprefix` K=50: 0.780 (Δ = -0.037)

Speaker-baking alone ("User: ..." prefix into embedded text) did NOT beat the SS substrate at matched K — it slightly underperformed. The substrate-level benefit only appears when paired with expand_context.

### Temporal-reasoning: substrate ceiling nearly untouched

- SS v2f K=50 on temporal-reasoning: 0.765
- EM v2f_expand_3 K=50: 0.768 (Δ = +0.003)
- EM v2f_expand_6 K=50: 0.771

The hypothesis that timestamp-walking via expand_context would break the 0.765 temporal-reasoning ceiling is NOT supported: lift is within noise (<1pp).  Temporal-reasoning questions appear to need actual temporal *reasoning*, not just temporal-adjacent chunk inclusion.

### Ensemble regresses on LME-hard

`em_ens_2_userprefix` K=50 overall = 0.735 vs `em_v2f_userprefix` 0.780 (Δ = -0.045). Adding the 7 type_enumerated cues over the 2 v2f cues dilutes the top-K ranking via sum_cosine: type_enumerated was designed for LoCoMo's scattered-constraint register (ARRIVAL, PREFERENCE, RESOLUTION, etc.) and its cues land on non-gold high-cosine distractors in LME.

## Verdict (LME-style corpora recipe)

1. Use EventMemory with speaker baking (`MessageContext.source = User|Assistant`).
2. Prepend `"User: "` to queries and cues before embedding.
3. Generate v2f cues (2 per question) from the natural question text.
4. **Use `expand_context=3`** at retrieval time — this is the primary win vs LoCoMo, where expand_context hurt.
5. Best single-call config on LME-hard: `em_v2f_expand_3` R@50 = 0.832 (vs SS 0.817 reference, Δ=+0.015).
6. Do NOT stack type_enumerated cues on LME — they regress sum_cosine.
7. Temporal-reasoning category remains the ceiling (~0.77 K=50); expand_context does not solve actual date arithmetic, only conversational adjacency.

## Outputs

- JSON: `results/em_lme_hard.json`
- Collections manifest: `results/em_lme_hard_collections.json`
- SQLite segment store: `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/eventmemory_lme.sqlite3`
- Qdrant collections: `arc_em_lmehard_v1_<question_id>` in namespace `arc_em_lmehard`
- Caches: `cache/emlme_v2f_llm_cache.json`, `cache/emlme_type_enum_llm_cache.json`
- Sources: `em_lme_setup.py`, `em_lme_eval.py`