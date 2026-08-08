# Session 2026-05-21 — K=40 stress test, decoupling ablation, prompt slim + robustness

Four investigations, all on the iteration stack (gpt-5-mini answerer +
gpt-5-mini judge, `mem0-bench`, MICRO = correct/1540 over cats 1-4).

## 1. K=40 / l=160 quality stress test

terse-decoupled-v2 and v22, pushed to `max_num_segments=40`,
`vector-search-limit=160` (far over the ≤350-tok budget — a diagnostic).

| run | K | tok/q | MICRO | vs own low-K |
|---|---|---|---|---|
| terse-v2 seg | 40 | 1387 | **94.09** | 91.17 @K=10 → +2.9 |
| terse-v2 rawev | 40 | 1843 | 94.03 | — |
| v22 seg | 40 | 1471 | 91.95 | 88.05 @K=7 → +3.9 |
| v22 rawev (K=40) | 40 | 1543 | 93.12 | 31.5 events/q |
| v22 rawev (K=56) | 56 | 2093 | 93.31 | 43.2 events/q |

**Quality does not just "remain high" — it RISES sharply with K.**
The low-token operating point (K≈9-10, ≤350t) leaves ~3pp on the table:
the gold evidence is already in the retrieved pool but ranked outside
the top-10. The ceiling at the budget is retrieval **ranking**, not
recall and not answering quality. (Caveat: the K=40 runs also widened
the pool to l=160; pool vs K not separated here.)

**terse-v2 seg (94.09) ≈ terse-v2 rawev (94.03).** Answering from the
terse `block.text` equals answering from the raw event text — the terse
rewrite is information-faithful.

**v22 rawev (93.12) > v22 seg (91.95), +1.2pp.** v22's rewritten
segment is lossy vs raw text. terse-v2's terse is not. The terse rewrite
is a better answer substrate than the v22 rewrite.

**Matched-events comparison** (granularity-independent retrieval
quality; both answered from raw events, deduped to the same event
count). terse-v2 rawev = 37.46 events/q. v22 rawev brackets it:
31.5 ev → 93.12, 43.2 ev → 93.31; interpolated to 37.5 ev ≈ **93.22**.
terse-v2 at 37.46 ev = **94.03** → **terse-v2 +0.81pp** pure
retrieval-quality edge over v22 (granularity- and substrate-neutral).

## 2. Decoupling ablation — do the 3 texts need to be separate?

`DecoupledRetrievalContext` gives each consumer its own text:
block.text (answerer) = terse; text_to_embed (embedder) = memory +
queries + raw chunk + dates; text_to_score_bm25 (BM25) = memory + dates.

Method: dump terse-v2's segmentation to a component cache
(`dump_terse_v2_cache.py`), replay it with re-assembled field
assignments (`probe_decoupling_ablation.py`, `CachedReassemblySegmenter`)
— every variant on the IDENTICAL segmentation, zero LLM cost. K=10.

| variant | block | embed | bm25 | MICRO | tok/q |
|---|---|---|---|---|---|
| cur (= terse-v2) | terse | M+Q+C+D | M+D | **91.17** | 356 |
| noc | terse | M+Q+D | M+D | 90.84 | 363 |
| coupledM | memory | M+Q+C+D | M+D | 90.65 | 432 |
| bm25terse | terse | M+Q+C+D | terse+D | 90.58 | 355 |
| noq | terse | M+C+D | M+D | 89.81 | 356 |
| coupledMsimple | memory | M+D | M+D | 89.48 | 428 |
| onetext | memory | M | M | 88.44 | 412 |

**All three separations earn their keep.** Collapsing to one text
(onetext) loses **2.73pp at MORE tokens**. Importance ranking:

1. **terse as a cheap answer text** — the biggest lever. Every
   memory-as-answer-text variant (coupledM/coupledMsimple/onetext)
   costs ~410-432 tok/q at K=10 yet scores ≤90.65: dominated on BOTH
   axes. terse is what lets K fit the budget, and (per §1) more K
   strongly raises accuracy.
2. **queries in text_to_embed** — −1.36pp when removed (noq).
3. **separate BM25 text (memory, not terse)** — −0.59pp when BM25
   scores the answer text instead (bm25terse). The fuller `memory` is a
   better lexical surface than the compressed `terse`.
4. **raw chunk in text_to_embed** — −0.33pp (noc), near noise.
   CORRECTS the earlier v3 claim of −2pp: that was LLM re-segmentation
   noise; the controlled cache-reassembly ablation isolates it at
   −0.33pp.

## 3. Prompt slim — redundancy removed

`PROMPT_TERSE_DECOUPLED_V2` had accreted to ~75 lines / 878 tokens.
Audit in `PROMPT_BLOAT_ANALYSIS.md`: the "particular" list stated 3x,
the retrieval objective framed 3x, a 19-line date section with two
redundant enumerations (10 relative-reference examples; 4 forbidden
date forms), "FAILURE" 7x.

- **slim_v1** (`probe_terse_decoupled_slim_v1.py`, 556 tok) removed the
  redundancy — but also dropped v2's anti-fragmentation guardrail and
  **over-segmented**: ~9.1k-11k segments across models vs v2's 5.2k.
  The segmentation diff showed one occasion split into one item per
  *clause* (e.g. a single book recommendation → 5 fragments).
- **slim_v2** (629 tok, −28%) restored anti-fragmentation as an
  objective principle (unit = TOPIC, not a sentence/particular).
  Granularity fixed (gpt-5.4-nano: 5283 vs v2's 5196) — but the gpt-5
  judge showed a **temporal (cat-2) regression**: per-category diff
  vs terse-v2 isolated it to c2 (86.0-87.9 vs v2's 88.8-89.7), c1/c4
  unchanged. The 19→6-line date-section compression had cut
  load-bearing date rules.
- **slim_v3** (`probe_terse_decoupled_slim_v3.py`, `PROMPT_SLIM_V3`,
  726 tok, **−17% vs v2**) restored the explicit date section (9
  relative-reference examples, EQUALS/DIFFERS branch, precision-matched
  ISO output) while keeping every other slim_v2 simplification.
  cat-2 recovered. **This is the final slim.**

**No regression — slim_v3, production config (gpt-5.4-nano @ low),
matched ~355t budget:**

| judge | slim_v3 K=11 | terse-v2 K=10 | Δ |
|---|---|---|---|
| gpt-5-mini | 91.43 @355t | 91.17 @356t | +0.26 |
| gpt-5 | 89.22 @355t | 89.35 @356t | −0.13 |

Both within run-to-run noise → the 17%-shorter prompt costs nothing.
(slim_v2 K=11 gpt-5 was 88.31 → −1.04 vs v2; slim_v3 fixed it.)

## 4. 6-model robustness matrix — slim_v3

slim_v3 ingested with 6 segmenter configs; searched + judged on the
iteration stack (K=10, mini).

| segmenter model | reasoning | MICRO | tok/q |
|---|---|---|---|
| gpt-5-mini | medium | 91.95 | 337 |
| gpt-5.4-nano | medium | 90.91 | 325 |
| gpt-5.4-nano | low | 90.45 | 323 |
| gpt-5-nano | low | 90.45 | 382 |
| gpt-5-mini | low | 90.00 | 335 |
| gpt-5-nano | medium | 89.87 | 378 |

**All six cells 89.9-92.0 — no model fails, all beat the v22 baseline
(88.05).** Spread 2.08pp, but slim_v2-vs-slim_v3 same-cell numbers
swing ±1.3pp from segmentation stochasticity alone — the single-ingest
noise floor is ~±1pp, so most of the spread is ingest noise, not model
brittleness. gpt-5-nano is NOT a laggard in slim_v3 (5n-l ties
54n-l at 90.45). The prompt is principle-based and robust.

## Verdict

- terse-decoupled-v2's decoupled 3-text architecture is validated end
  to end — every separation pays for itself (collapse = −2.73pp).
- **slim_v3 is the new prompt**: 17% shorter than the bloated v2,
  redundancy removed, principle-based, no regression on either judge
  at matched budget, robust across the 6-config model/reasoning grid.
- The real headroom is retrieval RANKING at the low-token budget
  (§1: ~94% is answerable at K=40; only ~91% is captured at K=10) —
  next lever is reranking / fusion, not segments or the answerer.
- Methodology note: small prompt deltas need the matched-budget gpt-5
  judge with a per-category diff — the mini judge missed slim_v2's
  temporal regression entirely.
