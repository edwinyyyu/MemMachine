# Session 2026-05-21 (cont.) — embed-format closure, noise floor, slim_v4

Continuation of the 2026-05-21 K=40 / decoupling / slim_v3 session. Goal:
raise the weakest model, reduce complexity, find the best embed/BM25
formats — without hurting performance.

## Headline outcomes

- **slim_v4 segmenter ships** (`probe_terse_decoupled_slim_v4.py`). The
  keep/drop rule was reframed from a closed filler enumeration into one
  objective dichotomy — a fact about the speaker's LIFE vs a move in the
  CONVERSATION. On the production cell (gpt-5.4-nano@low, nb8, n=6):
  **same accuracy as slim_v3, −43 tok/q (−14%), ~half the index.**
  Spending the freed budget on K: **K=12 = 88.97 @ 320t, K=13 = 89.03 @
  346t** (gpt-5 judge) — clears Mem0 (87.3) by +1.7pp.
- **Honest verdict:** slim_v4 is a SIMPLIFICATION win, not an accuracy
  win — Pareto-competitive with the prior best terse-v2 (K9 88.70@321t),
  +0.27 at matched budget = within noise.

## What was settled (all measured, not assumed)

- **Noise floor:** σ≈0.40pp, range ~1.1pp (5 identical `cur` runs). The
  yardstick for every delta since. Small effects need N runs (SE=σ/√N);
  consistent small gains stack.
- **Embed format — every axis a non-lever:** component order, separator
  string, label headers, dates-in-embed. `text_to_embed` = M+Q+C
  concatenated, any order. Additive lattice: ≥2 of {M,Q,C} required
  (1→2 = +2.2pp); 3rd component +0.4-0.7pp borderline; a 4th LLM framing
  (atomic / topic) does not help.
- **BM25 text — non-lever:** content/format/dates all within noise; raw
  chunk C as the BM25 anchor does not help.
- **Dates:** drop the programmatic `_date_aliases` — within noise even
  removing from BM25.
- **Neighbor window:** 8 is the peak (nb0→nb8 monotonic on accuracy AND
  tokens; nb16 regresses).
- **T-anchor:** T works as the embed/BM25 anchor as well as M — the
  `memory` field is droppable from STORAGE. But slim_v5 (drop the M
  field from the segmenter OUTPUT) was rejected: generating M is a
  load-bearing compression scaffold (+27 tok/q without it). Keep the
  3-field {memory,terse,queries} output.
- **Model gap:** ~0.6pp, mostly noise-inflated; the prompt is already
  fairly model-robust. The 6-model matrix's 2pp spread was noise + one
  outlier.

## Methodology notes added

- Don't self-throttle API concurrency (highest OpenAI tier).
- `rm` sqlite `-wal`/`-shm` sidecars when rebuilding a DB.
- Match ALL settings (esp. neighbor-window) across prompt comparisons —
  a confound cost one slim_v3-vs-v4 comparison.
- Mini-judge K-tuning gains don't fully transfer to the gpt-5 judge —
  confirm on gpt-5.

## Where this leaves the work

The in-scope memory-architecture space (segmenter / embed / BM25 / dates
/ neighbors) is explored to a validated local optimum — slim_v4 is the
consolidated best. The K=40 finding showed the remaining headroom is in
retrieval RANKING (gold in the pool but ranked outside top-K), which
segmentation cannot fix. Reranking is out of scope
(`feedback_retrieval_research_scope`); a query-side direction (HyDE /
multi-probe) would be the next lever but is a new arc needing a scope
decision.

Memories: `project_slimv4_segmenter`, `project_slimv3_neighbor_sweep`,
`feedback_eval_noise_floor`, `feedback_api_concurrency_tier`.
