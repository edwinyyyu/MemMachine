# Session 2026-05-20 — decoupled-retrieval architecture + terse display

## TL;DR

A Pareto win on LoCoMo at the ≤350 tok/q budget, on the Mem0-comparable
stack (gpt-5 judge, `mem0-bench` variant, micro-average):

| variant | K | tok/q | **micro** | macro |
|---|---|---|---|---|
| qkey-min3p (prior best) | 8 | 326 | 86.88 | 83.33 |
| terse-decoupled-v1 | 10 | 350 | 87.86 | 83.18 |
| **terse-decoupled-v2** | **9** | **321** | **88.70** | 84.85 |
| terse-decoupled-v2 | 10 | 356 | 89.35 | 85.35 |

At matched ~320 tok/q: **terse-decoupled-v2 = 88.70 vs qkey-min3p 86.88 →
+1.8pp**, and clears the Mem0 87.3 bar by +1.4pp while using ~1/20th of
Mem0's token budget (Mem0 = 91.56% micro at top-200 ≈ 6700 tok/q).

Iteration (gpt-5-mini judge) tracked the same ordering:
qkey-min3p K=8 89.16 → terse-v1 K=10 90.19 → terse-v2 K=10 91.17 micro.

## The metric is MICRO (correct/1540) — note on a transient slip

"c1234 avg" has always meant **micro** — number correct / total
questions over categories 1-4. Confirmed: `locomo_evaluate.py` only
prints per-category; Mem0's `mem0ai/memory-benchmarks`
`compute_overall_metrics` is `overall_accuracy = correct / total`
(published: 91.56% micro @ top-200, 82.66% @ top-50; cats [1,2,3,4]).

Transient error: the first `summarize_runs.py` written this session
computed the combined number as a macro-average (mean of per-category
accuracies), so several mid-session turns quoted macro. Caught and
fixed — the summarizer now prints both columns and **micro is the
headline**. All results below are micro.

## Negative result — sub-turn raw-text splitting fails (rejected)

Hypothesis tested: split raw turns into smaller chunks (recursive
splitter, ~110 char) so more segments fit the token budget → more
retrieval diversity. Built `WindowChunkSegmenter` /
`RawChunkRewriteSegmenter`.

Result: `wchunk110` 81.31 macro @ 311t and `dchunk110` 81.42 @ 311t —
**−3.8pp vs qkey-min3p 85.15@310**. K=12 was *worse* than K=9 (more K
hurts).

Mechanism (diagnosed by inspecting retrieved contexts):
1. **Sibling-fragment crowding** — a query about turn T retrieves
   several near-duplicate fragments *of T*; they cluster in the ranking,
   consume the K budget, and crowd out other turns. The answer-bearing
   fragment can lose its slot to its own siblings. Splitting *reduces*
   effective retrieval diversity.
2. **Raw text leaves relative dates unresolved** — the answerer reads
   raw "last week" and miscomputes; a rewrite resolves it to an
   absolute date. Worst hit: temporal (cat-2).

Conclusion: for LoCoMo, "more K via sub-turn splitting" is empirically
false. The fix is to make the *rewrite* cheaper, not to fragment.

## Architecture — `DecoupledRetrievalContext`

New core Context type (`data_types.py`). Decouples a segment's three
texts, each with a distinct consumer:

- `block.text` — verbatim text shown to the **answering model**; the
  ONLY text that costs answer tokens.
- `text_to_embed` — embedded for **semantic retrieval** (deriver passes
  it through; embedder is called once per derivative at ingest only).
- `text_to_score_bm25` — concatenated across a retrieved context and
  scored by **BM25** lexical retrieval.

Wiring: `_format_with_context` (both derivers) → `text_to_embed`;
`event_memory.py` `_segment_header` (renders `[ts] ` like RewriteContext),
`string_from_segment_context(for_bm25=...)` selects `text_to_score_bm25`,
`_fuse_bm25_scores` calls it with `for_bm25=True`. `locomo_search.py`
gained `--timestamp-format short` (Babel short date, ~5 tok vs ~14).

## The wins

**terse-decoupled** (`probe_terse_decoupled_v1.py`): qkey-min3p's
retrieval design unchanged (identical `text_to_embed` / BM25 text
formulas), but `block.text` is a separately-LLM-generated **terse**
statement — every concrete particular kept, only filler cut (~18%
fewer chars, ~accuracy-neutral per segment). Cheaper answer text →
more K fits the budget. + compact timestamp header.

**v2 date-alias** (`probe_terse_decoupled_v2.py`): diagnosed that ~82%
of hard failures are retrieval misses, a confirmed slice being temporal
date surface-form mismatch (query "August 2023" can't lexically bridge
memory "2023-08-15"). Fix: append natural-language date aliases
("August 2023", "August 15, 2023") to the **free** retrieval texts
(`text_to_embed`, `text_to_score_bm25`) — `block.text` untouched.
+1.5pp gpt-5-judge micro, concentrated in temporal (cat-2 +2.2,
cat-3 +5.2) — the category-targeted gain confirms the mechanism.

## Files changed

Core (the architecture — keep): `data_types.py`,
`deriver/text_deriver.py`, `deriver/llm_text_deriver.py`,
`event_memory.py`, `locomo/locomo_search.py`, `locomo/locomo_ingest.py`.
New probes (`longmemeval/llm_pipeline_probe/`):
`probe_terse_decoupled_v1.py`, `probe_terse_decoupled_v2.py`,
`probe_chunk_deriver_v1.py`, `probe_decoupled_chunk_v1.py`.

## Open directions

- Remaining ceiling: broad retrieval misses (cat-3 open-domain 74,
  cat-1 multi-hop 85.5 gpt-5-judge). Date-alias only addressed the
  temporal slice.
- terse compression is ~18% — the v22 statement is already tight;
  more aggressive compression risks fidelity loss.
- pool-size (vector-search-limit) diagnostic was run; it is a knob,
  not architecture.
