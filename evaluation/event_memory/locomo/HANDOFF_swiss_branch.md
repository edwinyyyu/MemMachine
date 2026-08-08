# HANDOFF — LoCoMo / segment-store benchmark work

**Read this in full before kicking off any benchmark run.**

Every rule below has been explained to me by the user multiple times. Forgetting
them has cost **hundreds of wasted runs** — a whole optimization session was run
against the wrong thing. This file exists so that stops happening. When the user
has to repeat something, it goes here.

---

## 0. The recurring mistakes — do not repeat these

These are actual mistakes that were made, not hypotheticals:

1. **Optimized on `--answer-with-raw-events` for an entire session**, treating a
   diagnostic as the architecture. Every accuracy number from it was invalid.
   88 search files + 118 evals are archived in `wrong_methodology_rawev/`.
2. **Faked raw-text answering with `--answer-with-raw-events`** instead of
   building a real raw-text-segment architecture. (Raw-text segments have
   since been built and properly tested — and REJECTED with evidence; see §7
   and memory `project_subturn_splitting_rejected`. Don't rebuild them.)
3. **Used a fixed K to compare against Mem0.** You cannot compare systems at a
   fixed K — different systems segment at different granularities, so "their
   top-10" is not "our K=10". A fixed-K comparison is valid ONLY between runs
   of the *same* system at *unchanged* segment granularity (see §3).
4. **Used `--separate-contexts` at `expand_context=0`.** Don't.
5. **Made an unjustified conclusion.** e.g. claiming a win from a gain within
   run-to-run noise (~±0.25pp), or "X doesn't work" from one weak variant. An
   unjustified conclusion is a serious error in ANY situation — every
   conclusion must be supported by the evidence in hand.
6. **Abandoned a promising idea after one negative variant** without
   diagnosing the failure and trying a targeted fix.
7. **Claimed "beat Mem0" off a gpt-5-mini-judge number** (mini ≠ Mem0-comparable).
8. **Ran benchmarks without recording their settings.** `eval_unknown_judge/` is
   a whole directory of eval files whose judge model and judge variant were
   never recorded — the numbers in it are unattributable and unreproducible.
9. **Tuned the BM25 score-fusion weight and presented the gain as a win** — when
   the task was to improve the LLM deriver. The BM25 fusion weight is a tuning
   knob, not the deriver; a gain from turning a knob is not a deriver or
   retrieval-architecture improvement, and chasing it is off-task.
10. **Swapped to text-embedding-3-large + gpt-5.4-mini (instead of nano) and
    presented it as an architectural win.** A bigger embedder or a bigger model
    is a model swap, not an architecture change — out of scope, and it proves
    nothing about the architecture.
11. **Computed the combined accuracy as a macro-average** (mean of the 4
    per-category accuracies) instead of **micro** (`correct / total` over all
    cat-1-4 questions). "c1234 avg" means micro. Macro over-weights the n=96
    open-domain category and runs ~3-4pp lower. Always aggregate as
    correct/1540; `summarize_runs.py` prints both — MICRO is the headline.
12. **Launched background jobs and didn't notice they silently hung.** An
    OpenAI API outage left ingests/searches stuck for ~2h with no error in
    the log (Python stdout is block-buffered to a file). Check process
    liveness / DB-file mtimes when a background job runs much longer than
    expected; don't assume "still running" means "still working".

If you are about to do any of these, stop.

---

## 1. Pre-run checklist (run through this every single time)

- [ ] Is this measuring the **real segment-answer path**? (NOT `--answer-with-raw-events`; NOT `--separate-contexts` at `expand_context=0`)
- [ ] Does the output filename record **every setting** — segmenter, deriver, embedder, answerer model, judge model, judge variant, `max_num_segments`, BM25 fusion mode + weight, pool size, `expand_context`? (No more `eval_unknown_judge/`.)
- [ ] Am I comparing at a **fixed token budget** (≤340 target, **≤350 hard max**), not a fixed K, and recording `avg(tokens/q)` next to every accuracy?
- [ ] Am I reading the **MICRO** accuracy (correct/1540 over cats 1-4), not the macro mean-of-categories? (`summarize_runs.py` prints both.)
- [ ] Iteration = gpt-5-mini ans + gpt-5-mini judge. Mem0-comparable = gpt-5 judge + `mem0-bench` (answerer may stay gpt-5-mini). Am I clear which this is?
- [ ] Is the gain attributable to **the thing I am actually working on** — not a knob (BM25 weight) and not a model/embedder swap?
- [ ] Am I chasing a large gain, not grinding ~0.1%? (A prompt idea gets a fair shot — diagnose + targeted fix — but no fixed iteration quota.)
- [ ] Have I checked a DB already exists before re-ingesting? (Ingest is slow and rarely needs redoing.)

---

## 2. Architecture — what the pipeline actually is

### Segmenter vs deriver — the distinction is VISIBILITY

Each memory unit has TWO separate blocks, with **disjoint consumers**:

| block | produced by | visible to | invisible to |
|-------|-------------|------------|--------------|
| **segment block** | the **segmenter** | **BM25** (lexical retrieval scores it) + the **answering model** (reads it) | the embedder |
| **derivative block** | the **deriver** | the **embedder** only (becomes the semantic-retrieval vector) | **BM25** + the **answering model** |

- The **segment block** is the visible artifact — `block.text` (for rewrite
  segmenters, the 3rd-person memory statement). BM25 matches query words against
  it; the answerer reads it; every word counts toward the answer token budget.
- The **derivative block** is the hidden artifact. Only the embedder sees it —
  it becomes the vector for semantic retrieval. BM25 never scores it; the
  answerer never sees it. (For the rewrite segmenter it is staged as
  `RewriteContext.text_to_embed`; the deriver passes it through verbatim.)
- Segments are NOT produced by the deriver. Say "segment" only for a segmenter
  output, "derivative" only for a deriver output.

Why conflating them caused mistakes:
- **The derivative is free.** Zero answer tokens (the answerer never sees it),
  zero effect on BM25. It may be arbitrarily long and may carry retrieval-only
  scaffolding (hypothetical queries, paraphrases) — optimize it purely for
  semantic-retrieval quality. Same K ⇒ same answer tokens regardless of it.
- **The segment is expensive and load-bearing.** Every word is read by the
  answerer (the ≤340/350 token budget) and matched by BM25 — keep it clean,
  correct, answer-ready, tight.
- **"Improving the deriver" only moves semantic retrieval.** It cannot change
  answer tokens, BM25, or what the answerer sees. A BM25-fusion-weight tweak is
  therefore NOT deriver work — BM25 is a different channel scoring a different
  block.

### Retrieval & answer path

- Two retrieval channels, fused (additive 0.5): **semantic** ranks by
  cosine(query, derivative embedding); **BM25** scores the segment block.
- The top-K **segments** are returned. At `expand_context=0`,
  `string_from_segment_context` renders them as `[timestamp] "block.text"` —
  that string is the answer context. Default prompt `ANSWER_PROMPT_SIMPLE`.
- `--answer-with-raw-events` is a **legitimate diagnostic** (not the
  architecture). It dedups retrieved segments back to the original raw events
  they came from and answers from that raw text. Two real uses: (a) does a
  segment lose information vs its raw source? (b) a granularity-independent
  retrieval comparison — any segmenter's retrieved segments dedup to the same
  underlying events. It is a *different thing* from a raw-segments architecture
  (where the segment block itself is raw text). The mistake was using its
  output as the headline architecture metric — not the diagnostic itself.
- `--separate-contexts` must NOT be used at `expand_context=0`.
- No segment→event dedup; the number of returned segments is fixed (= K).
- An expanded multi-segment candidate inherits its semantic score from the seed;
  BM25 still scores the full stringified context.

---

## 3. The metric — accuracy at a fixed TOKEN budget

- **The headline comparison axis is accuracy at a fixed token budget.** Tokens
  are counted on the `conversation_memories` text given to the answerer
  (excludes the question and prompt boilerplate).
- **Accuracy = MICRO** = `correct / total` over all cat-1-4 questions (1540).
  This is "c1234 avg"; it is what Mem0 reports. Not the macro mean of
  per-category accuracies. See §0 #11.
- **Budget: ≤340 tokens/question target, 350 tokens/question HARD MAXIMUM** when
  comparing to Mem0. A variant over 350 tok/q does not qualify, period.
  Derivation: Mem0 reports ~6700–7000 tok/q at **top-200** on LoCoMo;
  /200×10 ≈ 340 = est. of Mem0's top-10. (~51 tok per rendered segment ⇒
  K=6 ≈ 308t, K=7 ≈ 359t.)
- **The bar: Mem0's claimed LoCoMo number is 87.3%.** Beating Mem0 means beating
  87.3% at ≤350 tok/q.
- **Fixed-K comparison is valid only within one system at constant segment
  granularity.** You cannot compare at a fixed K across systems — Mem0 segments
  at a different granularity, so "their top-10" is not "our K=10"; compare to
  Mem0 only at the token budget. A fixed-K comparison is likewise invalid across
  any change that alters segment granularity (a new segmenter, different
  chunking). It IS valid for a **deriver-only change** with the segmenter held
  constant — the deriver is invisible to the answerer, so at the same K the
  answer context is the same K segments of the same granularity (≈ identical
  token budget) and only retrieval quality differs. When granularity changes,
  compare at the token budget; always report `avg(tokens/q)`.
- **Check retrieval didn't regress — how depends on what changed.** A low-token
  win can be a disguised trade: a variant may keep ~80% accuracy while cutting
  50% of tokens by shrinking segments.
  - **Deriver-only iteration (segmenter held constant):** a matched
    `max_num_segments` comparison IS valid — the segments are identical, only
    retrieval ranking changes; check the candidate never scores below baseline
    at matched `max_num_segments`.
  - **Segmenter / granularity change:** matched `max_num_segments` is NOT
    comparable (same flaw as fixed-K). Use the **raw-event diagnostic**
    (`--answer-with-raw-events`): dedup each system's retrieved segments back
    to the source raw events and compare there — granularity-independent, and
    it isolates retrieval quality from segment sizing/fidelity. (This is the
    diagnostic's real purpose; a *different thing* from a raw-segments
    architecture.)
  A real win is a Pareto improvement of the accuracy-vs-tokens curve.
- **Goal:** a Pareto improvement — beat our previous best AND beat Mem0 (87.3%)
  at ≤350 tok/q, without regressing at higher token counts.
- Never cut K to fit the budget and call it a win — that is the granularity
  game. Tighten per-segment content instead.

---

## 4. Eval stack

- **Iteration (relative measurement):** gpt-5-mini answerer + gpt-5-mini judge.
  Canonical flags:
  `--model gpt-5-mini --judge-model gpt-5-mini --judge-variant mem0-bench --no-reranker --bm25-fusion additive --bm25-fusion-weight 0.5 --vector-search-limit 28 --skip-category-5`
- **Final (Mem0-comparable):** the **judge** is what makes it comparable —
  **gpt-5 judge + `mem0-bench`** variant. The **answerer can stay gpt-5-mini**.
  If you instead use **gpt-5 as the answerer, you must also switch to the Mem0
  answering prompt** (not `ANSWER_PROMPT_SIMPLE`) — otherwise the comparison is
  not apples-to-apples.
- The iteration and final numbers differ; never conflate them, and never claim
  "beat Mem0" off a gpt-5-mini-judge number.

---

## 5. Research methodology

- **The focus is justified conclusions.** Every conclusion must be supported by
  the evidence. An unjustified conclusion is a serious error — in any
  situation. This is the actual rule; run count is only a means to it.
- **Run more items only as needed** — exactly enough to justify the conclusion,
  no more and no fewer. Don't reflexively double-run everything, and don't
  refuse a run a conclusion genuinely needs. A gain clearly outside run-to-run
  noise (~±0.25pp) is justified by one run; a gain within the noise band is not
  a win, and no number of runs makes it one.
- **Prioritize iteration speed and large gains.** Chase big gains; don't grind
  many iterations for ~0.1% movement — if a line of ideas only yields tiny
  gains it is tapped out, pivot to a bigger idea.
- Give a promising **idea** a fair shot — diagnose its failure mode and try a
  targeted fix before abandoning it — but a fair shot is a few real attempts,
  not a fixed quota and not deep grinding. Small benches are extra noisy.
- **Diagnose all failures AND successes** — hypothesize failure modes, reasons
  for success, and future directions.
- Prioritize **generalizable** approaches; do not optimize for specific eval
  cases.
- **Validate or reject with evidence, not guessing.**
- **Record every setting of every run** — segmenter, deriver, embedder, answerer
  model, judge model, judge variant, `max_num_segments`, BM25 fusion mode +
  weight, pool size, `expand_context` — in the output filename (and/or a
  sidecar). An unrecorded run is unattributable, unreproducible, and wasted.
  `eval_unknown_judge/` is a whole directory of exactly this failure.
- **Attribute every gain to the right cause.** If the task is to improve the LLM
  deriver, a gain from tuning the BM25 fusion weight, or from swapping to a
  bigger embedder/segmenter model, is NOT a deriver win — it is off-task. Knob
  tuning and model/provider swaps are out of scope for architecture and prompt
  work and must never be presented as architectural wins.
- Save each prompt iteration as its own versioned file. Never edit production in
  place.
- Audit your own methodology BEFORE running ablations or shipping results.
- When the user flags an outlier / magic-number / misleading-name concern, build
  a minimal probe that demonstrates it concretely before refactoring.

---

## 6. Practical notes

- **Ingest builds the DBs (`*.sqlite`). Ingest was never the wrong part** — a
  rerun search does NOT need a re-ingest. Check whether a DB already exists
  before ingesting.
- `wrong_methodology_rawev/` archives the 206 invalid (`--answer-with-raw-events`)
  runs. Their retrieval/recall is still informative; their answer-accuracy is
  not. See that directory's README.
- **Redo rule:** redoing an archived rawev benchmark on the segment-answer path
  is only a faithful *correction* of the same benchmark when the segment text
  equals the raw text. For rewrite segmenters (v22 / qkey / min3p) the segment
  is a 3rd-person rewrite ≠ raw text → there is no faithful "redo"; a
  segment-answer run there is new optimization work, not a cleanup correction.
- Pipeline scripts: `locomo_ingest.py` (build DB), `locomo_search.py`
  (retrieve + answer), `locomo_evaluate.py` (judge). Prompt-iteration probes
  live in `../longmemeval/llm_pipeline_probe/`.
- Use `uv run` for Python.
- Corresponding persistent memories: `feedback_locomo_architecture`,
  `feedback_locomo_eval_constraints`, `feedback_eval_stack_purpose`,
  `feedback_benchmark_hygiene`.

---

## 7. Current architecture and state (updated 2026-05-20)

**SHIPPED: the decoupled-retrieval architecture.** A new core Context type
`DecoupledRetrievalContext` carries three texts, each with its own consumer:
- `block.text` — the verbatim text shown to the **answerer**; the ONLY text
  that costs answer tokens.
- `text_to_embed` — embedded for **semantic** retrieval (deriver passes through).
- `text_to_score_bm25` — concatenated across a context, scored by **BM25**.

Best result (gpt-5 judge, `mem0-bench`, **micro** — Mem0's own metric):
**terse-decoupled-v2 K=9 = 88.70% @ 321 tok/q** (clears the 87.3 bar;
prior best qkey-min3p K=8 = 86.88 @ 326t → +1.8pp Pareto). The segmenter
(`probe_terse_decoupled_v2.py`) emits a compact `terse` block + qkey-min3p's
retrieval texts + deterministic date-alias enrichment. See
`SESSION_2026-05-20_decoupled_terse.md` and memory
`project_decoupled_retrieval_architecture`.

**Raw-text segments were TRIED and REJECTED — do not rebuild them.** Both
sub-turn raw splitting (sibling-fragment crowding wastes K; more K hurts) and
whole-turn raw segments (too token-expensive) lose to the rewrite segmenter.
Mechanism in memory `project_subturn_splitting_rejected`. The earlier HANDOFF
goal "build raw-text segments + a v22-like deriver" is superseded — raw text
for the answerer also leaves relative dates unresolved (hurts temporal). The
winning answerer text is a compact resolved rewrite (`terse`), not raw text.

**2026-05-21 session — K=40 stress test, decoupling ablation, prompt slim.**
See `SESSION_2026-05-21_k40_ablation_robustness.md`.
- K=40/l=160: accuracy RISES with K (terse-v2 91.17@K10 → 94.09@K40, mini).
  The ≤350-tok budget ceiling is retrieval RANKING (gold is in the pool but
  ranked outside top-10), not recall/answering. Next lever = rerank/fusion.
  (memory `project_k40_ranking_ceiling`)
- Decoupling ablation (cache-reassembly, `probe_decoupling_ablation.py`):
  all 3 text separations earn their keep; collapsing to one text = −2.73pp.
  Corrects the old "drop raw chunk = −2pp" (actually −0.33pp, was noise).
- Slim prompt: final = `probe_terse_decoupled_slim_v3.py`
  `PROMPT_SLIM_V3` — 17% shorter than the bloated v2 (726 vs 878 tok),
  no regression at matched ~355t budget (slim_v3 K=11 = 91.43 mini /
  89.22 gpt-5 vs terse-v2 K=10 91.17 / 89.35). Iteration: slim_v1
  over-segmented (cut anti-frag) → slim_v2 fixed granularity but
  regressed temporal on the gpt-5 judge (cut the date section) →
  slim_v3 restored the date section. 6-model robustness 89.9-92.0, no
  model fails. (memory `project_prompt_slim_and_robustness`)
- New ingest cases: `decoupling-ablation`,
  `terse-decoupled-slim-v1/v2/v3`. Events now carry `group_idx` in
  properties (for group-aware caching).

**The metric is MICRO** — "c1234 avg" = correct/1540 over cats 1-4 (verified:
`locomo_evaluate.py` prints only per-category, and Mem0's
`compute_overall_metrics` = `correct/total`). A macro mean-of-categories was
briefly mis-introduced in this session's `summarize_runs.py` and corrected;
micro is the headline metric.

**2026-05-21 (cont.) — noise floor, embed-format closure, T-anchor, nb-sweep,
prompt robustness.** All numbers gpt-5-mini judge unless noted.

- **Noise floor MEASURED** (5 identical-input `cur` reassembly runs):
  **σ≈0.40pp, range ~1.1pp**. A single-run delta <0.8pp is noise. BUT the
  mean of N runs has SE=σ/√N — small effects ARE resolvable with enough runs
  (0.3pp at 2σ ⇒ n≈7), and consistent small gains stack. Memory
  `feedback_eval_noise_floor`. This retroactively reclassified most of the
  embed-format work as noise.
- **Embed/BM25 format — fully closed, every axis a non-lever.**
  `text_to_embed` = M+Q+C: order (MQC vs MCQ 90.85 vs 90.79, n=7/5),
  separator (nl/sp/blank 0.13pp spread), label headers (0.39pp), and
  dates-in-embed (emb_nodate vs cur +0.10) are ALL non-levers.
  `text_to_score_bm25`: content/format/dates non-levers (raw chunk C as
  bm25 text tested — no help). Additive lattice: ≥2 of {M,Q,C} required
  (1→2 = +2.2pp, >5σ); the 3rd component +0.4-0.7pp is ~1.6σ, borderline;
  a 4th LLM framing (atomic decomposition / topic labels) does NOT help.
- **T-anchor: the `memory` field is droppable from the pipeline.** T (terse)
  works as the embedding AND BM25 anchor as well as M (embed_t 91.17, all_t
  90.77 — within noise of emb_nodate ~90.95). M and T come from ONE segmenter
  call; T = compressed M. slim_v5 (output `{statement,queries}` only, no M
  field) tests whether producing M is still a needed scaffold for a good T.
- **Neighbor-window sweep (slim_v3, 54n-l): window 8 is the peak.** nb0→nb8
  monotonic on BOTH accuracy (89.22→91.17) and token cost (354→309 tok/q);
  nb12 ties, nb16 regresses −1.6pp. Keep v22's window 8. Memory
  `project_slimv3_neighbor_sweep`.
- **Model-gap verified ~0.6pp, mostly noise-inflated.** The 6-model matrix's
  2pp spread was noise + one lucky mini run; replicated gap gpt-5-nano vs
  gpt-5-mini @ medium ≈ 0.5-0.7pp (~1.5σ). The prompt is already fairly
  model-robust.
- **slim_v4 — RESOLVED, ships as a simplification.**
  (`probe_terse_decoupled_slim_v4.py`) keep/drop rule reframed as a
  life-fact vs conversation-move dichotomy (diagnosed: gpt-5-nano
  over-retained conversational filler — praise/agreement/reactions — 19%
  more source messages than mini). On the production cell (54n-l @ nb8,
  n=6, matched): **same accuracy as slim_v3, −43 tok/q (−14%)**, half the
  index. Spending the freed budget on K: slim_v4 @ K=12 = 88.97 @ 320t,
  K=13 = 89.03 @ 346t (gpt-5 judge) — clears Mem0 87.3 by +1.7pp,
  Pareto-COMPETITIVE with prior best terse-v2 K9 88.70@321t (+0.27 at
  matched budget = within noise). Honest verdict: a SIMPLER system at
  equal accuracy, not an accuracy win. The keep/drop change did not raise
  weak-model accuracy directly (a leveling trade: nano +0.33, mini
  −0.51); its payoff is token efficiency. Mini-judge K-gains don't fully
  transfer — mini K12→K13 +0.29, gpt-5 +0.06. Memory
  `project_slimv4_segmenter`.
- **slim_v5 REJECTED** (`probe_terse_decoupled_slim_v5.py`, output
  `{statement,queries}` — no `memory` field). M is droppable from
  STORAGE (T-anchor) but generating M is a load-bearing compression
  scaffold: dropping it cost +27 tok/q (mini) at equal accuracy. Keep the
  3-field `{memory,terse,queries}` output.
- New ingest cases: `terse-decoupled-slim-v4`, `terse-decoupled-slim-v5`.
- Methodology notes added this session: don't self-throttle API concurrency
  (account is OpenAI's highest tier — run all variants in parallel,
  memory `feedback_api_concurrency_tier`); `rm` the sqlite `-wal`/`-shm`
  sidecars when rebuilding a DB (a stale WAL throws `disk I/O error`);
  match ALL settings (esp. neighbor-window) when comparing prompts.

## 8. SHIP DECISION — state at 2026-05-22 compaction (READ FIRST)

**The proposed ship is `slim_v3`** (the last *clean* win — the decoupled
architecture in its cleanest prompt form). slim_v4 was demoted to a
Pareto *trade*, not a clean win. Validation in progress — do NOT claim a
ship until the in-flight confirmation passes.

**Key numbers (mini judge unless noted; 54n-l = gpt-5.4-nano@low =
production segmenter cell; nb8):**
- terse-decoupled-v2 (last clean architectural win, +1.8pp Pareto over
  qkey-min3p): K=9 88.70@321t gpt-5 · K=10 91.17 mini · K=40 seg 94.09 ·
  K=40 rawev 94.03.
- slim_v3 (−17% prompt of terse-v2, no regression, 6-model 89.9-92.0):
  54n-l-nb8 K=10 = 90.80 mini (n=6). Budget point ≈ K=11 (~340t).
- slim_v4 (life-fact-vs-conversation-move keep/drop; HALVES the index):
  54n-l-nb8 K=10 90.85 mini@267t · K=12 91.24 mini / 88.97 gpt-5 @320t ·
  K=13 91.53 mini / 89.03 gpt-5 @346t. 6-MODEL ROBUST: matrix
  {5n,54n,5m}×{l,m} all 90.15-90.85, 0.70pp spread (tighter than v3).
  K=40 seg 92.66 (−1.43 vs terse-v2 — segment text lossy at high K).
  K=40 RAWEV 93.66 (−0.37 vs terse-v2 — retrieval/index coverage is
  FINE, near noise). So slim_v4's K=40 seg deficit is segment-text
  fidelity at high K, NOT a recall/index-coverage loss.
- slim_v4 vs slim_v3 matched (nb8, n=6): leveling trade — nano +0.33,
  mini −0.51, gap 1.38→0.54. Net ~neutral.
- Classic judge (gpt-4o-mini+mem0-classic) slim_v4: K12 88.00, K13
  88.57 — clears Mem0 87.3; "beats Mem0" is judge-robust (classic /
  gpt-5 / mini all clear it).
- Raw-text non-generative floor: text+whole K=7 82.14@319t (tsshort),
  text+sentence ~81-83. The LLM rewrite buys ~+9pp at low budget.
- Noise floor σ≈0.40pp (`feedback_eval_noise_floor`).

**slim_v4 honest verdict:** a Pareto TRADE — wins low budget (K=12-13,
leaner/terser → more fits), retrieval is fine (rawev), 6-model robust;
but K=40 seg ceiling −1.4pp (terse text lossy at high K). NOT a clean
win over terse-v2/slim_v3.

**WHY slim_v3 > slim_v4 — a prompt PRINCIPLE, not model bias** (both
nano AND mini dropped ~30% more under v4 — same behavior, so it's the
prompt). Evidence: K=40 seg-vs-rawev. terse-v2/slim_v3 seg≈rawev
(94.09≈94.03) — the `terse` block is FAITHFUL, answers as well as raw
source. slim_v4 seg<rawev (92.66<93.66, −1pp) — its `terse` is LOSSY.
rawev itself near-tied (−0.37) ⇒ index coverage fine; deficit is purely
segment-text fidelity. Principle: **the segment text must reconstruct
the source's answerable detail — faithfulness is non-negotiable,
leanness secondary.** Pushing harder compression/merging (v4: 1.05 vs
1.13 segs/msg, terser terse) buys low-budget tokens but makes the
segment lossy — invisible at low K, caps the high-K ceiling. Secondary:
a closed "drop only THESE" list (v3) is recall-safer than an open "keep
only what clears this bar" principle (v4) — dropping is irreversible,
keeping filler is cheap. Any future segmenter prompt must preserve
seg≈rawev faithfulness.

**IN-FLIGHT background batches at compaction (check these on resume):**
- `bpontbs0k` — slim_v3 DIRECT confirmation: re-search slim_v3-54n-l-nb8
  existing DBs at K=40 seg, K=40 rawev, K=11 (mini+gpt-5). Decides the
  ship: slim_v3 ships iff K=40 ≈94 and rawev ≈94 (i.e. no slim_v4-style
  ceiling loss — expected, slim_v3 did NOT halve the index).
- `bu89z3v5w` — embedding-model robustness: slim_v4-54n-l-nb8 re-ingested
  with embeddinggemma-300m + MiniLM-L6-v2 (local sentence-transformers),
  search K=12, vs 3-small 91.24 mini. "not much worse" = robust.

**New infra this session:** `embedder_factory.py` — wires
`--embedding-model` to OpenAI OR local sentence-transformers
(`embeddinggemma`, `minilm`); used by both locomo_ingest.py and
locomo_search.py (ingest+search must match). Uses the package's existing
`SentenceTransformerEmbedder`. ST models carry query/document prompts;
factory sets `default_prompt_name="document"` for the ingest side.

**POLL STATE (2026-05-22 00:10, post-compaction):** both batches still
running as detached procs (`run_slimv3_confirm.py` PID 44721,
`run_embedder_robust.py` PID 43893) — NOT harness-tracked (TaskList
empty post-compaction), so poll the logs, no notification will fire.
slim_v3 confirm: 12 searches ~75-100% done, evals not started → eval
JSONs `eval-tslimv3-54n-l-nb8-rep*-v160-*-l40-{seg,rawev}-mini-mb-c14.json`
and `eval-...-l11-seg-{mini,gpt5}-mb-c14.json`. embedder robust: 6
ingests done, 6 searches running width-2 → eval JSONs
`eval-tslimv4-54n-l-nb8-{gemma,minilm}-rep*-v28-...-mini-mb-c14.json`.

**ON RESUME:** collect `bpontbs0k` + `bu89z3v5w`. If slim_v3 confirm
passes (K=40 ≈94, rawev ≈94) → slim_v3 IS the clean, model-robust,
embedder-robust ship; record it, update memory, done. If slim_v3 also
shows a K=40 issue → fall back to terse-decoupled-v2 itself. slim_v4
stays documented as a low-budget Pareto trade, not the ship. The
prompt/segmentation line is at diminishing returns; the K=40 finding
says remaining headroom is retrieval RANKING (out of scope per
`feedback_retrieval_research_scope`).

## 9. SHIP-CONDITION VALIDATION — state at 2026-05-22 (READ FIRST)

slim_v3 confirm (`bpontbs0k`) PASSED: K=40 seg 93.92 ≈ rawev 93.90 (no
regression vs terse-v2 94.09/94.03, sub-noise); K=11 88.74 gpt-5 @341t
> Mem0 87.3. **slim_v4 is OUT** — K=40 seg 92.66 fails the explicit
"no K=40 regression" ship condition. slim_v3 is the only candidate.

**Budget point:** K=10 90.80 mini / 88.48 gpt-5 @310t; K=11 90.76 /
88.74 @341t. K=10 and K=11 tie on mini; K=11 +0.26 gpt-5 for +31 tok
(within noise). **K=10 @310t is the better budget point.** nb8 vs nb0
(K=10): +1.6/+2.25pp (mini/gpt-5) AND −44 tok — strict Pareto; nb0
gpt-5 86.23 is BELOW Mem0, so the neighbour window is load-bearing.

**Raw-text floor vs slim_v3 (matched token budget, both judges):**
floor (text seg + whole deriver) K=5/6/7 = 80.06/81.88/82.14 mini,
76.56/78.44/79.61 gpt-5 @230/274/319t. slim_v3 K=7/8/9/10 =
89.18/90.13/90.41/90.80 mini, 87.03/87.86/88.05/88.48 gpt-5 @
217/248/279/310t. **LLM rewrite buys +8.5-9pp mini / +9-10.5pp gpt-5**
at matched budget — larger on the stricter judge.

**LLM-model robustness — PROBLEM FOUND.** slim_v3 matrix {5n,54n,5m}x
{l,m} at nb8, K=10/11, mini judge: accuracy spread OK (1.2-1.7pp) but
SEGMENT VERBOSITY is not model-robust. Per-model terse length: 54n 21.4
tok, 5m 24.1, 5n 26.7 (+25%); 5n also +18% more segments. tok/q @K=10:
54n 310, 5m 341, **5n 382 — over the 350 budget**. gpt-5-nano on slim_v3
gpt-5 judge: 5n-l 87.47 / 5n-m 87.73 — barely clears Mem0, and only at
K=10 which is over budget (at the in-budget K=9 it would likely miss).

**Diagnosis (from segment dumps — prompt ambiguity, not model
weakness):** (1) the keep-rule was a CATEGORY LIST → weak models kept
pure-filler messages (greetings/apologies/generic Qs); (2) "fewest
words" is subjective → weak models restate one point several ways.

**FIX: slim_v6** (`probe_terse_decoupled_slim_v6.py`, ingest case
`terse-decoupled-slim-v6`) = slim_v3 + objective keep-GATE (item earned
only if ≥1 particular) + objective anti-redundancy ("state each point
once") + deletion-framed terse ("delete from memory, never paraphrase
up"). Everything else byte-identical to slim_v3. On the strong model
(54n) it changes little (expected — 54n already compressed well); the
real test is whether it pulls gpt-5-nano's terse down toward 54n's.

**EMBEDDING robustness (slim_v3-54n-l-nb8):** MiniLM-L6-v2 works,
~−3pp (K=10 87.84 mini vs 3-small 90.80). embeddinggemma re-running.

**OOM INCIDENT:** the machine ran out of RAM — 18-wide slim_v6 API
ingest + 2-wide local-embedder (gemma) batch concurrently. LESSON:
never run local sentence-transformers models concurrently (width 1
only); cap API batches ~4-6 wide. Memory `feedback_local_model_memory`.

**IN-FLIGHT (2026-05-22, post-OOM relaunch):**
- `buccql23l` — `run_slimv6_matrix.py`: 12 weak-cell slim_v6 ingests
  (54n ×6 already complete, skipped on resume) + search K=10/11 + K=40
  seg/rawev for 54n-l. Ingest width 4, search width 6.
- `btjh40q6d` — `run_slimv3_gemma_fix.py`: embeddinggemma arm of
  slim_v3 embedding robustness, width 1 (one local model at a time).

**ON RESUME:** collect both. If slim_v6 closes the token spread (5n
terse ≈ 54n terse, all cells ≤350 @K10) AND holds 54n accuracy/K=40 →
slim_v6 is the model-robust ship. If slim_v6 does NOT pull 5n down →
the verbosity is model weakness; ship slim_v3 and document gpt-5-nano
as the model floor. Either way the headline ship cell stays
gpt-5.4-nano@low.

---

## 10. ACTIVE MODEL STACK SWITCH — 2026-05-26 (READ FIRST ON RESUME)

The OpenAI account hit `insufficient_quota` mid-session. User directed
a switch to a self-hosted endpoint + local embeddings until told
otherwise. **All iteration from this point uses the new stack**;
gpt-5.4-nano/gpt-5-mini ship comparisons will be re-run at the end
after the OpenAI quota resets.

**Switch chronology (2026-05-26 → 2026-05-27):**
1. **qwen3-5-27b-1** + server `embedinggemma-300m` (initial request, both broken):
   - `qwen3-5-27b-1` was de-registered from `/v1/models` mid-ingest; bo-natural died at ~480 segs (400 BadRequest "Invalid model name"). All partial qwen-prefixed DBs/logs deleted.
   - Server `embedinggemma-300m` (registry name) returned 500 InternalServerError — registry had a typo (`embedinggemma`, single 'd'); the canonical google name `embeddinggemma-300m` (double 'd') works. **As of later in the session the registry typo was fixed by the server team.**
2. **gemma-4-31b-it-1 + LOCAL embeddinggemma** (interim, what produced the bo-natural baseline below).
3. **gemma-4-31b-it-1 + SERVER embeddinggemma-300m** (current target now that the embedding backend works).
4. **DNS dropped (2026-05-27)** — `api.vmnet4-200.eng.memverge.com` no longer resolves when working remotely / on a different VPN. User-provided IP-direct: `http://10.4.254.200/v1`. The nginx in front does virtual-host routing, so all requests must carry `Host: api.vmnet4-200.eng.memverge.com`. New CLI flag `--host-header` added to all three pipeline scripts; AsyncOpenAI clients pass it via `default_headers={"Host": ...}`. **This is now the standard endpoint URL going forward.**
5. **Embedding doc-prompt fix (2026-05-27)** — my first server-embed wrapper used `"task: search result | document: "` for ingest, parallel-structured with the query prompt. WRONG. embeddinggemma-300m's actual document prompt is asymmetric: `"title: none | text: "` (title-prefix format). cos(local-DOC, server-DOC) jumped from 0.892 → 0.999951 after the fix. Suspected to have caused the bo-natural egs c1 regression of ~5pp vs local-embed; re-ingesting now to confirm. See [[feedback_embeddinggemma_prompts]].
6. **HTTPS switch (2026-05-27) — REVERTED, did NOT help.** Tried `https://10.4.254.200/v1` + `--tls-no-verify` to bypass captive-portal HTTP interception. Theory was that captive portals can't silently MITM TLS. WRONG for this corporate network: Fortinet has SSL inspection enabled (corporate root CA pre-installed on managed laptops), so when the VPN session expires the portal forges a cert signed by the trusted corp CA, passes TLS validation, then drops POSTs same as HTTP. Probe confirmed: HTTPS `/v1/models` returned the captive portal HTML redirect (`window.location="https://169.254.1.1:1003/fgtauth?..."`) HTTP 200. Reverted standard URL back to `http://10.4.254.200/v1`. The `--tls-no-verify` flag stays in the code (still valid for other self-signed endpoints) but isn't used for this one.

**Use-server-when-available policy:**
- LLM (segmenter / answerer / judge): server-only. No local fallback — local hardware can't run 31B.
- Embeddings: prefer server. Fall back to local sentence-transformers (width-1 rule per [[feedback_local_model_memory]]) ONLY if the server endpoint is unavailable.

**Active stack (target):**

| role | model | how | flag |
|---|---|---|---|
| segmenter | gemma-4-31b-it-1 | server chat completions @ vmnet4-200 | `--segmenter-model gemma-4-31b-it-1 --base-url http://10.4.254.200/v1 --host-header api.vmnet4-200.eng.memverge.com --api-key none` |
| answerer  | gemma-4-31b-it-1 | same endpoint | `--model gemma-4-31b-it-1 --base-url ... --api-key none` |
| judge     | gemma-4-31b-it-1 | same endpoint | `--judge-model gemma-4-31b-it-1 --base-url ... --api-key none` |
| embedder  | embeddinggemma-300m | **server** OpenAI-compat embeddings @ vmnet4-200 | `--embedding-model embeddinggemma-300m` (added to `embedder_factory.py` 2026-05-26: routes through OpenAIEmbedder with a prompt-prepending wrapper since the server does NOT auto-apply embeddinggemma's `task: ...` prompts). |
| embedder fallback | google/embeddinggemma-300m | LOCAL sentence-transformers, width 1 | `--embedding-model embeddinggemma` (existing) |

**File naming convention on this stack:**
- LLM-using ingests (LLM segmenter / deriver): `locomo-{llm-tag}-{variant}-{embedder-tag}-nb{N}-rep{R}.{sqlite,vec.sqlite}` — `g31` for gemma-4-31b-it-1; embedder-tag is `eg` for LOCAL embeddinggemma and `egs` for SERVER embeddinggemma-300m. E.g. `locomo-g31-tslimv3bonatural-egs-nb8-rep1.sqlite`.
- LLM-less ingests (text segmenter + whole deriver, embedder-only): `locomo-{variant}-{embedder-tag}-rep{R}.{sqlite,vec.sqlite}` — **no LLM tag**, since the DB is stack-independent and can be reused across any LLM swap. E.g. `locomo-textwhole-eg-rep1.sqlite`.

**Cross-embedder caveat:** ingest and search MUST use the same embedder (and the same prompt handling). Don't mix local-embed DBs with server-embed searches or vice versa.

**gemma behavior probes (2026-05-26):**
- Clean chat completion: returned `'Paris'` (2 tokens, finish=stop) — no "Thinking Process" preamble. So the qwen-specific CoT issue does NOT apply.
- JSON schema strict: returned clean `{"name":"Julian Thorne","age":32}`.
- `extra_body.chat_template_kwargs.enable_thinking=false`: silently accepted (no-op on gemma; the patch stays in answerer for future thinking-model swaps).
- **Voluntary CoT on complex questions**: gemma emits step-by-step markdown working (bold, bullets, LaTeX math) when asked something multi-step in unconstrained mode. **A system prompt** ("Reply with ONLY the final answer ...") suppresses this cleanly — unlike qwen3, gemma respects system prompts. Added to `locomo_search.py` answerer call, gated on `--base-url` and skipped for `mem0v2` (which mandates reasoning by design).

**Bo-natural baseline on gemma + LOCAL embeddinggemma (n=1, mem0-classic):**

| metric | result | vs gpt-5-mini stack | Δ |
|---|---|---|---|
| c1 multi-hop | 82.62% | 71.63 | **+10.99pp** (gemma wins) |
| c2 temporal  | 79.75% | 79.44 | +0.31 |
| c3 open-domain | **52.08%** | 76.04 | **−23.96pp** (gemma refuses speculative) |
| c4 single-hop | 83.23% | 91.20 | −7.97 |
| **c124** | **82.34%** | 84.76 | −2.42 |
| **c1234** | 80.45% | 84.22 | −3.77 |

Tokens/q and full table TBD after rawev + textwhole evals. Note: this is the LOCAL-embed snapshot; server-embed re-runs are the next step now that `embeddinggemma-300m` works on the server.

**Segmentation density:** gemma-4-31b-it produces **3185 segs** for bo-natural vs gpt-5.4-nano's **5346** — gemma uses coarser topic-grouping (each segment carries more turns). Per-group counts evenly distributed.

**Code changes (rewind-immune; committed into the working tree):**

1. `packages/server/src/memmachine_server/common/language_model/openai_chat_completions_language_model.py` — `OpenAIChatCompletionsLanguageModelParams` gained `reasoning_effort: str | None = None` as a no-op pass-through field. Lets the 264 existing `OpenAIResponsesLanguageModel(...,reasoning_effort=...)` call sites in `locomo_ingest.py` keep their kwargs after a class-name swap.
2. `evaluation/event_memory/locomo/locomo_ingest.py` — all `OpenAIResponses*` → `OpenAIChatCompletions*` (sed; 264 replacements). New CLI flags `--base-url`, `--api-key`. `AsyncOpenAI` constructed against them when `--base-url` is set; otherwise falls back to `OPENAI_API_KEY`. Choices list now includes `terse-decoupled-slim-v3-bo-natural-sidecar`, `-sidecar-bm25only`, `-sidecar-resemb`, `-draft`, `-draftsidecar`.
3. `evaluation/event_memory/locomo/locomo_search.py` — same `--base-url` / `--api-key` flags; same fallback to `OPENAI_API_KEY`. The direct `client.chat.completions.create(...)` answerer call is already chat-completions-compatible.
4. `evaluation/event_memory/locomo/locomo_evaluate.py` — same `--base-url` / `--api-key` flags. `llm_judge.py` already uses `chat.completions.create` so no judge code change.

**Verified pre-launch (curl probes on 2026-05-26):**
- Plain chat completion: works.
- `response_format` JSON schema strict: works.
- `store=False` and unknown `reasoning_effort` kwargs: silently accepted.

**Active SIDECAR investigation context (carry across the switch):**

The 2026-05-26 SIDECAR finding on the **old stack** (gpt-5.4-nano + gpt-5-mini + text-embedding-3-small, mini-classic judge, K=10, vec-28, BM25 add 0.5, ts-short, nb8):

| variant | tok/q | c1 | c2 | c4 | c124 |
|---|---|---|---|---|---|
| bo-natural | 311 | 71.63 | 79.44 | 91.20 | 84.76 |
| CEILING raw-events | 483 | 70.92 | 83.80 | 92.87 | 86.57 |
| SIDECAR | 321 | 70.21 | 81.31 | 92.27 | 85.53 |

SIDECAR closed 42% of the c124 ceiling gap on the old stack — first variant to beat baseline AND lift cat2 (+1.87pp). Two queued iterations target the c1 regression:
- `-sidecar-bm25only`: dates → bm25 only, not embed. Tests embed-pollution hypothesis.
- `-sidecar-resemb`: verbatim memory (visible+bm25) + resolved memory (embed only) + sidecar dates. Tests verbatim-semantic-impoverishment hypothesis.

**In flight on the new stack (4 parallel ingests at switch time, all rep1):**
- bo-natural (`bac5ksoex`)
- SIDECAR (`b7uu5qeys`)
- SIDECAR_BM25_ONLY (`binu49n17`)
- SIDECAR_RESOLVED_EMBED (`bcu2j7dvy` after a fix-the-choices-list retry)

Once ingest lands → 4 segment searches + 1 raw-events search (on bo-natural's DB via `--answer-with-raw-events`) → 5 evals with qwen judge + mem0-classic.

**Methodology pin while on this stack:** continue n=1 mini-classic diagnostics per `feedback_eval_scope_minimal`. The qwen accuracy levels will be different (better or worse than gpt-5-mini — unknown until we measure), but RELATIVE deltas across SIDECAR variants on the same stack are what matters for iteration. The c124 view (excluding open-domain cat3) and the raw-events ceiling row stay in every comparison.

**When the OpenAI quota resets → switch back:**
- Drop `--base-url` / `--api-key` flags
- Swap models back: `--segmenter-model gpt-5.4-nano`, `--model gpt-5-mini`, `--judge-model gpt-5-mini`
- `--embedding-model text-embedding-3-small`
- Re-validate the SIDECAR winner from this round on the gpt stack to confirm it still beats bo-natural there.

---

# SWISS-TOURNAMENT RERANK BRANCH (experimental fork "swiss")

Branch goal: can a Swiss-system tournament of pairwise LM judgments
produce a top-k ranking that (a) beats the initial embedding ranking and
(b) eventually a frontier dedicated reranker, at sub-quadratic
super-linear cost (Theta(N log N))? Per user: if it can't beat a frontier
reranker it's pointless. Cost viability is an explicit gate.

## Setup (locked)
- Artifact: `swiss-textwhole-c2sub.sqlite` (+ `.vec`): recursive
  `TextSegmenter` + `WholeTextDeriver`, embeddings only, NO BM25 fusion,
  `text-embedding-3-small`, neighbor-window 2 both. Built LLM-free in 19s.
- Arms (all reorder the SAME embedding pool): `embedding` (baseline) vs
  `swiss`. Frontier-reranker arm DEFERRED (no AWS creds; sentence-
  transformer CE rejected by user as not a fair frontier stand-in).
- Comparator: gpt-5-nano, reasoning_effort low, single randomized A/B
  order (position-bias spread, not eliminated).
- Swiss: seed round-1 pairing by embedding score; R = ceil(log2(pool));
  adjacent pairing by standing, avoid rematch, bye=+0.5; final sort by
  (points, Buchholz, embed seed).
- Metric: judge-free recall@k + nDCG@k of gold evidence. Gold msg ts =
  session_dt + (turn-1)s (matches ingest). Pool item is gold if seed seg
  ts in gold set. `gold_ranks` stored per arm -> recompute recall@any-k
  post hoc.

## Files (all in evaluation/event_memory/locomo/)
- `swiss_rerank_probe.py` — standalone probe (no Reranker/EventMemory
  wiring touched; uses reranker=None EventMemory only to fetch the pool).
- `swiss-textwhole-c2sub.sqlite(.vec)` — the artifact.
- `log-swiss-probe-c1c2-n30.out`, `swiss-probe-c1c2-n30-k10.json` — first
  real run (pool 30, k 10, c1+c2, n=100).

## Early findings (smoke, n=5 — NOT conclusive)
- COST: pool-16 Swiss w/ gpt-5-nano ~= $2.9/1000q = 1.5x Cohere
  rerank-v3.5 (~$2/1000q). gpt-5-nano cheap enough to be same ballpark.
  Scales ~linearly with comparisons: pool-30 ~= 2.3x more -> ~3-4x Cohere.
- NOISE: identical seed+config, two runs gave Swiss nDCG 0.984 then 0.914
  (flipped from beating to losing vs embedding). gpt-5-nano pairwise
  judgments are stochastic; small-n estimates swing hard. Need n>=100 and
  likely variance-reduction (temp=0 if supported / both-orders / more
  rounds) before any conclusion.

## Open questions / next
1. Does Swiss beat embedding on recall@10 / nDCG@10 at n=100, pool 30
   (recall-pressure regime where gold is sometimes outside top-k)?
2. Variance: is the gain (if any) larger than run-to-run nano noise?
3. Frontier-reranker quality bar still UNTESTED — needs Cohere creds or a
   frontier API reranker. Until then we only know vs embedding + cost.
4. If promising: cross-encoder-seeded short Swiss (hybrid) is the likely
   cost-effective winner (seed by reranker, spend LM only on top contenders).

## RESULTS (n=100, c1+c2, pool 30, k=10, seed 42) — VERIFIED

Embedding baseline ordering verified correct (first-gold-rank decays from
rank 0; not reversed). gold_not_in_pool = 19/119 (reranking can't fix).

| arm                       | recall@10 | nDCG@10 | $/1000q | latency/call p50 |
|---------------------------|-----------|---------|---------|------------------|
| embedding                 | 0.729     | 0.575   | ~0      | n/a              |
| Swiss 5-nano effort=low   | 0.981     | 0.868   | 6.59    | 1.29s            |
| Swiss 5-nano effort=min   | 0.920     | 0.782   | 1.10    | 0.82s            |
| Cohere rerank-v3.5         | PENDING   | PENDING | 2.00*   | ~0.1-0.3s batch  |
| Swiss 5.4-nano effort=none| PENDING   | PENDING | ~3.5**  | 0.45s            |

*Cohere list price (flat per query <=100 docs). **5.4-nano input $0.20/1M
(4x 5-nano) so none-effort is input-dominated -> pricier than 5-nano-min.

### Findings
- Swiss MASSIVELY beats embedding: +25pp recall (low), +19pp (minimal).
  VERIFIED real (19/19 moved into top10, 0 out; far above noise at n=100).
  The n=5 smoke "noise" was a small-sample artifact, NOT comparator noise.
- Reasoning buys quality: low->minimal loses 6pp recall / 9pp nDCG, saves 6x $.
- COST (corrected per-model prices): 5-nano-min = $1.1 (0.55x Cohere);
  5-nano-low = $6.6 (3.3x); 5.4-nano-none input-dominated ~$3.5 (1.8x).
  5-nano-minimal is the cost play.
- LATENCY is the structural problem: Swiss runs R=5 rounds SEQUENTIALLY.
  Real wall time ~10.3s/question (minimal) -> within-round concurrency only
  ~6-7x, not 15x. vs cross-encoder single batch ~0.1-0.3s. 10-50x latency
  tax inherent to tournament structure. Fine offline; likely dealbreaker online.
- 5.4-nano ~2x faster than 5-nano at every effort (faster model).

### STILL OPEN (the actual bar)
- Does Swiss beat COHERE on recall@10/nDCG@10? (cohere run pending) — this
  is the real viability question; beating embedding is the easy half.
- Caching lever (untested): pool-in-prefix + index-reference + randomized
  suffix could cut input ~10x (clears 1024-tok threshold, keeps position-bias
  randomization). Risk: index-reference may degrade judge accuracy.

## DECISIVE RESULT: Swiss BEATS Cohere rerank-v3.5 (n=100, c1+c2, pool30, k10)

| arm                    | recall@10 | nDCG@10 | $/1000q | lat/call | beats Cohere |
|------------------------|-----------|---------|---------|----------|--------------|
| embedding              | 0.729     | 0.575   | ~0      | -        | -            |
| Cohere rerank-v3.5     | 0.934     | 0.782   | 2.00    | ~0.2s    | (bar)        |
| Swiss 5-nano minimal   | 0.920     | 0.782   | 1.10    | 0.82s    | NO (recall -1.4pp, nDCG tie) |
| Swiss 5.4-nano none    | 0.944     | 0.808   | 3.58    | 0.45s    | YES +1.0/+2.6pp |
| Swiss 5-nano low       | 0.966*    | 0.848   | 6.59    | 1.29s    | YES +3.2/+6.6pp |

*Swiss-low: 0.981 (run1) / 0.966 (run2) — ~1.5pp noise; both beat Cohere -> robust.

VERDICT:
- QUALITY: Swiss beats the frontier reranker (decisive at low, modest at
  5.4-none, ~parity at minimal). Both categories. Clears the user's bar.
- COST: 0.55x (minimal, ~parity quality) to 3.3x (low). Single-digit mult.
- LATENCY: R=5 sequential rounds -> 2-10s/query vs Cohere ~0.2s. 10-50x tax,
  inherent to tournament structure. The real viability blocker.
- VIABLE for OFFLINE/BATCH rerank (5.4-nano-none = sweet spot: beats Cohere,
  fastest/call, 1.8x cost). NOT for low-latency online serving as-is.

OPEN FOLLOW-UPS:
1. End-to-end QA validation: does +3pp recall@10 -> higher c124 answer acc?
   (main-thread says ranking is the dominant lever; this is the payoff test)
2. Latency levers: fewer rounds (resolve top-k only, skip tail rounds);
   smaller pool. Round-sequential floor remains.
3. Caching lever (input cost): pool-in-prefix + index-ref + randomized suffix
   (~10x input cut, keeps position-bias randomization; risk: index-ref accuracy).
4. Position bias: only single randomized order tested. both-orders-tie untested.
5. Generalize beyond c1+c2 (c4 single-hop), beyond pool30/k10, beyond c2sub.

## SINGLE-PHASE SPARSE RESULT (n=100, c1+c2, pool30, k10, d=5, 5-nano low)

One phase of 75 random comparisons -> 4 denoisers in-memory (free). Same
budget+comparator as Swiss-low. Edges UNIFORM RANDOM (seed-independent);
seed enters only at denoising (HodgeRank lambda-I, PPR teleport, Borda tiebreak).

| arm                  | recall@10 | nDCG@10 | $/1000q | vs Cohere |
|----------------------|-----------|---------|---------|-----------|
| embedding            | 0.729     | 0.575   | ~0      | -         |
| Cohere rerank-v3.5   | 0.934     | 0.784   | 2.00    | (bar)     |
| sparse hodge_seeded  | 0.977     | 0.743   | 6.33    | +4.3pp R  |
| sparse hodge         | 0.963     | 0.729   | 6.33    | +2.9pp R  |
| sparse borda         | 0.953     | 0.765   | 6.33    | +1.9pp R  |
| sparse rc_seeded(PPR)| 0.929     | 0.750   | 6.33    | -0.5pp R  |
| Swiss 5-nano low(ref)| 0.966     | 0.848   | 6.33    | +3.2pp R  |

mean_intransitivity = 0.23 (robust). elapsed 834s/100q = 8.3s/q.

FINDINGS:
- Non-adaptive single-phase MATCHES/BEATS adaptive Swiss on recall@10
  (0.977 vs 0.966) at identical budget+comparator. Premise validated.
- LATENCY: 8.3s/q vs Swiss 23.3s/q = 2.8x faster (no round barriers).
  Same cost. The online-serving latency objection is largely removed.
- DENOISER (one run, 4 denoisers): seeded HodgeRank wins recall (0.977),
  seeding helps it (+1.4pp vs unseeded). PPR worst (bakes cycles into walk).
  Borda best nDCG. Intransitivity 0.23 explains it: HodgeRank projects out
  curl, Borda SST-robust; PPR corrupted by cycles. Theory confirmed.
- CAVEAT: sparse nDCG@10 (~0.74-0.77) < Swiss 0.848. Single-phase gets gold
  INTO top-10 (recall) but orders coarser WITHIN top-10. Fine for QA (top-10
  to answerer); worse if intra-top-k order matters.

CARRY FORWARD: sparse single-phase + seeded HodgeRank. Comparator 5-nano-low.
NEXT (user sequencing: base works -> now decrease): drop budget d=4,d=3
(seed fills disconnection); try cheaper/faster comparator 5.4-nano-none;
later seed-biased (boundary) edge selection. Watch recall AND nDCG.

## ZERO-REASONING SWEEP: sparse + seeded-HodgeRank vs comparator (n=100, d=5)

| comparator        | intransit | recall@10 | nDCG@10 | $/1000q | s/q  | vs Cohere |
|-------------------|-----------|-----------|---------|---------|------|-----------|
| Cohere rerank-v3.5| -         | 0.934     | 0.782   | 2.00    | <1s  | (bar)     |
| 5-nano minimal    | 0.39      | 0.924     | 0.619   | 1.10    | 4.6  | -1.0pp X  |
| 5.4-nano none     | 0.31      | 0.965     | 0.756   | 3.58    | 3.9  | +3.1pp OK |
| 5-nano low        | 0.23      | 0.977     | 0.743   | 6.33    | 8.3  | +4.3pp OK |

KEY FINDINGS:
- INTRANSITIVITY (HodgeRank curl, measured free) PREDICTS recall monotonically:
  0.23->0.977, 0.31->0.965, 0.39->0.924. Free leading indicator of judge quality.
- Robust denoiser rescues zero-reasoning ONLY if judge good enough: 5.4-none
  (intrans .31) -> 0.965 > Cohere; 5-nano-min (intrans .39) -> 0.924 < Cohere
  (denoiser lifts it but noise ceiling beats it). Earlier "rescue" claim holds
  for 5.4-none, NOT 5-nano-min.
- WINNER: 5.4-nano-none + sparse single-phase + seeded HodgeRank. Beats Cohere
  (+3.1pp), fastest (3.9s/q; sub-1s reachable w/ full fan-out), $3.58/1000q.
- Cost-floor 5-nano-min ($1.10) does NOT clear the bar. Need >=5.4-none judge.
- LATENCY still concurrency-capped (~6-7x). True sub-1s needs raising the
  AsyncOpenAI/httpx connection cap so all ~75 comparisons fan out at once
  (per-call floor 0.45s for 5.4-none). Engineering, not algorithm.

DENOISER across all runs: seeded HodgeRank consistently best on recall;
seeding helps; PPR weakest under high intransitivity. Borda best nDCG.

## LATENCY FIX: drop + raised pool (THE latency objection is closed)

Diagnosis (burst test, 5.4-none): the >1s wall was NOT uniform slowdown.
Median call stays ~0.5s at any burst; the killer was (a) client-side ~10
concurrency ceiling and (b) SDK backoff-RETRIES on transient timeouts ->
15s stragglers. asyncio.gather blocks on the slowest.
Fix: http-pool=200 + max_retries=0 -> burst75 wall 15.3s->2.7s, eff_conc
4.2->19.3. Then per-call timeout=1.5s DROPS stragglers (HodgeRank lam-I
tolerates incomplete graph).

OPTIMIZED WINNER (n=100): sparse d=5, 5.4-nano-none, seeded HodgeRank,
pool200, drop@1.5s:
  recall@10=0.976 (+4.2pp vs Cohere 0.934), nDCG@10=0.716, $3.47/1000q,
  median query-rerank=1.61s, dropped=2.3/75 (3%), intransitivity=0.29.
  Dropping stragglers did NOT hurt (recall 0.965->0.976) -- robust denoiser
  absorbs missing edges for free. Latency 3.9s->1.6s; ~15x faster than
  Swiss-low (23s). Now in Cohere's latency class.

New CLI: --http-pool N, --call-timeout S. Summary adds median_query_rerank_s,
mean_dropped_per_q.

REMAINING: (1) strict sub-1s via timeout=1.0s (drop ~5-8%, likely fine);
(2) nDCG 0.72 < Swiss 0.85 (coarse intra-top-10); (3) END-TO-END QA: does
+4pp recall -> higher c124 answer accuracy? = the real payoff, still untested.
(4) budget reduction d=3,4 (seed fills disconnection) for lower cost/latency.

## STRICT SUB-1s ACHIEVED (hosted OpenAI, beats Cohere)

Degree-reduction sweep FAILED both ways: latency is timeout-bound not
degree-bound (d=2 m=30 and d=4 m=60 both ~1.05s at pool200), AND cutting
degree cratered quality (d=4=0.931, d=3=0.903 < Cohere 0.934; vs d=5=0.976).
Comparison density is load-bearing; the seed does NOT fill the gap below d=5.
(d=2 intransitivity 0.073 is misleading: too few edges to observe cycles.)

CORRECT lever = keep d=5 (redundant 75-cmp graph) + tighten timeout-drop:

| d=5 timeout | rerank_s | dropped/75 | recall@10 | nDCG | $/1000q | vs Cohere |
|-------------|----------|------------|-----------|------|---------|-----------|
| 1.5s        | 1.61     | 2.3 (3%)   | 0.976     | .716 | 3.47    | +4.2pp    |
| 0.8s        | 0.91     | 17.2 (23%) | 0.950     | .684 | 2.75    | +1.6pp OK |
| 0.7s        | 0.79     | 25.5 (34%) | 0.934     | .686 | 2.36    | tie       |

WINNER (strict sub-1s, hosted): d=5 + 0.8s drop -> 0.91s/q, recall 0.950
(+1.6pp vs Cohere), $2.75/1000q. 5.4-nano-none + seeded HodgeRank + pool200.
Drop 23% of edges -> lose only 2.6pp (redundancy + HodgeRank lam-I reconstruct).
0.7s too aggressive (recall falls to Cohere parity).

LATENCY VERDICT: hosted sub-1s reranker that BEATS frontier Cohere on recall.
Cohere still ~2-4x faster absolute (~0.2-0.5s dedicated batched serving);
true parity needs self-host + continuous batching (vLLM) or distill the
5.4-none pairwise judge into a small batchable model / cross-encoder.

NEXT (the real payoff, still untested): END-TO-END QA -- does +1.6 to +4pp
recall@10 over Cohere -> higher c124 answer accuracy?

## COST CORRECTION: dropped calls ARE billed (probe undercounts)

FLAW: probe records dropped (timed-out) calls as 0 tokens (never receives
usage obj). But the HTTP request was SENT -> OpenAI bills server-side work
regardless of client cancellation. So reported $/1000q EXCLUDE dropped
stragglers = UNDERCOUNT.

All 75 comparisons are sent regardless of timeout; dropping only stops the
CLIENT WAITING, not the server billing. So true cost ~FLAT across timeouts:
  timeout 1.5 -> ~$3.58   0.8 -> ~$3.57   0.7 -> ~$3.58  (vs reported 3.47/2.75/2.36)
TIMEOUT IS A LATENCY LEVER, NOT A COST LEVER. Earlier "cheaper at tighter
timeout" was an accounting artifact.

Corrected: sub-1s winner d=5+0.8s drop = 0.91s/q, recall 0.950 (+1.6pp vs
Cohere), TRUE cost ~$3.6/1000q (~1.8x Cohere $2.00), not $2.75.

Uncertainty: OpenAI billing of client-cancelled requests undocumented; input
tokens ~certainly billed, output maybe-partial. ~$3.6 assumes full input bill.
Verify via usage dashboard vs probe-counted tokens. TODO: have probe estimate
dropped-call input cost (input tok deterministic = ~212 x dropped).

## PROMPT SENSITIVITY (qualifies all "+4pp vs Cohere" numbers)

Comparator prompt is NOT a free variable. d=5 5.4-none seeded-HodgeRank:
| prompt                       | seeded-hodge | best denoiser | intransitivity |
| domain-specific (orig)       | 0.976        | hodge 0.976   | 0.291          |
| fully generic v1             | 0.930        | hodge 0.947   | 0.310          |
| generic v2 (+constraint enum)| 0.934        | borda 0.947   | 0.327          |
Cohere bar 0.934.

FINDINGS:
- The +4pp-over-Cohere edge was substantially LoCoMo-PROMPT-SPECIFIC
  ("memories from a person's chat history; right time, right subject").
  Fully generic prompt ~ties Cohere (0.934 seeded / 0.947 borda).
- Genericness raises intransitivity monotonically (0.29->0.33): vaguer task
  -> noisier/more-cyclic judge. Domain prompt = more consistent comparisons.
- Best DENOISER shifts with judge noise: low intrans -> seeded HodgeRank;
  high intrans (generic) -> Borda (SST-robust counting). Confirms theory.
- A truly generic reranker should expect ~Cohere parity, NOT +4pp. The edge
  comes from domain adaptation.

RECONCILIATION (TODO): generic STRUCTURE + one-line {domain_hint} param
("items are: <X>") -- domain-agnostic code, per-deployment priming. Standard
general-reranker pattern (Cohere/Voyage have optional instruction fields).
Likely recovers most of the gap without hardcoding. = the version to ship/distill.

## PROMPT FRAMING RESOLVED: "memories from a history" is the shippable prompt

d=5 5.4-none, n=100. Framing ladder (seeded HodgeRank recall@10):
  generic "piece of text"          0.930  (none)      ~tie Cohere
  generic + constraint enum        0.934  (none)      ~tie
  "memories from a history" +constr 0.960  (mem+hist)  +2.6pp  <-- SHIP
  original "person's chat history"  0.976  (chat+person) +4.2pp

DECOMPOSITION of the +4.2pp edge: +2.6pp from just "these are MEMORIES
retrieved from a HISTORY" (domain-general: logs/transcripts/timelines, not
just chat); +1.6pp residual from LoCoMo specifics (person/chat/right-time).

SHIPPABLE: "memories from a history" framing + generic constraint-matching
("match every entity/name/time/place/quantity/attribute/condition the QUERY
expresses, not just topic"). Zero chat/person assumptions, beats Cohere +2.6pp.
Optional {context_block} slot (default "") recovers last ~1.6pp per-deployment
via a one-line CONTEXT: hint. New --domain-hint plumbing partially wired
(pairwise_judge has domain_hint param; not yet threaded through tournament/
sparse callers -- TODO if per-deployment hint wanted).

Intransitivity predicts recall WITHIN a fixed judge (effort sweep clean) but
NOISY ACROSS prompts (memhist curl 0.339 > original 0.291 yet recall 0.960<0.976).

## NOISE FLOOR MEASURED (3 seeds) -- validates framing, invalidates small deltas

Config d=5 5.4-none seeded-HodgeRank, n=100, seeds 42/43/44:
  memhist:   0.960/0.966/0.980  mean 0.9685  sd 0.0083  range[.960,.980]
  genericv2: 0.934/0.948/0.952  mean 0.9445  sd 0.0078  range[.934,.952]

NOISE FLOOR sigma ~= 0.8pp (n=100, stochastic 5.4-none judge + drop). So
single-run deltas < ~1.6pp (2 sigma) are NOISE.

VALIDATED (real, above noise):
- Framing memhist vs genericv2: +2.4pp, non-overlapping ranges, ~3.6 sigma. REAL.
- memhist beats Cohere: +3.5pp mean, worst seed 0.960 > 0.934. REAL.
- embedding(0.73) -> tournament(0.97). REAL (huge).

INVALIDATED / DOWNGRADED (within noise, were single-run):
- GENERIC prompt only TIES Cohere: genericv2 mean 0.9445 = +1.1pp (~1.4 sigma),
  worst seed = 0.934 exactly. NOT a reliable win. The FRAMING makes the win.
- Swiss "5.4-none +1.0pp over Cohere" (0.944 vs 0.934): within noise, not reliable.
- ~1pp denoiser gaps between runs: mostly noise. (seeded-Hodge vs Borda choice
  is still directionally tied to intransitivity, but small per-run gaps are noisy.)

METHOD NOTE: most earlier branch numbers were SINGLE-RUN. Headline claims that
survive replication: tournament>>embedding; memhist framing real; memhist>Cohere
by ~3.5pp. Generic-prompt>Cohere is NOT established (parity).
New: --prompt-variant {memhist,genericv2}, --seed.

## PRIOR ART: this is NOT novel -- independent re-derivation of 2023-25 work

The branch's core ideas are published. Do NOT claim novelty.
- PRP (Qin 2023, arxiv 2306.17563): pairwise ranking prompting, the foundation.
- PRP-Graph (ACL 2024, aclanthology 2024.acl-long.313): pairwise LLM comparisons
  selected by SWISS-SYSTEM -> ranking graph -> graph aggregation. ~EXACTLY our
  Swiss-tournament+graph-agg design (they even say "Swiss-System"; their twist =
  LLM output-probability as edge confidence).
- JointRank (SIGIR ICTIR 2025, arxiv 2506.22262): SINGLE-PASS parallel blocks ->
  implicit pairwise -> aggregate via Winrate/PageRank(=Rank Centrality); latency
  21s->8s. = our single-phase + rank-agg + latency arc, published.
- LLM-RankFusion (2406.00231): LLM ranking intransitivity + aggregation fix.
- Batching&Caching paper (2505.24643): the prompt-caching cost lever.
- Real-Time Pairwise Reranking (2511.07555): the latency angle.

Mildly differentiated (NOT clearly novel): HodgeRank aggregator + curl as
intransitivity diagnostic (vs PRP-Graph confidence-graph / JointRank Winrate-
PageRank); embedding-seeded aggregation; straggler-drop. HodgeRank rank-agg is
classical (Jiang 2011). Don't bank on novelty without deeper search.

VALUE of this branch = empirical characterization on memory/LoCoMo (noise floor,
cost-accounting correction, prompt-framing decomposition), NOT the algorithm.
If pursued: start from PRP-Graph/JointRank, don't reinvent. Open Q for us =
untested end-to-end QA payoff.

## REPRODUCTION VERDICT: at N=30, 1 listwise call BEATS the pairwise tournament

Reproduced JointRank/RankGPT-family (single listwise call ranks whole pool)
vs our pairwise tournament. n=100, 5.4-none, memhist framing:

| method                 | calls/q | recall@10 | nDCG@10 | lat   | $/1000q |
| embedding              | 0       | 0.729     | 0.575   | -     | ~0      |
| Cohere rerank-v3.5     | 1 API   | 0.934     | 0.782   | <1s   | 2.00    |
| sparse tournament      | 75      | 0.969     | 0.757   | ~1.6s | ~3.6    |
| LISTWISE 1-call        | 1       | 0.960     | 0.860   | 1.05s | 0.39    |

Single listwise call: ties tournament recall (0.960 vs 0.969, ~1 sigma),
BEATS nDCG (+10pp, best of any method), ~9x cheaper, faster. => at N=30 the
pairwise tournament is OVERKILL; just show all 30 to the model at once.
nDCG win because listwise produces a true total order (tournament's weakness).

Pairwise/aggregation only earns its keep when N EXCEEDS one context window
(JointRank blocking / RankGPT sliding-window regime). At pool-30 reranking,
listwise dominates.

CAVEATS: listwise recall is 1 run (low end of memhist band [.960,.980]) -
replicate to confirm recall tie; nDCG/cost/latency wins decisive regardless.
N=30 specific; flips toward pairwise/blocks as N grows + listwise lost-in-middle.

IMPLICATION: for memory reranking at modest pool sizes, ship a single listwise
call, not the tournament. PRP/PRP-Graph/our-tournament are for large-N.

## LARGE-N (count) REPRODUCTION: single-listwise dominates up to 200; blocking did NOT win

LoCoMo c1+c2, 5.4-none, memhist, top_k=10:
| method            | p100 R@10/nDCG | p200 R@10/nDCG | lat    | $/1k    | calls |
| embedding         | .602/.495      | .572/.476      | -      | ~0      | 0     |
| Cohere (pointwise)| .824/.716      | (chunked)      | <1s    | 2       | N/100 |
| single-listwise   | .892/.820      | .857/.791      | 1.2-1.4| 1-1.8   | 1     |
| JointRank pr-seed | .916/.744      | .867/.645      | 6-10s  | 3.9-7.7 | 15-30 |

FINDINGS:
- single-listwise degrades gracefully w/ count: .960(p30)/.892(p100)/.857(p200),
  best nDCG throughout, 1 call, cheap, fast. Beats Cohere at every pool.
- JointRank recall edge SHRANK +2.4pp(p100)->+1.0pp(p200)=noise; nDCG WORSE
  (.744->.645); 9x slower (fan-out tail, 30 calls), 4x cost. At p200 single-
  listwise wins outright (tied recall, better nDCG, 9x faster, 4x cheaper).
- Blocking didn't pay off b/c LoCoMo docs ~45tok -> listwise of 200 = ~9k tok,
  still inside reliable-ranking range; lost-in-middle not biting yet.
- JointRank latency = fan-out TAIL (max-of-N): min block call 1.3s ~= single
  listwise, but waits for slowest of 15-30 -> p50 6-10s. Fixable w/ drop (not
  applied here), but fan-out structurally can't beat 1 round-trip.

KEY: crossover (blocking>listwise) governed by TOTAL PROMPT TOKENS (N x docsize),
NOT count. LoCoMo (200x45=9k) listwise wins. User's regime (200-300 x hundreds
of tok = 60-90k) is where blocking SHOULD win -- UNTESTED (LoCoMo docs too small).
Need larger-doc corpus to verify candidate-SIZE dimension.

OVERALL BRANCH VERDICT: for memory reranking at modest pools/small docs, ONE
listwise call beats tournament/jointrank/cohere on cost+latency+nDCG, ties on
recall. Tournament/blocking are large-token-budget tools. None of this is novel
(PRP/PRP-Graph/JointRank/RankGPT). Untested payoff: end-to-end QA.
