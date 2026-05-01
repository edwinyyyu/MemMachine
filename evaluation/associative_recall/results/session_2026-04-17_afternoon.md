# Session 2026-04-17 afternoon — 2026-04-18 morning — Retrieval Architecture Research

Continuation from overnight 2026-04-16/17 session. Focus: architectural exhaustion of retrieval wins within the `text-embedding-3-small` + `gpt-5-mini` substrate; generalization beyond LoCoMo; shape-robustness to non-question inputs.

## Final production recipe (per-corpus conditional)

```
For each query:
1. If query mentions a conversation participant (regex, zero-LLM)
   → two_speaker_filter  (LoCoMo: +13.6pp K=20, +3.4pp K=50, shape-robust)

2. Else if corpus has timestamp metadata AND query has temporal constraint
   → timestamp-filter only if downstream wants narrow answer (NOT for recall@K benchmarks)

3. Else if ingest-time critical_info tagging fired
   → v2f + critical_info_store.always_top_M  (synthetic fact-dense: +3pp K=20)

4. Else: v2f alone  (universal base, shape-coerces OK)

For K=50 ceiling push on ambiguous queries:
   → gated_threshold_0.7  (confidence-gated displacement from 6 channels, +3.3pp LoCoMo K=50)

Max-effort K=50 ceiling:
   → ens_all_plus_crit  (5 specialists + critical-info, +4.3pp overall K=50, 10× cost)
```

## Benchmark ceilings achieved this session

| Benchmark | K | v2f baseline | Shipped | Oracle |
|---|---|---|---|---|
| LoCoMo-30 | 20 | 0.756 | **0.892** (two_speaker) | 0.908 |
| LoCoMo-30 | 50 | 0.858 | **0.917** (composition_v2) | 0.933 |
| Synthetic-19 | 50 | 0.851 | 0.897 (ens_all+crit) | 0.938 |
| Puzzle-16 | 50 | 0.917 | 0.949 (ens_all+crit) | 0.976 |
| LME-hard-90 | 50 | 0.817 | 0.817 (v2f alone) | — |
| Overall K=50 | 50 | 0.879 | 0.922 (ens_all+crit) | 0.952 |

**96.3% of oracle captured at K=50.** Substrate-saturated.

## New architectural wins this wave

1. **two_speaker_filter** (LoCoMo K=20 +13.6pp, K=50 +3.4pp, zero per-query LLM) — biggest single win. Triggered by participant name mention in query. Shape-robust.
2. **gated_threshold_0.7** (LoCoMo K=50 +3.3pp, 1W/29T/0L) — confidence-gated displacement from 6 channels. LLM outputs per-channel confidence; only channels ≥0.7 fire; they replace v2f's weakest picks (not top picks).
3. **meta_router** (dispatch between two_speaker and gated) — cleanest production recipe, inherits two_speaker's shape-robustness, zero LLM on 60% of queries.
4. **composition_v2_all** (LoCoMo K=50 0.917) — full stack of speaker + router + ens_2 + critical + alias + context + clause.
5. **intent_parser_full** (LoCoMo K=50 +1.7pp, modest) — structured intent parsing with constraint propagation. Weaker than gated but exposes unique signals (preference, list, aggregation).
6. **adaptive ensemble τ=0.1** — cost-saver (99.3% of ens_5 at 33.5% cost).

## Durable substrate insights added to memory

1. **v2f is near a local optimum for single-call cue gen** — elaborations (few-shot, anti-paraphrase, verbatim, MMR, speaker-conditional) consistently no-op or degrade. Minimalism preserves imagination; elaboration pulls cues toward query-space.
2. **Gold is off-center from query but DIFFUSE** (only 36% kNN-adjacency). Local-neighborhood architectures fail; multi-probe approaches win.
3. **Proximity to gold ≠ recall.** Iterative query refinement provably moved queries closer to gold on 100% of questions yet recall dropped — attractor pulls into denser non-gold topic mass.
4. **Dispersion > convergence ONLY across distinct modes.** Ensemble specialists (different prompts) win; MMR cue selection (same prompt) loses.
5. **Merge-strategy hierarchy: confidence-gated displacement > max-score merge > pure-stacked > linear fusion.** Linear fusion (multichannel_weighted) dilutes v2f's positional strength even with sensible LLM-chosen weights.
6. **Dialog entities are mostly VOCATIVES, not subjects.** Entity-mention retrieval inverts the semantic signal. Speaker filtering works because it uses role metadata, not name-in-text match.
7. **Architectures deriving probes from v2f-reachable content uniformly fail.** Anchor expansion, inverse query, iterative refinement, topic segmentation, spreading activation, pronoun resolution all found 0-3.5% novel gold.
8. **Corpus-specific signals don't generalize across benchmarks.** two_speaker fires 0% on LME (no named participants). critical_info tag rate 0.03% on LME (no structured medical/commitment facts). v2f itself generalizes cleanly.
9. **Nano has a capacity floor below what spec + repair can rescue.** Model-agnostic cue spec works for mini (no regression) but doesn't lift nano to ship threshold.
10. **Temporal-reasoning is universally hard** — 0.765 on LME, similar weakness on LoCoMo temporal. Embedding substrate cannot express "after X but before Y". Timestamp metadata doesn't fix it on recall@K benchmarks because gold is set-oriented.

## This wave's abandons (negative results worth keeping as evidence)

**Query-side**: prompt generalization (task-shape doesn't rescue via prompt), contrastive retrieval (v2f already filters paraphrase neighborhood), inverse query generation (0-3.5% novel), anchor-turn expansion (0% novel), iterative query refinement (modal-topic attractor), query-direction projection (queries already off-interrogative axis), MMR cue selection (within-mode dispersion loses relevance), few-shot cue exemplars (LLM imitates exemplar style, fabricates content), anti-paraphrase / verbatim-quote (coercion failure).

**Ingest-side**: regex alt-keys (97% FP), LLM alt-keys (competes with cue gen), pair-level embedding (0% novel, max-score merge hurts), pronoun resolution (stacked has no empty slots), topic segmentation (summaries lose vocabulary), dialogue-act tagging (corpus lacks density), entity-mention exact-match (vocative bug), stacked-merge alias alt-keys (doesn't capture alias_expand_v2f's per-variant mechanism), salience pruning (LLM says everything is askable).

**Integration**: multichannel_weighted linear fusion (dilutes v2f), query-time alias injection (LLM hedges across options), speaker-conditional cue gen (regresses two_speaker alone), clause decomposition (synth multi-clause only), timestamp scoring on LME (too narrow for set-oriented benchmark gold).

## Compositions

- Narrow wins don't stack additively. Once speaker + ensemble are present, other supplements contribute 0pp in ablation.
- `gated_threshold_0.7` and `gated_v2_intent_only` both hit exactly 0.8917 LoCoMo K=50 with entirely different channel sets — the GATED MECHANISM is the pattern, not the specific channels.
- Meta-router (speaker → two_speaker, else → gated) is the cleanest production compromise: matches K=50 ceiling, inherits K=20 breakthrough, 60% zero-LLM.

## Open directions (not pursued this wave)

- Temporal-reasoning beyond date filtering: timeline-construction LLM pass, temporal-aware re-ranker on v2f's broader pool.
- Cross-session entity linking for LME multi-session category.
- Preference_marker channel for LME single-session-preference (predicted to help ~+5pp based on per-signal attribution from intent_parser).
- Downstream reader-grounded eval (we've been on recall@K; actual answer quality may be saturated at lower recall).
- Fine-grained per-query routing (oracle K=20 shows 10.3pp headroom via per-query routing we don't have infrastructure for).

## Session status

**Converged.** K=50 substrate-saturated at 0.952 oracle / 0.922 shipped (96.3% captured). K=20 break-through on LoCoMo via two_speaker. LongMemEval hard categories partly generalize v2f but NOT speaker/critical — need corpus-specific signal channels. Meta-router is the cleanest shipped recipe.
