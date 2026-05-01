# Context-enriched embeddings (stacked merge)

Ingest-time: for each turn, embed `{prev} [SEP] {curr} [SEP] {next}` (and
wider / asymmetric variants). Query-time: run v2f, then stacked-append
context-index hits in score order — no displacement of v2f's own top-K
picks. Zero per-query LLM overhead beyond v2f. Addresses the earlier
failure of max-score merge with context-enriched / alt-key supplements.

## Index stats

| variant | dataset | unique entries | convs |
|---|---|---:|---:|
| window_1  | locomo_30q     | 2687 | 8 |
| window_1  | synthetic_19q  | 462  | 5 |
| window_2  | locomo_30q     | 2687 | 8 |
| window_2  | synthetic_19q  | 462  | 5 |
| prev_only | locomo_30q     | 2687 | 8 |
| prev_only | synthetic_19q  | 462  | 5 |

(One enriched entry per turn per variant; 1 turn-text produced no
prev_only entry for turn 0 in 5 synth convs but was then filled by the
curr-only fallback — final counts all match 462.)

## Fair-backfill recall (arch vs meta_v2f baseline)

| Arch | Dataset | v2f r@20 | ctx r@20 | Δ@20 | v2f r@50 | ctx r@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f                 | locomo_30q    | 0.756 | —     |  —     | 0.858 | —     |  —     | 1.0 |
| contextemb_w1_stacked    | locomo_30q    | 0.756 | 0.756 | +0.000 | 0.858 | **0.867** | **+0.009** | 1.0 |
| contextemb_w2_stacked    | locomo_30q    | 0.756 | 0.756 | +0.000 | 0.858 | 0.858 | +0.000 | 1.0 |
| contextemb_prev_stacked  | locomo_30q    | 0.756 | 0.756 | +0.000 | 0.858 | 0.858 | +0.000 | 1.0 |
| contextemb_w1_bonus      | locomo_30q    | 0.756 | 0.756 | +0.000 | 0.858 | **0.867** | **+0.009** | 1.0 |
| meta_v2f                 | synthetic_19q | 0.613 | —     |  —     | 0.851 | —     |  —     | 1.0 |
| contextemb_w1_stacked    | synthetic_19q | 0.613 | 0.613 | +0.000 | 0.851 | **0.863** | **+0.012** | 1.0 |
| contextemb_w2_stacked    | synthetic_19q | 0.613 | 0.613 | +0.000 | 0.851 | 0.856 | +0.005 | 1.0 |
| contextemb_prev_stacked  | synthetic_19q | 0.613 | 0.613 | +0.000 | 0.851 | 0.854 | +0.003 | 1.0 |
| contextemb_w1_bonus      | synthetic_19q | 0.613 | 0.613 | +0.000 | 0.851 | **0.863** | **+0.012** | 1.0 |

K=20 is always 0-delta: v2f produces ~30 segments in stacked order, so
its top-20 fills the budget before any context-hits are appended. K=50
is where the mechanism can fire.

## Mechanism firing rate

| variant | dataset | mean novel ctx-hits/q | fired at K=20 | fired at K=50 | ctx-hit contributed gold @K=50 |
|---|---|---:|---:|---:|---:|
| w1_stacked  | locomo_30q    | 4.50 | 0/30 | 29/30 | 1/30 |
| w1_stacked  | synthetic_19q | 2.68 | 0/19 | 15/19 | 4/19 |
| w2_stacked  | locomo_30q    | 5.47 | 0/30 | 30/30 | 0/30 |
| w2_stacked  | synthetic_19q | 3.16 | 0/19 | 17/19 | 4/19 |
| prev_stacked| locomo_30q    | 4.13 | 0/30 | 30/30 | 0/30 |
| prev_stacked| synthetic_19q | 1.95 | 0/19 | 16/19 | 3/19 |
| w1_bonus    | locomo_30q    | 4.50 | 0/30 | 29/30 | 1/30 |
| w1_bonus    | synthetic_19q | 2.68 | 0/19 | 15/19 | 4/19 |

The context-index hits DO enter top-50 almost every query (29–30/30
LoCoMo, 15–17/19 synth). But they rarely contribute gold — 0–4/total.
That is: context enrichment is retrieving plausible turns that weren't
in v2f's top-30, but those turns are rarely the ones needed.

## Orthogonality vs meta_v2f gold (K=50)

| variant | dataset | total_gold | novel vs v2f | frac_novel |
|---|---|---:|---:|---:|
| w1_stacked  | locomo_30q    | 37 | 1 | 2.7% |
| w1_stacked  | synthetic_19q | 123 | 3 | 2.4% |
| w2_stacked  | locomo_30q    | 36 | 0 | 0.0% |
| w2_stacked  | synthetic_19q | 122 | 3 | 2.5% |
| prev_stacked| locomo_30q    | 36 | 0 | 0.0% |
| prev_stacked| synthetic_19q | 121 | 2 | 1.7% |
| w1_bonus    | locomo_30q    | 37 | 1 | 2.7% |
| w1_bonus    | synthetic_19q | 123 | 3 | 2.4% |

Low novelty. Context index is mostly rediscovering turns v2f already
finds. A handful of novel gold hits (1 LoCoMo, 3 synth for w1) drive
the +0.009 / +0.012 deltas.

## Per-category Δ vs meta_v2f at K=50 (window_1 variant)

| cat | dataset | n | v2f r@50 | ctx r@50 | Δ |
|---|---|---:|---:|---:|---:|
| locomo_single_hop | locomo_30q    | 10 | 0.825 | **0.850** | **+0.025** |
| locomo_multi_hop  | locomo_30q    | 4  | 0.875 | 0.875 | +0.000 |
| locomo_temporal   | locomo_30q    | 16 | 0.875 | 0.875 | +0.000 |
| completeness      | synthetic_19q | 4  | 0.865 | **0.885** | **+0.019** |
| inference         | synthetic_19q | 3  | 0.939 | **0.970** | **+0.030** |
| procedural        | synthetic_19q | 2  | 0.661 | **0.690** | **+0.030** |
| conjunction       | synthetic_19q | 3  | 1.000 | 1.000 | +0.000 |
| control           | synthetic_19q | 3  | 1.000 | 1.000 | +0.000 |
| proactive         | synthetic_19q | 4  | 0.643 | 0.643 | +0.000 |

w2_stacked shows -0.048 on synthetic `conjunction` (1 question loss;
wider context adds noise) — tradeoff of more context.

## Bonus variant (+0.05 score) vs pure stacked

w1_bonus produces identical results to w1_stacked. Expected: at K=50
there are always enough empty slots for all 10 context hits to be
appended regardless of score ordering; the bonus only affects the
order within the appended tail, which doesn't change the set of turns
present in the top-50.

## Comparison to prior LLM / regex alt-key tests

Those lost because max-score merge displaced v2f's clean hop0 picks.
Stacked merge (this test) loses 0 at K=20 and gains small-but-nonzero
at K=50. The substrate insight — stacked append > max-score — is
confirmed: context-enrichment is not *harmful*, it's just not
*impactful*.

## Sample retrievals (window_1)

- LoCoMo single-hop "What is Caroline's identity?" (gold=[4]):
  v2f finds turn 4 via cue "I found my true self — I'm a trans woman";
  ctx appends turns [3, 1, 185, 78] — neighbors of v2f hits, not gold.
- Synth completeness "List all smart home devices..." (gold=[8,14,...]):
  v2f finds 4/5; ctx appends one additional gold turn via window_1 that
  was on the boundary of v2f's cue2 retrieval.

## Verdict

**NARROW**. Context-enriched embeddings + stacked merge is a
marginally-positive addition:
- Never hurts at K=20 (never displaces v2f's primary picks).
- +0.009 LoCoMo K=50 (1 novel gold hit across 30 questions; noise-level).
- +0.012 synth K=50 (3 novel gold hits; borderline significant at n=19).

**Best variant**: `window_1` (symmetric 1-wide context). `window_2` is
neutral to slightly harmful (wider context = more generic embedding,
more redundant hits). `prev_only` is neutral-to-null.

**Not ship as primary**. The mechanism fires (context hits enter top-50
in 80-100% of queries) but those hits rarely *are* the gold turns that
v2f missed. Low orthogonality (2-3% novel) means the context index is
mostly retrieving the same conversational neighborhood v2f already
finds via cues.

**Could ship as part of an ensemble** alongside other low-cost stacked
supplements (stacked_alias, critical_info_store) — individually small,
collectively non-trivial at K=50. Zero per-query LLM overhead, ingest
cost ≈ $0.06 per dataset.

**Reinforces the stacked-merge insight**: max-score merge (prior
alt-key experiments) loses because it displaces v2f; stacked merge
gains (small) because it only supplements. The merge strategy is the
decisive factor.

## Output files

- `results/context_embedding.json` — raw summaries / diagnostics /
  orthogonality
- `results/context_embedding.md` — this report
- `results/context_embedding_<arch>_<ds>.json` — per-(arch,dataset)
  details including fair_backfill, ctx_appended_turn_ids, gold_found_at_K
- `results/context_embedding_index.json` — per-variant index samples
- `context_embedding.py`, `contextemb_eval.py` — source

## Index / budget realisation

- 3 variants × 3149 turns ≈ 9447 enriched embeddings ingested (one
  batch per store/variant, cached in `cache/contextemb_embedding_cache.json`)
- Zero new LLM calls (v2f cues served from warm cache — our LLM cache
  read order mirrors antipara_cue_gen for bitwise-identical v2f runs)
- Spend: embeddings ~$0.19; LLM $0. Under $0.50 cap.
