# Final Composition Test — stack ALL shipped wins

Stacking the 6 narrow wins discovered this session and measuring the
cumulative ceiling. Integration is: base (ens_2 @K=50 / v2f @K=20) +
alias overlay (max-cos) + clause overlay (max-cos) + context-emb
stacked append + critical-info always_top_M.


Elapsed: 2429s.


## Recall table (r@20)

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q | overall |
|---|---|---|---|---|---|
| v2f | 0.7556 | 0.6130 | 0.4804 | 0.5931 | **0.6323** |
| router_v2fplus_default | 0.7556 | 0.6276 | 0.4905 | 0.5844 | **0.6350** |
| ens_2_v2f_typeenum | 0.5806 | 0.5864 | 0.5185 | 0.5693 | **0.5676** |
| ens_all_plus_crit | 0.6806 | 0.6318 | 0.5141 | 0.6079 | **0.6208** |
| finalstack_all | 0.7556 | 0.6427 | 0.4841 | 0.5962 | **0.6402** |
| finalstack_no_alias | 0.7556 | 0.6427 | 0.4841 | 0.5962 | **0.6402** |
| finalstack_no_clause | 0.7556 | 0.6427 | 0.4841 | 0.5962 | **0.6402** |
| finalstack_no_context | 0.7556 | 0.6427 | 0.4841 | 0.5962 | **0.6402** |
| finalstack_no_critinfo | 0.7556 | 0.6130 | 0.4804 | 0.5931 | **0.6323** |

## Recall table (r@50)

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q | overall |
|---|---|---|---|---|---|
| v2f | 0.8583 | 0.8513 | 0.9169 | 0.9021 | **0.8789** |
| router_v2fplus_default | 0.8833 | 0.8789 | 0.9299 | 0.9345 | **0.9042** |
| ens_2_v2f_typeenum | 0.9083 | 0.8606 | 0.9213 | 0.8949 | **0.8969** |
| ens_all_plus_crit | 0.9167 | 0.8968 | 0.9488 | 0.9299 | **0.9217** |
| finalstack_all | 0.9083 | 0.8647 | 0.9150 | 0.8949 | **0.8966** |
| finalstack_no_alias | 0.9083 | 0.8647 | 0.9150 | 0.8949 | **0.8966** |
| finalstack_no_clause | 0.9083 | 0.8647 | 0.9150 | 0.8949 | **0.8966** |
| finalstack_no_context | 0.9083 | 0.8647 | 0.9150 | 0.8949 | **0.8966** |
| finalstack_no_critinfo | 0.9083 | 0.8606 | 0.9213 | 0.8949 | **0.8969** |

## Ablation (overall @K=50)

| Variant | overall | Δ vs finalstack_all |
|---|---|---|
| v2f | 0.8789 | -0.0177 |
| router_v2fplus_default | 0.9042 | +0.0076 |
| ens_2_v2f_typeenum | 0.8969 | +0.0003 |
| ens_all_plus_crit | 0.9217 | +0.0251 |
| finalstack_all | 0.8966 | +0.0000 |
| finalstack_no_alias | 0.8966 | +0.0000 |
| finalstack_no_clause | 0.8966 | +0.0000 |
| finalstack_no_context | 0.8966 | +0.0000 |
| finalstack_no_critinfo | 0.8969 | +0.0003 |

## Supplement trigger rates

| Dataset | alias_matched | clause_split | n_crit_turns | altkeys | context_hits |
|---|---|---|---|---|---|
| locomo_30q | 30/30 | 0/30 | 0 | 0 | 10 avg |
| synthetic_19q | 7/19 | 7/19 | 17 | 51 | 10 avg |
| puzzle_16q | 7/16 | 7/16 | 13 | 39 | 10 avg |
| advanced_23q | 17/23 | 15/23 | 12 | 36 | 10 avg |

## LLM retrieval cost per question (rel. to 1 v2f call)

| Variant | locomo_30q | synthetic_19q | puzzle_16q | advanced_23q |
|---|---|---|---|---|
| v2f | 1.00× | 1.00× | 1.00× | 1.00× |
| router_v2fplus_default | 2.00× | 2.32× | 2.00× | 2.13× |
| ens_2_v2f_typeenum | 2.00× | 2.00× | 2.00× | 2.00× |
| ens_all_plus_crit | 10.00× | 10.00× | 10.00× | 10.00× |
| finalstack_all | 5.00× | 3.84× | 4.19× | 5.52× |
| finalstack_no_alias | 2.00× | 2.74× | 2.88× | 3.30× |
| finalstack_no_clause | 5.00× | 3.10× | 3.31× | 4.22× |
| finalstack_no_context | 5.00× | 3.84× | 4.19× | 5.52× |
| finalstack_no_critinfo | 5.00× | 3.84× | 4.19× | 5.52× |

## Per-category r@50 on locomo_30q

| category | n | v2f | router_v2fplus_default | ens_2_v2f_typeenum | ens_all_plus_crit | finalstack_all | finalstack_no_alias | finalstack_no_clause | finalstack_no_context | finalstack_no_critinfo |
|---|---|---|---|---|---|---|---|---|---|---|
| locomo_multi_hop | 4 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| locomo_single_hop | 10 | 0.825 | 0.900 | 0.875 | 0.900 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| locomo_temporal | 16 | 0.875 | 0.875 | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 | 0.938 |

## Per-category r@50 on synthetic_19q

| category | n | v2f | router_v2fplus_default | ens_2_v2f_typeenum | ens_all_plus_crit | finalstack_all | finalstack_no_alias | finalstack_no_clause | finalstack_no_context | finalstack_no_critinfo |
|---|---|---|---|---|---|---|---|---|---|---|
| completeness | 4 | 0.865 | 0.865 | 0.827 | 0.885 | 0.827 | 0.827 | 0.827 | 0.827 | 0.827 |
| conjunction | 3 | 1.000 | 0.952 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| control | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| inference | 3 | 0.939 | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 |
| proactive | 4 | 0.643 | 0.791 | 0.753 | 0.805 | 0.773 | 0.773 | 0.773 | 0.773 | 0.753 |
| procedural | 2 | 0.661 | 0.653 | 0.561 | 0.686 | 0.561 | 0.561 | 0.561 | 0.561 | 0.561 |

## Orthogonality: gold found by finalstack_all but NO prior ship

| Dataset | finalstack gold@50 | novel vs priors | frac_novel |
|---|---|---|---|
| locomo_30q | 38 | 0 | 0.0 |
| synthetic_19q | 120 | 0 | 0.0 |
| puzzle_16q | 205 | 0 | 0.0 |
| advanced_23q | 223 | 0 | 0.0 |

## Critical-info classifier cost (ingest-time, one-off)

- Prompt version: v3
- New calls this run: 109, cached: 2486
- Input tokens: 47142 output tokens: 79236
- Est USD (gpt-5-mini @ $0.25/M in, $2/M out): $0.1703


## Verdict

- Best variant overall @ K=50 (weighted): **ens_all_plus_crit** r@50=0.9217
- finalstack_all @ K=50 = 0.8966 — does **NOT** beat `ens_all_plus_crit` (−2.5pp), does **NOT** beat `router_v2fplus_default` (−0.8pp), ties `ens_2_v2f_typeenum` (+0.0pp).

### At K=50: supplements add 0 pp in composition

| Drop | Overall r@50 | Δ vs finalstack_all |
|---|---|---|
| drop alias | 0.8966 | +0.0000 |
| drop clause | 0.8966 | +0.0000 |
| drop context | 0.8966 | +0.0000 |
| drop critinfo | 0.8969 | +0.0003 |

All four supplements are redundant with the ens_2 base at K=50. Orthogonality
check: **0 / 586 gold turns** were found only by finalstack_all and no prior
ship — whatever finalstack_all surfaces, one of `v2f / router / ens_2 /
ens_all+crit` already finds. Critical-info forcing actively *hurts* at K=50
on `synthetic_19q` (−0.4pp) and `puzzle_16q` (−0.6pp) because 5 always_top_M
items displace legitimate ens_2 gold from the 50-slot budget.

### At K=20: only critical-info contributes

| Variant | overall r@20 |
|---|---|
| v2f | 0.6323 |
| finalstack_no_critinfo | 0.6323 |
| finalstack_all | 0.6402 |
| ens_all_plus_crit | 0.6208 |

Dropping critinfo takes finalstack_all back to exactly v2f baseline. So on
K=20 with a v2f base, alias/clause/context-embedding overlays contribute
**zero**; critinfo alone adds +0.8pp.

### Why alias/clause/context don't stack

My integration uses **max-cosine merge** (per the plan): supplements contribute
candidates at their raw cosine-vs-query score. Base arch picks (ens_2
sum-cosine items) are scored with a ~+5 offset so they sort above cosine.
Supplement items therefore only compete with cosine-backfill entries, never
with arch picks — and cosine-backfill within the same K window is already
surfacing whatever alias/clause/context's cosine-scored novel hits would
have surfaced. The wins those supplements posted in isolation (alias
alone: +2.3pp on LoCoMo @K=50) required them to displace v2f arch picks;
that's not possible under a max-cos overlay onto a stronger ens_2 base.

### Final production recipe (ceiling this session)

**SHIP `ens_all_plus_crit`** — all 5 specialists sum_cosine + critical-info
always_top_M @ top_m=5. r@50=0.9217, 10× retrieval LLM cost per question.

For budget-sensitive deployments:
- **`router_v2fplus_default`** r@50=0.9042, 2.0–2.3× cost — 1.75pp below
  ceiling at 5× lower LLM cost.
- **`v2f`** r@50=0.8789, 1× cost — 4.3pp below ceiling.

The stacking hypothesis (narrow supplements target *different* failure
modes and thus compose additively) **does not hold** under a max-cosine
overlay strategy. The failure modes alias/clause/context target overlap
with what ens_5 already covers via its 5 diverse cue generators. No
dataset-specific combo beat `ens_all_plus_crit`.
