# Stacked-merge alias alt-keys

Ingest-time alias substitution + separate-index + stacked-merge retrieval. Cheaper query-time alternative to `alias_expand_v2f` (which runs v2f on each alias variant, ~3x LLM). This test moves the alias work to ingest and keeps per-query cost equal to plain v2f.

## Alt-key index stats

### locomo_30q

- alt-keys raw: 6722
- alt-keys unique: 6691
- convs with alt-keys: 8/8
- turns with >=1 alias match: 1364

### synthetic_19q

- alt-keys raw: 731
- alt-keys unique: 731
- convs with alt-keys: 5/5
- turns with >=1 alias match: 222

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| stacked_alias | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| stacked_alias | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.855 | +0.032 | 1.0 |

**Reference baselines** (from `alias_expansion.json`): v2f LoCoMo r@20=0.756 r@50=0.858; alias_expand_v2f LoCoMo r@20=0.694 r@50=0.881 (the latter costs ~3x LLM/query).

## Per-category (stacked_alias)

### locomo_30q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| locomo_multi_hop | 4 | +0.125 | +0.375 | 3/1/0 |
| locomo_single_hop | 10 | +0.567 | +0.700 | 8/2/0 |
| locomo_temporal | 16 | +0.312 | +0.125 | 2/14/0 |

### synthetic_19q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| completeness | 4 | +0.016 | +0.058 | 2/2/0 |
| conjunction | 3 | +0.143 | +0.048 | 1/2/0 |
| control | 3 | +0.000 | +0.000 | 0/3/0 |
| inference | 3 | +0.043 | +0.000 | 0/3/0 |
| proactive | 4 | +0.018 | +0.057 | 2/1/1 |
| procedural | 2 | +0.067 | +0.000 | 0/2/0 |

## Alias mechanism diagnostics

### locomo_30q

- n queries: 30
- mean raw alt-key hits / query: 30.0
- mean alias-turn hits (deduped) / query: 17.5
- mean alias-turn hits novel vs v2f / query: 11.17
- queries where >=1 alias turn entered top-20: 0/30
- queries where >=1 alias turn entered top-50: 30/30
- queries where an alias-appended turn hit gold @K=20: 0/30
- queries where an alias-appended turn hit gold @K=50: 0/30

### synthetic_19q

- n queries: 19
- mean raw alt-key hits / query: 30.0
- mean alias-turn hits (deduped) / query: 9.58
- mean alias-turn hits novel vs v2f / query: 2.32
- queries where >=1 alias turn entered top-20: 0/19
- queries where >=1 alias turn entered top-50: 9/19
- queries where an alias-appended turn hit gold @K=20: 0/19
- queries where an alias-appended turn hit gold @K=50: 1/19

## Orthogonality vs meta_v2f

| Dataset | K | total_gold | novel_vs_v2f | frac_novel |
|---|---:|---:|---:|---:|
| locomo_30q | 20 | 29 | 0 | 0.0 |
| locomo_30q | 50 | 36 | 0 | 0.0 |
| synthetic_19q | 20 | 72 | 0 | 0.0 |
| synthetic_19q | 50 | 121 | 1 | 0.0083 |

## Cost comparison (per-query LLM calls)

| Arch | avg LLM/q (LoCoMo) | avg LLM/q (synthetic) | ingest LLM |
|---|---:|---:|---:|
| meta_v2f | 1.0 | 1.0 | none |
| stacked_alias | 1.0 | 1.0 | 1 per conv (shared w/ alias_expand) |

Reference: `alias_expand_v2f` costs ~3 LLM/query (1 v2f per alias variant). `stacked_alias` matches `meta_v2f` at 1 LLM/query.

## Verdict

**ABANDON (tie with v2f, mechanism dry)**: stacked_alias ties v2f on LoCoMo K=50 (0.858 == 0.858) and alias alt-keys surface only **1/157 novel gold turns** across both datasets at K=50 (0.6%). Alias alt-keys embedded in isolation (per-turn substitution) add no retrieval signal that v2f+cosine does not already find. A score-bonus variant would let alt-keys displace v2f items but the mechanism has nothing useful to displace them with — alias_expand_v2f's +2.3pp lift comes from per-variant v2f cue generation (which generates semantically different cues), not from cosine retrieval on substituted turn text. The ingest-time-cheap version of alias expansion does not recover the alias_expand_v2f lift.
