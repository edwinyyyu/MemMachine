# Entity-mention exact-match index — non-cosine retrieval signal

Motivation: v2f's cosine retrieval is fuzzy. Gold turns that share a specific entity with the query (names, IDs, numbers) can be missed even when obvious. An inverted index {entity -> turns} built at ingest, queried for query entities, boosts exact-match turns via final_score = cosine + beta * I(turn mentions query entity). This is NOT BM25 — only named entities are indexed.

## Entity extraction stats

| Dataset | Extractor | turns | turns w/ent | % | ent/turn | unique ents |
|---|---|---:|---:|---:|---:|---:|
| locomo_30q | regex | 2687 | 2158 | 80% | 10.82 | 8002 |
| locomo_30q | llm | 419 | 57 | 14% | 0.17 | 32 |
| synthetic_19q | regex | 462 | 361 | 78% | 2.05 | 430 |
| synthetic_19q | llm | 462 | 156 | 34% | 0.66 | 197 |

## Query-entity & boost coverage

| Arch | Dataset | queries | w/ents | ent/q | boosted turns/q |
|---|---|---:|---:|---:|---:|
| entity_regex_b0.05 | locomo_30q | 30 | 30 | 1.13 | 85.4 |
| entity_regex_b0.05 | synthetic_19q | 19 | 15 | 1.11 | 6.6 |
| entity_regex_b0.1 | locomo_30q | 30 | 30 | 1.13 | 85.4 |
| entity_regex_b0.1 | synthetic_19q | 19 | 15 | 1.11 | 6.6 |
| entity_regex_b0.2 | locomo_30q | 30 | 30 | 1.13 | 85.4 |
| entity_regex_b0.2 | synthetic_19q | 19 | 15 | 1.11 | 6.6 |
| entity_llm_b0.1 | locomo_30q | 30 | 28 | 1.17 | 15.5 |
| entity_llm_b0.1 | synthetic_19q | 19 | 13 | 0.89 | 2.3 |
| entity_regex_plus_v2f | locomo_30q | 30 | 30 | 1.13 | 85.4 |
| entity_regex_plus_v2f | synthetic_19q | 19 | 15 | 1.11 | 6.6 |

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q | embed/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 | 4.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 | 4.0 |
| entity_regex_b0.05 | locomo_30q | 0.383 | 0.250 | -0.133 | 0.508 | 0.383 | -0.125 | 0.0 | 2.0 |
| entity_regex_b0.05 | synthetic_19q | 0.569 | 0.583 | +0.014 | 0.824 | 0.824 | +0.000 | 0.0 | 2.0 |
| entity_regex_b0.1 | locomo_30q | 0.383 | 0.200 | -0.183 | 0.508 | 0.250 | -0.258 | 0.0 | 2.0 |
| entity_regex_b0.1 | synthetic_19q | 0.569 | 0.592 | +0.022 | 0.824 | 0.824 | +0.000 | 0.0 | 2.0 |
| entity_regex_b0.2 | locomo_30q | 0.383 | 0.067 | -0.317 | 0.508 | 0.133 | -0.375 | 0.0 | 2.0 |
| entity_regex_b0.2 | synthetic_19q | 0.569 | 0.565 | -0.004 | 0.824 | 0.824 | +0.000 | 0.0 | 2.0 |
| entity_llm_b0.1 | locomo_30q | 0.383 | 0.300 | -0.083 | 0.508 | 0.467 | -0.042 | 0.0 | 2.0 |
| entity_llm_b0.1 | synthetic_19q | 0.569 | 0.569 | +0.000 | 0.824 | 0.824 | +0.000 | 0.0 | 2.0 |
| entity_regex_plus_v2f | locomo_30q | 0.383 | 0.467 | +0.083 | 0.508 | 0.589 | +0.081 | 1.0 | 4.0 |
| entity_regex_plus_v2f | synthetic_19q | 0.569 | 0.626 | +0.057 | 0.824 | 0.826 | +0.002 | 1.0 | 4.0 |

## Orthogonality vs v2f

Fraction of gold turns found by the variant that v2f did NOT find.

| Arch | Dataset/K | gold_found | novel_vs_v2f | frac_novel |
|---|---|---:|---:|---:|
| entity_regex_b0.05 | locomo_30q_K20 | 8 | 0 | 0.000 |
| entity_regex_b0.05 | locomo_30q_K50 | 13 | 0 | 0.000 |
| entity_regex_b0.05 | synthetic_19q_K20 | 65 | 16 | 0.246 |
| entity_regex_b0.05 | synthetic_19q_K50 | 115 | 2 | 0.017 |
| entity_regex_b0.1 | locomo_30q_K20 | 6 | 0 | 0.000 |
| entity_regex_b0.1 | locomo_30q_K50 | 8 | 0 | 0.000 |
| entity_regex_b0.1 | synthetic_19q_K20 | 66 | 16 | 0.242 |
| entity_regex_b0.1 | synthetic_19q_K50 | 115 | 2 | 0.017 |
| entity_regex_b0.2 | locomo_30q_K20 | 2 | 0 | 0.000 |
| entity_regex_b0.2 | locomo_30q_K50 | 4 | 0 | 0.000 |
| entity_regex_b0.2 | synthetic_19q_K20 | 65 | 16 | 0.246 |
| entity_regex_b0.2 | synthetic_19q_K50 | 115 | 2 | 0.017 |
| entity_llm_b0.1 | locomo_30q_K20 | 10 | 0 | 0.000 |
| entity_llm_b0.1 | locomo_30q_K50 | 16 | 0 | 0.000 |
| entity_llm_b0.1 | synthetic_19q_K20 | 63 | 15 | 0.238 |
| entity_llm_b0.1 | synthetic_19q_K50 | 115 | 2 | 0.017 |
| entity_regex_plus_v2f | locomo_30q_K20 | 17 | 1 | 0.059 |
| entity_regex_plus_v2f | locomo_30q_K50 | 23 | 0 | 0.000 |
| entity_regex_plus_v2f | synthetic_19q_K20 | 71 | 11 | 0.155 |
| entity_regex_plus_v2f | synthetic_19q_K50 | 115 | 1 | 0.009 |

## Qualitative trios (entity_regex_plus_v2f, LoCoMo, K=50)

Each row: query, extracted query entities, boosted turn ids, gold found.

- **Q:** When did Caroline go to the LGBTQ support group?
  - Query entities: ['LGBTQ', 'Caroline']
  - Num boosted turns: 147
  - Boosted turn ids (sample): [1, 2, 3, 9, 15, 17, 18, 20, 24, 26, 29, 30, 34, 35, 36, 38, 40, 42, 44, 50]
  - Gold found: [2]

- **Q:** What fields would Caroline be likely to pursue in her educaton?
  - Query entities: ['Caroline']
  - Num boosted turns: 127
  - Boosted turn ids (sample): [1, 3, 9, 15, 17, 18, 20, 24, 26, 30, 34, 36, 38, 40, 42, 44, 50, 52, 56, 59]
  - Gold found: [10]

- **Q:** What is Caroline's identity?
  - Query entities: ["Caroline's"]
  - Num boosted turns: 0
  - Boosted turn ids (sample): []
  - Gold found: [4]

- **Q:** When did Melanie run a charity race?
  - Query entities: ['Melanie']
  - Num boosted turns: 56
  - Boosted turn ids (sample): [12, 14, 21, 33, 35, 51, 58, 60, 62, 66, 72, 74, 80, 96, 104, 130, 132, 134, 145, 157]
  - Gold found: [18]

- **Q:** When is Melanie planning on going camping?
  - Query entities: ['Melanie']
  - Num boosted turns: 56
  - Boosted turn ids (sample): [12, 14, 21, 33, 35, 51, 58, 60, 62, 66, 72, 74, 80, 96, 104, 130, 132, 134, 145, 157]
  - Gold found: [24]

## Top categories by Δr@50


### entity_regex_b0.05 on locomo_30q

Gaining:
  - (none with Δ > 0.001)
Losing:
  - locomo_multi_hop (n=4): Δ=-0.250 W/T/L=0/3/1
  - locomo_temporal (n=16): Δ=-0.125 W/T/L=0/14/2
  - locomo_single_hop (n=10): Δ=-0.075 W/T/L=0/8/2

### entity_regex_b0.1 on locomo_30q

Gaining:
  - (none with Δ > 0.001)
Losing:
  - locomo_multi_hop (n=4): Δ=-0.375 W/T/L=0/2/2
  - locomo_temporal (n=16): Δ=-0.312 W/T/L=0/11/5
  - locomo_single_hop (n=10): Δ=-0.125 W/T/L=0/8/2

### entity_regex_b0.2 on locomo_30q

Gaining:
  - (none with Δ > 0.001)
Losing:
  - locomo_temporal (n=16): Δ=-0.500 W/T/L=0/8/8
  - locomo_multi_hop (n=4): Δ=-0.500 W/T/L=0/1/3
  - locomo_single_hop (n=10): Δ=-0.125 W/T/L=0/8/2

### entity_llm_b0.1 on locomo_30q

Gaining:
  - (none with Δ > 0.001)
Losing:
  - locomo_temporal (n=16): Δ=-0.062 W/T/L=0/15/1
  - locomo_single_hop (n=10): Δ=-0.025 W/T/L=0/9/1

### entity_regex_plus_v2f on locomo_30q

Gaining:
  - locomo_single_hop (n=10): Δ=+0.342 W/T/L=4/5/1
Losing:
  - locomo_temporal (n=16): Δ=-0.062 W/T/L=0/15/1

### entity_regex_plus_v2f on synthetic_19q

Gaining:
  - conjunction (n=3): Δ=+0.048 W/T/L=1/2/0
Losing:
  - proactive (n=4): Δ=-0.025 W/T/L=0/3/1

## Verdict

**ABANDON**: no entity variant beats v2f on LoCoMo-30 @K=50 (v2f=0.858; entity_regex_b0.05=LC0.383, entity_regex_b0.1=LC0.250, entity_regex_b0.2=LC0.133, entity_llm_b0.1=LC0.467, entity_regex_plus_v2f=LC0.589).


## Outputs
- `results/entity_mention_study.md` — this report
- `results/entity_mention_study.json` — raw metrics + stats
- `results/turn_entities.json` — per-turn extracted entities
- `results/entity_<arch>_<dataset>.json` — per-question detail
