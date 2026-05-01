# Co-temporal Graph Expansion — Retrieval Evaluation

Corpus: 145 docs, 115 queries (base+adversarial+cotemporal merged).
Wall: 64.2s. LLM cost: $0.0000.

## Graph stats

- Nodes: 145
- Edges (undirected): 813
- Avg degree: 11.21
- Max degree: 20
- Threshold: 0.3, cap/node: 20
- Raw pairs above threshold: 1091 / 10385
- Sem-bridge edges kept (no-temporal-signal docs): 4
- Hub docs (top-10 by degree): [('adv_a5_2', 20), ('adv_a8_1', 20), ('adv_r4_1', 20), ('cot_2_connected', 20), ('doc_decade_1', 20), ('adv_a1_1', 20), ('adv_a1_2', 20), ('adv_a2_0', 20), ('adv_a2_1', 20), ('adv_a3_1', 20)]

## Retrieval metrics per subset

| Subset | N | Direct R@5 | Cot R@5 | Δ R@5 | Direct R@10 | Cot R@10 | Direct MRR | Cot MRR | Direct NDCG | Cot NDCG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 55 | 0.394 | 0.387 | -0.008 | 0.505 | 0.530 | 0.776 | 0.749 | 0.509 | 0.516 |
| adversarial_full | 40 | 0.876 | 0.914 | +0.038 | 0.971 | 0.986 | 0.776 | 0.790 | 0.803 | 0.821 |
| adversarial_S8 | 1 | 0.500 | 0.500 | +0.000 | 0.500 | 0.500 | 1.000 | 1.000 | 0.613 | 0.613 |
| cotemporal | 20 | 0.675 | 0.600 | -0.075 | 0.725 | 0.733 | 0.787 | 0.688 | 0.665 | 0.599 |

## Expansion source breakdown (cotemporal subset, top-5)

- Total slots: 100
- Direct-or-mixed: 100 (100.0%)
- Expansion-only: 0 (0.0%)

## Cotemporal subset — per-query R@5

| qid | direct R@5 | cot R@5 | Δ |
|---|---:|---:|---:|
| `q_cot_1_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_1_b` | 0.33 | 0.67 | +0.33 |
| `q_cot_2_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_2_b` | 0.67 | 0.33 | -0.33 |
| `q_cot_3_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_3_b` | 0.50 | 0.50 | +0.00 |
| `q_cot_4_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_4_b` | 1.00 | 1.00 | +0.00 |
| `q_cot_5_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_5_b` | 1.00 | 1.00 | +0.00 |
| `q_cot_6_a` | 0.50 | 0.00 | -0.50 |
| `q_cot_6_b` | 1.00 | 1.00 | +0.00 |
| `q_cot_7_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_7_b` | 1.00 | 1.00 | +0.00 |
| `q_cot_8_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_8_b` | 1.00 | 1.00 | +0.00 |
| `q_cot_9_a` | 0.50 | 0.50 | +0.00 |
| `q_cot_9_b` | 1.00 | 0.50 | -0.50 |
| `q_cot_10_a` | 0.50 | 0.00 | -0.50 |
| `q_cot_10_b` | 1.00 | 1.00 | +0.00 |

## S8 adversarial recovery

### q_s8_0
- gold: ['adv_s8_2018', 'adv_s8_meet']
- direct top-5: ['adv_s8_meet', 'adv_r4_0', 'cot_1_connected', 'doc_decade_1', 'adv_r1_0']
- cot top-5: ['adv_s8_meet', 'cot_1_connected', 'adv_r4_0', 'adv_r1_0', 'doc_decade_1']
- direct hit: ['adv_s8_meet'] | cot hit: ['adv_s8_meet']

## Base-set regressions (queries where cot-R@5 < direct-R@5)

Count: 8

- `q_rel_day_3`: direct=0.33, cot=0.00, diff=-0.33
- `q_rel_day_4`: direct=0.40, cot=0.20, diff=-0.20
- `q_rec_1`: direct=0.67, cot=0.33, diff=-0.33
- `q_rec_2`: direct=0.67, cot=0.33, diff=-0.33
- `q_rec_3`: direct=0.23, cot=0.15, diff=-0.08
- `q_rec_7`: direct=0.67, cot=0.33, diff=-0.33
- `q_rec_8`: direct=0.17, cot=0.00, diff=-0.17
- `q_crit_1`: direct=0.40, cot=0.20, diff=-0.20

## Topic drift diagnosis

- Drift docs in corpus: 8
- Expansion candidates (all cot queries): 1133
- …of which on drift docs: 126 (11.1%)
- Drift docs in top-10 of cot queries: 17 / 200 slots

## Cost

- Input tokens: 0, output tokens: 0
- Estimated cost: $0.0000
- Wall clock: 64.2s

## Ship recommendation

Δ R@5 by subset: base=-0.008, adversarial=+0.038, S8=+0.000, cotemporal=-0.075.

- DEPRIORITIZE: graph expansion does not lift co-mention queries.
