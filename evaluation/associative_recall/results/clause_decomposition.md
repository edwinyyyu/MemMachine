# Query Clause Decomposition

Mechanical split of multi-part queries on sentence boundaries, semicolons, and safe conjunctions. Per-clause retrieval, unioned. No LLM decomposition (context_tree_v2 failed because an LLM couldn't decompose without losing intent — this is cheap and preserves literal clause tokens).

## Clause distribution (n3 cap)

| Dataset | n queries | n=1 | n=2 | n>=3 |
|---|---:|---:|---:|---:|
| locomo_30q | 30 | 30 | 0 | 0 |
| synthetic_19q | 19 | 12 | 5 | 2 |

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.00 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.00 |
| clause_cosine_n2 | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.508 | +0.000 | 0.00 |
| clause_cosine_n2 | synthetic_19q | 0.569 | 0.555 | -0.014 | 0.824 | 0.813 | -0.011 | 0.00 |
| clause_cosine_n3 | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.508 | +0.000 | 0.00 |
| clause_cosine_n3 | synthetic_19q | 0.569 | 0.555 | -0.014 | 0.824 | 0.809 | -0.015 | 0.00 |
| clause_v2f_n2 | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.00 |
| clause_v2f_n2 | synthetic_19q | 0.569 | 0.608 | +0.039 | 0.824 | 0.861 | +0.037 | 1.40 |
| clause_plus_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.00 |
| clause_plus_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.837 | +0.013 | 1.00 |

## Multi-clause slice (queries with >=2 clauses)

| Arch | Dataset | n | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | W/T/L@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0 | - | - | - | - | - | - | - |
| meta_v2f | synthetic_19q | 0 | - | - | - | - | - | - | - |
| clause_cosine_n2 | locomo_30q | 0 | - | - | - | - | - | - | - |
| clause_cosine_n2 | synthetic_19q | 7 | 0.612 | 0.574 | -0.038 | 0.857 | 0.829 | -0.029 | 0/6/1 |
| clause_cosine_n3 | locomo_30q | 0 | - | - | - | - | - | - | - |
| clause_cosine_n3 | synthetic_19q | 7 | 0.612 | 0.574 | -0.038 | 0.857 | 0.818 | -0.040 | 0/5/2 |
| clause_v2f_n2 | locomo_30q | 0 | - | - | - | - | - | - | - |
| clause_v2f_n2 | synthetic_19q | 7 | 0.612 | 0.644 | +0.032 | 0.857 | 0.894 | +0.036 | 2/4/1 |
| clause_plus_v2f | locomo_30q | 0 | - | - | - | - | - | - | - |
| clause_plus_v2f | synthetic_19q | 7 | 0.612 | 0.657 | +0.045 | 0.857 | 0.830 | -0.027 | 1/5/1 |

## Per-category (clause_plus_v2f)

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
| proactive | 4 | +0.018 | +0.037 | 1/2/1 |
| procedural | 2 | +0.067 | -0.133 | 0/1/1 |

## Sample splits

### locomo_30q


### synthetic_19q

- **control** `What is Bob allergic to? Please include any updates or corrections mentioned later in the conversation.`  
  -> ['What is Bob allergic to?', 'Please include any updates or corrections mentioned later in the conversation']
- **conjunction** `I need to buy a birthday gift for Bob. When is his birthday and what are his interests?`  
  -> ['I need to buy a birthday gift for Bob', 'When is his birthday and what are his interests?']
- **completeness** `List ALL dietary restrictions and food preferences for every guest at the Saturday dinner party, including any updates o`  
  -> ['List ALL dietary restrictions', 'food preferences for every guest at the Saturday dinner party', 'any updates or corrections']
- **completeness** `What are all of the user's current medications, including dosages and what they're for? Include any recent changes.`  
  -> ["What are all of the user's current medications", "dosages and what they're for?", 'Include any recent changes']
- **inference** `Based on everything in the conversation, what medication interactions and health concerns should the user bring up with `  
  -> ['Based on everything in the conversation, what medication interactions', 'health concerns should the user bring up with Dr. Patel at their January 25th appointment?']
- **proactive** `I want to cook dinner for Bob tonight. What should I keep in mind?`  
  -> ['I want to cook dinner for Bob tonight', 'What should I keep in mind?']

## Per-category (clause_v2f_n2)

### locomo_30q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| locomo_multi_hop | 4 | +0.125 | +0.375 | 3/1/0 |
| locomo_single_hop | 10 | +0.567 | +0.700 | 8/2/0 |
| locomo_temporal | 16 | +0.312 | +0.125 | 2/14/0 |

### synthetic_19q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| completeness | 4 | +0.119 | +0.096 | 2/2/0 |
| conjunction | 3 | -0.024 | +0.048 | 1/2/0 |
| control | 3 | +0.000 | +0.000 | 0/3/0 |
| inference | 3 | -0.048 | +0.030 | 1/2/0 |
| proactive | 4 | +0.102 | +0.037 | 1/2/1 |
| procedural | 2 | +0.033 | -0.033 | 0/1/1 |

## Verdict

### clause_plus_v2f
- locomo_30q K=50: meta_v2f=0.858, clause_plus_v2f=0.858, Δ=+0.000 (llm/q 1.00 -> 1.00)
- synthetic_19q K=50: meta_v2f=0.851, clause_plus_v2f=0.837, Δ=-0.014 (llm/q 1.00 -> 1.00)
  - multi-clause subset (n=7) K=50: Δ=-0.027 W/T/L=1/5/1
### clause_v2f_n2
- locomo_30q K=50: meta_v2f=0.858, clause_v2f_n2=0.858, Δ=+0.000 (llm/q 1.00 -> 1.00)
- synthetic_19q K=50: meta_v2f=0.851, clause_v2f_n2=0.861, Δ=+0.009 (llm/q 1.00 -> 1.40)
  - multi-clause subset (n=7) K=50: Δ=+0.036 W/T/L=2/4/1
### clause_cosine_n2
- locomo_30q K=50: meta_v2f=0.858, clause_cosine_n2=0.508, Δ=-0.350 (llm/q 1.00 -> 0.00)
- synthetic_19q K=50: meta_v2f=0.851, clause_cosine_n2=0.813, Δ=-0.038 (llm/q 1.00 -> 0.00)
  - multi-clause subset (n=7) K=50: Δ=-0.029 W/T/L=0/6/1
