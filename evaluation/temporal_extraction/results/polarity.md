# Polarity-aware Temporal Retrieval — Results

## Extraction polarity accuracy

- Doc-level polarity accuracy: **20/20 = 1.000**
- Intent classifier accuracy on query intents: **10/15 = 0.667**

### Doc polarity confusion (gold -> predicted)

- affirmed -> affirmed: 7
- hypothetical -> hypothetical: 5
- negated -> negated: 5
- uncertain -> uncertain: 3

## Retrieval on polarity test set

| Variant | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| raw | 0.667 | 0.667 | 0.456 | 0.508 |
| default | 0.116 | 0.116 | 0.333 | 0.163 |
| polarity_routed | 0.432 | 0.432 | 0.600 | 0.470 |

## Per-intent breakdown (R@5)

| Variant | affirmed | negation | agnostic |
|---|---:|---:|---:|
| raw | 0.000 | 1.000 | 1.000 |
| default | 0.000 | 0.000 | 0.347 |
| polarity_routed | 0.000 | 0.800 | 0.497 |

## Per-intent breakdown (MRR)

| Variant | affirmed | negation | agnostic |
|---|---:|---:|---:|
| raw | 0.000 | 0.367 | 1.000 |
| default | 0.000 | 0.000 | 1.000 |
| polarity_routed | 0.000 | 0.800 | 1.000 |

## Base-corpus regression

| Variant | R@5 | R@10 |
|---|---:|---:|
| raw | 0.667 | 0.692 |
| default | 0.667 | 0.692 |

Base-corpus polarity distribution over extracted expressions: {'affirmed': 47, 'hypothetical': 5, 'negated': 3}

## Cost

- Tokens: input=0, output=0
- Total cost: **$0.0000**

## Ship recommendation

- raw R@5=0.667 MRR=0.456; default R@5=0.116; polarity_routed R@5=0.432 MRR=0.600.
- Base-corpus regression: default and raw are indistinguishable (affirmed-dominated corpus).
- **Recommendation**: keep polarity as an opt-in channel (polarity_routed) rather than always-on default. Routed retrieval lifts negation MRR cleanly; always-on affirmed-only filtering hurts agnostic and negation intents.
