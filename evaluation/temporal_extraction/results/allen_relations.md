# Allen-Relation Retrieval — Results

## Extraction quality (query side)
- Queries: 20
- Relation correct: 20 (100.0%)
- Anchor similar (substring match): 20 (100.0%)
- Both correct: 20 (100.0%)

## Event resolver hit rate
- Queries whose anchor resolved to a corpus event: 85.0%

## Retrieval — Overall
| Metric | Base hybrid | Allen | Δ |
|---|---:|---:|---:|
| R@5 | 0.950 | 1.000 | +0.050 |
| R@10 | 0.975 | 1.000 | +0.025 |
| MRR | 0.604 | 0.662 | +0.058 |
| NDCG@10 | 0.705 | 0.762 | +0.057 |

## Retrieval — per relation
| Relation | N | Base R@5 | Allen R@5 | Base MRR | Allen MRR | Base NDCG@10 | Allen NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| before | 4 | 1.000 | 1.000 | 0.833 | 0.833 | 0.875 | 0.875 |
| after | 4 | 1.000 | 1.000 | 0.708 | 0.708 | 0.783 | 0.783 |
| during | 4 | 1.000 | 1.000 | 0.438 | 0.438 | 0.579 | 0.596 |
| overlaps | 4 | 0.750 | 1.000 | 0.333 | 0.583 | 0.496 | 0.739 |
| contains | 4 | 1.000 | 1.000 | 0.708 | 0.750 | 0.794 | 0.815 |

## Queries now answerable that weren't before
- **q_overlaps_wedding** (overlaps): base R@5 0.50 → Allen R@5 1.00
  - gold: {'c_wedding_contains_1', 'c_wedding_during_1'}; base top-5: ['a_wedding', 'c_wedding_after_1', 'a_honeymoon', 'c_wedding_contains_1', 'c_grad_contains_1']; allen top-5: ['a_wedding', 'd_abs_1', 'c_wedding_contains_1', 'c_wedding_during_1', 'c_move_before_1']
- **q_overlaps_honeymoon** (overlaps): base R@5 0.50 → Allen R@5 1.00
  - gold: {'c_honeymoon_overlaps_1', 'c_honeymoon_during_1'}; base top-5: ['a_honeymoon', 'c_wedding_after_1', 'c_honeymoon_during_1', 'c_wedding_contains_1', 'a_wedding']; allen top-5: ['c_honeymoon_during_1', 'c_honeymoon_overlaps_1', 'c_wedding_contains_1', 'c_wedding_during_1', 'a_honeymoon']

## Failure analysis — Allen losses vs base
(none)

## Cost
- This run (fully cached): $0.00.
- Initial cold run: extractor $0.4586, event-resolver pair-extraction ~$0.13, **total ~$0.59** for 36 docs + 20 queries. Well under the $1.50 budget.
