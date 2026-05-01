# Temporal Extraction + Retrieval — Results

## Extraction quality
- Overall F1: **0.860** (precision 0.870, recall 0.851)
- Docs F1: 0.913 (tp=42, fp=6, fn=2)
- Queries F1: 0.809 (tp=38, fp=6, fn=12)
- Resolution MAE on matched pairs (80 pairs): mean=5627655s, median=0s, p95=30844800s

## Retrieval
| Condition | Recall@5 | Recall@10 | MRR | NDCG@10 | Critical top-1 |
|-----------|---------:|----------:|----:|--------:|---------------:|
| T | 0.460 | 0.460 | 0.625 | 0.476 | 5/5 |
| S | 0.418 | 0.494 | 0.763 | 0.500 | 5/5 |
| T_and_S | 0.555 | 0.590 | 0.918 | 0.652 | 5/5 |

## Cost
- Total LLM tokens: input=97,247, output=99,122
- Estimated LLM cost (gpt-5-mini @ $0.25/M in, $2.00/M out): $0.2226

## Verdict
Temporal structure helps retrieval. Wins:

- T beats S on recall@5 (0.460 vs 0.418)
- T_and_S beats S on recall@5 (0.555 vs 0.418)
- T_and_S beats S on recall@10 (0.590 vs 0.494)
- T_and_S beats S on mrr (0.918 vs 0.763)
- T_and_S beats S on ndcg@10 (0.652 vs 0.500)
