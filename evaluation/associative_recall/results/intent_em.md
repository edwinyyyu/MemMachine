# intent_parser on EventMemory (LoCoMo-30)

## Schema available for intent-driven filtering

- **Filterable Context fields**: `context.source` (speaker name), `context.type` (always 'message' for LoCoMo — useless).
- **EM reserved base field**: `timestamp` (synthesized; see note).
- **User metadata stored, NOT payload-indexed**: `arc_conversation_id`, `turn_id`, `role`.

> Only context.source is semantically useful for intent-driven filtering. EM timestamps are synthesized (2023-01-01 + 60s per turn_id) so LoCoMo real-world temporal references ('4 years ago', 'last weekend') do NOT map. Role duplicates context.source for two-speaker LoCoMo.

## Constraint firing rates across 30 LoCoMo queries

| Constraint | Fired | % |
| --- | --- | --- |
| `speaker` | 8 | 26.7% |
| `temporal_relation` | 2 | 6.7% |
| `negation` | 1 | 3.3% |
| `answer_form` | 30 | 100.0% |
| `needs_aggregation` | 5 | 16.7% |

Only `speaker` maps to a usable EM filter; `temporal_relation` is dropped (synthesized timestamps don't align with LoCoMo's real-world dates); `negation` is applied as a post-retrieval score nudge; `answer_form` / `needs_aggregation` have no schema support on this corpus.

## Recall matrix

| Variant | R@20 | R@50 |
| --- | --- | --- |
| em_v2f_speakerformat (ref) | 0.8170 | 0.8920 |
| em_two_speaker_filter (ref) | 0.8420 | 0.9000 |
| em_two_speaker_query_only (ref K=50 leader) | 0.8000 | **0.9330** |
| **intent_em_speaker_only** | **0.7583** | **0.9000** |
| **intent_em_full_filter** | **0.7583** | **0.9000** |
| **intent_em_filter_no_cues** | **0.7500** | **0.9000** |
| **intent_em_with_speakerformat_cues** | **0.8167** | **0.9083** |

## Filter-firing vs recall (intent_em_speaker_only)

- Filter fired on 8/30 queries.
- Queries with filter: R@20=0.875 R@50=1.0
- Queries without filter: R@20=0.7159 R@50=0.8636

## Verdict

- Best intent_em variant R@50=0.9083 UNDER em_two_speaker_query_only (0.9330): LLM speaker parsing is either underfiring or hallucinating participants.

- `intent_em_full_filter` (0.9000) ≈ `intent_em_filter_no_cues` (0.9000) at K=50: cues neither help nor hurt meaningfully on top of the filter.

## Outputs

- `results/intent_em.json`
- `results/intent_em.md`
- Source: `intent_em.py`, `intemf_eval.py`