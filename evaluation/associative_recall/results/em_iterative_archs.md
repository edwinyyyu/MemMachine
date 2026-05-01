# Iterative/Agentic Architectures on EventMemory (speakerformat)

## Setup

- n_questions = 30 (benchmark=locomo, first 30)
- Backend: EventMemory (`text-embedding-3-small`, `gpt-5-mini`, `max_text_chunk_length=500`, `derive_sentences=False`, `reranker=None`).
- Speaker-baked embeddings: `"{source}: {text}"`. All cue-gen prompts use the V2F_SPEAKERFORMAT style (cues must start with `"<speaker_name>: "`).
- `*_filter` variants apply `property_filter(context.source=<speaker>)` post-hoc when the query mentions one participant (mirrors `em_two_speaker_filter`).
- Dedicated caches: `cache/iter_{hypothesis_driven,v15_conditional_hop2,v15_rerank,working_memory_buffer}_sf_cache.json`.

## Prior SS-era baselines (reference, on LoCoMo K=50)

| SS arch | SS K=20 | SS K=50 |
| --- | --- | --- |
| hypothesis_driven | n/a | 0.842 |
| v15_conditional_hop2 | n/a | 0.822 |
| v15_rerank | 0.772 | 0.772 |
| working_memory_buffer | n/a | 0.717 |

## Prior EM baselines (for direct comparison)

| EM baseline | R@20 | R@50 |
| --- | --- | --- |
| em_v2f_speakerformat | 0.8167 | 0.8917 |
| em_two_speaker_filter (v2f+filter) | 0.8417 | 0.9000 |
| em_two_speaker_query_only | 0.8000 | 0.9333 |

## Results (this run)

| Variant | R@20 | R@50 | avg LLM calls/query | time (s) |
| --- | --- | --- | --- | --- |
| `em_hypothesis_driven_sf` | 0.7583 | 0.8833 | 2.40 | 628.8 |
| `em_v15_conditional_hop2_sf` | 0.8167 | 0.8750 | 2.00 | 498.8 |
| `em_v15_rerank_sf` | 0.7833 | 0.8833 | 2.00 | 796.9 |
| `em_working_memory_buffer_sf` | 0.7500 | 0.8167 | 2.00 | 952.9 |
| `em_hypothesis_driven_sf_filter` | 0.8083 | 0.9333 | 2.43 | 83.7 |
| `em_v15_conditional_hop2_sf_filter` | 0.8167 | 0.9417 | 2.00 | 39.9 |
| `em_v15_rerank_sf_filter` | 0.8000 | 0.9167 | 2.00 | 42.3 |
| `em_working_memory_buffer_sf_filter` | 0.8167 | 0.8833 | 2.00 | 47.7 |

## W/T/L vs em_v2f_speakerformat (all variants)

| Variant | K=20 W/T/L | K=50 W/T/L |
| --- | --- | --- |
| `em_hypothesis_driven_sf` | 1/25/4 | 1/27/2 |
| `em_v15_conditional_hop2_sf` | 1/28/1 | 0/29/1 |
| `em_v15_rerank_sf` | 1/27/2 | 0/29/1 |
| `em_working_memory_buffer_sf` | 0/27/3 | 0/26/4 |
| `em_hypothesis_driven_sf_filter` | 2/25/3 | 3/26/1 |
| `em_v15_conditional_hop2_sf_filter` | 1/28/1 | 3/27/0 |
| `em_v15_rerank_sf_filter` | 1/28/1 | 2/27/1 |
| `em_working_memory_buffer_sf_filter` | 2/26/2 | 2/26/2 |

## W/T/L vs em_two_speaker_filter (_filter variants)

| Variant | K=20 W/T/L | K=50 W/T/L |
| --- | --- | --- |
| `em_hypothesis_driven_sf_filter` | 0/28/2 | 1/29/0 |
| `em_v15_conditional_hop2_sf_filter` | 2/25/3 | 2/28/0 |
| `em_v15_rerank_sf_filter` | 1/27/2 | 1/28/1 |
| `em_working_memory_buffer_sf_filter` | 3/23/4 | 1/27/2 |

## Deltas vs em_v2f_speakerformat / em_two_speaker_filter / em_two_speaker_query_only

| Variant | dR@20 vs v2f_sf | dR@50 vs v2f_sf | dR@20 vs two_sf_filter | dR@50 vs two_sf_filter |
| --- | --- | --- | --- | --- |
| `em_hypothesis_driven_sf` | -0.0584 | -0.0084 | -0.0834 | -0.0167 |
| `em_v15_conditional_hop2_sf` | +0.0000 | -0.0167 | -0.0250 | -0.0250 |
| `em_v15_rerank_sf` | -0.0334 | -0.0084 | -0.0584 | -0.0167 |
| `em_working_memory_buffer_sf` | -0.0667 | -0.0750 | -0.0917 | -0.0833 |
| `em_hypothesis_driven_sf_filter` | -0.0084 | +0.0416 | -0.0334 | +0.0333 |
| `em_v15_conditional_hop2_sf_filter` | +0.0000 | +0.0500 | -0.0250 | +0.0417 |
| `em_v15_rerank_sf_filter` | -0.0167 | +0.0250 | -0.0417 | +0.0167 |
| `em_working_memory_buffer_sf_filter` | +0.0000 | -0.0084 | -0.0250 | -0.0167 |

## Cost efficiency: pp-gain-per-extra-LLM-call vs em_v2f_speakerformat

em_v2f_speakerformat uses 1 LLM call/query (v2f prompt). Extra LLM calls above that are the iterative overhead.

| Variant | extra LLM calls | dR@50 | dR@50 per extra call |
| --- | --- | --- | --- |
| `em_hypothesis_driven_sf` | 1.40 | -0.0084 | -0.0060 |
| `em_v15_conditional_hop2_sf` | 1.00 | -0.0167 | -0.0167 |
| `em_v15_rerank_sf` | 1.00 | -0.0084 | -0.0084 |
| `em_working_memory_buffer_sf` | 1.00 | -0.0750 | -0.0750 |
| `em_hypothesis_driven_sf_filter` | 1.43 | +0.0416 | +0.0291 |
| `em_v15_conditional_hop2_sf_filter` | 1.00 | +0.0500 | +0.0500 |
| `em_v15_rerank_sf_filter` | 1.00 | +0.0250 | +0.0250 |
| `em_working_memory_buffer_sf_filter` | 1.00 | -0.0084 | -0.0084 |

## Findings

- Best R@20 variant: `em_v15_conditional_hop2_sf` = 0.8167
- Best R@50 variant: `em_v15_conditional_hop2_sf_filter` = 0.9417

Ceilings to beat:
  - R@20: em_two_speaker_filter = 0.8417
  - R@50: em_two_speaker_query_only = 0.9333

## Outputs

- `results/em_iterative_archs.json`
- `results/em_iterative_archs.md`
- Source: `em_iterative_archs.py`, `iter_eval.py`
- Caches: `cache/iter_{hypothesis_driven,v15_conditional_hop2,v15_rerank,working_memory_buffer}_sf_cache.json`
