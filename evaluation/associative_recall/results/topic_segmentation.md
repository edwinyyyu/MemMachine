# Topic-Segment Hierarchical Retrieval — Eval Report

## Motivation

Corpus-geometry analysis showed LoCoMo gold turns are off-center and diffuse
(only 36% of gold have another gold in their embedding-kNN top-10). Topic
segmentation hypothesizes: segment summaries operate at a coarser scale; if
one turn in a segment is gold, the segment's summary may be retrievable even
when the gold turn's own embedding is far from the query. Summary hits expand
to all constituent turns.

## Segmentation statistics

| Dataset / Variant | #Convs | Total segments | Avg turns/seg |
|---|---|---|---|
| locomo_30q / fixed_n10 | 3 | 146 | 9.94 |
| locomo_30q / fixed_n15 | 3 | 98 | 14.81 |
| locomo_30q / llm_w40   | 3 | 154 | 9.42 |
| synthetic_19q / fixed_n10 | 5 | 48 | 9.62 |
| synthetic_19q / fixed_n15 | 5 | 33 | 14.00 |
| synthetic_19q / llm_w40   | 5 | 76 | 6.08 |

LLM-driven segmentation produces more (finer-grained) segments with tighter
topic boundaries than fixed-size chunks.

## Recall table (fair-backfill)

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | W/T/L@50 |
|---|---|---|---|---|---|---|---|---|
| **v2f (reference)** | locomo_30q     | 0.3833 | **0.7556** | +0.3722 | 0.5083 | **0.8583** | +0.3500 | 13/17/0 |
| **v2f (reference)** | synthetic_19q  | 0.5694 | 0.6130 | +0.0436 | 0.8238 | 0.8513 | +0.0276 | 4/14/1 |
| topic_seg_fixed_n10 | locomo_30q     | 0.3833 | 0.3167 | -0.0667 | 0.5083 | 0.6111 | +0.1028 | 6/22/2 |
| topic_seg_fixed_n10 | synthetic_19q  | 0.5694 | 0.5212 | -0.0482 | 0.8238 | 0.8050 | -0.0188 | 3/11/5 |
| topic_seg_fixed_n15 | locomo_30q     | 0.3833 | 0.3167 | -0.0667 | 0.5083 | 0.5250 | +0.0167 | 3/25/2 |
| topic_seg_fixed_n15 | synthetic_19q  | 0.5694 | 0.4678 | -0.1016 | 0.8238 | 0.7654 | -0.0583 | 1/11/7 |
| topic_seg_llm_m3    | locomo_30q     | 0.3833 | 0.3944 | +0.0111 | 0.5083 | 0.5611 | +0.0528 | 6/21/3 |
| topic_seg_llm_m3    | synthetic_19q  | 0.5694 | 0.4933 | -0.0761 | 0.8238 | 0.8105 | -0.0132 | 3/13/3 |
| topic_seg_llm_m5    | locomo_30q     | 0.3833 | 0.3944 | +0.0111 | 0.5083 | 0.5611 | +0.0528 | 8/17/5 |
| topic_seg_llm_m5    | synthetic_19q  | 0.5694 | 0.4933 | -0.0761 | 0.8238 | 0.8297 | +0.0060 | 5/11/3 |

Columns: `baseline` = cosine top-K on raw turns; `arch` = topic_seg (or v2f)
top-K after fair-backfill; `Δ` = arch − baseline.

## Best topic_seg variant vs v2f (locomo_30q, r@50)

| | r@50 |
|---|---|
| cosine baseline      | 0.5083 |
| topic_seg_fixed_n10  | 0.6111 |
| topic_seg_llm_m3 / m5 | 0.5611 |
| **v2f**              | **0.8583** |

Topic segmentation beats the *cosine* baseline at K=50 (+0.10 pp for the best
variant) but remains **0.25 pp below v2f** on locomo r@50, and is slightly
below v2f on synthetic too.

## Category deltas (locomo_30q, vs cosine baseline)

`topic_seg_llm_m5`:
| Category | n | Δ@50 |
|---|---|---|
| locomo_multi_hop  | 4  | +0.3750 |
| locomo_single_hop | 10 | +0.2083 |
| locomo_temporal   | 16 | -0.1250 |

`topic_seg_fixed_n10` (best overall for locomo):
| Category | n | Δ@50 |
|---|---|---|
| locomo_multi_hop  | 4  | +0.2500 |
| locomo_single_hop | 10 | +0.1083 |
| locomo_temporal   | 16 | +0.0625 |

### Top 3 gains (across variants, category × dataset)
1. locomo_multi_hop (n=4), topic_seg_llm_m5: +0.3750 — segment context
   assembles multiple related turns for multi-hop questions.
2. locomo_multi_hop (n=4), topic_seg_fixed_n10: +0.2500
3. locomo_single_hop (n=10), topic_seg_llm_m5: +0.2083

### Top 2 losses
1. synthetic_procedural (n=2), topic_seg_llm_m5: -0.1705 — fine-grained
   procedural task lists span many turns; 5 summaries pull in too much noise.
2. synthetic_inference (n=3), topic_seg_llm_m5: -0.1515
3. locomo_temporal (n=16), topic_seg_llm_m5: -0.1250 — temporal questions
   need specific dated turns; segment-level expansion dilutes the top-K with
   same-topic chatter.

## Verdict: **ABANDON** on its own; niche utility for multi-hop

- v2f beats all topic_seg variants by 0.22–0.34 pp on locomo r@50.
- Topic segmentation does beat the *cosine* baseline (+0.05–0.10 pp on
  locomo r@50), but that's a much weaker target than v2f.
- The mechanism clearly helps on locomo_multi_hop (+0.25–0.38 pp Δ) and
  locomo_single_hop; it hurts locomo_temporal and synthetic_procedural
  because expansion pulls in off-topic same-segment turns.

### Why it fails vs v2f
v2f uses an LLM to generate *specific-vocabulary cues* (dates, names,
tool-identifiers) that land directly on the gold turn. Topic summaries are
**1-2 abstract sentences** that share vocabulary with the whole segment;
when a query needs a specific date or name, the summary embedding is too
lossy and the summary-expansion dilutes the top-K with many non-gold turns
from the hit segment. In other words: the coarse layer is too coarse to
disambiguate.

### Plausible stacking with ens_2
The +0.25 pp locomo_multi_hop gain (topic_seg_llm_m5) is **orthogonal** to
v2f's gains (v2f already saturates temporal/single-hop). In principle,
topic_seg could contribute a distinct column to an ensemble for multi-hop.
**Not pursued here** (per plan: "don't force it"); the absolute levels are
too far behind v2f for the ensemble merge to keep topic_seg's hits. A fair
test would require changing ensemble_retrieval's merging to boost multi-hop
specifically.

## Output paths

- Results JSON: `results/topic_segmentation.json`
- This report: `results/topic_segmentation.md`
- Segmentations (reusable):
  - `results/topic_segments_locomo_30q_fixed_n10.json`
  - `results/topic_segments_locomo_30q_fixed_n15.json`
  - `results/topic_segments_locomo_30q_llm_w40.json`
  - `results/topic_segments_synthetic_19q_fixed_n10.json`
  - `results/topic_segments_synthetic_19q_fixed_n15.json`
  - `results/topic_segments_synthetic_19q_llm_w40.json`
- Source: `topic_segment.py`, `topic_segment_eval.py`
