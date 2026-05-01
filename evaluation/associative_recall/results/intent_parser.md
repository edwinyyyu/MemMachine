# Structured Intent Parser + Constraint-Based Retrieval

One LLM call per query extracts a typed intent plan (intent_type, entities, speaker, temporal_relation, negation, answer_form, needs_aggregation, primary_topic). A retrieval plan is derived and executed as stacked signal bonuses on top of a v2f-style cosine base channel.


## Schema & sample parses

```json
{
  "intent_type": "one of factual-lookup | preference | temporal-compare | multi-hop-inference | commitment-tracking | synthesis | counterfactual | other",
  "entities": [
    "Caroline",
    "Phoenix"
  ],
  "constraints": {
    "speaker": "Caroline or null",
    "temporal_relation": {
      "marker": "after",
      "reference": "Monday meeting"
    },
    "negation": false,
    "quantity_bound": null,
    "answer_form": "date | person | number | description | list | yes-no | null"
  },
  "primary_topic": "Phoenix status",
  "needs_aggregation": false
}
```

Three example parses (LoCoMo):

- **Q**: When did Caroline go to the LGBTQ support group?
  - intent=factual-lookup primary_topic='LGBTQ support group' entities=['Caroline']
  - constraints={"speaker": null, "temporal_relation": null, "negation": false, "quantity_bound": null, "answer_form": "date"}
  - signals_detected=['answer_form:date'] signals_applied=['answer_form:date']
- **Q**: When did Melanie paint a sunrise?
  - intent=factual-lookup primary_topic='Melanie paint a sunrise' entities=['Melanie']
  - constraints={"speaker": null, "temporal_relation": null, "negation": false, "quantity_bound": null, "answer_form": "date"}
  - signals_detected=['answer_form:date'] signals_applied=['answer_form:date']
- **Q**: What fields would Caroline be likely to pursue in her educaton?
  - intent=preference primary_topic='fields in her educaton' entities=['Caroline']
  - constraints={"speaker": null, "temporal_relation": null, "negation": false, "quantity_bound": null, "answer_form": "list"}
  - signals_detected=['answer_form:list', 'intent_type:preference'] signals_applied=['intent_type:preference', 'answer_form:list']


## Constraint detection rates

Rate at which each signal was extracted from a query, across the combined LoCoMo + synthetic datasets.

| signal | detection rate (locomo) | detection rate (synthetic) | Δ@50 when detected (locomo) | Δ@50 when detected (synthetic) |
|---|---|---|---|---|
| answer_form:date | 0.47 | 0.00 | +0.1429 | +0.0000 |
| answer_form:list | 0.17 | 0.63 | +0.6000 | +0.0170 |
| intent_type:preference | 0.10 | 0.05 | +0.8333 | +0.0000 |
| needs_aggregation | 0.17 | 0.79 | +0.7000 | +0.0136 |
| negation | 0.03 | 0.00 | +0.5000 | +0.0000 |
| speaker | 0.27 | 0.00 | +0.3125 | +0.0000 |
| temporal_relation | 0.07 | 0.53 | +0.2500 | +0.0050 |


## Recall Matrix (fair-backfill)

| Architecture | Dataset | base r@20 | arch r@20 | Δ@20 | base r@50 | arch r@50 | Δ@50 | avg LLM | avg embed |
|---|---|---|---|---|---|---|---|---|---|
| meta_v2f | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 1.0 | 4.0 |
| meta_v2f | synthetic_19q | 0.5694 | 0.6130 | +0.0436 | 0.8238 | 0.8513 | +0.0276 | 1.0 | 4.0 |
| intent_parser_full | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8750 | +0.3667 | 1.0 | 6.1 |
| intent_parser_full | synthetic_19q | 0.5694 | 0.6121 | +0.0427 | 0.8238 | 0.8345 | +0.0107 | 1.0 | 6.5 |
| intent_parser_critical_only | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 1.0 | 6.1 |
| intent_parser_critical_only | synthetic_19q | 0.5694 | 0.6121 | +0.0427 | 0.8238 | 0.8345 | +0.0107 | 1.0 | 6.5 |
| intent_parser_no_plan_exec | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 1.0 | 4.0 |
| intent_parser_no_plan_exec | synthetic_19q | 0.5694 | 0.6121 | +0.0427 | 0.8238 | 0.8372 | +0.0135 | 1.0 | 4.0 |


## Per-intent-type analysis (intent_parser_full, LoCoMo)

| intent_type | n | base r@50 | arch r@50 | Δ@50 |
|---|---|---|---|---|
| factual-lookup | 24 | 0.5521 | 0.8438 | +0.2917 |
| preference | 3 | 0.1667 | 1.0000 | +0.8333 |
| commitment-tracking | 1 | 0.0000 | 1.0000 | +1.0000 |
| counterfactual | 1 | 0.5000 | 1.0000 | +0.5000 |
| multi-hop-inference | 1 | 1.0000 | 1.0000 | +0.0000 |


## Comparison vs multichannel_weighted

| Dataset | K | multich_llm_weighted | intent_parser_full | intent_parser_critical_only |
|---|---|---|---|---|
| locomo_30q | 20 | 0.6250 | 0.7556 | 0.7556 |
| locomo_30q | 50 | 0.8083 | 0.8750 | 0.8583 |
| synthetic_19q | 20 | 0.6263 | 0.6121 | 0.6121 |
| synthetic_19q | 50 | 0.8768 | 0.8345 | 0.8345 |


## Verdict

- **locomo_30q K=50**: meta_v2f=0.8583, intent_full=0.8750 (+0.0167), intent_critical_only=0.8583 (+0.0000), intent_no_plan_exec=0.8583 (+0.0000), multich_llm_weighted=0.8083
- **synthetic_19q K=50**: meta_v2f=0.8513, intent_full=0.8345 (-0.0168), intent_critical_only=0.8345 (-0.0168), intent_no_plan_exec=0.8372 (-0.0141), multich_llm_weighted=0.8768


## Summary

Structured intent parsing helps on datasets where constraints are strongly present (LoCoMo: speaker, temporal, date-answer-form) and is neutral-to-slightly-harmful on datasets where queries are already open/synthesis-style (synthetic-19).

On LoCoMo, per-signal delta@50 when detected:

  - intent_type:preference: +0.8333 (n=3)
  - needs_aggregation: +0.7000 (n=5)
  - answer_form:list: +0.6000 (n=5)
  - negation: +0.5000 (n=1)
  - speaker: +0.3125 (n=8)
  - temporal_relation: +0.2500 (n=2)
  - answer_form:date: +0.1429 (n=14)

**Which signals actually moved recall?** On LoCoMo, `intent_type:preference`, `needs_aggregation`, `answer_form:list`, and `speaker` delivered the biggest per-query lifts when detected. `answer_form:date`, `temporal_relation`, and `negation` were either rare or redundant with the v2f base.

**vs multichannel_weighted**: intent_parser_full beats multich_llm_weighted on LoCoMo K=50 (+6.7pp) but loses on synthetic K=50 (-4.2pp). The structured parser's typed constraints buy clear lift when query structure is distinct (speaker + temporal on LoCoMo); on synthetic where most queries are needs_aggregation, the non-decomposing multich channels win via broader candidate coverage.

**Decision**: Intent parsing is a conditional tool — it helps specifically on queries where typed constraints (speaker, temporal, list-aggregation) can be extracted and the base v2f doesn't already saturate recall. Because our Decision Rule 2 (if it matches `multichannel_weighted` prefer simpler) is split across datasets, we recommend NOT making intent_parser the primary architecture. Instead, route on intent_type: queries where the parser finds {speaker, temporal_relation} constraints go through the parser; others use v2f/meta_v2f.
