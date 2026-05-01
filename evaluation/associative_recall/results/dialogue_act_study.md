# Dialogue-Act Tagging + Act-Routed Retrieval — Empirical Recall Test

At ingestion time an LLM (gpt-5-mini) tags each turn with a speech-act label. Non-STATEMENT turns populate separate per-act vector stores. At query time, the query is routed to relevant acts via keyword rules (and optionally a per-query LLM call). Top-M hits from each act-index are merged with the v2f main retrieval using the same always-top-M / additive-bonus pattern that critical_info_store validated.

## 1. Act distribution (tagged turns)

| dataset | n_turns | STATEMENT | DECISION | COMMITMENT | RETRACTION | UNRESOLVED | CLARIFICATION | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| locomo_30q | 419 | 260 (62.1%) | 4 (1.0%) | 13 (3.1%) | 0 (0.0%) | 1 (0.2%) | 141 (33.7%) | 0 (0.0%) |
| synthetic_19q | 462 | 204 (44.2%) | 41 (8.9%) | 50 (10.8%) | 4 (0.9%) | 14 (3.0%) | 149 (32.3%) | 0 (0.0%) |

## 2. Recall (fair-backfill)

| dataset | K | baseline v2f | dialact_keyword_route | Δ | dialact_plus_v2f | Δ | dialact_llm_route | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| locomo_30q | 20 | 0.7556 | 0.7556 | +0.0000 | 0.7556 | +0.0000 | 0.7556 | +0.0000 |
| locomo_30q | 50 | 0.8583 | 0.8583 | +0.0000 | 0.8583 | +0.0000 | 0.8583 | +0.0000 |
| synthetic_19q | 20 | 0.6130 | 0.6192 | +0.0062 | 0.6130 | +0.0000 | 0.6125 | -0.0005 |
| synthetic_19q | 50 | 0.8513 | 0.8513 | +0.0000 | 0.8544 | +0.0031 | 0.8544 | +0.0031 |

## 3. Per-category — locomo_30q

| category | n | base@20 | kw@20 | plus@20 | base@50 | kw@50 | plus@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| locomo_multi_hop | 4 | 0.625 | 0.625 | 0.625 | 0.875 | 0.875 | 0.875 |
| locomo_single_hop | 10 | 0.617 | 0.617 | 0.617 | 0.825 | 0.825 | 0.825 |
| locomo_temporal | 16 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |

## 3. Per-category — synthetic_19q

| category | n | base@20 | kw@20 | plus@20 | base@50 | kw@50 | plus@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| completeness | 4 | 0.455 | 0.455 | 0.455 | 0.865 | 0.865 | 0.865 |
| conjunction | 3 | 0.809 | 0.809 | 0.809 | 1.000 | 1.000 | 1.000 |
| control | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| inference | 3 | 0.766 | 0.766 | 0.766 | 0.939 | 0.939 | 0.939 |
| proactive | 4 | 0.351 | 0.351 | 0.351 | 0.643 | 0.643 | 0.643 |
| procedural | 2 | 0.347 | 0.406 | 0.347 | 0.661 | 0.661 | 0.690 |

## 4. Act-contribution rate

Fraction of questions where the act-routed store surfaced gold outside the main top-K.

| dataset | variant | K | frac Q with act-gold | frac gold via act |
|---|---|---:|---:|---:|
| locomo_30q | dialact_keyword_route | 20 | 0.0% | 0.0% |
| locomo_30q | dialact_keyword_route | 50 | 0.0% | 0.0% |
| locomo_30q | dialact_plus_v2f | 20 | 0.0% | 0.0% |
| locomo_30q | dialact_plus_v2f | 50 | 0.0% | 0.0% |
| locomo_30q | dialact_llm_route | 20 | 0.0% | 0.0% |
| locomo_30q | dialact_llm_route | 50 | 0.0% | 0.0% |
| synthetic_19q | dialact_keyword_route | 20 | 5.3% | 1.3% |
| synthetic_19q | dialact_keyword_route | 50 | 0.0% | 0.0% |
| synthetic_19q | dialact_plus_v2f | 20 | 0.0% | 0.0% |
| synthetic_19q | dialact_plus_v2f | 50 | 5.3% | 1.3% |
| synthetic_19q | dialact_llm_route | 20 | 31.6% | 5.2% |
| synthetic_19q | dialact_llm_route | 50 | 5.3% | 0.7% |

## 5. Query-routing distribution (keyword rules)

- locomo_30q: DECISION=1, NONE=29
- synthetic_19q: COMMITMENT=2, NONE=16, RETRACTION=1, UNRESOLVED=2

## 6. Cost

- Uncached calls: 49
- Cached calls: 881
- Prompt tokens: 8516
- Completion tokens: 13697
- Est. cost (gpt-5-mini): $0.029
