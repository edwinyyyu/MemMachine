# LLM-Weighted Multi-Channel Retrieval

This study tests whether letting the LLM choose per-query channel weights (an **LLM conductor**) outperforms both the baseline (cosine only or v2f) and a uniform-weight multi-channel control.


## Channels

- **cosine_baseline** — raw cosine; general-purpose, works for most queries
- **v2f_cosine** — LLM-imagined cue cosine; best for open queries with complex intent
- **speaker_filter** — filter/boost turns spoken by a named person in the query
- **alias_context** — boost when query mentions an entity with known aliases
- **critical_info** — boost turns containing facts of enduring importance (dates, preferences, commitments)
- **temporal_tokens** — boost turns with dates/time/sequence words; use for temporal queries
- **entity_exact_match** — boost turns exact-matching proper nouns in query


## Recall Matrix (fair-backfill)

| Architecture | Dataset | base r@20 | arch r@20 | Δ@20 | base r@50 | arch r@50 | Δ@50 | avg LLM | avg embed |
|---|---|---|---|---|---|---|---|---|---|
| meta_v2f | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 1.0 | 4.0 |
| meta_v2f | synthetic_19q | 0.5694 | 0.6130 | +0.0436 | 0.8238 | 0.8513 | +0.0276 | 1.0 | 4.0 |
| multich_llm_weighted | locomo_30q | 0.3833 | 0.6250 | +0.2417 | 0.5083 | 0.8083 | +0.3000 | 2.0 | 4.4 |
| multich_llm_weighted | synthetic_19q | 0.5694 | 0.6263 | +0.0569 | 0.8238 | 0.8768 | +0.0531 | 2.0 | 17.2 |
| multich_uniform | locomo_30q | 0.3833 | 0.5694 | +0.1861 | 0.5083 | 0.7972 | +0.2889 | 1.0 | 7.5 |
| multich_uniform | synthetic_19q | 0.5694 | 0.6330 | +0.0636 | 0.8238 | 0.8588 | +0.0350 | 1.0 | 18.5 |
| multich_binary | locomo_30q | 0.3833 | 0.4417 | +0.0583 | 0.5083 | 0.6250 | +0.1167 | 1.2 | 2.5 |
| multich_binary | synthetic_19q | 0.5694 | 0.6235 | +0.0540 | 0.8238 | 0.8784 | +0.0546 | 1.7 | 15.8 |


## Weight patterns (multich_llm_weighted)

### locomo_30q

n=30. Average LLM-chosen weight per channel (0.0 = never engaged, 1.0 = always fully engaged):

| channel | avg weight | zero-rate |
|---|---|---|
| cosine_baseline | 0.562 | 0.00 |
| v2f_cosine | 0.305 | 0.00 |
| speaker_filter | 0.598 | 0.00 |
| alias_context | 0.018 | 0.90 |
| critical_info | 0.485 | 0.13 |
| temporal_tokens | 0.503 | 0.40 |
| entity_exact_match | 0.757 | 0.00 |

Sample queries with LLM-chosen weights:

- **[locomo_temporal]** When did Caroline go to the LGBTQ support group?
  - Weights (non-zero): {'cosine_baseline': 0.6, 'v2f_cosine': 0.2, 'speaker_filter': 0.4, 'critical_info': 0.5, 'temporal_tokens': 0.9, 'entity_exact_match': 0.7}
  - Reasoning: This is a temporal question about a specific person and event — prioritize temporal_tokens and entity_exact_match, boost turns by/ about Caroline (speaker_filter) and those containing factual dates (c
- **[locomo_temporal]** When did Melanie paint a sunrise?
  - Weights (non-zero): {'cosine_baseline': 0.6, 'v2f_cosine': 0.1, 'speaker_filter': 0.2, 'critical_info': 0.9, 'temporal_tokens': 0.9, 'entity_exact_match': 0.8}
  - Reasoning: User requests a date for 'Melanie' painting a 'sunrise' so prioritize entity exact-match and temporal/critical-info boosts, keep a strong baseline, modest speaker and v2f signals, and no alias boost.
- **[locomo_multi_hop]** What fields would Caroline be likely to pursue in her educaton?
  - Weights (non-zero): {'cosine_baseline': 0.6, 'v2f_cosine': 0.9, 'speaker_filter': 0.7, 'critical_info': 0.5, 'entity_exact_match': 0.8}
  - Reasoning: This is an open-ended inference about a named person so prioritize LLM-imagined cues and general semantic matching, boost turns spoken by or exactly mentioning Caroline, and moderately favor turns wit

### synthetic_19q

n=19. Average LLM-chosen weight per channel (0.0 = never engaged, 1.0 = always fully engaged):

| channel | avg weight | zero-rate |
|---|---|---|
| cosine_baseline | 0.603 | 0.00 |
| v2f_cosine | 0.563 | 0.00 |
| speaker_filter | 0.187 | 0.68 |
| alias_context | 0.055 | 0.74 |
| critical_info | 0.618 | 0.10 |
| temporal_tokens | 0.471 | 0.16 |
| entity_exact_match | 0.489 | 0.32 |

Sample queries with LLM-chosen weights:

- **[control]** What is Bob allergic to? Please include any updates or corrections mentioned later in the conversation.
  - Weights (non-zero): {'cosine_baseline': 0.8, 'v2f_cosine': 0.6, 'speaker_filter': 0.5, 'critical_info': 0.7, 'temporal_tokens': 0.6, 'entity_exact_match': 0.8}
  - Reasoning: Use strong general (cosine_baseline) and exact-match signals to find mentions of 'Bob', v2f for nuanced intent, speaker_filter to prioritize Bob's own statements, critical_info because allergies are e
- **[control]** What content management system (CMS) is Acme Corp currently using for their website?
  - Weights (non-zero): {'cosine_baseline': 0.6, 'v2f_cosine': 0.25, 'alias_context': 0.2, 'temporal_tokens': 0.5, 'entity_exact_match': 0.9}
  - Reasoning: Prioritize exact-entity matches and temporal tokens for the time-sensitive 'currently' question, with general cosine retrieval as backbone and modest v2f/alias boosts; speaker and critical-info channe
- **[control]** What is the user's most recent A1C level and how has it been trending?
  - Weights (non-zero): {'cosine_baseline': 0.4, 'v2f_cosine': 0.25, 'alias_context': 0.2, 'critical_info': 0.6, 'temporal_tokens': 0.7, 'entity_exact_match': 0.3}
  - Reasoning: This query requests a recent lab value and its trend, so prioritize channels that surface exact lab mentions and temporal context (critical_info, temporal_tokens, entity_exact_match) while keeping gen


## Cost Comparison (avg LLM calls per query)

| Architecture | LoCoMo-30 | Synthetic-19 |
|---|---|---|
| meta_v2f | 1.00 | 1.00 |
| multich_llm_weighted | 2.00 | 2.00 |
| multich_uniform | 1.00 | 1.00 |
| multich_binary | 1.23 | 1.74 |


## Verdict

### locomo_30q

- **K=20**: meta_v2f=0.7556, llm_weighted=0.6250 (-0.1306), uniform=0.5694 (-0.1862), binary=0.4417 (-0.3139)
- **K=50**: meta_v2f=0.8583, llm_weighted=0.8083 (-0.0500), uniform=0.7972 (-0.0611), binary=0.6250 (-0.2333)

### synthetic_19q

- **K=20**: meta_v2f=0.6130, llm_weighted=0.6263 (+0.0133), uniform=0.6330 (+0.0200), binary=0.6235 (+0.0105)
- **K=50**: meta_v2f=0.8513, llm_weighted=0.8768 (+0.0255), uniform=0.8588 (+0.0075), binary=0.8784 (+0.0271)
