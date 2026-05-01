# Confidence-Gated Conditional Channel Overlay

Primary channel is v2f (always active). Supplement channels REPLACE v2f's weakest slots when their per-query LLM confidence clears the threshold. Supplement ordering is preserved within the displaced tail; v2f's strongest picks are retained.


## Supplement channels

- **speaker_filter** - boost turns spoken by a specific named person; confidence high only if query names a person by first name
- **alias_context** - substitute entity aliases; confidence high if query mentions an entity with known aliases (e.g. 'Dr. Smith' / 'John Smith')
- **critical_info** - high if query seeks an enduring fact (medication, deadline, commitment, preference)
- **temporal_tokens** - high if query has temporal constraint (when, after, during, by, specific date)
- **entity_exact_match** - high if query has distinctive proper noun (not common names); irrelevant for generic queries


## Recall matrix (fair-backfill)

| Architecture | Dataset | base r@20 | arch r@20 | Δ@20 | base r@50 | arch r@50 | Δ@50 | avg LLM | avg embed |
|---|---|---|---|---|---|---|---|---|---|
| meta_v2f | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 1.0 | 4.0 |
| meta_v2f | synthetic_19q | 0.5694 | 0.6130 | +0.0436 | 0.8238 | 0.8513 | +0.0276 | 1.0 | 4.0 |
| gated_threshold_0.7 | locomo_30q | 0.3833 | 0.7583 | +0.3750 | 0.5083 | 0.8917 | +0.3833 | 4.0 | 7.1 |
| gated_threshold_0.7 | synthetic_19q | 0.5694 | 0.5675 | -0.0019 | 0.8238 | 0.8332 | +0.0095 | 4.0 | 31.9 |
| gated_threshold_0.5 | locomo_30q | 0.3833 | 0.7250 | +0.3417 | 0.5083 | 0.8917 | +0.3833 | 4.0 | 7.1 |
| gated_threshold_0.5 | synthetic_19q | 0.5694 | 0.5675 | -0.0019 | 0.8238 | 0.8332 | +0.0095 | 4.0 | 32.8 |
| gated_replace_strict_0.85 | locomo_30q | 0.3833 | 0.7583 | +0.3750 | 0.5083 | 0.8917 | +0.3833 | 4.0 | 7.0 |
| gated_replace_strict_0.85 | synthetic_19q | 0.5694 | 0.5675 | -0.0019 | 0.8238 | 0.8332 | +0.0095 | 4.0 | 31.9 |
| gated_critical_only | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 4.0 | 7.0 |
| gated_critical_only | synthetic_19q | 0.5694 | 0.6094 | +0.0400 | 0.8238 | 0.8394 | +0.0157 | 4.0 | 31.3 |


## W/T/L vs meta_v2f (per-question)

| Variant | Dataset | W/T/L @20 | W/T/L @50 |
|---|---|---|---|
| gated_threshold_0.7 | locomo_30q | 4/23/3 | 1/29/0 |
| gated_threshold_0.7 | synthetic_19q | 3/11/5 | 3/13/3 |
| gated_threshold_0.5 | locomo_30q | 4/22/4 | 1/29/0 |
| gated_threshold_0.5 | synthetic_19q | 3/11/5 | 3/13/3 |
| gated_replace_strict_0.85 | locomo_30q | 4/23/3 | 1/29/0 |
| gated_replace_strict_0.85 | synthetic_19q | 3/11/5 | 3/13/3 |
| gated_critical_only | locomo_30q | 0/30/0 | 0/30/0 |
| gated_critical_only | synthetic_19q | 3/11/5 | 4/12/3 |


### Firing stats: gated_threshold_0.7 / locomo_30q

n=30; avg firing channels/query: 1.93

Firing-count distribution (channels per query):
- 1 channels: 7
- 2 channels: 18
- 3 channels: 5

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.977 | 0.97 | 0.97 |
| alias_context | 0.027 | 0.03 | 0.03 |
| critical_info | 0.390 | 0.30 | 0.00 |
| temporal_tokens | 0.567 | 0.57 | 0.57 |
| entity_exact_match | 0.080 | 0.07 | 0.07 |


### Firing stats: gated_threshold_0.7 / synthetic_19q

n=19; avg firing channels/query: 1.79

Firing-count distribution (channels per query):
- 0 channels: 4
- 1 channels: 2
- 2 channels: 7
- 3 channels: 6

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.211 | 0.21 | 0.00 |
| alias_context | 0.158 | 0.10 | 0.05 |
| critical_info | 0.632 | 0.63 | 0.47 |
| temporal_tokens | 0.658 | 0.58 | 0.37 |
| entity_exact_match | 0.263 | 0.26 | 0.10 |


### Firing stats: gated_threshold_0.5 / locomo_30q

n=30; avg firing channels/query: 2.17

Firing-count distribution (channels per query):
- 1 channels: 3
- 2 channels: 19
- 3 channels: 8

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.977 | 1.00 | 1.00 |
| alias_context | 0.027 | 0.03 | 0.03 |
| critical_info | 0.390 | 0.47 | 0.00 |
| temporal_tokens | 0.567 | 0.57 | 0.57 |
| entity_exact_match | 0.080 | 0.10 | 0.10 |


### Firing stats: gated_threshold_0.5 / synthetic_19q

n=19; avg firing channels/query: 2.05

Firing-count distribution (channels per query):
- 0 channels: 3
- 1 channels: 1
- 2 channels: 7
- 3 channels: 8

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.211 | 0.21 | 0.00 |
| alias_context | 0.158 | 0.21 | 0.16 |
| critical_info | 0.632 | 0.63 | 0.47 |
| temporal_tokens | 0.658 | 0.74 | 0.53 |
| entity_exact_match | 0.263 | 0.26 | 0.10 |


### Firing stats: gated_replace_strict_0.85 / locomo_30q

n=30; avg firing channels/query: 1.87

Firing-count distribution (channels per query):
- 1 channels: 8
- 2 channels: 18
- 3 channels: 4

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.977 | 0.93 | 0.93 |
| alias_context | 0.027 | 0.00 | 0.00 |
| critical_info | 0.390 | 0.30 | 0.00 |
| temporal_tokens | 0.567 | 0.57 | 0.57 |
| entity_exact_match | 0.080 | 0.07 | 0.07 |


### Firing stats: gated_replace_strict_0.85 / synthetic_19q

n=19; avg firing channels/query: 1.79

Firing-count distribution (channels per query):
- 0 channels: 4
- 1 channels: 2
- 2 channels: 7
- 3 channels: 6

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.211 | 0.21 | 0.00 |
| alias_context | 0.158 | 0.10 | 0.05 |
| critical_info | 0.632 | 0.63 | 0.47 |
| temporal_tokens | 0.658 | 0.58 | 0.37 |
| entity_exact_match | 0.263 | 0.26 | 0.10 |


### Firing stats: gated_critical_only / locomo_30q

n=30; avg firing channels/query: 0.30

Firing-count distribution (channels per query):
- 0 channels: 21
- 1 channels: 9

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.977 | 0.00 | 0.00 |
| alias_context | 0.027 | 0.00 | 0.00 |
| critical_info | 0.390 | 0.30 | 0.00 |
| temporal_tokens | 0.567 | 0.00 | 0.00 |
| entity_exact_match | 0.080 | 0.00 | 0.00 |


### Firing stats: gated_critical_only / synthetic_19q

n=19; avg firing channels/query: 0.63

Firing-count distribution (channels per query):
- 0 channels: 7
- 1 channels: 12

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.211 | 0.00 | 0.00 |
| alias_context | 0.158 | 0.00 | 0.00 |
| critical_info | 0.632 | 0.63 | 0.47 |
| temporal_tokens | 0.658 | 0.00 | 0.00 |
| entity_exact_match | 0.263 | 0.00 | 0.00 |


### Sample W/L (gated_threshold_0.7 vs meta_v2f @ K=50, locomo_30q)

Wins:
- **[locomo_temporal]** How long ago was Caroline's 18th birthday?
  - v2f=0.000 -> gated=1.000
  - fired: ['speaker_filter', 'critical_info', 'temporal_tokens']
  - contributed: ['speaker_filter', 'temporal_tokens']

Losses:


### Sample W/L (gated_threshold_0.7 vs meta_v2f @ K=50, synthetic_19q)

Wins:
- **[completeness]** List ALL dietary restrictions and food preferences for every guest at the Saturday dinner party, including any updates o
  - v2f=0.692 -> gated=0.769
  - fired: ['critical_info', 'temporal_tokens']
  - contributed: ['temporal_tokens']
- **[inference]** Based on everything in the conversation, what medication interactions and health concerns should the user bring up with 
  - v2f=0.818 -> gated=1.000
  - fired: ['critical_info', 'temporal_tokens', 'entity_exact_match']
  - contributed: ['critical_info']
- **[proactive]** Help me prepare a list of topics to discuss with Dr. Patel at my January 25th appointment.
  - v2f=0.500 -> gated=0.600
  - fired: ['critical_info', 'temporal_tokens', 'entity_exact_match']
  - contributed: ['critical_info', 'temporal_tokens', 'entity_exact_match']

Losses:
- **[conjunction]** What content should be included in the presentation for the Acme Corp client meeting on Wednesday?
  - v2f=1.000 -> gated=0.714
  - fired: ['alias_context', 'temporal_tokens', 'entity_exact_match']
  - contributed: ['alias_context']
- **[completeness]** What are all of the user's current medications, including dosages and what they're for? Include any recent changes.
  - v2f=1.000 -> gated=0.833
  - fired: ['critical_info', 'temporal_tokens']
  - contributed: ['critical_info']
- **[proactive]** What needs to happen to set up the bedroom smart home features?
  - v2f=0.625 -> gated=0.375
  - fired: []
  - contributed: []


## Verdict

### locomo_30q

- **K=20**: meta_v2f=0.7556, gated_threshold_0.7=0.7583 (+0.0027), gated_threshold_0.5=0.7250 (-0.0306), gated_replace_strict_0.85=0.7583 (+0.0027), gated_critical_only=0.7556 (+0.0000)
- **K=50**: meta_v2f=0.8583, gated_threshold_0.7=0.8917 (+0.0334), gated_threshold_0.5=0.8917 (+0.0334), gated_replace_strict_0.85=0.8917 (+0.0334), gated_critical_only=0.8583 (+0.0000)

### synthetic_19q

- **K=20**: meta_v2f=0.6130, gated_threshold_0.7=0.5675 (-0.0455), gated_threshold_0.5=0.5675 (-0.0455), gated_replace_strict_0.85=0.5675 (-0.0455), gated_critical_only=0.6094 (-0.0036)
- **K=50**: meta_v2f=0.8513, gated_threshold_0.7=0.8332 (-0.0181), gated_threshold_0.5=0.8332 (-0.0181), gated_replace_strict_0.85=0.8332 (-0.0181), gated_critical_only=0.8394 (-0.0119)
