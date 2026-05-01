# Gated Overlay v2

v2f-primary retrieval with confidence-gated supplement displacement (same mechanism as gated_overlay v1). v2 extends v1 with three new supplement channels (preference_markers, negation_markers, date_answer_boost) and a K-expansion signal (list_aggregation) derived from intent_parser's unique signal coverage.


## Channel catalog

- **speaker_filter** - boost turns spoken by a specific named person; confidence high only if query names a person by first name
- **alias_context** - substitute entity aliases; confidence high if query mentions an entity with known aliases (e.g. 'Dr. Smith' / 'John Smith')
- **critical_info** - high if query seeks an enduring fact (medication, deadline, commitment, preference)
- **temporal_tokens** - high if query has temporal constraint (when, after, during, by, specific date)
- **entity_exact_match** - high if query has distinctive proper noun (not common names); irrelevant for generic queries
- **preference_markers** - high if query asks 'what do I like / prefer / always / never / hate / usually'; boosts first-person self-statements
- **negation_markers** - high if query asks about what was NOT done / refused / avoided; boosts turns containing never/not/didn't/decline/refuse
- **date_answer_boost** - high if query asks for a specific calendar date or a 'when exactly' / 'how long ago' answer; boosts turns containing month names or numeric dates (tighter than temporal_tokens)
- **list_aggregation** (K-expander) - high if query asks for 'all / every / total / list of / overall'; NOT a candidate pool - instead EXPANDS K by 50% before truncating to increase coverage of list-style golds


## Recall matrix (fair-backfill)

| Architecture | Dataset | base r@20 | arch r@20 | Δ@20 | base r@50 | arch r@50 | Δ@50 | avg LLM | avg embed |
|---|---|---|---|---|---|---|---|---|---|
| meta_v2f | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 1.0 | 4.0 |
| meta_v2f | synthetic_19q | 0.5694 | 0.6130 | +0.0436 | 0.8238 | 0.8513 | +0.0276 | 1.0 | 4.0 |
| gated_threshold_0.7 | locomo_30q | 0.3833 | 0.7583 | +0.3750 | 0.5083 | 0.8917 | +0.3833 | 4.0 | 7.1 |
| gated_threshold_0.7 | synthetic_19q | 0.5694 | 0.5675 | -0.0019 | 0.8238 | 0.8332 | +0.0095 | 4.0 | 31.9 |
| intent_parser_full | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8750 | +0.3667 | 1.0 | 6.1 |
| intent_parser_full | synthetic_19q | 0.5694 | 0.6121 | +0.0427 | 0.8238 | 0.8345 | +0.0107 | 1.0 | 6.5 |
| gated_v2_all | locomo_30q | 0.3833 | 0.6833 | +0.3000 | 0.5083 | 0.8917 | +0.3833 | 4.0 | 7.1 |
| gated_v2_all | synthetic_19q | 0.5694 | 0.6156 | +0.0462 | 0.8238 | 0.8372 | +0.0135 | 4.0 | 25.5 |
| gated_v2_intent_only | locomo_30q | 0.3833 | 0.7889 | +0.4056 | 0.5083 | 0.8917 | +0.3833 | 4.0 | 7.0 |
| gated_v2_intent_only | synthetic_19q | 0.5694 | 0.6121 | +0.0427 | 0.8238 | 0.8372 | +0.0135 | 4.0 | 7.0 |
| gated_v2_minus_critical | locomo_30q | 0.3833 | 0.6833 | +0.3000 | 0.5083 | 0.8917 | +0.3833 | 4.0 | 7.1 |
| gated_v2_minus_critical | synthetic_19q | 0.5694 | 0.6080 | +0.0386 | 0.8238 | 0.8372 | +0.0135 | 4.0 | 7.2 |


## W/T/L of v2 variants vs meta_v2f (per-question)

| v2 variant | Dataset | W/T/L @20 | W/T/L @50 |
|---|---|---|---|
| gated_v2_all | locomo_30q | 3/22/5 | 1/29/0 |
| gated_v2_all | synthetic_19q | 4/12/3 | 3/14/2 |
| gated_v2_intent_only | locomo_30q | 1/29/0 | 1/29/0 |
| gated_v2_intent_only | synthetic_19q | 4/12/3 | 3/14/2 |
| gated_v2_minus_critical | locomo_30q | 3/22/5 | 1/29/0 |
| gated_v2_minus_critical | synthetic_19q | 3/13/3 | 3/14/2 |


## W/T/L of v2 variants vs gated_threshold_0.7 (per-question)

| v2 variant | Dataset | W/T/L @20 | W/T/L @50 |
|---|---|---|---|
| gated_v2_all | locomo_30q | 0/27/3 | 0/30/0 |
| gated_v2_all | synthetic_19q | 5/12/2 | 1/17/1 |
| gated_v2_intent_only | locomo_30q | 3/24/3 | 0/30/0 |
| gated_v2_intent_only | synthetic_19q | 6/10/3 | 1/17/1 |
| gated_v2_minus_critical | locomo_30q | 0/27/3 | 0/30/0 |
| gated_v2_minus_critical | synthetic_19q | 5/11/3 | 1/17/1 |


## W/T/L of v2 variants vs intent_parser_full (per-question)

| v2 variant | Dataset | W/T/L @20 | W/T/L @50 |
|---|---|---|---|
| gated_v2_all | locomo_30q | 3/22/5 | 1/28/1 |
| gated_v2_all | synthetic_19q | 1/17/1 | 1/17/1 |
| gated_v2_intent_only | locomo_30q | 1/29/0 | 1/28/1 |
| gated_v2_intent_only | synthetic_19q | 0/19/0 | 1/17/1 |
| gated_v2_minus_critical | locomo_30q | 3/22/5 | 1/28/1 |
| gated_v2_minus_critical | synthetic_19q | 0/18/1 | 1/17/1 |


### Firing stats: gated_v2_all / locomo_30q

n=30; avg firing channels/query: 2.30; list_expander fire rate: 0.07

Firing-count distribution (channels per query):
- 1 channels: 7
- 2 channels: 9
- 3 channels: 12
- 4 channels: 2

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.967 | 0.97 | 0.97 |
| alias_context | 0.050 | 0.03 | 0.03 |
| critical_info | 0.350 | 0.30 | 0.00 |
| temporal_tokens | 0.567 | 0.57 | 0.57 |
| entity_exact_match | 0.067 | 0.07 | 0.07 |
| preference_markers | 0.000 | 0.00 | 0.00 |
| negation_markers | 0.033 | 0.03 | 0.03 |
| date_answer_boost | 0.417 | 0.33 | 0.33 |
| list_aggregation | 0.067 | 0.07 | (K-expander, no displacement) |


### Firing stats: gated_v2_all / synthetic_19q

n=19; avg firing channels/query: 1.68; list_expander fire rate: 0.63

Firing-count distribution (channels per query):
- 0 channels: 3
- 1 channels: 5
- 2 channels: 7
- 3 channels: 3
- 4 channels: 1

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.211 | 0.21 | 0.00 |
| alias_context | 0.184 | 0.10 | 0.00 |
| critical_info | 0.526 | 0.47 | 0.26 |
| temporal_tokens | 0.605 | 0.58 | 0.21 |
| entity_exact_match | 0.237 | 0.21 | 0.05 |
| preference_markers | 0.026 | 0.00 | 0.00 |
| negation_markers | 0.026 | 0.00 | 0.00 |
| date_answer_boost | 0.105 | 0.10 | 0.05 |
| list_aggregation | 0.658 | 0.63 | (K-expander, no displacement) |


### Firing stats: gated_v2_intent_only / locomo_30q

n=30; avg firing channels/query: 0.37; list_expander fire rate: 0.07

Firing-count distribution (channels per query):
- 0 channels: 19
- 1 channels: 11

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.967 | 0.00 | 0.00 |
| alias_context | 0.050 | 0.00 | 0.00 |
| critical_info | 0.350 | 0.00 | 0.00 |
| temporal_tokens | 0.567 | 0.00 | 0.00 |
| entity_exact_match | 0.067 | 0.00 | 0.00 |
| preference_markers | 0.000 | 0.00 | 0.00 |
| negation_markers | 0.033 | 0.03 | 0.03 |
| date_answer_boost | 0.417 | 0.33 | 0.33 |
| list_aggregation | 0.067 | 0.07 | (K-expander, no displacement) |


### Firing stats: gated_v2_intent_only / synthetic_19q

n=19; avg firing channels/query: 0.10; list_expander fire rate: 0.63

Firing-count distribution (channels per query):
- 0 channels: 17
- 1 channels: 2

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.211 | 0.00 | 0.00 |
| alias_context | 0.184 | 0.00 | 0.00 |
| critical_info | 0.526 | 0.00 | 0.00 |
| temporal_tokens | 0.605 | 0.00 | 0.00 |
| entity_exact_match | 0.237 | 0.00 | 0.00 |
| preference_markers | 0.026 | 0.00 | 0.00 |
| negation_markers | 0.026 | 0.00 | 0.00 |
| date_answer_boost | 0.105 | 0.10 | 0.10 |
| list_aggregation | 0.658 | 0.63 | (K-expander, no displacement) |


### Firing stats: gated_v2_minus_critical / locomo_30q

n=30; avg firing channels/query: 2.00; list_expander fire rate: 0.07

Firing-count distribution (channels per query):
- 1 channels: 11
- 2 channels: 9
- 3 channels: 9
- 4 channels: 1

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.967 | 0.97 | 0.97 |
| alias_context | 0.050 | 0.03 | 0.03 |
| critical_info | 0.350 | 0.00 | 0.00 |
| temporal_tokens | 0.567 | 0.57 | 0.57 |
| entity_exact_match | 0.067 | 0.07 | 0.07 |
| preference_markers | 0.000 | 0.00 | 0.00 |
| negation_markers | 0.033 | 0.03 | 0.03 |
| date_answer_boost | 0.417 | 0.33 | 0.33 |
| list_aggregation | 0.067 | 0.07 | (K-expander, no displacement) |


### Firing stats: gated_v2_minus_critical / synthetic_19q

n=19; avg firing channels/query: 1.21; list_expander fire rate: 0.63

Firing-count distribution (channels per query):
- 0 channels: 4
- 1 channels: 8
- 2 channels: 6
- 3 channels: 1

| channel | avg confidence | fire rate | contribution rate |
|---|---|---|---|
| speaker_filter | 0.211 | 0.21 | 0.00 |
| alias_context | 0.184 | 0.10 | 0.00 |
| critical_info | 0.526 | 0.00 | 0.00 |
| temporal_tokens | 0.605 | 0.58 | 0.21 |
| entity_exact_match | 0.237 | 0.21 | 0.05 |
| preference_markers | 0.026 | 0.00 | 0.00 |
| negation_markers | 0.026 | 0.00 | 0.00 |
| date_answer_boost | 0.105 | 0.10 | 0.10 |
| list_aggregation | 0.658 | 0.63 | (K-expander, no displacement) |


### Sample W/L (gated_v2_all vs gated_threshold_0.7 @ K=50, locomo_30q)

Wins:

Losses:


### Sample W/L (gated_v2_all vs gated_threshold_0.7 @ K=50, synthetic_19q)

Wins:
- **[completeness]** What are all of the user's current medications, including dosages and what they're for? Include any recent changes.
  - ref=0.833 -> v2=1.000
  - fired: ['critical_info', 'temporal_tokens']
  - contributed: []
  - list_expander: True

Losses:
- **[inference]** Based on everything in the conversation, what medication interactions and health concerns should the user bring up with Dr. Patel at their J
  - ref=1.000 -> v2=0.909
  - fired: ['critical_info', 'temporal_tokens', 'date_answer_boost']
  - contributed: ['critical_info']
  - list_expander: True


## Verdict

### locomo_30q

- **K=20**: meta_v2f=0.7556, gated_threshold_0.7=0.7583, intent_parser_full=0.7556; gated_v2_all=0.6833 (vs v1 -0.0750, vs ip -0.0723); gated_v2_intent_only=0.7889 (vs v1 +0.0306, vs ip +0.0333); gated_v2_minus_critical=0.6833 (vs v1 -0.0750, vs ip -0.0723)
- **K=50**: meta_v2f=0.8583, gated_threshold_0.7=0.8917, intent_parser_full=0.8750; gated_v2_all=0.8917 (vs v1 +0.0000, vs ip +0.0167); gated_v2_intent_only=0.8917 (vs v1 +0.0000, vs ip +0.0167); gated_v2_minus_critical=0.8917 (vs v1 +0.0000, vs ip +0.0167)

### synthetic_19q

- **K=20**: meta_v2f=0.6130, gated_threshold_0.7=0.5675, intent_parser_full=0.6121; gated_v2_all=0.6156 (vs v1 +0.0481, vs ip +0.0035); gated_v2_intent_only=0.6121 (vs v1 +0.0446, vs ip +0.0000); gated_v2_minus_critical=0.6080 (vs v1 +0.0405, vs ip -0.0041)
- **K=50**: meta_v2f=0.8513, gated_threshold_0.7=0.8332, intent_parser_full=0.8345; gated_v2_all=0.8372 (vs v1 +0.0040, vs ip +0.0027); gated_v2_intent_only=0.8372 (vs v1 +0.0040, vs ip +0.0027); gated_v2_minus_critical=0.8372 (vs v1 +0.0040, vs ip +0.0027)
