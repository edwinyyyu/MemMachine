# Two-speaker attributed retrieval

Extends `speaker_user_filter` (user-side only) to cover BOTH
participants. At ingest: one LLM call per conv identifies both
the user-role and assistant-role speakers (LoCoMo's 'assistant'
is often a second human). At query time: if the query mentions
exactly one side, apply the speaker-aware transform to that
role; if both or neither, no transform.

## Two-speaker identification

- Conversations scanned: 13
- Both identified: 3 (23.1%)
- One side identified: 4
- Neither identified: 6
- Any-side hit rate: 53.8%

| Conversation | user | assistant |
|---|---|---|
| beam_4 | Christina | UNKNOWN |
| beam_5 | Craig | UNKNOWN |
| beam_6 | Crystal | UNKNOWN |
| beam_7 | UNKNOWN | UNKNOWN |
| beam_8 | Darryl | UNKNOWN |
| locomo_conv-26 | Caroline | Melanie |
| locomo_conv-30 | Jon | Gina |
| locomo_conv-41 | John | Maria |
| synth_medical | UNKNOWN | UNKNOWN |
| synth_personal | UNKNOWN | UNKNOWN |
| synth_planning | UNKNOWN | UNKNOWN |
| synth_technical | UNKNOWN | UNKNOWN |
| synth_work | UNKNOWN | UNKNOWN |

## Query coverage by side

| Dataset | n | user-only | assistant-only | both | none |
|---|---:|---:|---:|---:|---:|
| locomo_30q | 30 | 18 (60.0%) | 12 (40.0%) | 0 (0.0%) | 0 (0.0%) |
| synthetic_19q | 19 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 19 (100.0%) |

### Per-category side counts

**locomo_30q**: {'locomo_temporal': {'user': 9, 'assistant': 7}, 'locomo_multi_hop': {'user': 4}, 'locomo_single_hop': {'user': 5, 'assistant': 5}}

**synthetic_19q**: {'control': {'none': 3}, 'conjunction': {'none': 3}, 'completeness': {'none': 4}, 'inference': {'none': 3}, 'proactive': {'none': 4}, 'procedural': {'none': 2}}

## Fair-backfill recall (full question sets)

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| speaker_user_filter | locomo_30q | 0.383 | 0.839 | +0.456 | 0.508 | 0.892 | +0.383 | 1.0 |
| speaker_user_filter | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| two_speaker_filter | locomo_30q | 0.383 | 0.892 | +0.508 | 0.508 | 0.892 | +0.383 | 1.0 |
| two_speaker_filter | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| two_speaker_boost_0.05 | locomo_30q | 0.383 | 0.625 | +0.242 | 0.508 | 0.892 | +0.383 | 1.0 |
| two_speaker_boost_0.05 | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |

## Subset recall tables

Each row restricts to a subset of queries (by side-mention) and
compares the two-speaker arch vs a reference (meta_v2f or
speaker_user_filter).

| Arch | Subset key | n | ref@20 | arch@20 | Δ@20 | ref@50 | arch@50 | Δ@50 | W/T/L@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| two_speaker_boost_0.05 | locomo_30q__user_only_vs_v2f | 18 | 0.806 | 0.583 | -0.222 | 0.889 | 0.944 | +0.056 | 1/17/0 |
| two_speaker_boost_0.05 | locomo_30q__assistant_only_vs_v2f | 12 | 0.681 | 0.688 | +0.007 | 0.812 | 0.812 | +0.000 | 0/12/0 |
| two_speaker_boost_0.05 | locomo_30q__assistant_only_vs_spf | 12 | 0.681 | 0.688 | +0.007 | 0.812 | 0.812 | +0.000 | 0/12/0 |
| two_speaker_boost_0.05 | locomo_30q__transform_fired_vs_v2f | 30 | 0.756 | 0.625 | -0.131 | 0.858 | 0.892 | +0.033 | 1/29/0 |
| two_speaker_filter | locomo_30q__user_only_vs_v2f | 18 | 0.806 | 0.944 | +0.139 | 0.889 | 0.944 | +0.056 | 1/17/0 |
| two_speaker_filter | locomo_30q__assistant_only_vs_v2f | 12 | 0.681 | 0.812 | +0.132 | 0.812 | 0.812 | +0.000 | 0/12/0 |
| two_speaker_filter | locomo_30q__assistant_only_vs_spf | 12 | 0.681 | 0.812 | +0.132 | 0.812 | 0.812 | +0.000 | 0/12/0 |
| two_speaker_filter | locomo_30q__transform_fired_vs_v2f | 30 | 0.756 | 0.892 | +0.136 | 0.858 | 0.892 | +0.033 | 1/29/0 |

## Per-category (two_speaker_filter vs base cosine)

### locomo_30q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| locomo_multi_hop | 4 | +0.375 | +0.375 | 3/1/0 |
| locomo_single_hop | 10 | +0.775 | +0.700 | 8/2/0 |
| locomo_temporal | 16 | +0.375 | +0.188 | 3/13/0 |

### synthetic_19q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| completeness | 4 | +0.016 | +0.058 | 2/2/0 |
| conjunction | 3 | +0.143 | +0.048 | 1/2/0 |
| control | 3 | +0.000 | +0.000 | 0/3/0 |
| inference | 3 | +0.043 | +0.000 | 0/3/0 |
| proactive | 4 | +0.018 | +0.037 | 1/2/1 |
| procedural | 2 | +0.067 | +0.000 | 0/2/0 |

## Verdict

- Both-side ID hit rate: 23.1%
- locomo_30q side coverage: user-only=60.0%, assistant-only=40.0%, both=0.0%, none=0.0%
- synthetic_19q side coverage: user-only=0.0%, assistant-only=0.0%, both=0.0%, none=100.0%
- LoCoMo two_speaker_filter vs speaker_user_filter: Δ@20=+0.053, Δ@50=+0.000
- LoCoMo two_speaker_filter vs meta_v2f: Δ@20=+0.136, Δ@50=+0.033

**SHIP two_speaker_filter** — extends speaker_user_filter to the assistant side and improves on it (Δ@20=+0.053, Δ@50=+0.000) at ~0 extra per-query LLM cost.
