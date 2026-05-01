# Speaker-attributed retrieval

At ingest: identify which real name corresponds to the `user` role
per conversation (1 LLM call per conv). At query time: regex-detect
capitalized first-name tokens; if a token matches the conversation's
user-name, apply a speaker-aware transform (boost or filter role=user
turns). Rationale: dialog entities appear as vocatives ('Hey Caroline!')
not subjects; turns ABOUT Caroline are often SPOKEN by Caroline, and
her name isn't in the turn text.

## Speaker identification

- Conversations scanned: 13
- Identified (non-UNKNOWN): 7
- Hit rate: 53.85%

| Conversation | user speaker |
|---|---|
| beam_4 | Christina |
| beam_5 | Craig |
| beam_6 | Crystal |
| beam_7 | UNKNOWN |
| beam_8 | Darryl |
| locomo_conv-26 | Caroline |
| locomo_conv-30 | Jon |
| locomo_conv-41 | John |
| synth_medical | UNKNOWN |
| synth_personal | UNKNOWN |
| synth_planning | UNKNOWN |
| synth_technical | UNKNOWN |
| synth_work | UNKNOWN |

## Query mention coverage

Fraction of queries in each dataset that mention the conv-user's first name (case-insensitive, stop-word-filtered).

| Dataset | n | mentions conv-user | frac |
|---|---:|---:|---:|
| locomo_30q | 30 | 18 | 0.600 |
| synthetic_19q | 19 | 0 | 0.000 |

### Mentions by category

**locomo_30q**: {'locomo_temporal': 9, 'locomo_multi_hop': 4, 'locomo_single_hop': 5}

**synthetic_19q**: {}

## Fair-backfill recall (full question sets)

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| speaker_boost_0.02 | locomo_30q | 0.383 | 0.622 | +0.239 | 0.508 | 0.892 | +0.383 | 1.0 |
| speaker_boost_0.02 | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| speaker_boost_0.05 | locomo_30q | 0.383 | 0.622 | +0.239 | 0.508 | 0.892 | +0.383 | 1.0 |
| speaker_boost_0.05 | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| speaker_user_filter | locomo_30q | 0.383 | 0.839 | +0.456 | 0.508 | 0.892 | +0.383 | 1.0 |
| speaker_user_filter | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |

## Mentioned-subset recall (speaker arch vs meta_v2f)

On the SUBSET of queries that mention the conv-user. This is the
subset where the mechanism can possibly activate.

| Arch | Dataset | n | v2f@20 | arch@20 | Δ@20 | v2f@50 | arch@50 | Δ@50 | W/T/L@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speaker_boost_0.02 | locomo_30q | 18 | 0.806 | 0.583 | -0.222 | 0.889 | 0.944 | +0.056 | 1/17/0 |
| speaker_boost_0.05 | locomo_30q | 18 | 0.806 | 0.583 | -0.222 | 0.889 | 0.944 | +0.056 | 1/17/0 |
| speaker_user_filter | locomo_30q | 18 | 0.806 | 0.944 | +0.139 | 0.889 | 0.944 | +0.056 | 1/17/0 |

## Per-category (speaker_boost_0.02 vs v2f)

### locomo_30q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| locomo_multi_hop | 4 | +0.125 | +0.375 | 3/1/0 |
| locomo_single_hop | 10 | +0.267 | +0.700 | 8/2/0 |
| locomo_temporal | 16 | +0.250 | +0.188 | 3/13/0 |

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

- Speaker-ID hit rate: 53.8% (7/13)
- LoCoMo-30 conv-user mention coverage: 60.0%
- Synthetic-19 conv-user mention coverage: 0.0%
- Best variant on LoCoMo K=50 full-set: **speaker_boost_0.02** (Δ vs v2f = +0.033)

**SHIP speaker_boost_0.02**: beats v2f on LoCoMo K=50 full-set (Δ=+0.033) at ~0 extra per-query LLM cost (ingest-time speaker ID only).
