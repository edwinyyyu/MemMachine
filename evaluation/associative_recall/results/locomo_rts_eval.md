# intent_parser on EventMemory with REAL LoCoMo timestamps

## Schema changes

- `timestamp` on each event now = session_date_time (parsed) + microsecond offset per in-session turn.
- Per-event metadata adds `session_idx`, `in_session_idx`, `session_date_time`, `dia_id`.

## Constraint firing rates (LoCoMo-30)

| Constraint | Fired | % |
| --- | --- | --- |
| `speaker` | 8 | 26.7% |
| `temporal_relation` | 2 | 6.7% |
| `negation` | 1 | 3.3% |
| `answer_form` | 30 | 100.0% |

## Temporal-filter smoke test

- locomo_conv-26: window [2023-07-12T16:33:00+00:00, 2023-08-23T15:31:00+00:00] — unfiltered hits 30 (16 sessions), filtered hits 30 (7 sessions), ok=True, violations=0.
- locomo_conv-30: window [2023-03-23T19:28:00+00:00, 2023-06-13T20:29:00+00:00] — unfiltered hits 30 (15 sessions), filtered hits 30 (6 sessions), ok=True, violations=0.
- locomo_conv-41: window [2023-04-10T18:13:00+00:00, 2023-07-05T18:59:00+00:00] — unfiltered hits 30 (18 sessions), filtered hits 30 (11 sessions), ok=True, violations=0.

## Recall matrix

| Variant | R@20 | R@50 |
| --- | --- | --- |
| em_v2f_speakerformat (ref) | 0.8170 | 0.8920 |
| em_two_speaker_filter (ref) | 0.8420 | 0.9000 |
| em_two_speaker_query_only (ref) | 0.8000 | 0.9330 |
| intent_em_with_speakerformat_cues (ref) | 0.8167 | 0.9083 |
| intent_em_filter_no_cues (ref) | 0.7500 | 0.9000 |
| **intent_rts_full** | **0.7833** | **0.9083** |
| **intent_rts_temporal_only** | **0.7417** | **0.8833** |
| **intent_rts_speaker_only** | **0.8167** | **0.9083** |

## Per-category (intent_rts_full)

| Category | n | R@20 | R@50 |
| --- | --- | --- | --- |
| locomo_multi_hop | 4 | 0.7500 | 1.0000 |
| locomo_single_hop | 10 | 0.6500 | 0.8250 |
| locomo_temporal | 16 | 0.8750 | 0.9375 |

## locomo_temporal focus (all variants)

| Variant | n | R@20 | R@50 |
| --- | --- | --- | --- |
| intent_rts_full | 16 | 0.8750 | 0.9375 |
| intent_rts_temporal_only | 16 | 0.8125 | 0.9375 |
| intent_rts_speaker_only | 16 | 0.9375 | 0.9375 |

## Firing vs recall

### intent_rts_full

- Filter fired on 10/30 queries (temporal filter: 2/30).
- K=20: with_filter=0.7 without_filter=0.825 with_temporal=0.0
- K=50: with_filter=0.95 without_filter=0.8875 with_temporal=0.75

### intent_rts_temporal_only

- Filter fired on 2/30 queries (temporal filter: 2/30).
- K=20: with_filter=0.0 without_filter=0.7946 with_temporal=0.0
- K=50: with_filter=0.75 without_filter=0.8929 with_temporal=0.75

### intent_rts_speaker_only

- Filter fired on 8/30 queries (temporal filter: 0/30).
- K=20: with_filter=0.875 without_filter=0.7955 
- K=50: with_filter=1.0 without_filter=0.875 


## Verdict

- Best variant R@50=0.9083 UNDER em_two_speaker_query_only (0.9330): temporal filter is too narrow or too rarely resolvable; net negative.
- `intent_rts_temporal_only` R@50=0.8833 ≤ em_v2f_speakerformat (0.8920): temporal signal alone does not beat speakerformat cues.

## Outputs

- `results/locomo_rts.json`
- `results/locomo_rts_eval.md`
- `results/locomo_rts_ingest.md`
- `results/eventmemory_locomo_rts_collections.json`
- Source: `em_setup_rts.py`, `intent_rts.py`, `intent_rts_eval.py`