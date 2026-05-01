# LoCoMo-30 topic+summary stacked dual-view (Part A)

## References

| Variant | R@20 | R@50 |
| --- | --- | --- |
| em_v2f_topic (topic-only ingest)              | 0.8333 | 0.9333 |
| em_v2f_summ (summary-only dual-view)          | 0.9083 | 0.9167 |
| em_v2f_summ_sf_spkfilter (summary-only + spkf)| 0.8917 | 0.9417 |
| em_topic_plus_speaker_filter (topic-only)     | 0.8667 | 0.9333 |

## Stacked (topic + summary) recall

| Variant | R@20 | R@50 | time (s) |
| --- | --- | --- | --- |
| `em_cosine_baseline_topicsumm` | 0.8500 | 0.9083 | 7.8 |
| `em_v2f_topicsumm` | 0.8583 | 0.9250 | 314.2 |
| `em_v2f_topicsumm_sf_spkfilter` | 0.8750 | 0.9250 | 287.1 |

## View coverage (top-50, gold-credited)

| Variant | gold credited | raw wins | summary wins | summary share |
| --- | --- | --- | --- | --- |
| `em_cosine_baseline_topicsumm` | 40 | 1 | 39 | 97.50% |
| `em_v2f_topicsumm` | 40 | 6 | 34 | 85.00% |
| `em_v2f_topicsumm_sf_spkfilter` | 40 | 5 | 35 | 87.50% |

## Decision rules (from plan)

- em_v2f_topicsumm > em_v2f_summ (0.9083/0.9167) by >=1pp -> stacks additively
- ties em_v2f_summ -> summary captures what topic baking did
- em_v2f_topicsumm_sf_spkfilter breaks 0.9417 (K=50) -> new LoCoMo ceiling

## Verdict

**Topic + summary do NOT stack additively.** Summary captures (and exceeds)
what topic baking does:

- `em_cosine_baseline_topicsumm` 0.8500/0.9083 is IDENTICAL to
  `em_cosine_baseline_summ` 0.8500/0.9083 -- the topic prefix on the raw
  view adds nothing when the summary view is also present.
- `em_v2f_topicsumm` 0.8583/0.9250 REGRESSES vs `em_v2f_summ`
  0.9083/0.9167 at R@20 (-5pp) and slightly beats at R@50 (+0.83pp).
  The regression at R@20 is mild; the topic prefix appears to add noise
  when cues already aim at specific content.
- `em_v2f_topicsumm_sf_spkfilter` 0.8750/0.9250 UNDERPERFORMS
  `em_v2f_summ_sf_spkfilter` 0.8917/0.9417 at both K's. 0.9417 ceiling
  is NOT cracked -- it's lost (-1.67pp R@50).
- View-coverage: 85-97% of gold credits go to the summary view -> the
  raw view (with or without topic prefix) is mostly redundant once
  summary is indexed. Topic prefix crowds the raw-view embedding with
  token noise ("[topic: X]") that slightly hurts specificity.

Recommendation: use `em_v2f_summ_sf_spkfilter` for LoCoMo K=50 (0.9417
ceiling); do NOT stack topic baking on top of turn summary.

Part C (reflective on top of topic+summary) was SKIPPED per plan:
"If Part A shows topic+summary lifts, try reflmem on top." Part A
shows no lift, so reflective iteration on this ingest is not the
right next step.

## Outputs

- Collections manifest: `results/eventmemory_topicsumm_collections.json`
- SQLite store: `results/eventmemory_topicsumm.sqlite3`
- Sources: `em_setup_topicsumm.py`, `compose_eval.py`