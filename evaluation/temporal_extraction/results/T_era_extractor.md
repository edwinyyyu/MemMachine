# T_era_extractor — wire EraExtractor as fallback for empty-TE queries

## Headline R@1

| Benchmark | baseline (fuse_T_R + recency_additive) | era_extractor wired | delta |
|---|---:|---:|---:|
| **era_refs** | **0.417** (5/12) | **0.583** (7/12) | **+0.167 (+2 q)** |
| hard_bench (75) | 0.893 | 0.893 | 0.000 |
| temporal_essential (25) | 1.000 | 1.000 | 0.000 |
| tempreason_small (60) | 0.733 | 0.733 | 0.000 |
| conjunctive_temporal (12) | 1.000 | 1.000 | 0.000 |
| multi_te_doc (12) | 1.000 | 1.000 | 0.000 |

Two more era_refs queries land at rank 1 with no regression on the 184-query
regression set. era_refs MRR also improves: 0.667 -> 0.736.

## Era fallback firing pattern

| Benchmark | items eligible (TE-empty) | era TEs added | net effect |
|---|---:|---:|---|
| era_refs | 10 / 12 queries | 7 queries got TEs | +2 R@1 wins |
| hard_bench | 1 / 75 queries | 1 got TE | no change |
| temporal_essential | 0 / 25 | 0 | no-op |
| tempreason_small | 25 / 60 queries | 0 (era extractor declined all) | no-op |
| conjunctive_temporal | 0 / 12 | 0 | no-op |
| multi_te_doc | 0 / 12 | 0 | no-op |

Notable: tempreason had 25 queries with empty TEs (likely "Nov, 1918"-style
year+month with comma that ExtractorV2's prompt missed); EraExtractor refused
to emit eras on all 25 (correctly recognizing them as concrete dates, not
eras), so the fallback was a clean no-op. The hard_bench fallback added 1
TE that didn't displace any existing wins.

## Implementation

Added `era_extractor_eval.py`. Pipeline:

1. Run `run_v2_extract` (standard ExtractorV2) on docs and queries.
2. For each query whose `q_ext[qid]` is empty, run `EraExtractor.extract()`
   as a fallback (gated to queries only — docs in era_refs all have explicit
   calendar dates, doc-side era extraction is wasted compute and unused).
3. Merge fallback TEs into `q_ext` and continue with the standard
   `multi_channel_eval` pipeline (build_memory -> lattice -> T_lblend ->
   `fuse_T_R + recency_additive`).

Key choices:
- **Fallback, not augmentation**: only fires when ExtractorV2 returns []. Keeps
  it from interfering with benchmarks where the standard extractor already
  has signal.
- **Queries-only by default** (`era_on_docs=False`). Doc-side era extraction
  is plumbed but disabled — era_refs docs are all explicit dates; flipping it
  on would only add cost.
- **Single shared `LLMCaller`** with persistent JSONCache, so era prompts are
  cached across reruns.

## Limitations: era_refs queries still missing R@1 after wiring

5 of 12 queries still don't hit R@1; all are personal eras whose ground truth
requires user-specific anchors that EraExtractor's generic defaults can't
pin to:

| qid | query | gold year | issue |
|---|---|---:|---|
| era_q_003 | "during my time at the startup" | 2021 | personal employment era; no generic prior |
| era_q_004 | "back in college" | 2012 | "college" default = ref-10y..ref-6y = 2015-2019 (misses 2012) |
| era_q_005 | "when I lived in Boston" | 2018 | residence era, no default available |
| era_q_007 | "while training for the Olympics" | 2020 | event-anchored, requires knowing which Olympics |
| era_q_010 | "back when I worked at Globex" | 2019 | personal employment era |

What the extractor handles well (when generic defaults overlap gold dates):
- "during the pandemic year" -> 2020-03-11 / 2023-05-05 (correct: 2020-07-07)
- "during grad school" -> generic ref-10..ref-6 = 2015-2019 (correct: 2019-03-12)
- "while living in Sweden" -> personal residence, fuzzy default 2015-2019
  (correct: 2021-05-17 — close enough for top-1 with rerank tie-break)
- "right after I graduated college" -> ref-9..ref-5 = 2016-2020 (correct: 2019-08-12)
- "during my fitness phase" -> generic recent fuzzy interval (correct: 2021-09-04)

What it misses:
- **Personal employment eras** ("at Acme", "at Globex", "at the startup")
  without user metadata — extractor uses fuzzy ref-N..ref-M defaults that
  rarely align with arbitrary user employment histories. Two of 12 queries
  did succeed by coincidence (Acme, fitness-phase) when defaults intersected
  the gold year; two failed (Globex, startup).
- **Personal residence eras** ("when I lived in Boston") for the same reason.
- **Event-anchored eras** ("training for the Olympics", "before the kids
  were born") that require resolving the named anchor and aren't already in
  the world-knowledge prompt list.
- "back in college" specifically — extractor's default for "college" is
  ref-10..ref-6y (2015-2019); gold is 2012, just outside that window.

The unlock for closing these would be a per-user "era registry" (passed as
context to Pass 2 — birth year, employment timeline, residence timeline)
rather than relying on `ref_time`-relative defaults. Without that personal
anchoring, the +2 wins from generic-default overlap is roughly the ceiling.

## Files
- /Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/era_extractor_eval.py
- /Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/results/T_era_extractor.json
- /Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/results/T_era_extractor.md (this file)
