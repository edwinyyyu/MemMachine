# LongMemEval hard-category baseline

Evaluates retrieval architectures on the 3 HARD LongMemEval categories
(multi-session, single-session-preference, temporal-reasoning) with 30
questions per category (n=90 total). Per-question conversation scoping.
Fair-backfill recall at K=20 and K=50. Embedding model `text-embedding-3-small`,
cues via `gpt-5-mini`.

## Sample composition

| Category | n | mean haystack turns | gold turns (mean/min/max) |
|---|---:|---:|---|
| multi-session | 30 | 454 | 28.5 / 20 / 60 |
| single-session-preference | 30 | 491 | 14.3 / 12 / 20 |
| temporal-reasoning | 30 | 453 | 26.9 / 10 / 58 |

Total turns embedded: 41,952. Critical-info classifier (gpt-5-mini v3 prompt)
tagged **14/41,952 turns (0.03%)** as CRITICAL — effectively nothing to
anchor a critical-info store on, compared to LoCoMo's richer density.

## Measured recall (real cue generation)

The only architectures for which this evaluation successfully ran real
per-query cue generation are `cosine_baseline` and `two_speaker_filter`.
Because the two-speaker role filter fires on **0/90** questions on LME
(LongMemEval questions are first-person and never name a participant), its
retrieval pipeline reduces exactly to `meta_v2f`'s (v2f cues + hop-0 cosine).
So the `two_speaker_filter` column is the real `meta_v2f` signal on LME hard.

| Architecture | r@20 | r@50 |
|---|---:|---:|
| cosine_baseline | 0.549 | 0.754 |
| meta_v2f (via two_speaker_filter fallthrough) | 0.605 | 0.817 |

Δ meta_v2f vs cosine: **+0.056 @K=20, +0.063 @K=50**.

### Per-category

| Architecture | multi-session | single-session-preference | temporal-reasoning |
|---|---:|---:|---:|
| cosine_baseline r@20 | 0.521 | 0.636 | 0.489 |
| meta_v2f    r@20 | 0.552 | 0.755 | 0.507 |
| **Δ r@20**  | +0.031 | +0.119 | +0.018 |
| cosine_baseline r@50 | 0.760 | 0.787 | 0.716 |
| meta_v2f    r@50 | 0.818 | 0.868 | 0.765 |
| **Δ r@50**  | +0.058 | +0.081 | +0.049 |

## Status of the other architectures

`ens_2_v2f_typeenum`, `critical_info_store`, `ens_all_plus_crit` were NOT
successfully measured in this eval. The shipped `build_specialist` forces
cache-only LLM mode; since every LME-hard question is novel, every specialist
returned `DONE` on cache-miss and collapsed to cosine. A follow-up run with
real cue generation hit API-side rate-limits and did not complete in the
available time (was still on question 5/90 after ~12 min of eval-phase time).

The shape of the fix is known — override `arch.llm_call` to a real
pass-through, not cache-only — but the full re-run was not feasible within
this session's time budget. The `meta_v2f` proxy above is a clean signal of
what v2f achieves; the ensemble deltas (expected small on LME given
critical-info's 0.03% tag rate) remain open.

## Per-category verdict

### multi-session
- cosine r@50 = 0.760, v2f = 0.818 (+0.058). v2f helps modestly.
- Compared to LoCoMo (v2f r@50 = 0.858), multi-session LME is **slightly
  harder in absolute terms** but responds to cue generation similarly.
- Verdict: **v2f generalizes fine. The baseline is already strong because
  LME haystacks have a clear target session; cues pull in the few missing
  turns from that session.**

### single-session-preference
- cosine r@50 = 0.787, v2f = 0.868 (+0.081). Largest v2f lift.
- LME preferences ("I prefer X", "My favorite Y is Z") are fact-dense; cues
  catch the specific vocabulary.
- Verdict: **v2f wins here, as it does on LoCoMo preference-style questions.
  Critical-info is expected to add little (only 14 tagged turns across 42K).**

### temporal-reasoning
- cosine r@50 = 0.716, v2f = 0.765 (+0.049). **Smallest v2f lift, lowest
  ceiling.**
- Consistent with LoCoMo temporal being the weakest category — the
  embedding substrate cannot express temporal relationships
  ("after X but before Y", "the first time I did Z").
- At K=20 temporal reasoning is 0.507 for v2f — substantial gap.
- Verdict: **temporal-reasoning is the fundamentally-hard category.** v2f
  doesn't close the gap; the substrate is the bottleneck. This mirrors
  LoCoMo and is the primary open frontier. Cue generation can surface
  candidate turns, but ranking them into temporal order requires either
  (a) a temporal-aware embedding, (b) a timestamp-aware scoring layer, or
  (c) a timeline-construction pass before retrieval.

## Two-speaker filter coverage

Two-speaker filter fired on **0/90** questions (0.0%). This confirms
the pre-experiment prediction — LongMemEval questions do not name speakers
third-person. The specialist's speaker-ID sub-prompt returned
user=UNKNOWN/assistant=UNKNOWN for essentially every LME conversation,
and no query mentioned any identified name. **Skip two_speaker_filter
on LongMemEval benchmarks.**

## Comparison vs LoCoMo

| Category (analogous) | LoCoMo v2f r@50 (reference) | LME v2f r@50 |
|---|---:|---:|
| LoCoMo overall | 0.858 | (n/a) |
| LME multi-session | — | 0.818 |
| LME single-session-preference | — | 0.868 |
| LME temporal-reasoning | — | 0.765 |

LoCoMo's ceiling on LoCoMo-30 is 0.922 with `ens_all_plus_crit` — +6.4pp
over v2f. On LME, critical-info is effectively empty (0.03% density vs
LoCoMo's richer mentions of medications, allergies, family). The ensemble
probably adds at most a few points on LME.

## Most promising next direction

**Temporal-reasoning is the remaining frontier.** Both on LoCoMo and LME
this category has the lowest v2f ceiling and the smallest v2f-vs-cosine
gap. The embedding substrate's weakness at temporal relations is the
bottleneck. Worth exploring:
1. Timestamp-aware scoring: parse `question_date`, score segments
   whose `haystack_dates` match a temporal predicate.
2. Timeline construction: LLM first builds a session-date timeline from
   gold segments, then answers against the timeline.
3. Duration/interval embeddings: explicitly embed "between X and Y" spans.

None of these are architectural wins we have shipped. LoCoMo's
temporal-reasoning results already foreshadowed this gap.

## Output paths

- `data/questions_longmemeval_hard.json` (90 questions, 103KB)
- `data/longmemeval_hard_segments.npz` (41,952 turn embeddings, 300MB)
- `results/longmemeval_hard_baseline.json` (per-question raw)
- `results/longmemeval_hard_baseline.md` (this file)
- Script: `longmemeval_hard_setup.py`, `longmemeval_hard_eval.py`
- Dedicated caches: `lmehard_embedding_cache.json`,
  `lmehard_llm_cache.json` (contains 42K critical-info decisions + 90
  speaker-ID calls).
