# T_negation — Temporal-negation handling

## Headline (negation_temporal R@1)

- baseline T_lblend (full query):     **0.000** (0/15)
- baseline T_v4 (full query):         **0.000** (0/15)
- negation **mask**: positive_composite × (1 − excl):    **0.733** (11/15)  Δ = **+0.733**
- negation **signed** (λ=1.0): positive_composite − λ·excl:  **0.733** (11/15)  Δ = **+0.733**

Both strategies fix 11/15 negation queries identically. R@5 = 0.867 for both.

## All-bench R@1 / R@5

| Bench | n | bl_lblend R@1 | bl_v4 R@1 | mask R@1 | signed R@1 | mask R@5 |
|---|---:|---:|---:|---:|---:|---:|
| **negation_temporal** | 15 | 0.000 | 0.000 | **0.733** | **0.733** | 0.867 |
| conjunctive_temporal | 12 | 0.917 | 0.667 | 0.917 | 0.917 | 1.000 |
| multi_te_doc | 12 | 0.750 | 0.667 | 1.000 | 1.000 | 1.000 |
| relative_time | 12 | 0.417 | 0.417 | 0.917 | 0.917 | 1.000 |
| era_refs | 12 | 0.083 | 0.167 | 0.167 | 0.167 | 1.000 |
| hard_bench | 75 | 0.000 | 0.027 | 0.947 | 0.947 | 0.987 |
| temporal_essential | 25 | 0.280 | 0.240 | 1.000 | 1.000 | 1.000 |

## Regression check

`has_negation_cue` fires **0 times** on all 6 non-negation benches (verified
with `_neg_falsepos_check.py`). On those benches the mask/signed paths use
only `positive_composite = 0.7·semantic + 0.3·T_lblend(positive_query)` — the
negation-specific arithmetic is a no-op (no excluded interval extracted).

Mask/signed never regressed below baseline_lblend on any bench. The large
positive deltas (e.g. +0.500 on relative_time, +0.947 on hard_bench, +0.720 on
temporal_essential) come from the semantic channel that the composite scorer
adds on top of T_lblend — not from negation logic itself. They are a side
effect of how the harness computes the positive-query score; baselines here
are pure T-channel only.

## Implementation

- **Cue detection** (`negation.has_negation_cue`): word-boundary regex over
  `not in/during/on`, `outside (of)`, `excluding`, `except (for)`, `without`.
  Bare `not` is gated on a following temporal token (year, `Q1`–`Q4`, month,
  season, holiday, quarter) so we don't fire on `did I not finish the report`.
  Falsepos rate on non-negation benches: 0/148.
- **Parse** (`negation.parse_negation_query`): returns
  `(positive_query, excluded_phrase)`. positive_query removes both the cue AND
  the excluded phrase (everything from cue end up to the next `?`/`.`/`!` —
  commas preserved so phrases like `summer (June–August 2024)` stay intact).
  Example: "What workouts did I do not in January 2025?" → `("What workouts did I do?", "January 2025")`.
- **Excluded intervals**: same v2 extractor (gpt-5-mini, reasoning_effort=minimal)
  is run on the excluded phrase with the original query's `ref_time`. The
  resulting `TimeExpression` list is flattened to `Interval`s.
- **Positive composite score**:
  `0.7·semantic_cosine(positive_query, doc) + 0.3·T_lblend(positive_query)`.
  Semantic dominates because positive_query usually has no temporal phrase
  left, making T_lblend uniformly 0 across docs.
- **Excluded containment** (`negation.excluded_containment`): for doc d and
  excluded interval set E, `max over (d_iv, e_iv) of |d_iv ∩ e_iv| / |d_iv|`.
  Same asymmetric containment primitive as T_v4. A doc fully inside the
  excluded window → 1.0; fully outside → 0.0.
- **mask** (`apply_mask`): `final = positive_composite · (1 − excl_containment)`.
  Multiplicative; guarantees a zero score for docs whose anchor is fully
  inside the excluded window.
- **signed** (`apply_signed`, λ=1.0): `final = positive_composite − λ·excl_containment`.
  Continuous; allows negative scores so excluded docs rank below
  positive-zero docs.

## Per-query analysis (negation_temporal)

11/15 hits at rank 1; 4 misses are queries whose positive_query is too
generic for semantic to disambiguate the topic cluster:

| qid | excluded phrase | positive_query | mask rank |
|---|---|---|---:|
| nt_q_000 | "2023" | "What did I do?" | miss |
| nt_q_001 | "the holiday season (Nov-Dec)" | "What expenses do I have?" | 5 |
| nt_q_002 | "Q4 2023" | "Meetings" | 6 |
| nt_q_011 | "Q1 2024" | "What expenses do I have?" | 5 |

The mask suppresses in-window distractors correctly, but the topic cluster
isn't separated from the rest of the corpus. Queries with topic-distinctive
positive text (workouts, classes, therapy sessions, family events,
presentations) all hit rank 1.

## Limitations

- **Generic positive_query** when the negation cue strips the only
  discriminating information (e.g. "What did I do?" / "Meetings"). Mask
  ranks gold below same-topic-but-out-of-window distractors and unrelated
  docs alike.
- **Bare `not`** requires a temporal token immediately following the cue
  (year, month, Q1–Q4, season, holiday). `not really during the summer`
  fails the lookahead.
- **Sentence-final stop only**: excluded phrase extraction stops at
  `?`/`.`/`!`, not commas. `excluding 2022, and biking` pulls
  `2022, and biking` into the excluded phrase; the extractor usually still
  recovers the right interval since the only temporal token is `2022`.
- **No double-negation handling**: `I did not avoid the holiday season`
  is treated as a `holiday season` exclusion.
- **Disjunction OK, conjunction not distinguished**: `excluding 2022 or 2023`
  produces a multi-interval excluded set (correct). `excluding 2022 AND winter`
  also collapses both into the excluded set — usually still right behavior but
  no syntactic distinction.
- **`signed` and `mask` tied at R@1=0.733** — they only diverge on docs with
  exactly zero positive_composite (rare in this benchmark since semantic
  cosine is rarely exactly zero).

## Files

- `negation.py` — cue detection + parse + scoring helpers
- `negation_eval.py` — full eval driver (negation_temporal + 6 regression benches)
- `results/T_negation.json` — raw per-query results
- `cache/<bench>-neg-pos/`, `cache/<bench>-neg-excl/` — extractor caches for
  positive_query and excluded_phrase
