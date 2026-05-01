# T_lblend Missing Patterns

Four 15-query benchmarks targeting query patterns that the current `T_lblend`
scorer (`0.2·interval + 0.2·tag + 0.6·lattice` from `make_t_scores`) does not
handle. Each benchmark probes a distinct missing capability.

Synth scripts: `causal_relative_synth.py`, `latest_recent_synth.py`,
`open_ended_date_synth.py`, `negation_temporal_synth.py`.
Eval: `missing_patterns_eval.py`.
Per-query JSON: `results/missing_patterns_eval.json`.

## R@1 Summary

| Benchmark           | rerank_only R@1 | T_lblend R@1 |
|---------------------|-----------------|--------------|
| causal_relative     | 7 / 15  (0.467) | 0 / 15  (0.000) |
| latest_recent       | 2 / 15  (0.133) | 0 / 15  (0.000) |
| open_ended_date     | 4 / 15  (0.267) | 0 / 15  (0.000) |
| negation_temporal   | 0 / 15  (0.000) | 0 / 15  (0.000) |

**Headline: T_lblend hits 0/60 across all four patterns.** Each benchmark
isolates a different reason. Rerank_only also degrades significantly: it
saturates only on causal_relative (where the topic words alone disambiguate
gold) and collapses on negation (where the in-window distractors are *more*
on-topic than gold).

## Sample queries (3 per benchmark)

**causal_relative**
- "What did Sarah say after the migration was complete?"
- "What happened before the launch?"
- "What did Maya report since the last review?"

**latest_recent**
- "What's my latest project Alpha status update?"
- "When was my last appointment with Dr. Patel?"
- "What's the most recent feedback on the dashboard design?"

**open_ended_date**
- "What did I work on after 2022?"
- "What did I work on before the pandemic (January 2020)?"
- "What's my activity since I moved in June 2023?"

**negation_temporal**
- "What did I do not in 2023?"
- "What expenses do I have outside of the holiday season (November–December)?"
- "Meetings excluding Q4 2023"

## Per-benchmark diagnosis

### 1. causal_relative — T has no event-anchored time semantics
- **Failure mode**: Gold has T=0 in 14/15 queries. interval=0 in 15/15,
  lattice=0 in 14/15, tag=0 in 14/15.
- **Why**: The query strings ("after the migration", "before the launch")
  contain *no absolute date or duration*. The v2 extractor produces no
  `TimeExpression` from such phrases, so `q_ivs`, `q_tags`, and lattice
  expansion are all empty. T_lblend collapses to a uniform-zero score.
- **What's missing**: The system has no notion of an *event anchor* — a
  reference to a named event in the corpus that itself has a date. To handle
  this it would need to (a) detect the anchor reference, (b) resolve it to a
  doc whose date provides the anchor, then (c) score candidates by their
  *relative position* to that resolved anchor (after / before / since).
  None of these three steps exist.

### 2. latest_recent — T has no recency / "last instance" semantics
- **Failure mode**: Gold has T=0 in 12/15. interval=0 in 15/15.
- **Why**: Queries are deliberately date-free ("latest", "last", "most
  recent"). The extractor returns no intervals/tags. Even when something
  *is* extracted (3/15 nonzero gold T), the score is no higher than older
  instances of the same topic — there's no "prefer-newer" preference.
  Reranker also can't help (2/15 R@1) because all 5 candidate docs have
  identical text modulo date.
- **What's missing**: No max-over-dates / argmax-by-recency primitive. T
  computes *similarity* between query intervals and doc intervals, never
  *the latest* of a candidate set. The query's `ref_time` is computed but
  never used as a sort key against doc dates.

### 3. open_ended_date — T scores by interval *overlap*, not *containment*
- **Failure mode**: Gold has nonzero T in all 15/15 queries (extractor *does*
  produce something from "after 2022", "before January 2020"). But T's
  top-1 is the wrong doc in 15/15 queries, and the wrong pick has nonzero
  T in 15/15 cases (i.e., the system is *confidently* wrong, not silent).
- **Why**: `interval_pair_best` uses `score_jaccard_composite`. "After 2022"
  apparently extracts as a closed interval centered on/inside 2022, so the
  best-overlap doc is one *in* 2022, not *after* it. Q[1] makes this
  extreme: gold (July 2019) gets iv=0 / tag=0.07; the in-window distractor
  (January 2020) gets iv=0.276, tag=0.667, lat=0.591 — T(distractor)=0.688
  vs T(gold)=0.014.
- **What's missing**: Open-ended (one-sided / unbounded) intervals. The
  extractor and the scorer both assume two-sided intervals. There is no
  `(−∞, 2020-01)` representation, and no `containment` or `before/after`
  predicate that would let T say "doc inside the open ray" → match.

### 4. negation_temporal — T has no exclusion semantics, AND text-rerank inverts
- **Failure mode**: Gold has nonzero T in 15/15 (extractor pulls "2023",
  "Q4 2023" etc.), but the in-window distractors *always* outrank gold:
  active wrong pick in 15/15. Rerank_only is even worse: 0/15 R@1 because
  the in-window distractor docs (which copy the query's excluded date
  words) get *higher* CE scores than the out-of-window gold.
- **Why**: All three T components are positive-match-only:
  - `interval_pair_best` is monotone in Jaccard overlap → larger when
    doc-date is *inside* the query interval.
  - `tag_score` is positive overlap → matches share more tags.
  - `lattice_retrieve_multi` only inserts *positive* candidates from
    `query_by_tags`. There is no exclude-set step.
  Result: every component pushes the in-window distractors *higher*, not
  lower. Q[2] illustrates: gold (Feb 2024 meeting) gets T=0.029; in-window
  distractor (Oct 2023 meeting) gets T=0.674.
- **What's missing**: No way to express "exclude this set of dates". T is a
  similarity blend; negation is a logical operator that doesn't compose
  with similarity. Both retrieval (lattice) and scoring (interval/tag)
  would need a separate negative pathway.

## Concrete proposals

### causal_relative → event-anchor resolution path
- Detect anchor references in the query (`the migration`, `the launch`,
  `the last review`) — likely an LLM rewrite step that takes the query
  plus a peek at the corpus index and rewrites as a 2-step plan:
  1. Resolve "the migration" to the doc/date in the corpus that describes
     the named event (anchor doc).
  2. Filter / re-score candidates by `before` / `after` / `since` relative
     to the resolved anchor date.
- Without this, T_lblend cannot help. Even with this, the answer doc text
  must still be *recovered by semantic+rerank* — the temporal step only
  reorders within the on-topic candidates. The 7/15 rerank_only baseline
  shows a sufficiently distinctive topic gold doc can be picked by
  semantics alone (when on-topic gold is more semantically salient than
  the in-corpus anchor doc itself).
- Bench is suitable for testing such an anchor-resolution module.

### latest_recent → recency primitive
- Add a per-query recency feature when the query contains
  "latest" / "last" / "most recent" / "newest" / etc. Rank by `doc_date`
  ascending-from-`ref_time` *within the on-topic candidate set*.
- This benchmark is intentionally unsolvable without that primitive.
  Reranker peaks at 2/15 because it picks *one* of the 5 identical-template
  docs effectively at random by tiny tokenizer differences in dates.
- Implementation could be: cue-detector → if "latest"-type, take rerank
  top-K and re-sort by date desc, keep K=1 if cue is unambiguous.

### open_ended_date → one-sided interval representation
- Two pieces:
  1. Extract `(after 2022)` as `Interval(start=2023-01-01, end=+∞)` (or
     equivalent); `(before Jan 2020)` as `Interval(start=−∞, end=2019-12-31)`.
  2. Scorer needs *containment* not *overlap*. A doc with date `d` and
     a query interval `[s, e]` should score 1.0 if `s ≤ d ≤ e`, else 0
     (or some monotone-with-distance variant). Jaccard is the wrong shape
     for unbounded intervals — Jaccard with one infinite endpoint is
     either undefined or zero.
- This is also a fix for closed-interval queries like "in Q3 2023" but
  the failure mode there is masked by the lattice channel (which
  *does* do containment via tag inclusion); the open-ended case has no
  fallback.

### negation_temporal → separate exclude-set path
- This benchmark is a clean stress test that motivates a query with
  *positive and negative components*. Two viable architectures:
  - **Two-pass**: extract positive intent ("meetings", "expenses") and
    negative intent ("not in 2023"). Run T+R for the positive, get
    candidates. Filter out candidates whose date intersects the negative
    set.
  - **Negative-tag query**: extend the lattice query API to take
    `must_have_tags` and `must_not_have_tags`. Both interval and tag
    scoring become signed: docs in the excluded interval get
    *score − penalty* rather than *score + bonus*.
- Note: rerank_only also fails (0/15) — the distractors copy the
  excluded date phrasing, so the cross-encoder *up-ranks* them. The
  benchmark cleanly shows that *no* current channel (semantic, rerank,
  T_lblend) handles negation. It needs a dedicated mechanism.

## Files

- Data: `data/{causal_relative,latest_recent,open_ended_date,negation_temporal}_{docs,queries,gold}.jsonl`
- Synth: `{causal_relative,latest_recent,open_ended_date,negation_temporal}_synth.py`
- Eval: `missing_patterns_eval.py` (R@1/R@5 + per-query T-component breakdown)
- Per-query JSON: `results/missing_patterns_eval.json`
- Diagnostic helpers: `_diagnose.py`, `_diagnose2.py`
