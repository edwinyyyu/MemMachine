# T_lblend Edge-Case Failure Modes

Setup: T_lblend = 0.2*interval_jaccard + 0.2*tag_jaccard + 0.6*lattice on four
synthetic adversarial benchmarks (12 queries each, 60 docs each, ~1 gold doc
per query, 4-5 same-topic distractors at off-target dates). Compared against
rerank_only (CE union of T-top-50 + S-top-50) and semantic_only.

## Headline numbers (R@1 / R@5 / MRR)

| benchmark            | T_lblend          | rerank_only       | semantic_only     |
|----------------------|-------------------|-------------------|-------------------|
| era_refs             | 0.083 / 0.250 / 0.16 | 0.250 / 1.000 / 0.47 | 0.167 / 1.000 / 0.49 |
| relative_time        | 0.417 / 0.917 / 0.63 | 0.250 / 1.000 / 0.53 | 0.333 / 1.000 / 0.58 |
| conjunctive_temporal | 0.917 / 0.917 / 0.93 | 1.000 / 1.000 / 1.00 | 0.417 / 1.000 / 0.67 |
| multi_te_doc         | 0.667 / 0.917 / 0.79 | 1.000 / 1.000 / 1.00 | 1.000 / 1.000 / 1.00 |

T_lblend wins R@1 only on relative_time, and only because the topic words
collide across distractors there. Everywhere else it under-performs the
trivial semantic baseline.

## Per-benchmark failure mode

### 1. era_refs — extraction collapse (all three components contribute zero)
T component non-zero rate across top-5 entries: **iv=7/60, tag=10/60,
lattice=2/60.** For 11/12 queries the gold doc and every retrieved doc
score 0.000 across all three T components — so T_lblend ranking is just
"whatever order make_t_scores iterates the dict in."

Failure cause: era phrases like "during grad school" / "back when I worked at
Acme" are not extracted by extractor_v2 into intervals, axis tags, or lattice
cells — they have no surface date tokens, no quarter/month/year/season tags,
nothing for the lattice to bind. The query side of T is empty, so all three
sub-scores degrade to zero (interval_pair_best returns 0.0 on empty q_ivs;
tag_jaccard collapses to 0; lattice retrieval has no probe). Doc side is fine
— docs all have explicit dates — but with empty queries the channel is dead.

**Concrete example** (q `era_q_000` "When did Sarah Park take her European
backpacking trip during grad school?"):
```
#1 iv=0.00 tag=0.00 lat=0.00 tot=0.00 | Kim Patel got the promotion on August 14, 2021.
#2 iv=0.00 tag=0.00 lat=0.00 tot=0.00 | Quinn Reeves adopted his dog on July 7, 2020.
#3 iv=0.00 tag=0.00 lat=0.00 tot=0.00 | Marcus Davis hosted his first dinner party on November 8, 2016.
#4 iv=0.00 tag=0.00 lat=0.00 tot=0.00 | Sara Lee moved into her first apartment on August 26, 2023.
#5 iv=0.00 tag=0.00 lat=0.00 tot=0.00 | Felix Wood ran the half-marathon on January 4, 2022.
[GOLD@46] Sarah Park took her European backpacking trip on March 12, 2019.
```
**Culprit: ALL three components.** The pipeline has no era→interval mapping.

### 2. relative_time — partial coverage (T modestly beats semantic)
T_lblend lifts R@1 to 0.417 vs semantic 0.333. Wins are queries where v2
extractor resolves the relative phrase against ref_time (e.g., "last year",
"last month" → year/month tags). Failures are mostly fine-grained anchors
("yesterday", "yesterday afternoon", "three weeks ago") where extractor
either anchors at day granularity that misses ±1 day, or fails to bind to
ref_time at all.

**Concrete example** (q `rel_q_000` "renew her gym membership, yesterday?",
ref_time 2025-04-15):
```
#1 iv=1.00 tag=1.00 lat=0.45 tot=0.67 | Tom Reed wrapped the all-hands on April 14, 2025.
[GOLD#2] iv=1.00 tag=1.00 lat=0.45 tot=0.67 | Sarah Park renewed her gym membership on April 14, 2025.
```
Gold and a topic-irrelevant distractor score IDENTICALLY on every T component
because both share the date 2025-04-14 — T cannot disambiguate by topic.
**Culprit: by design.** This is the expected limitation — T should not score
topic, but here it surfaces the structural asymmetry that T+R works (the
reranker resolves the tie correctly when the gold is in the union).

### 3. conjunctive_temporal — interval-sum-of-best penalizes covering-both
For "Q3 2023 AND Q1 2024" style queries, the gold doc covers BOTH dates and
the single-anchor distractors cover ONE. The interval channel's
sum-of-best is fair (both query intervals find a perfect doc match → high
sum). But the **tag channel actively penalizes** the gold:

**Concrete example** (q `conj_q_001` "Marcus Davis's quarterly reviews
between March and August 2024"):
```
#1 iv=1.00 tag=0.50 lat=0.59 tot=0.65 | Sara Lee had puppy training sessions on March 12, 2024.
#2 iv=0.99 tag=0.50 lat=0.59 tot=0.65 | Marcus Davis had his quarterly reviews on March 15, 2024.
#3 iv=1.00 tag=0.44 lat=0.59 tot=0.64 | Sara Lee had puppy training sessions on March 12, 2024 and on June 19, 2024.
#4 iv=0.91 tag=0.50 lat=0.59 tot=0.64 | Layla Smith had tax filings on April 4, 2024.
#5 iv=0.90 tag=0.50 lat=0.59 tot=0.63 | Olivia Roberts had yoga classes on April 6, 2024.
[GOLD@7] iv=0.99 tag=0.35 lat=0.59 tot=0.62 | Marcus Davis had his quarterly reviews on March 15, 2024 and on August 20, 2024.
```
Gold tag = **0.35**, single-anchor distractors tag = **0.50**. The doc-side
tag union for the two-date gold contains *more* tags (March-2024 ∪ August-2024
= {Q1, Mar, Q3, Aug, ...}), so jaccard with the query's union shrinks. Doc is
*correctly more informative* and the metric **inverts** that into a penalty.
**Culprit: tag_jaccard's symmetric-union denominator.** T_lblend ranks the
single-Q1 distractor *above* the both-quarters gold, plus a totally-wrong
person at #1 because Sara Lee's interval matches the March-2024 anchor as
strongly as Marcus's does and her tags happen to align with the query.

### 4. multi_te_doc — interval-sum and tag-jaccard both dilute on long docs
Gold doc is a meeting note containing 5 dates (gold date plus 4 unrelated).
Distractors are meeting notes with 5 unrelated dates each. T loses 4/12
because some distractor's first date matches the query date as a coincidence,
and dilution drags gold's normalised score below it.

**Concrete example** (q `mte_q_000` "What did Sarah Park do on March 12,
2024?"):
```
#1 iv=1.00 tag=0.32 lat=0.45 tot=0.54 | Meeting notes for Aiden Park. ...status update on March 12, 2024...
[GOLD#2] iv=0.85 tag=0.30 lat=0.45 tot=0.50 | Meeting notes for Sarah Park. On March 12, 2024, Sarah Park completed the dental cleaning. Other items: project review on January 5, 2024, ...
```
Aiden-Park doc has March-12-2024 as one of five dates listed (no actual
event); gold has the same date but as the *focal* event. Both have lattice =
0.45 and tag ≈ 0.31. Gold's iv_norm = **0.85** vs distractor's **1.00** —
because `interval_pair_best` divides by the global max of the *raw* sum, and
some doc with an even tighter date match for the single query interval
becomes the normaliser. **Culprit: interval channel + tag-jaccard both
dilute when doc-side has many TEs.** Tag union over 5 dates dilutes the
March-12 component to 1/5 of doc-tag-mass; interval sum-of-best is
unweighted by focus/centrality, so the focal date and the bullet-point
date count equally. Semantic (which weights focal nouns/verbs) gets this
right at R@1 = 1.000.

## Pattern summary — what new signals would fix each

| failure mode | what's missing | concrete fix |
|---|---|---|
| era_refs | no era→interval translation; T is dead on era queries | era extractor (era_extractor.py exists already) → resolve "during grad school" against a per-user life-event timeline (or LLM era→date-range mapping at query time); emit interval/tags/lattice cells from the resolved range |
| relative_time | partial; the residual failures are extractor granularity errors on "yesterday" / "N days/weeks ago" | extractor patch: explicit relative-phrase parser anchored on ref_time with day-granularity; gives T a clean interval. T then ties with same-date distractors → relies on R to break |
| conjunctive_temporal | tag-Jaccard's union-denominator inverts on multi-anchor gold | replace doc-side tag union with **per-query-anchor coverage**: for each query interval, compute max-overlap doc tag; combine via product or min, not jaccard. (Equivalent to "AND-coverage" instead of "set-similarity".) Keep interval sum-of-best — it already does this on the interval side; just bring tag in line |
| multi_te_doc | every TE in a long doc contributes equally to tag union and interval set; focal/peripheral dates count the same | weight TEs by **centrality**: salience score per TE (already prototyped in salience_extractor.py / salience_eval.py) → up-weight the TE that anchors the doc's focal predicate, down-weight bullet-listed/calendrical mentions. Alternatively, switch tag aggregation from union → max (max-pool query-side AND doc-side); switch interval norm denominator from global-max to per-query-anchor max |

Cross-cutting pattern: **all four failures trace back to T's set-/union-based
aggregations on the doc side.** Tag-jaccard symmetric union, interval
sum-of-best with global normaliser, and lattice all-tags-on-doc-side all
treat every TE in a doc as equally informative. Two adversarial regimes
break this:
1. doc has ONE focal TE plus extras → other channels get noise (multi_te_doc).
2. doc has MULTIPLE focal TEs and query asks for all → set-similarity inverts
   the gain into a penalty (conjunctive_temporal).

The unifying fix is **per-anchor coverage scoring with TE-salience weighting**:
- Score each query anchor against its best doc TE independently (already done
  for intervals; do it for tags/lattice too).
- Combine query anchors by **min/product** ("AND") not arithmetic mean.
- Weight doc TEs by extracted-salience so peripheral/bullet dates are
  discounted before set-formation.

The era_refs collapse is orthogonal: it's an extractor coverage gap, not a
scoring bug. era_extractor.py exists in the repo but isn't wired into
make_t_scores; the highest-value short-term unlock is plugging it in.

## Files
- Synth:    `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/edge_synth.py`
- Eval:     `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/edge_eval.py`
- Data:     `data/edge_{era_refs,relative_time,conjunctive_temporal,multi_te_doc}_{docs,queries,gold}.jsonl`
- Per-q dumps: `results/edge_T_components_{name}.json`
- Summary:  `results/edge_summary.json`
