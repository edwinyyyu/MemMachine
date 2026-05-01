# Future Ideas — Post-Representation

After utterance anchor (R2g) landed (+22pp on utterance-divergence subset), the
open questions span scoring, structure, and extraction robustness. Brainstorm
sorted by expected leverage.

## Already known headroom

- **E3 LLM-judge upper bound** showed ~11pp R@5 gap between hand-crafted
  scoring and a gpt-5-mini judge. Concrete ceiling, untapped.
- **Extraction recall holes** — `last month`, `this month`, eras in personal
  narratives. Extractor v2–v6 currently running; results pending.
- **Corpus shape**: current synthetic data doesn't stress events (so E1 was
  neutral) or narrative sequences (so Allen-relations untested).

## Top new ideas to test

### F1 — Learned scoring from E3 labels ★
Attack the 11pp headroom directly. E3 already collected ~400 pairwise
judgments. Build a small feature-based ranker:
- Features: jaccard, best-proximity (sec), granularity-gap, semantic cosine,
  has-anchor-match, has-recurrence-instance-match, query-length,
  num-expressions-query, num-expressions-doc, anchor-vs-referent path.
- Model: logistic regression + tiny MLP (2-layer, 16 hidden). Train on 80%,
  test on 20%. Cross-validated AUC + R@5 lift on the 20 held-out queries.
- Output: `scorer_learned.py` that can be plugged in where
  `jaccard_composite` is today.
- Expected: 3–8pp R@5 recovery on the E3-labeled subset; extrapolate to full
  corpus. If <3pp, model is under-capacity; go to neural ranker. If >8pp, we
  close most of the judged headroom with a small model.

### F2 — Hierarchical granularity tags (alternative to fuzzy brackets)
Each extracted time emits tags at every containing granularity:
- `March 15, 2024` → `["day:2024-03-15", "month:2024-03", "quarter:2024-Q1",
  "year:2024", "decade:2020s"]`
- `the 90s` → `["decade:1990s", "year:1990..1999"]` (one per year, or just
  decade tag with expansion-at-query)
- Store per doc as list of tags; query extracts same tag format.
- Score: `|shared_tags| / |union_tags|` — clean Jaccard over discrete tags;
  no bracket fuzziness needed.
- This sidesteps the continuous-interval scoring problem entirely and
  reduces matching to set intersection (fast, interpretable).
- Expected: parity or slight lift on base; bigger lift on mixed-granularity
  queries. Catastrophic failure mode: "between 2019 and 2022" becomes 4 year
  tags — tractable.

### F3 — Negation / polarity awareness
`"We didn't meet last week"` currently indexes `last week` the same as
`"We met last week"`. Add a polarity bit to each TimeExpression:
- Pass 2 outputs `polarity ∈ {affirmed, negated, uncertain}`.
- Retrieval: default to affirmed-only; negated matches flagged as low-rank
  or filtered by query intent.
- Small fix, catches real retrieval bug on natural text.

### F4 — Event-time joint enrichment on BOTH sides
E1 failed because queries had null event spans (all "what happened on X").
Retry with event enrichment forced on queries too:
- Pass 2 on queries: if query is temporal-only, LLM augments with plausible
  event types ("activities", "messages", "meetings") as implicit event
  hooks.
- Match events across sides; now event-binding has both sides populated.
- Expected: revives E1 on this corpus; tests whether the architecture truly
  helps.

### F5 — Allen-relation retrieval channel
Current system handles "overlaps with X" but not "before X" / "after X" /
"during X". Add a retrieval path that uses explicit relations:
- Extract `(relation, anchor_event_or_time)` pairs: "before the meeting" →
  `(before, meeting)`.
- Resolve `meeting` via co-reference or entity-name match.
- Score via Allen inequality over resolved intervals.
- Expected: unlocks a query class the current system can't answer.

### F6 — Confidence-weighted aggregation
LLM already emits `confidence` per expression; currently unused in scoring.
Weight each pair contribution by `min(q.confidence, d.confidence)`:
- Low-confidence extractions don't dominate ranking.
- Test: does this reduce noise from LLM FP extractions?

### F7 — Multi-granularity retrieval cascade (efficiency, defer to scale)
First pass: match at year granularity (fast). Second pass: refine within
top-N candidates using day-grain overlap. Not urgent at prototype scale.

### F8 — Document-type classifier for scoring profile selection
Calendar entries → anchor-dominant scoring. Narrative → referent-dominant
scoring. News → publication-date-dominant. Today R2e routes on *query*
intent; extension routes on *doc* type. Higher lift on mixed corpora.

### F9 — Event-lifecycle merge across docs
"I started at Company X in 2020" + "I left Company X in 2023" → merged
interval `[2020, 2023]` tagged to entity Company X. Enables "when did I work
at X?" queries.
Complex; requires entity linking. Defer but keep on the radar.

### F10 — Temporal coherence loss in extraction
When a doc contains N times, penalize internally inconsistent extraction:
if "last Thursday" and "three days ago" both appear at the same ref_time
and resolve to different days, flag. Self-consistency check at doc level.

## Priorities

Run **F1 (learned scoring)** immediately — biggest proven-headroom attack.
After that, if time permits: **F2 (hierarchical tags)** for representation
comparison; **F3 (polarity)** for a real-bug catch; **F5 (Allen relations)**
for new query class.

Deferred: F4 (until E1 corpus is rebuilt with events), F6 (marginal), F7
(scale-bound), F8 (needs doc-type data), F9 (complex), F10 (self-consistency
is a quality check, not retrieval).

## Wave 2 — post-F1/F2 findings

F1 (learned scoring) revealed label-limited, not architecture-limited — with
only 400 labels across 20 queries, the held-out 4-query test was too small.
F2 (hierarchical tags) lost to brackets by discarding continuous proximity
signal. These results suggest bracket-based scoring is already near the
ceiling for the feature set it uses, and learned scoring can only help with
more labels + richer features.

### F11 — Temporal query rewriting with multi-probe union
Take a query like "what did I do 2 years ago?" and LLM-rewrite into K
variants ("in 2024", "around the end of 2023 / start of 2024", "a couple
years back", "during the year before last"). Run retrieval on each variant
independently, union candidates, score by max or fused rank. Classical
query expansion applied to temporal — might close where temporal brackets
alone miss.
**Cost**: +K LLM calls per query.
**Expected**: +3-5pp R@5 via coverage of varied phrasing.

### F12 — Doc-level temporal centroid as a tiebreaker feature
For each doc with N extracted expressions, compute centroid (mean of bests,
weighted by confidence). Query similarly. When multiple docs tie on
Jaccard, sort by centroid distance. Cheap feature, low risk.
**Expected**: resolves ranking ties in center-matching subsets; 1-3pp lift.

### F13 — Active-learning for judge labels
Current 400 E3 labels concentrate on 20 queries. Better approach: select
the most-informative query-doc pairs (high-uncertainty under the learned
scorer) and label those next. Over 2000 labels acquired this way, we
expect the F1 learned scorer to close much more of the 11pp judge-oracle
headroom than uniform sampling.
**Cost**: ~$2-5 for 2000 additional judgments; plus labeling time.
**Expected**: 5-10pp additional R@5 over random sampling.

### F14 — Cross-encoder semantic rerank as a feature
A cross-encoder reranker (ms-marco-style) applied to top-K temporal
candidates would provide fine-grained semantic signal that current
"filter-then-cosine-rerank" lacks. Slightly out of the spirit of "rank
fusion with vector similarity" (user flagged this out of scope) because
cross-encoder is a different operator — worth clarifying with user before
pursuing.
**Deferred**: pending clarification.

### F15 — Time-normalized document embedding
Prepend canonical date string ("[DATE: 2024-04-23]") to text before
embedding. Vectors absorb temporal signal. Plays well with future rank
fusion.
**Expected**: modest lift to pure-semantic baseline; enables rank fusion
without needing structured match.

### F16 — Docs indexed at both raw-text AND structured-time representations
Tri-view: (raw text, temporal structure, events). Each gets its own
retrieval path; scores fused. Generalizes R1/R2 beyond just referent+anchor
to include content as a third signal.

### F17 — Per-expression confidence calibration via LLM chain
Have the LLM emit confidence on a 3-way scale (high/medium/low) instead of
0-1 continuous. Three-way is easier to calibrate and more discriminative
in practice.

## New priorities

Wave 2 picks:
- **F3 (polarity)** — small, catches real bug (pending launch)
- **F5 (Allen relations)** — new query class (running)
- **F11 (query rewriting)** — classical technique worth testing
- **F12 (centroid tiebreaker)** — cheap feature addition
- **F13 (active learning for judge labels)** — attacks F1's bottleneck

Defer: F14 (scope clarification), F15/F16 (rank fusion territory), F17
(minor calibration).

