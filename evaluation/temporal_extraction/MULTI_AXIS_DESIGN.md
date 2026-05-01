# Multi-Axis + Distributional Time Representation

F2 (hierarchical tags) tested only CONTAINING-granularity tags (day → week → month → year → decade). That's one axis. Real times live on many orthogonal axes simultaneously — year, month-of-year, day-of-month, day-of-week, hour-of-day, quarter, season, part-of-day, weekend-vs-weekday.

This design combines cross-axis tagging with per-axis continuous distributions, so a single time is represented by:

1. **Absolute interval** (current bracketed fuzzy instant)
2. **Per-axis categorical distributions**: `P(axis_value | time)` for each axis
3. **Cross-axis tags**: the set of `{axis}:{value}` pairs with mass above a threshold

This unlocks query classes neither interval-overlap nor hierarchical-tags can serve cleanly:

- "What have I done on Thursdays?" → matches on `weekday:Thursday` regardless of year
- "March events across any year" → `month-of-year:March`
- "Afternoon meetings" → `part-of-day:afternoon`
- "Weekend events" → `weekend:yes`
- "Q2 anything" → `quarter:Q2`
- "Tuesday afternoons in March 2024" → multi-axis specific query

## Axes to represent

| Axis | Values | Specificity |
|---|---|---|
| year | 1900..2100 | high |
| month | 1..12 (January..December) | medium |
| day-of-month | 1..31 | medium-low (noise-heavy) |
| weekday | 0..6 (Mon..Sun) | medium |
| hour | 0..23 | medium (when specified) |
| quarter | Q1..Q4 | low |
| decade | 1900s..2020s | low |
| season | winter/spring/summer/autumn | low |
| part-of-day | night/morning/afternoon/evening | low |
| weekend | yes/no | very low |

## Distribution generation from a FuzzyInstant

Given `FuzzyInstant(earliest, latest, best, granularity)`:

1. Discretize [earliest, latest] at the smallest relevant grain (day for most, hour for time-of-day expressions).
2. Assign per-point weight via Gaussian centered on `best` with σ = (latest − earliest) / 4. For uniform distributions (no `best`), use uniform weights.
3. For each axis, aggregate point weights by axis-value to form a histogram.
4. Normalize to sum to 1.

### Special cases

- Expression specifies a full datetime: axes are deltas — `year, month, day-of-month, weekday, hour` all concentrated on single values.
- Expression says "last Thursday": `weekday` concentrates on Thursday; other axes drift based on the resolved absolute date.
- Expression says "the 90s": `year` spreads [1990..1999] uniform; `decade=1990s` point mass; `weekday/month/day-of-month` uniform.
- Expression says "3pm today": `year/month/day/weekday` point mass on today; `hour=15` point mass.
- Recurrence "every Thursday": `weekday=Thursday` point mass; all other axes uniform over the recurrence window.

## Tag generation

From the distributions, emit a tag for every axis value with P > threshold (default 0.1). This gives discrete matchable labels AND preserves continuity through the underlying distribution.

## Scoring

### Per-axis match

Given query axis distribution `p_q` and doc axis distribution `p_d`:

- Bhattacharyya coefficient: `BC(p, q) = Σᵢ √(pᵢ · qᵢ)`. Range 0..1. Continuous.
- Entropy gating: if either distribution is ~uniform (entropy close to max), treat as "uninformative on this axis" — skip in the score.

### Combined score

```
axis_score = geomean over informative axes of BC(p_q[axis], p_d[axis])
tag_score  = |shared_tags ∩ informative_axis_tags| / |union|
interval_score = existing quarter/jaccard/sum (for temporal brackets)

total = α · interval_score + β · axis_score + γ · tag_score
```

With default `α = 0.5, β = 0.35, γ = 0.15` to start, sweep later.

## Expected wins

- Queries that specify ONE axis dimension only (weekday, month, hour) can now retrieve docs that share that axis, regardless of absolute time.
- Multi-axis specific queries ("Tuesday afternoons in March") become discriminable — interval-overlap alone can't isolate Tuesdays from Mondays within the same week, but weekday axis can.
- Recurrence queries ("do I have anything on Thursdays?") retrieve via weekday axis, cheaper than expanding instances.

## Expected null cases

- Pure absolute queries ("March 15, 2024") — interval already handles these well; axes add noise.
- Short-burst recent queries ("last week") — interval matches well; axes add little.

## Implementation files (new)

- `axis_distributions.py` — convert `FuzzyInstant` to per-axis categorical distributions
- `multi_axis_tags.py` — generate tag set from distributions
- `multi_axis_scorer.py` — Bhattacharyya per axis, combined scoring
- `axis_synth.py` — 15 docs + 20 axis-specific queries
- `multi_axis_eval.py` — orchestrate and compare vs base hybrid, bracket-only, tag-only

## Success criteria

- On the new axis-specific query subset: R@5 lift ≥ +15pp vs base hybrid (axis-only queries cannot be answered via interval overlap alone — this is a new query class).
- On the base corpus: no regression >1pp R@5.
- On the hierarchical-tags test from F2: match or exceed T2.

If multi-axis hits both, ship the combined scorer as the new default. If it hits the subset criterion but regresses on base, ship as a routed channel (query-intent-based selection).
