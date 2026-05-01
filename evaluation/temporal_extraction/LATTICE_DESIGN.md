# Temporal Lattice Inverted Index

Replaces doc-doc co-temporal edges (O(n²) worst-case) with a **lattice inverted index** where docs are tagged at their NATIVE precision only, and queries walk the lattice to expand the search space.

## Motivating problem

Doc-doc edges scale badly — a date in a popular year produces many edges.
Our F2 hierarchical-tag experiment over-tagged (every time got tags at every
containing granularity), which made specific-day queries collapse into
ranking ties because all day-level docs shared their year/quarter/decade
tags.

Right model: **precision-native tagging, lattice-walk lookup.**

## Lattice structure

### Absolute lattice (totally ordered by containment)

```
century        (e.g. "20th century")
  └─ decade    (e.g. "1990s")
       └─ year (e.g. "1999")
            └─ quarter (e.g. "1999-Q1")
                 └─ month (e.g. "1999-01")
                      └─ week (e.g. "1999-W01")
                           └─ day (e.g. "1999-01-01")
                                └─ hour (e.g. "1999-01-01T15")
                                     └─ minute (e.g. "1999-01-01T15:30")
```

### Cyclical axes (orthogonal to absolute)

```
weekday:     Mon / Tue / … / Sun
month-of-y:  Jan / Feb / … / Dec
day-of-mon:  1 / 2 / … / 31
hour-of-day: 0 / 1 / … / 23
season:      winter / spring / summer / autumn
part-of-day: morning / afternoon / evening / night
weekend:     yes / no
```

A single time can have tags on BOTH the absolute and cyclical axes
(e.g., "Jan 1, 1999" has `day:1999-01-01` + `weekday:Friday` +
`month-of-year:January`, etc.).

## Tagging policy (the critical design choice)

Tag docs at **only** their native precision on the absolute lattice, plus
**all applicable cyclical axes**.

| Time expression | Absolute tags | Cyclical tags |
|---|---|---|
| "Jan 1, 1999" | `day:1999-01-01` | `weekday:Friday, month-of-year:January, day-of-month:1, season:winter, weekend:no, part-of-day:ALL` |
| "January 1999" | `month:1999-01` | `month-of-year:January, season:winter` |
| "1999" | `year:1999` | — |
| "the 1990s" | `decade:1990s` | — |
| "every Thursday at 3pm" | — (recurrence — no absolute cell) | `weekday:Thursday, hour-of-day:15, part-of-day:afternoon` |
| "a couple years ago" (fuzzy) | `year:2024, year:2023, year:2022` (a few years surrounding best) | — |

**Never tag upward**. A 1999 doc does NOT get tagged `decade:1990s` — it
gets ONLY `year:1999`. The decade match is discovered at lookup time by
walking the lattice.

## Lookup (bidirectional expansion)

Given a query TimeExpression with native precision P:

1. **Tag the query** at precision P + cyclical axes.
2. **Expand UP** from P to broader absolute cells (parent/ancestors) — these
   are the cells the query point ALSO belongs to.
3. **Expand DOWN** from P to cells contained within the query interval —
   these are the cells whose contents fall within the query's window.
4. **Keep cyclical tags exactly** (no expand up/down).
5. **Look up each expanded cell in the inverted index**, collect candidates
   with their cell match level.

Example — query "Jan 1, 1999" (precision=day):
- UP: `month:1999-01, quarter:1999-Q1, year:1999, decade:1990s, century:20th`
- DOWN: `hour:1999-01-01T00..23` (trivial — contained within the day)
- Cyclical: `weekday:Friday, month-of-year:January, day-of-month:1, …`

All of these look up in the index; ANY doc sharing ANY cell is a candidate.

Example — query "the 1990s" (precision=decade):
- UP: `century:20th`
- DOWN: `year:1990, year:1991, …, year:1999, decade:1990s` (self)
  Plus finer levels? In principle yes — `month:1990-01, …, day:1999-12-31`.
  In practice, expanding down to day level is explosive. **Cap expansion at
  1 level below query precision** (so "decade" expands down to `year` only).
- Cyclical: none (decade has no cyclical specificity).

## Scoring

For each candidate, compute:

```
cell_score = 1 / log2(2 + |cell_span| / |query_span|)      # specificity: narrower cell = higher score
direction_bonus = 1.0 if doc_cell ⊆ query_cell else 0.5    # prefer docs nested in query
semantic_cosine = text-embedding cosine                     # topical fit
axis_overlap = Bhattacharyya on cyclical axes               # continuous fine score

candidate_score = 0.4·cell_score + 0.2·direction_bonus + 0.25·semantic + 0.15·axis_overlap
```

The `cell_score` favors narrow cells: a query-to-doc match at `day` is
stronger than at `decade`. This fixes F2's collapse-to-ties — two docs
sharing only `decade:1990s` are ranked low because the cell is wide.

## Advantages vs current / proposed alternatives

| Approach | Storage | Query cost | Handles "the 1990s" doc? | Handles "Jan 1, 1999" query → "1999" doc? |
|---|---|---|---|---|
| Doc-doc edges (cotemporal) | O(n·M) edges | O(M + n_direct) | Yes but explodes if year has many docs | Yes via shared year edge |
| F2 containing-tag | O(n·log n) — every doc tagged at all levels | O(k) lookup | No — would falsely tag at day/month | Yes |
| **This: lattice + native tag** | O(n) tags | O(levels·k) lookup | **Yes — exactly decade:1990s** | **Yes via lattice walk** |

## Integration with V7 SCORE-BLEND

The lattice retrieval produces a ranked candidate list with `candidate_score`.
This becomes a new **L channel** in the blend:

```
final = 0.3·T + 0.3·S + 0.1·A + 0.1·E + 0.2·L
```

Or we can REPLACE the `axis_tag_Jaccard` component inside the multi-axis
scorer with lattice-tag scoring (cleaner — they do the same job, the lattice
version does it better).

## Evaluation

New query test set:
- **Cross-precision queries** (15-20 queries):
  - Precise-query / broad-doc: "what happened on March 15, 2015?" → doc "things were tough in 2015"
  - Broad-query / precise-doc: "anything from the 90s?" → doc "Jan 1, 1999 …"
  - Same-precision: "March 2020?" → doc "March 2020 was …"

- **Doc-doc sharing** (adversarial S8-like):
  - Doc A: "I met my wife at the 2012 retreat" (year)
  - Doc B: "My wife loves hiking"  (no time)
  - Doc C: "2012 retreat was incredible" (year)
  - Query: "when did I meet my wife?" → must link via shared `year:2012`

- **Cyclical-match queries**:
  - "Thursday events?" → any doc tagged `weekday:Thursday` regardless of absolute year
  - "Afternoon meetings?" → any doc with `hour-of-day:13..17` or `part-of-day:afternoon`

Metrics: R@5/R@10/MRR/NDCG@10 per subset; compare to:
- V7 SCORE-BLEND baseline
- V7 + lattice (L channel)
- Lattice replacing multi-axis tag component

## Deliverables

- `lattice_cells.py` — tag generation at native precision + cyclical
- `lattice_store.py` — inverted index schema + SQLite
- `lattice_retrieval.py` — expansion + scoring
- `lattice_synth.py` — cross-precision test data
- `lattice_eval.py` — orchestration
- `results/lattice.md` + `.json`
