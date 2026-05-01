# Temporal Extraction + Retrieval — Design Spec

## Goal

Extract every temporal reference in natural-language text, resolve each to a structured fuzzy-time representation, and build an index that supports efficient retrieval: given query text at query-time, find documents whose temporal references overlap with the query's temporal references.

**Concrete success criterion** — a document that says "2 weeks from now" (authored at T1) must be retrievable by a query saying "2 weeks ago" issued at T2 = T1 + 4 weeks, because both resolve to the same absolute day.

## Non-goals

- Non-Gregorian calendars.
- Timezones beyond a single configured TZ (assume UTC for synthetic data; TZ is a mechanical extension).
- Adversarial recurrence (Fibonacci gaps, recursive rules).
- Cross-lingual extraction.

## Borrowed standards

- **iCalendar RFC 5545** for recurrence. Use the RRULE string language directly — parse with `python-dateutil.rrule.rrulestr`.
- **TIMEX3 value language** as a mental model for fuzziness (we don't serialize TIMEX3 — we serialize our own schema, but the vocabulary of `PAST_REF / 201X / P1W / R/P1W` informs the types we handle).
- **ISO 8601** for datetimes and durations.
- **Allen's interval algebra** as the mental model for retrieval (we use "intersects" — the `o, oi, d, di, s, si, f, fi, =` relations collapsed to a single overlap predicate).

## Data model

### Primitives

```python
Granularity = Literal[
    "second", "minute", "hour", "day", "week",
    "month", "quarter", "year", "decade", "century",
]

@dataclass
class FuzzyInstant:
    """A moment in time, represented as a bracketed absolute interval.

    For retrieval, `[earliest, latest)` is the match window.
    For ranking, `best` is the point estimate.
    `granularity` drives how we expand vague expressions (e.g., "around 2010"
    expands to a year-wide window centered on 2010-07-01).
    """
    earliest: datetime
    latest:   datetime
    best:     datetime | None
    granularity: Granularity

@dataclass
class FuzzyInterval:
    """A span [start, end) where each endpoint can itself be fuzzy."""
    start: FuzzyInstant
    end:   FuzzyInstant
```

### Expression

```python
@dataclass
class Recurrence:
    rrule:   str                         # iCalendar RRULE, validated via dateutil
    dtstart: FuzzyInstant                # anchor (first occurrence)
    until:   FuzzyInstant | None         # recurrence end, None = open-ended
    exdates: list[FuzzyInstant]          # cancelled instances
    # overrides not modeled in v1 (rare + complex; defer)

@dataclass
class TimeExpression:
    """Top-level extracted temporal reference."""
    kind: Literal["instant", "interval", "duration", "recurrence"]
    surface: str                         # original span from source text
    reference_time: datetime             # "now" at utterance
    confidence: float = 1.0
    # exactly one of these non-None, per kind:
    instant:    FuzzyInstant | None = None
    interval:   FuzzyInterval | None = None
    duration:   timedelta | None = None  # unanchored (for retrieval: skip)
    recurrence: Recurrence | None = None
```

**Design rationale**:
- *Every temporal value gets resolved to absolute wall-clock time at extraction.* Retrieval doesn't need to re-resolve. If we later want to re-resolve (e.g., because the source was ambiguous and we learn more), we keep `surface + reference_time` so we can.
- *Fuzziness is bracketed, not distributional.* Overlap is the retrieval predicate — a distribution buys us nothing we'd use. Granularity captures fuzziness shape.
- *Durations are stored but not indexed.* "Two weeks" without anchor has no absolute location; we keep it for semantic match but skip it in the temporal index.

## Fuzziness table — how to bracket each expression type

| Expression              | earliest              | latest                 | best               | granularity |
|-------------------------|-----------------------|------------------------|--------------------|-------------|
| "March 15, 2026"        | 2026-03-15 00:00      | 2026-03-16 00:00       | 2026-03-15 12:00   | day         |
| "March 2026"            | 2026-03-01 00:00      | 2026-04-01 00:00       | 2026-03-15 00:00   | month       |
| "2026"                  | 2026-01-01 00:00      | 2027-01-01 00:00       | 2026-07-01 00:00   | year        |
| "the 2010s"             | 2010-01-01            | 2020-01-01             | 2015-01-01         | decade      |
| "around 2010"           | 2008-01-01            | 2013-01-01             | 2010-01-01         | year        |
| "the 20th century"      | 1900-01-01            | 2000-01-01             | 1950-01-01         | century     |
| "about 20 years ago"    | ref − 25y             | ref − 15y              | ref − 20y          | decade      |
| "a few years ago"       | ref − 5y              | ref − 2y               | ref − 3y           | year        |
| "last year"             | start-of(ref.year − 1)| start-of(ref.year)     | mid-of(ref.year-1) | year        |
| "yesterday"             | start-of(ref.date−1)  | start-of(ref.date)     | noon(ref.date−1)   | day         |
| "2 weeks ago"           | ref − 21d             | ref − 7d               | ref − 14d          | week        |
| "last week"             | start-of-week(ref−1w) | end-of-week(ref−1w)    | mid-week           | week        |
| "next Thursday"         | next TH 00:00         | next TH 23:59:59       | next TH 12:00      | day         |
| "3pm"                   | today 15:00:00        | today 15:01:00         | today 15:00        | minute      |
| "morning"               | today 06:00           | today 12:00            | today 09:00        | hour        |
| "the first week of May" | May first Mon 00:00   | May first Mon+7d 00:00 | mid-week           | week        |

"About" / "around" / "roughly" inflate the window by +1 level of granularity. "Exactly" or concrete numbers shrink it.

### Correction: counted relative expressions are fuzzy by default

A critical insight — natural-language relative expressions like "2 weeks ago", "3 months ago", "5 years ago" DO NOT mean exactly N units. They're colloquial and carry a fuzziness that scales with N. Treat them as fuzzy unless a precision anchor is present.

**Default rule** — for any expression matching `\bN?\s*(day|week|month|year|decade)s?\s+(ago|from now|later|earlier)\b`:

```
fuzziness_half_width ≈ 0.25 × N × unit           (minimum 0.5 × unit, maximum 50% of N×unit)
earliest = ref ± N×unit − fuzziness_half_width
latest   = ref ± N×unit + fuzziness_half_width
best     = ref ± N×unit
granularity = unit (or larger if N is large)
```

Examples (at `ref_time = 2026-04-23`):

| Surface           | earliest     | latest       | best         | granularity |
|-------------------|--------------|--------------|--------------|-------------|
| "2 days ago"      | 2026-04-20   | 2026-04-22   | 2026-04-21   | day         |
| "3 days ago"      | 2026-04-19   | 2026-04-22   | 2026-04-20   | day         |
| "2 weeks ago"     | 2026-04-02   | 2026-04-16   | 2026-04-09   | week        |
| "a few weeks ago" | 2026-03-19   | 2026-04-16   | 2026-04-02   | week        |
| "2 months ago"    | 2026-02-08   | 2026-03-10   | 2026-02-23   | month       |
| "2 years ago"     | 2023-10-23   | 2024-10-23   | 2024-04-23   | year        |
| "5 years ago"     | 2020-01-01   | 2022-12-31   | 2021-04-23   | year        |
| "20 years ago"    | 2003-04-23   | 2011-04-23   | 2006-04-23   | decade      |
| "a decade ago"    | 2013-04-23   | 2018-04-23   | 2016-04-23   | decade      |

**Precision anchors** — if the surface or its surrounding context includes any of the following, BRACKETS TIGHTEN to ±0.5 unit (or sharper):

1. Explicit precision words: "exactly", "precisely", "N days to the day", "N weeks on the dot"
2. Time-of-day preserved: "2 weeks ago at 3pm", "5 years ago today"
3. Day-of-week preserved: "2 weeks ago Thursday" (weekday match forces exact weekly alignment)
4. Anchor to a known recurring event: "at the meeting 2 weeks ago", "at last month's sync", "at our last Thursday"
5. Time math inside the same conversation: if an earlier message says "March 15" and a later one says "2 weeks later" referencing that same event, they compose exactly

The LLM must judge presence of a precision anchor during Pass 2 and narrow the bracket accordingly.

**Special tight cases**:

- "today", "yesterday", "tomorrow" — tight (±0 day, interval spans the whole day)
- "last week", "this week", "next week" — tight to the calendar week
- "last month", "this month", "next month" — tight to the calendar month
- "last year", "this year", "next year" — tight to the calendar year

These aren't counted-relative — they're named-relative, and natural usage treats them as calendar-unit boundaries, not fuzzy.

**Retrieval implication** — with proper fuzziness, the "2 weeks from now ↔ 2 weeks ago" pair still matches (both windows center on the same day, both are 2-week-wide windows, they overlap strongly). Precision anchors actually make matching EASIER because both ends tighten consistently.

## Recurrence storage — expansion strategy

Recurrences can't be naively interval-indexed — they cover infinitely many discrete moments. Strategy:

1. Store the RRULE + dtstart + until + exdates for each recurrence expression as canonical record.
2. **Expand to instances** within a fixed window around current clock time (default: `[now − 10y, now + 2y]`, configurable). Each instance is indexed as a separate interval row, linked back to its parent `Recurrence` id.
3. On query, match against the pre-expanded instances.
4. **Re-expansion** happens lazily: if the indexer detects a query range outside any recurrence's currently-expanded envelope, it re-expands that recurrence to cover the query range. Ingest-time expansion is the common path; re-expand is the rare path.

**Why not expand everything globally**: a weekly recurrence over 1000 years is 52k rows per recurrence. Bounding to ±10y → 520 rows. Still manageable, matches human memory access patterns.

Each expanded instance is a `FuzzyInstant` with granularity set per the RRULE's finest field (e.g., weekly at 15:00 → granularity=hour; daily, no time → granularity=day). Apply exdate cancellation at expansion time (skip the rows).

## Storage substrate

SQLite with two tables:

```sql
CREATE TABLE expressions (
    expr_id        INTEGER PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    kind           TEXT NOT NULL,          -- instant|interval|duration|recurrence
    surface        TEXT NOT NULL,
    ref_time       INTEGER NOT NULL,       -- unix us
    confidence     REAL NOT NULL,
    rrule          TEXT,                   -- only for kind=recurrence
    dtstart_us     INTEGER,
    until_us       INTEGER,
    duration_us    INTEGER,                -- for kind=duration
    payload        TEXT                    -- JSON dump of the full TimeExpression
);

CREATE TABLE intervals (
    iv_id          INTEGER PRIMARY KEY,
    expr_id        INTEGER NOT NULL REFERENCES expressions(expr_id),
    doc_id         TEXT NOT NULL,
    earliest_us    INTEGER NOT NULL,       -- unix us, inclusive lower
    latest_us      INTEGER NOT NULL,       -- unix us, exclusive upper
    best_us        INTEGER,
    granularity    TEXT NOT NULL,
    is_instance    INTEGER NOT NULL DEFAULT 0   -- 1 if this is an expanded recurrence instance
);

CREATE INDEX idx_iv_earliest ON intervals(earliest_us);
CREATE INDEX idx_iv_latest   ON intervals(latest_us);
CREATE INDEX idx_iv_doc      ON intervals(doc_id);
```

**Why SQLite**: 
- Zero deployment cost for research. 
- Range queries fast enough up to ~1M intervals with B-tree indexes (~ms-scale).
- Can be swapped for Postgres `tstzrange` + GIST later without changing the schema meaningfully.

**Why two indexes (earliest, latest) not a single combined one**: SQLite's range query `earliest <= Q_latest AND latest >= Q_earliest` benefits from either one; for most workloads (short-bounded queries against many longer intervals), the `earliest` index is sufficient.

For higher scale, consider:
- Postgres with `tstzrange + GIST` — single range-aware index, ideal for production.
- Interval tree (Python `intervaltree` lib) — in-memory, O(log n + k) query.
- Qdrant payload filter (range) — when co-located with vector search.

## Retrieval algorithm

Given query text `Q` at `ref_time`:

1. Extract `TimeExpression`s from `Q`.
2. Flatten each to one or more absolute intervals (instants → 1 interval; intervals → 1 interval; recurrences → N instances; durations → skip).
3. For each query interval `[q_e, q_l]`:
   - SQL: `SELECT DISTINCT doc_id FROM intervals WHERE earliest_us < :q_l AND latest_us > :q_e`
   - Each hit gets a score.
4. Merge across query intervals. A doc_id's final score is the sum (or max) of per-interval match scores.

### Scoring a single (query, stored) interval pair

```python
def score(q: Interval, s: Interval) -> float:
    if not overlaps(q, s):
        return 0.0
    # 1) overlap magnitude (Jaccard over intervals)
    overlap = min(q.latest, s.latest) - max(q.earliest, s.earliest)
    union   = max(q.latest, s.latest) - min(q.earliest, s.earliest)
    jaccard = overlap / union
    # 2) best-point proximity (0..1, 1 if bests coincide, 0 at interval edges)
    if q.best and s.best:
        span = max(q.latest - q.earliest, s.latest - s.earliest, timedelta(seconds=1))
        proximity = max(0, 1 - abs(q.best - s.best) / span)
    else:
        proximity = 0.5
    # 3) granularity compatibility (penalize matching a century to a minute)
    gap = abs(GRANULARITY_ORDER[q.granularity] - GRANULARITY_ORDER[s.granularity])
    gran_score = max(0, 1 - gap / 5)
    return 0.5 * jaccard + 0.3 * proximity + 0.2 * gran_score
```

### Hybrid with semantic

Temporal filter → narrowed candidate set → semantic cosine rerank within candidates. For the initial evaluation we compare three conditions:

- **T** — temporal-only retrieval
- **S** — semantic cosine over full text (baseline)
- **T ∧ S** — semantic rerank within top-K temporal

## LLM extractor — two-pass with deterministic post-processing

### Pass 1 — span identification

Model: `gpt-5-mini`, JSON mode, prompt caching on system message.

Input: source text + reference_time (ISO 8601 + weekday).
Output: `list[{surface, kind_guess, context_hint}]`.

**Prompt sketch** (domain-neutral):
```
You identify every temporal reference in a passage.

A temporal reference is any span that refers to a moment, span, duration, or
recurring pattern in time. It can be absolute ("March 5, 2026"), relative
("yesterday", "2 weeks ago"), vague ("around 2010", "a decade ago"), or
recurring ("every Thursday at 3pm").

For each reference, output:
- surface: the exact span from the text, verbatim
- kind_guess: one of [instant, interval, duration, recurrence]
- context_hint: a short (≤12 word) note describing what this refers to

Reference time: {ref_time} ({weekday})
Passage: {text}

Output a single JSON object: {"refs": [...]}
```

### Pass 2 — normalization

For each identified ref, second call with the full schema.

Model: `gpt-5-mini`, structured output via `response_format={"type": "json_schema", ...}`.

Input: the surface + kind_guess + context_hint + reference_time + surrounding sentence(s).
Output: a structured `TimeExpression` object.

**Prompt sketch**:
```
Resolve a single temporal reference to a structured form.

Reference time: {ref_time}  (use this to resolve all relative expressions)
Surrounding context: {sentence}
Reference: "{surface}"
Kind hint: {kind_guess}

Resolve to ABSOLUTE wall-clock time using the reference time. Compute
carefully — check weekday alignment, month length, and year rollovers.

Output JSON matching this schema (omit fields not relevant to the kind):
{
  "kind": "instant" | "interval" | "duration" | "recurrence",
  "surface": string,
  "instant": { "earliest": ISO, "latest": ISO, "best": ISO|null, "granularity": string } | null,
  "interval": { "start": {...instant...}, "end": {...instant...} } | null,
  "duration": { "seconds": int } | null,
  "recurrence": {
    "rrule": string,                   // valid iCalendar RRULE, no trailing semicolon
    "dtstart": {...instant...},
    "until": {...instant...} | null,
    "exdates": [{...instant...}]
  } | null,
  "confidence": float (0..1)
}

Granularity is one of: second, minute, hour, day, week, month, quarter, year, decade, century.

Rules:
- For "about", "around", "roughly", or similar hedges, widen the interval by one granularity level and set best to the centered point estimate.
- For recurrences with no explicit end, set until to null.
- For "every Thursday at 3pm", rrule is "FREQ=WEEKLY;BYDAY=TH;BYHOUR=15;BYMINUTE=0".
- Use UTC ISO 8601 with 'Z' suffix for all datetimes.
```

### Deterministic post-processing

After Pass 2, run each `TimeExpression` through:

1. **ISO validation** — parse every datetime field; reject with a retry if any fail.
2. **Granularity consistency** — check `latest − earliest` matches the stated granularity (±2×). If mismatch, log and keep LLM's bracket (it's usually right for fuzzy-hedge cases the table can't anticipate).
3. **RRULE validation** — `dateutil.rrule.rrulestr("DTSTART:... \n RRULE:" + rrule)`. Reject on exception.
4. **Arithmetic sanity check for relative expressions** — for surfaces matching `\b\d+\s+(day|week|month|year)s?\s+ago\b` or `\bin\s+\d+\s+(day|week|month|year)s?\b`, compute the expected `best` deterministically and warn if LLM's `best` disagrees by more than the granularity window. For an MVP, just log; future: auto-correct.

### Cost model

- 20 docs × avg 3 temporal refs = 60 refs → 60 Pass-2 calls + 20 Pass-1 calls = 80 calls.
- gpt-5-mini ~$0.15/M input, ~$0.60/M output. Pass 1 ≈ 500in / 200out per doc. Pass 2 ≈ 300in / 300out per ref.
- Total ~60k input + 22k output tokens → ~$0.03 for the whole benchmark. Cheap.

## Synthetic data

Generate 30 documents + 60 queries with ground truth.

### Document template categories (aim for balanced coverage)

1. **Absolute date, recent** (5 docs): "On March 15, 2026 I visited..."
2. **Absolute date, distant** (3 docs): "In 1987, grandma..."
3. **Fuzzy decade** (3 docs): "Back in the 90s we used to..."
4. **Relative recent** (5 docs): "Yesterday I..." / "Last week..." / "Two weeks ago..."
5. **Relative distant** (3 docs): "About 20 years ago..." / "A few decades back..."
6. **Explicit interval** (3 docs): "From March 5 to March 12 I was..." / "Our trip lasted from ... to ..."
7. **Recurrence, simple** (3 docs): "I have book club every Thursday..."
8. **Recurrence with start** (2 docs): "Starting in June, we'll meet monthly..."
9. **Recurrence with cancellation** (2 docs): "...every Monday except Jan 15 and Feb 5..."
10. **Multiple times per doc** (5 docs): compound, dense.

Each doc has:
- `text`: natural-language prose
- `ref_time`: utterance timestamp
- `gold_expressions`: manually authored `TimeExpression` ground truth (because we're generating, we know)

### Query templates (aim for overlap-with-some-docs)

1. **Specific-day queries** (10): "What happened on April 1, 2026?"
2. **Relative-day queries** (10): "What did I do yesterday?" / "... 2 weeks ago?" — at varied ref_times so these collide with specific docs' relative times.
3. **Fuzzy-period queries** (10): "What happened in 2015?" / "... in the 90s?" / "... around 1998?"
4. **Recurrence-probe queries** (10): "What's on for next Thursday?" / "Do I have anything this month?"
5. **Interval-probe queries** (5): "What happened during the first week of May 2026?"
6. **No-time queries** (5): semantic-only probes (to verify temporal extractor correctly returns empty list and retrieval falls back gracefully)
7. **"2 weeks ago" matching "2 weeks from now"** — critical test (5): pair a doc saying "N days from now" at T1 with a query saying "N days ago" at T1+2N.

### Ground truth

For each query, compute the oracle match set by:
1. Taking each doc's gold temporal intervals.
2. Taking the query's gold temporal intervals.
3. A doc is relevant if ANY doc-interval overlaps ANY query-interval with a Jaccard ≥ 0.05 OR both bests are within the union granularity.

### Metrics

- **Extraction F1** — predicted expressions vs gold, matched by surface span overlap.
- **Resolution MAE** — for matched expressions, `|predicted.best − gold.best|` in seconds, summarized as median and P95.
- **Retrieval recall@5, recall@10, MRR, NDCG@10** — T vs S vs T ∧ S.
- **Critical-case accuracy** — the "2 weeks ago ↔ 2 weeks from now" pairs retrieved at top-1? yes/no.

## Implementation layout

```
evaluation/temporal_extraction/
├── DESIGN.md                  # this file
├── schema.py                  # dataclasses + JSON (de)serializers
├── extractor.py               # two-pass LLM with gpt-5-mini
├── resolver.py                # deterministic post-processing + RRULE validation
├── expander.py                # recurrence → instances expansion
├── store.py                   # SQLite interval store + range queries
├── scorer.py                  # per-interval scoring + per-doc aggregation
├── synth_data.py              # generate docs + queries + gold
├── eval.py                    # run extraction, index, query, compute metrics
├── baselines.py               # semantic-only (text-embedding-3-small cosine)
├── results/
│   ├── extraction_quality.json
│   ├── retrieval_results.json
│   └── REPORT.md              # human-readable summary
└── data/
    ├── docs.jsonl
    ├── queries.jsonl
    └── gold.jsonl
```

## Implementation notes

- Use `openai.AsyncOpenAI`. Model: `gpt-5-mini`. Embedder: `text-embedding-3-small` (for the semantic baseline).
- `uv run` for all execution. No new deps beyond `python-dateutil`, `openai`, `sqlite3` (stdlib). Don't add `icalendar` unless needed — `dateutil.rrule` handles parsing.
- Concurrency: `asyncio.Semaphore(10)` for LLM calls.
- All datetimes stored as **microseconds since epoch** (int64) in SQLite for portability. Convert at the boundary.
- For "now" in the synthetic data, use a fixed reference `2026-04-23T12:00:00Z` by default, but vary it across docs so relative expressions resolve to different absolute times.

## Out of scope (future)

- Timezones and DST
- Allen-algebra qualitative retrieval ("before", "during") as separate operators
- Duration-anchored retrieval (match a 3-hour window against a 3-hour duration)
- Uncertainty fusion — if one doc is confident and another is fuzzy, ranking should prefer the confident one
- Adversarial parsing ("every other Tuesday except when the Tuesday is in a month that contains a national holiday")
- Multi-language

## Decision summary

- **Schema**: unified `TimeExpression` with 4 kinds; fuzzy primitives bracketed + best-point.
- **Recurrence**: iCalendar RRULE, expand to instances within a ±window, index instances.
- **Storage**: SQLite with two B-tree indexed interval tables; swappable for Postgres tstzrange later.
- **Retrieval**: range overlap → Jaccard + proximity + granularity-compat composite score.
- **Extraction**: two-pass gpt-5-mini + deterministic validation.
- **Eval**: 30 docs / 60 queries synthetic; extraction F1 + retrieval recall@K vs semantic baseline.
