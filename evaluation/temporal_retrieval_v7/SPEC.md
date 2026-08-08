# Temporal Retrieval V7 — Specification

**Status:** Design document. No implementation yet.
**Sibling of:** `temporal_retrieval/` (current research), `temporal_retrieval_min/` (production shipped).
**Goal:** Principled rewrite of the temporal layer with cleaner semantics, smaller surface area, and correct handling of compound queries / engagement-relevance / open-ended ranges.

---

## 1. Motivation

The current temporal layer (`temporal_retrieval_min` shipped, `temporal_retrieval` research) has accumulated three architectural debts:

1. **Same-anchor-binding bug on compound queries.** `evaluate_dnf_match` aggregates per-leaf factors via `min` (AND) / `max` (OR). For multi-leaf AND clauses, different doc anchors can "satisfy" different leaves separately, producing false positives. E.g., query "not in 2020 or 2022" + doc with anchors `[March 2020, May 2022]` (neither outside both) → naive per-leaf existential says "passes" (separate anchors satisfy separate leaves), but no single anchor binds both constraints. V1 strict accidentally avoids this via its overzealous filter; softer variants (V2/V2.7/V5) inherit the bug.

2. **Doc-side relation asymmetry.** Doc extraction produces only calendar intervals — no relation. A doc saying "Since the v3 launch in March 2024, the team has been busy" gets only `[March 2024]` as its envelope. Query "after the v3 launch" → range `(March 2024, +∞)`. They barely touch, don't overlap → doc fails to match a query its content is exactly relevant to.

3. **Filter and scoring duplicate the constraint logic.** `doc_passes_filter` does per-leaf strict EXISTS checks; `evaluate_dnf_match` does per-leaf factor aggregation. They use different rules for the same constraints, can disagree at boundaries (engagement docs leak through filter via `build_pool` top-up but score 0). The duplication is also where V1 strict's "accidental correctness" hides.

V7 fixes all three by collapsing to a single primitive: **TimeRange** (a list of disjoint half-open intervals, possibly with ±∞ endpoints). Both the doc side and the query side speak the same language. Retrieval is one overlap-quality computation.

## 2. Core Design Principles

1. **TimeRange is the canonical data type.** Both query and doc produce a list of TimeRefs (each ref a TimeRange).
2. **No relations on either side.** Relations (`intersect/after/before/disjoint`) absorb into TimeRange shapes (closed, half-open with ±∞, complement).
3. **Each TimeRef = one anchor's binding.** Query side: "what one anchor must satisfy". Doc side: "where one mention anchors".
4. **Scoring: per-query-ref max over doc-refs, sum across query-refs, normalize.** Final score in [0, 1].
5. **Filter is derived.** `temporal_pass = score > 0`. No separate filter function.
6. **Recurrence is NOT first-class.** Bounded recurrences enumerate to explicit intervals at extraction; unbounded ones are skipped. See §3 for the rationale (over-extension risk + no update mechanism).

## 3. Data Structures

### Interval

```python
@dataclass
class Interval:
    """Half-open temporal interval [earliest_us, latest_us)."""
    earliest_us: int
    latest_us: int
```

Sentinel for ±∞: `NEG_INF = -(2**62)`, `POS_INF = (2**62)`. Sentinel chosen so that any plausible date measure (microseconds since epoch is ~2^54 in 2024) stays well below it, making sentinel arithmetic numerically robust.

### TimeRange

A **TimeRange** is the canonical representation of "a set of moments in time" used by both queries and docs. Conceptually it's a subset of R̄ (extended real line for microsecond timestamps, including ±∞).

```python
@dataclass
class TimeRange:
    """A set of moments in time, represented as a canonical list of
    disjoint half-open intervals (possibly with ±∞ endpoints)."""
    intervals: list[Interval] = field(default_factory=list)
```

Canonical form invariants:
- Sorted by `earliest_us` ascending
- No two intervals overlap (overlapping ones are merged)
- No two intervals are adjacent at a boundary (`a.hi == b.lo` becomes one interval `[a.lo, b.hi)`)
- All intervals have `earliest_us < latest_us` (no empty intervals)

That's the whole data structure. No lazy recurrence patterns, no separate materialization step. The TimeRange is always an explicit interval list.

### On recurrence (intentionally not first-class)

Recurrences like "every Tuesday" or "every March" are NOT a first-class TimeRange feature in V7. Reasons:

1. **Over-extension risk.** Without explicit bounds, "I walk my dog every Tuesday" naively becomes a recurrence with no `end` — claiming the user has been walking the dog every Tuesday from -∞ to +∞. This over-matches arbitrary queries far outside the doc's actual scope.

2. **No update mechanism.** Recurrence patterns encode claims about open-ended past or future. If the user stops the activity or the schedule changes, there's no way to revoke an extracted recurrence without re-extracting from a later doc — and we'd need conflict-resolution logic.

3. **Bounded cases work via enumeration.** "Every March 2020-2024" (explicitly bounded) → extractor enumerates 5 explicit interval instances at construction time and emits a TimeRange with those 5 intervals. No lazy pattern needed.

4. **Unbounded cases get skipped.** "Every Tuesday" with no bounds → extractor does NOT emit a temporal anchor. The recurrence-pattern tokens ("every Tuesday", "weekly") are still in the doc text, so the semantic layer (cosine, reranker) can still surface the doc for queries semantically about repeated activity. The temporal layer just doesn't make a calendrical claim about specific dates.

**Extraction rules for recurrence surface forms:**

| Surface | Extracted as |
|---|---|
| "every March 2020-2024" (bounded) | 5 explicit intervals in one TimeRange |
| "every Monday in 2024" (bounded) | ~52 explicit intervals |
| "annual reviews from 2020 until 2024" (bounded) | 5 intervals |
| "since 2020 I've done annual reviews" (partially bounded, end implicit) | Enumerate from 2020 to ref_time → bounded N intervals |
| "every Tuesday" (unbounded) | **Skip — emit no temporal anchor** |
| "we meet on Fridays" (unbounded) | **Skip — emit no temporal anchor** |
| "annually" (unbounded) | **Skip — emit no temporal anchor** |

**Safeguard**: cap explicit-interval enumeration at MAX_INSTANCES_PER_RANGE (e.g., 100). "Every minute in 2024" → exceeds cap → either skip OR coarsen to one big interval `[2024]`. Pathological cases skipped entirely.

If a future bench shows recurrence-heavy queries underperforming, recurrence-as-first-class can be added back as an extension. For V7 initial scope, the bounded-enumerate / unbounded-skip rule covers the use cases that matter.

### Query / Doc

```python
@dataclass
class Query:
    text: str
    ref_time: datetime
    time_refs: list[TimeRange]     # one TimeRange per query template

@dataclass
class Doc:
    id: str
    text: str
    ref_time: datetime
    time_refs: list[TimeRange]     # one TimeRange per extracted temporal mention
```

**Terminology**: a "time ref" is one element of the `time_refs` list — i.e., one TimeRange used in the role of "a single temporal binding template" (query side) or "a single extracted temporal mention" (doc side). The data type is `TimeRange`; "time ref" is the role-flavored name when discussing the per-element semantics. Throughout this spec the data type is always called `TimeRange`; "time ref" refers to one element in a `time_refs: list[TimeRange]`.

## 4. Operations on TimeRange

A TimeRange supports a closed algebra of set operations on explicit interval lists. All ops produce a TimeRange in canonical form. Pure integer arithmetic on the underlying microsecond sentinels — no calendar logic at this layer, no lazy types.

### 4.1 Core set operations

```python
def union(A: TimeRange, B: TimeRange) -> TimeRange:
    """A ∪ B. Merge sorted, collapse overlapping/adjacent.
    O(|A.intervals| + |B.intervals|)."""

def intersect(A: TimeRange, B: TimeRange) -> TimeRange:
    """A ∩ B. Linear two-pointer sweep over sorted disjoint intervals.
    O(|A.intervals| + |B.intervals|)."""

def complement(A: TimeRange) -> TimeRange:
    """¬A on (-∞, +∞). Half-open: complement of [lo, hi) is 
    (-∞, lo) ∪ [hi, +∞). Walks A's intervals and emits gaps.
    O(|A.intervals|).
    
    Produces unbounded intervals when A has finite extent — this is
    natural and unavoidable for 'disjoint X' / 'outside X' queries."""

def difference(A: TimeRange, B: TimeRange) -> TimeRange:
    """A \\ B = A ∩ complement(B). Derived."""

def symmetric_difference(A: TimeRange, B: TimeRange) -> TimeRange:
    """A △ B = (A ∪ B) \\ (A ∩ B). Derived."""

def measure(A: TimeRange) -> int:
    """Sum of (hi - lo) over A's intervals. Returns an int that may 
    reach the sentinel-large range when intervals touch ±∞. Callers
    check `m >= SENTINEL_THRESHOLD` to detect saturation."""

SENTINEL_THRESHOLD = 2**60  # any real measure stays below this
```

### 4.2 Combining intervals (no recurrence cases needed)

All combinations reduce to standard set ops on explicit interval lists:

| Construction | Result |
|---|---|
| `union([Mar 2020], [Mar 2024])` | TimeRange `[Mar 2020, Mar 2024]` (2 intervals) |
| `intersect([2024], complement([summer 2024]))` | `[Jan-May 2024) ∪ [Sep-Dec 2024 + Jan 1 2025)` (2 intervals) |
| `difference([2024], [Mar 2024])` | `[Jan-Feb 2024) ∪ [Apr-Dec 2024 + Jan 1 2025)` |
| `complement([summer 2024])` | `(-∞, Jun 1 2024) ∪ [Sep 1 2024, +∞)` |

The extractor enumerates bounded recurrences ("every March 2020-2024") at construction time into explicit intervals before these ops see them. Unbounded recurrences are skipped at extraction.

### 4.3 Unbounded intervals — when and how to use

Unbounded intervals (with `±INF` sentinel endpoints) are FIRST-CLASS in V7. They're necessary in two cases:

1. **Query-side directional patterns**: "after 2024" → `[Jan 1 2025, +∞)`. "before 2020" → `(-∞, Jan 1 2020)`. These map naturally to half-bounded intervals and can't be cleanly expressed otherwise — picking an arbitrary upper/lower bound (e.g., +50 years) is just a hidden over-extension.

2. **Complement results**: `complement([2024])` produces `(-∞, Jan 1 2024) ∪ [Jan 1 2025, +∞)`. The disjoint relation's complement-based interpretation REQUIRES unbounded intervals to be well-defined.

The set-algebra ops handle them uniformly through sentinel arithmetic. The pair_overlap scoring handles them through the empty-gate + both-infinite shortcut (see §5.1).

### 4.4 Unbounded-interval pitfalls — when to bound

Like recurrence, unbounded intervals on the DOC side carry over-extension risk. A doc "Since 2020 I've been remote" extracted as `[2020, +∞)` would match queries about "in 2050" with full credit — but the doc makes no claim about 2050.

**Extractor convention for doc-side directional surface forms:**

| Doc surface | Extract as |
|---|---|
| "Since X" / "After X" / "From X onwards" (open-right) | **`[X, ref_time)`** — bound by when the doc was written. The doc can only assert what was true through its own ref_time. |
| "Until X" / "Before X" (open-left at the *end* of a backward span) | **`[X − implicit_lookback, X)`** if context allows; otherwise `(-∞, X)`. Practical default: use `(-∞, X)` since recovering the implicit start is hard. Accept some over-extension on the left. |
| "Throughout X" / "All of X" | bounded `[X.start, X.end)` |
| Anaphoric event ("the launch") | resolved interval via corpus anchor; bounded |

**Query-side directional surface forms** stay unbounded as intended:
- "after X" → `[X.end, +∞)`
- "before X" → `(-∞, X.start)`
- "outside X" → `complement([X])`

This asymmetry matches the asymmetry of intent: queries express "I don't care about the upper/lower bound", which is a CLAIM about user intent. Docs express "this is what the content says", which has a real authorship time and shouldn't make claims past it.

### 4.5 Computational rationale

For retrieval we never need to MANIPULATE recurrences or calendar-aware periods symbolically — we need to compute OVERLAP on intervals. Keeping the runtime ops on integer microseconds means:

- All set ops are O(n) integer-arithmetic sweeps
- Sentinel ±∞ is trivial (compare-and-min against `2^62`)
- No datetime / dateutil dependency in the scoring hot path
- The result of any op is just another canonical interval list

Calendar-aware resolution (e.g., recognizing "March 2024" as `[Mar 1, Apr 1)`) lives in the EXTRACTOR (which already does LLM-based natural language reasoning). The extractor converts datetime to microsecond ints once at the boundary; the rest of the pipeline never sees calendar logic.

## 5. Scoring

### 5.1 Pair Overlap (TimeRange × TimeRange → [0, 1])

```python
def pair_overlap(A: TimeRange, B: TimeRange) -> float:
    """Frac_min on TimeRanges. Symmetric. Returns value in [0, 1]."""
    
    inter = intersect(A, B)
    if not inter.intervals:
        return 0.0                          # HARD GATE: no overlap → 0
    
    a_w = measure(A)
    b_w = measure(B)
    inter_w = measure(inter)
    
    # Both-infinite shortcut (only after empty-check passes):
    # any non-empty real overlap counts as full match.
    # Materialized form preserves the ±∞ sentinels for endpoints that
    # extend beyond the window — so the measure reflects unboundedness.
    a_inf = a_w >= SENTINEL_THRESHOLD
    b_inf = b_w >= SENTINEL_THRESHOLD
    if a_inf and b_inf:
        return 1.0
    
    # Standard frac_min: if exactly one side is infinite, min picks the
    # finite one. Both finite: regular frac_min.
    denom = min(a_w, b_w)
    if denom <= 0:
        return 0.0
    return min(1.0, inter_w / denom)
```

**Note:** no recurrence materialization step is needed at scoring time. Bounded recurrences are enumerated to explicit intervals at extraction/construction; unbounded recurrences are skipped at extraction. By the time pair_overlap sees a TimeRange, it's always a plain list of explicit half-open intervals.

### 5.2 Final Score (Query × Doc → [0, 1])

```python
def final_score(query: Query, doc: Doc) -> float:
    """Per-query-ref max over doc-refs, summed across query-refs,
    normalized to [0, 1]."""
    
    if not query.time_refs:
        return 1.0                          # no temporal constraint
    if not doc.time_refs or _is_universal(doc):
        return 1.0                          # timeless doc matches everything
    
    n = len(query.time_refs)
    total = 0.0
    for qref in query.time_refs:
        best = 0.0
        for dref in doc.time_refs:
            f = pair_overlap(qref, dref)
            if f > best:
                best = f
        total += best
    return total / n                        # ∈ [0, 1]
```

### 5.3 Filter

```python
def temporal_pass(query: Query, doc: Doc) -> bool:
    return final_score(query, doc) > 0.0
```

No separate filter primitive. The score IS the filter.

### 5.4 Combination with cosine / rerank / recency

The temporal score `∈ [0, 1]` composes additively with other ranking signals:

```python
combined = base_norm + final_score(query, doc) + recency_norm
# all components in [0, 1]; combined in [0, 3] (or smaller with weighting)
```

Coefficients can be tuned. The default equal-weight composition matches what the current production retriever does.

## 6. Planner

### 6.1 Output

Planner output: a list of TimeRanges. Each TimeRange corresponds to one binding template the user wants the doc to satisfy.

The internal AST may use DNF shape (list of clauses, each AND-of-leaves) — this is a convenient generation form for the LLM and matches the current `expr: list[list[Constraint]]`. The CONVERTER evaluates the AST to a list of TimeRanges:

```python
def planner_ast_to_refs(plan: PlanAST) -> list[TimeRange]:
    """Convert each AND-clause to a single TimeRange via range composition.
    AND-of-leaves → intersect leaf ranges.
    OR-of-clauses → list[TimeRange] (one ref per clause).
    Leaf ranges: intersect→[T], disjoint→complement([T]), 
                 after→(T.end, +∞), before→(-∞, T.start)."""
    refs = []
    for clause in plan.expr:
        leaf_ranges = [leaf_to_range(leaf) for leaf in clause]
        clause_range = intersect_all(leaf_ranges)
        if not is_empty(clause_range):  # skip empty (impossible) clauses
            refs.append(clause_range)
    return refs
```

### 6.2 Compatibility detection rule

Planner prompt addition (one rule):

> "Multi-leaf clauses describe what ONE temporal reference must jointly satisfy. If conjuncts have date ranges that can't possibly overlap (e.g., `[2020]` and `[2024]` are incompatible — no time is in both), put each leaf in its own clause instead. Use multiple clauses for surface-form 'in X and Y' when X and Y are disjoint dates."

The planner detects incompatibility at the LLM level. As a safety net, the AST-to-refs converter checks: if `intersect_all(leaf_ranges)` is empty, the converter splits the clause into separate refs (one per leaf).

### 6.3 Surface-to-relation table

| Surface | Planner emits (one clause) | TimeRange |
|---|---|---|
| "in March 2024" | `intersect("March 2024")` | `[Mar 1, Apr 1 2024)` |
| "after 2024" | `after("2024")` | `[Jan 1 2025, +∞)` |
| "before 2020" | `before("2020")` | `(-∞, Jan 1 2020)` |
| "outside summer 2024" | `disjoint("summer 2024")` | `(-∞, Jun 1 2024) ∪ [Sep 1 2024, +∞)` |
| "in 2024 not in summer" | `intersect("2024"), disjoint("summer 2024")` | `[Jan-May 2024) ∪ [Sep-Dec 2024 + Jan 1 2025)` (one clause, composed range, one ref) |
| "in 2020 and 2024" | DETECTED INCOMPATIBLE → two clauses | `[[2020]], [[2024]]` (two refs) |
| "not in 2020 or 2022" | `disjoint("2020"), disjoint("2022")` | one clause with intersected complements (one ref) |
| "in Q1 or Q4 of 2023" | two clauses | two refs |

### 6.4 Surface-form negation rule (added to planner prompt)

> "When the query has 'did not' or 'didn't' before a verb (event-polarity negation), IGNORE it for temporal classification — emit the same relation as if the verb were affirmative. Only flip the relation when 'not' / 'outside' / 'excluding' attaches DIRECTLY to a temporal preposition (e.g., 'not in', 'outside summer'). Examples: 'what did NOT happen in 2024' → intersect(2024). 'what happened outside 2024' → disjoint(2024)."

## 7. Extractor

### 7.1 Output

Doc extraction returns `list[TimeRange]`. Each TimeRange corresponds to one extracted temporal mention.

The extractor extends the current envelope-based output: instead of returning `list[TimeEnvelope]` (each a single closed interval), returns TimeRanges that may be:
- Single-interval closed (most common): "in March 2024" → `[Mar 1, Apr 1 2024)`
- Open-ended one side: "since March 2024" / "after the launch" → `[Mar 1 2024, +∞)`
- Open-ended other side: "before 2020" → `(-∞, Jan 1 2020)`
- Bounded recurrence: "every March 2020-2024" → enumerated explicit intervals at construction time (5 Marches). Unbounded ("every March", no bounds) → skipped, no anchor emitted.

### 7.2 Extractor prompt additions (sketch)

Extend the current v3.3 extractor prompt to:
- Recognize directional surface forms ("since X", "after Y", "until Z", "before W") and emit open-ended bounds
- Recognize recurrence patterns ("every March", "every Monday", "annually"): if bounded, enumerate to explicit intervals; if unbounded, skip emission (no temporal anchor)
- Continue to skip purely topical mentions ("lessons from X", "aftermath of Y")
- For event-negation ("X did NOT happen on D"), emit `[D]` as a normal anchor — let the semantic layer handle the polarity

### 7.3 What the extractor does NOT do

- Does NOT classify mentions as "anchor" vs "reference" (every named interval is just an interval)
- Does NOT carry polarity flag (`AFFIRMS`/`DENIES`) — irrelevant to retrieval, handled by semantic layer
- Does NOT carry granularity hint (was descriptive in v3.3; not used in scoring)

## 8. Considered Cases

### 8.1 Simple intersect

Query "in March 2024" → 1 ref `[Mar 1, Apr 1 2024)`.
Doc "I went hiking in March 2024" → 1 ref `[Mar 1, Apr 1 2024)`.
- pair_overlap = frac_min(identical) = 1.0. final_score = 1.0. ✓

### 8.2 Doc anchor narrower than query window

Query "in 2024" → `[Jan 1 2024, Jan 1 2025)`.
Doc "March 2024" → `[Mar 1, Apr 1 2024)`.
- inter = March 2024. min(|D|=1mo, |Q|=12mo) = 1mo. frac_min = 1.0. ✓
- (Doc is fully inside query window; full credit.)

### 8.3 Open-ended doc + query crossing the boundary

Query "before 2030" → `(-∞, Jan 1 2030)`.
Doc "Since March 2024" → `[Mar 1 2024, +∞)`.
- inter = `[Mar 1 2024, Jan 1 2030)` ≈ 5.8 years (finite).
- |Q| = sentinel-large. |D| = sentinel-large. Both infinite shortcut → 1.0. ✓

Query "before 2020" → `(-∞, Jan 1 2020)`.
Doc "Since March 2024" → `[Mar 1 2024, +∞)`.
- inter = empty (March 2024 ≥ 2020). final_score = 0. ✓

### 8.4 Compound AND intersect + disjoint (same-anchor binding)

Query "in 2024 not in summer" → 1 ref `[Jan-May 2024) ∪ [Sep-Dec 2024 + Jan 1 2025)`.
Doc "March 2024" → 1 ref `[Mar 1, Apr 1 2024)`.
- inter = March 2024 (March ∈ Jan-May). frac_min = 1.0. ✓

Doc "July 2024, January 2025" → 2 refs `[Jul 1, Aug 1)`, `[Jan 1, Feb 1 2025)`.
- qref vs dref1=July: July ∉ Jan-May, July ∉ Sep-Dec. inter empty. 0.
- qref vs dref2=Jan 2025: Jan 2025 ∉ Jan-May 2024, Jan 1 2025 is the half-open exclusive boundary of Sep-Dec 2024 (i.e., dec range is `[Sep 1 2024, Jan 1 2025)`, so Jan 1 2025 itself is OUT). inter empty. 0.
- max = 0. final_score = 0. **Filtered.** ✓ (V5 false-positive correctly killed by V7.)

### 8.5 AND of multiple disjoints

Query "not in 2020 or 2022" → 1 ref (single clause with two disjoint leaves):
- Composed range = complement([2020]) ∩ complement([2022])
                 = complement([2020] ∪ [2022])
                 = `(-∞, Jan 1 2020) ∪ [Jan 1 2021, Jan 1 2022) ∪ [Jan 1 2023, +∞)`

Doc `[March 2020, May 2022]` (two refs, both inside excluded years):
- qref vs Mar 2020: inter empty. 0.
- qref vs May 2022: inter empty. 0.
- max = 0. final_score = 0. ✓

Doc `[October 2023]`:
- qref vs Oct 2023: Oct 2023 ∈ `[Jan 1 2023, +∞)`. frac_min ≈ 1.0 (doc fully inside query's third interval). final_score = 1.0. ✓

Doc `[March 2020, October 2023]`:
- qref vs Mar 2020: 0.
- qref vs Oct 2023: 1.0.
- max = 1.0. final_score = 1.0. ✓

### 8.6 Colloquial-and-as-or

Query "in 2020 and 2024" → planner detects [2020] ∩ [2024] = ∅, emits TWO refs: `[2020]`, `[2024]`.

Doc `[March 2020]`:
- qref1=[2020] vs Mar 2020 → 1.0. qref2=[2024] vs Mar 2020 → 0.
- per-query-ref max: qref1=1.0, qref2=0. Sum=1.0. Normalize / 2 = 0.5.

Doc `[June 2024]`:
- qref1=0, qref2=1.0. Sum=1.0. Normalize / 2 = 0.5.

Doc `[March 2020, June 2024]`:
- qref1=1.0 (best dref is Mar 2020), qref2=1.0 (best dref is Jun 2024). Sum=2.0. Normalize / 2 = **1.0**. ✓
- Doc matching BOTH ranks higher than docs matching one.

### 8.7 OR clauses

Query "in Q1 or Q4 of 2023" → planner emits two clauses, two refs: `[Q1 2023]`, `[Q4 2023]`.

Doc `[February 2023]`:
- qref1=[Q1 2023] vs Feb 2023 → frac_min: inter=Feb (1mo), min(month, 3 months)=1mo → 1.0. qref2=0. Sum=1.0/2=0.5.

Doc `[Q1 2023]` (3 months):
- qref1=Q1 vs Q1: inter=Q1, min=3 months, frac_min=1.0. qref2=0. Sum=1.0/2=0.5.

Doc `[Feb 2023, Nov 2023]`:
- qref1: Feb in Q1 → 1.0. qref2: Nov in Q4 → 1.0. Sum=2.0/2=**1.0**. ✓

### 8.8 Bounded recurrence (eager enumeration)

Query "every March 2020-2024" → extractor/planner recognizes the explicit bounds (2020-2024) and enumerates → 1 TimeRange with 5 explicit intervals `[Mar 1 2020, Apr 1 2020), [Mar 1 2021, Apr 1 2021), ..., [Mar 1 2024, Apr 1 2024)`.

Doc "March 2022" → 1 TimeRange `[Mar 1 2022, Apr 1 2022)`.
- intersect = `[Mar 1 2022, Apr 1 2022)`. measure = 1 month.
- min(|Q|=5mo, |D|=1mo) = 1mo. frac_min = 1.0. ✓

Doc "2022" (full year):
- intersect = `[Mar 1 2022, Apr 1 2022)`. 1 month.
- min(|Q|=5mo, |D|=12mo) = 5mo. frac_min = 1/5 = 0.2.
- Year-doc partially fits; only March 2022 is in the query's set.

### 8.9 Bounded recurrence × interval (intersection at construction)

Query "every Monday in March 2024" → extractor enumerates Mondays bounded by March 2024 → 1 TimeRange with 4 explicit intervals (the 4 March Mondays).

Doc "March 18, 2024 meeting" → 1 TimeRange `[Mar 18 2024, Mar 19 2024)`.
- intersect = `[Mar 18]`. min(|Q|=4d, |D|=1d) = 1d. frac_min = 1.0. ✓

### 8.10 Bounded recurrence × interval (difference at construction)

Query "every March 2020-2024 except 2022" → extractor enumerates Marches, removes March 2022 → 1 TimeRange with 4 explicit intervals: `[Mar 2020, Mar 2021, Mar 2023, Mar 2024]`.

Doc "March 2023": intersect = `[Mar 2023]`. frac_min = 1.0. ✓
Doc "March 2022": intersect = ∅. Score = 0. ✓

### 8.11 Unbounded recurrence (skipped at extraction)

Doc text "I walk my dog every Tuesday" → **extractor emits NO temporal anchor** for the recurrence (it's unbounded; over-extension risk).

The doc's `time_refs` for this sentence is empty (unless there are other dated mentions in the same sentence). The doc text still contains the tokens "every Tuesday", so the semantic layer (cosine, reranker) can surface it for queries semantically about repeated weekly activity. But the temporal layer makes no calendrical claim about Tuesdays.

Query "What did I do on Tuesday March 12, 2024?" → 1 TimeRange `[Mar 12 2024, Mar 13 2024)`.
- doc.time_refs = ∅ (no anchor extracted). final_score special case: empty doc → 1.0 (timeless / no temporal anchor, doesn't get filtered out — semantic ranking takes over).
- This is the right behavior: the dog-walking doc surfaces for semantic similarity, but doesn't claim to be specifically about March 12.

### 8.12 Engagement (doc mentions excluded period in contrast)

Query "outside summer 2024" → 1 ref complement([summer 2024]) = `(-∞, Jun 1 2024) ∪ [Sep 1 2024, +∞)`.

Doc "Unlike summer 2024, in October 2024 I focused on writing" → 2 refs:
- dref1 = `[Jun 1, Sep 1 2024)` (summer 2024)
- dref2 = `[Oct 1, Nov 1 2024)` (October)

- qref vs dref1: inter = summer ∩ complement(summer) = ∅. 0.
- qref vs dref2: inter = Oct. min(|D|=1mo, |Q|=∞) = 1mo. frac_min = 1.0.
- max = 1.0. final_score = 1.0. ✓ (Engagement gold correctly surfaces.)

### 8.13 Retrospective (doc written inside excluded, content outside)

Query "outside Q2 2024" → 1 ref complement([Q2 2024]).
Doc with ref_time=2024-05-15 (inside Q2), content = "Reflecting on Q3 2023 trip + December 2023 reunion" → 2 refs `[Q3 2023]`, `[Dec 2023]`. (Note: doc.ref_time is metadata, not a temporal ref.)
- qref vs Q3 2023: Q3 2023 ∈ complement(Q2 2024) → 1.0.
- qref vs Dec 2023: similarly → 1.0.
- max = 1.0. final_score = 1.0. ✓ (Retrospective doc correctly surfaces because content-anchors, not ref_time, drive scoring.)

### 8.14 Event-negation

Query "what did NOT happen on May 3, 2024?" → 1 ref `[May 3 2024]` (the planner ignores "did NOT happen" verb-polarity and emits intersect for May 3).
Doc "The product launch did NOT happen on May 3, 2024" → 1 ref `[May 3 2024]`.
- pair_overlap = 1.0. final_score = 1.0.

Other docs:
- "The launch happened on May 3" → also 1 ref `[May 3]` → also 1.0.
- "Standup on May 3" → also 1 ref `[May 3]` → also 1.0.

**The temporal layer cannot distinguish event-affirmation from event-negation.** All three docs score equally on temporal. The semantic layer (cosine reads "did NOT"; reranker reads it more strongly) must disambiguate. This is correct — polarity is semantic, not calendrical.

### 8.15 Universal / empty / timeless

Doc with no extracted refs (timeless): `doc.time_refs = []`. final_score returns 1.0 (special case at top of function). Matches every query.

Doc with universal range (-∞, +∞) as its only ref: same special case → 1.0.

Query with no refs (no temporal constraint): same special case → 1.0 for every doc.

### 8.16 Both-end unbounded with finite gap

A = `(-∞, -1) ∪ [1, +∞)` (a TimeRange with infinite measure but a finite gap).
B = `[-1, 1)` (a single bounded interval inside A's gap).

- intersect via two-pointer sweep:
  - `(-∞, -1)` vs `[-1, 1)`: max=-1, min=-1, -1<-1 false → no overlap.
  - `[1, +∞)` vs `[-1, 1)`: max=1, min=1, 1<1 false → no overlap.
- inter = empty. **HARD GATE: empty → 0.** ✓

The empty-check fires FIRST. The both-infinite shortcut never reaches (which would have incorrectly returned 1.0).

### 8.17 Opposite-end unbounded with finite overlap

A = `(-∞, 2030)`, B = `(2020, +∞)`. A is left-open, B is right-open.

- inter = `(2020, 2030)` = 10 years finite.
- Both have infinite measure (their endpoints reach ±∞).
- Both-infinite shortcut after empty-check passes → 1.0. ✓

This is the case the user explicitly flagged. The shortcut applies AFTER the empty-check; here the inter is non-empty, so the shortcut fires.

A = `(-∞, 2020)`, B = `(2024, +∞)` (gap between 2020 and 2024):
- inter = empty (2024 ≥ 2020 means no overlap).
- HARD GATE → 0. ✓

### 8.18 Both-end unbounded with multi-piece TimeRange × bounded

A = `(-∞, -1) ∪ [1, +∞)` (∞ measure, multi-piece).
B = `[-2, 2)` (bounded, measure 4, partially overlaps A's both halves).
- inter = `[-2, -1) ∪ [1, 2)` (two slivers, total measure 2).
- a_w = sentinel-large; b_w = 4. Only B is finite → frac_min branch.
- min = 4. frac_min = 2 / 4 = 0.5. ✓ (Half of B fits within A.)

## 9. Filter & Scoring Semantics — Complete Reference

### 9.1 Pair-overlap matrix by unboundedness shape

| A shape | B shape | inter shape | a_w | b_w | min | path | result |
|---|---|---|---|---|---|---|---|
| bounded | bounded | bounded (or empty) | finite | finite | finite | standard frac_min | inter/min ∈ [0, 1] |
| bounded | one-sided unbounded | bounded (or empty) | finite | sentinel | finite | standard frac_min | inter/finite ∈ [0, 1] |
| both-sided unbounded (=universal) | bounded | bounded | sentinel | finite | finite | standard frac_min | inter/finite ∈ [0, 1] (typically 1.0 since B ⊆ A) |
| one-sided unbounded × one-sided unbounded (same end) | unbounded | sentinel | sentinel | sentinel | sentinel | shortcut | 1.0 |
| one-sided unbounded × one-sided unbounded (opposite end, real overlap) | finite | sentinel | sentinel | sentinel | sentinel | shortcut | 1.0 |
| one-sided unbounded × one-sided unbounded (opposite end, gap) | empty | sentinel | sentinel | sentinel | sentinel | HARD GATE | 0 |
| universal × universal | universal | 2·sentinel | 2·sentinel | sentinel | sentinel | shortcut | 1.0 |
| multi-piece infinite × bounded | bounded (or empty) | sentinel | finite | finite | finite | standard frac_min | inter/finite ∈ [0, 1] |
| multi-piece infinite × multi-piece infinite | bounded or unbounded | sentinel | sentinel | sentinel | sentinel | shortcut (if non-empty) | 1.0 |
| anything × anything | empty | * | * | * | * | HARD GATE | 0 |

### 9.2 Half-open semantics throughout

All intervals are `[lo, hi)` — lo inclusive, hi exclusive. The complement of `[lo, hi)` is `(-∞, lo) ∪ [hi, +∞)`. Overlap check uses strict `<`. This means:

- `[Mar 2024]` and `[Apr 2024]` do NOT overlap (adjacent but not overlapping).
- `[March 2024]` and `[April 2024]` are sequential — March ends at Apr 1 exclusive, April starts at Apr 1 inclusive. No shared point.
- The complement of `[2024]` is `(-∞, Jan 1 2024) ∪ [Jan 1 2025, +∞)`. A doc anchored at Jan 1 2025 IS in the complement (not in 2024).

This eliminates boundary-touch false positives.

### 9.3 Normalization

Final score = `(sum over qrefs of max over drefs of pair_overlap) / |query.time_refs|`.

Properties:
- Range: [0, 1]
- Doc matching ALL query refs perfectly → 1.0
- Doc matching HALF the query refs perfectly → 0.5
- Doc matching NONE → 0.0
- Within a single query, "matching more refs" → higher score (ranking property)
- Across queries, scores are comparable (both bounded in [0, 1])

### 9.4 Filter

`temporal_pass = score > 0`. The filter is fully derived from the score function. No separate primitive.

This unifies what was historically two functions (`doc_passes_filter` and `evaluate_dnf_match`) into one.

## 10. What V7 Fixes vs. V1 (Current Production)

| Issue | V1 (current) | V7 |
|---|---|---|
| Same-anchor binding | Buggy: filter-side strict EXISTS gives accidentally-correct behavior on most benches; soft variants (V2/V2.7/V5) inherit a same-anchor binding bug | Fixed: range composition within a clause; each clause is one TimeRange; per-clause overlap is between one composed range and the doc's anchor set |
| Doc-side relations | Missing: doc extraction only emits closed intervals; "since X" / "after Y" / "before Z" surface forms lose direction | Fixed: extractor emits open-ended TimeRanges that match what the planner would produce on the query side |
| Engagement-relevance | Failed: doc with summer anchor + October anchor gets V1 strict-filter-killed for "outside summer" queries because of the summer anchor | Fixed: per-doc-ref scoring — the October anchor satisfies the query, the summer anchor's presence doesn't hurt |
| Event-negation | Unhandled (was never the temporal layer's job; doc with "X did NOT happen on May 3" gets May 3 anchor like any other) | Same — explicitly delegated to the semantic layer; the temporal layer treats event-polarity as out of scope |
| Filter / score duplication | Two functions, inconsistent rules at boundaries | One function, filter = score > 0 |
| AND/OR aggregator | Mostly dead weight on production (per the audit) — flat_mean tied per-bench-exactly | Goes away — replaced by per-ref scoring + cross-ref additivity |
| DNF AST | Planner output shape, but evaluator does per-leaf factor aggregation (wrong semantics) | Planner AST can remain DNF-shaped internally; evaluator does range composition per clause, producing TimeRange per ref |

## 11. What V7 Costs

| Cost | Magnitude |
|---|---|
| Extractor must handle directional surface forms (since/after/until/before) | LLM-prompt growth + extraction stochasticity; potentially testable as a smaller ablation first |
| Extractor must handle recurrence patterns | Modest prompt growth; RecurringPattern data type adds complexity |
| TimeRange data structure + set ops | ~150 LOC + unit tests |
| New retrieval evaluator | ~50 LOC |
| Planner prompt update (compatibility rule + verb-vs-temporal-negation rule) | ~10 lines added to prompt |
| Validation: A/B vs current on 35-bench + engagement_disjoint | ~2 days work |

Total LOC delta: roughly equivalent or slightly less than current temporal_retrieval. The complexity moves from "many overlapping concepts" (relations + DNF aggregator + filter) to "one data type with clean ops" (TimeRange).

## 12. Implementation Plan

Phased rollout (no production change until phase 7):

1. **Phase 1 — TimeRange + ops**: `time_range.py` (Interval, TimeRange, intersect, union, complement, measure, canonicalize). Unit tests covering all unboundedness shapes.
2. **Phase 2 — Scoring**: `scoring.py` (pair_overlap, final_score, temporal_pass). Unit tests covering all cases in §8.
3. **Phase 3 — Recurrence**: `recurrence.py` (RecurringPattern, materialize). Unit tests.
4. **Phase 4 — Planner adapter**: take existing planner output (DNF AST), convert to `list[TimeRange]`. Keep current planner LLM/prompt unchanged for now (test V7 with existing planner outputs first).
5. **Phase 5 — Extractor adapter**: take existing extractor output (`list[TimeEnvelope]`), wrap each as a single-interval TimeRange. Test with current extractor.
6. **Phase 6 — Validation**: A/B test V7 vs current V1 strict on:
   - 35-bench (no expected regression on macro; specific bench reports)
   - `engagement_disjoint` bench (V7 expected to win clearly)
7. **Phase 7 — Production wiring** (only if Phase 6 wins): update `retriever.py` to call V7. Keep V1 path under a flag for rollback.
8. **Phase 8 — Extractor extensions**: add directional surface-form handling. Test as separate A/B (extractor change isolated from V7's main effect).
9. **Phase 9 — Planner prompt update**: add compatibility detection rule + verb-negation rule. Test as separate A/B.

Each phase produces measurable artifacts before the next. Production temporal_retrieval_min stays untouched throughout. Phases 8 and 9 are optional refinements after V7 ships.

## 13. Decision Criteria for Shipping V7

Required before any production wiring:

1. **35-bench macro R@1 ≥ V1 R@1** (no macro regression)
2. **35-bench macro R@5 ≥ V1 R@5 − 0.005** (within tight tolerance; one-bench notin_multi_interval regression is expected and acceptable if other gains compensate)
3. **No individual bench regresses by > 0.05 R@1** (no catastrophic per-bench failure)
4. **engagement_disjoint bench R@1 strictly > V1 R@1** (clean win on the engagement / compound / event-negation cases the new bench is designed to test)
5. **Unit test coverage** on TimeRange ops and pair_overlap edge cases (all §8 cases pass deterministically)

If any criterion fails, V7 stays as research code. The architectural recipe stands as a future reference but isn't shipped.

## 14. Open Questions for Future Refinement

1. **Salient window for both-infinite × both-infinite-multi-piece**: my spec uses the "both-infinite shortcut → 1.0" rule. An alternative is to clip to a salient window (e.g., `ref_time ± 50y`) and compute frac_min on the clipped versions. This gives a fractional score that varies with overlap size. Probably not worth the complexity at v1.

2. **Recurrence beyond fixed-period**: "first Monday of each month" requires filter masks beyond `(base, period, end)`. RFC 5545 RRULE handles this; we don't initially.

3. **Anaphoric event references** ("after the launch", "since the v3 release") still depend on corpus retrieval to anchor the event date. This is orthogonal to V7's range-composition logic — the anchored interval gets plugged into the same TimeRange pipeline once resolved.

4. **Confidence / uncertainty on extracted ranges**: extractor might want to express "this date is approximate" or "I'm not sure". Currently the spec is binary (range or no range). Could be added as a per-interval weight if proves useful (probably not needed).

5. **Cross-template scoring weights**: per-query-ref contributes equally (sum / |refs|). Could weight refs by some prior of importance. Probably not justified by current evidence.

## 15. References & Lineage

This spec consolidates conclusions from research sessions throughout 2024-2026:

- AND/OR distinction audit (project_temporal_and_or_dropped): 92.6% queries are flat/empty; aggregator choice is no-op on production
- Disjoint variants (project_disjoint_keep_strict, project_disjoint_v4_v6_range_composition): V1 strict wins on synthetic benches; V4 ref_time +0.006 macro; V5 has same-anchor binding bug
- frac_min leaf factor (project_temporal_dnf_compose): +0.011 R@1 macro on 35-bench
- Engagement-relevance probe (project_disjoint_v4_v6_range_composition): V6 range composition wins 3/3 on compound-disjoint scenarios
- Slim planner (project_slim_planner_marginal): planner prompt is mostly load-bearing; can't aggressively simplify
- Architectural reframing (this session, 2026-05-21): list-of-time-refs + range composition + additive scoring as the unifying recipe
- New bench (engagement_disjoint_*.jsonl): 10 queries × ~4 docs covering all the architectural test cases

The bench file lives at `evaluation/temporal_extraction/data/engagement_disjoint_{docs,queries,gold}.jsonl` — committed independently of any V7 implementation.
