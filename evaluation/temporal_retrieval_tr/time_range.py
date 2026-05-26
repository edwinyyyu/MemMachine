"""IntervalSet: list of disjoint half-open intervals on R̄ (with ±∞ sentinels).

The canonical temporal data type for the temporal-retrieval engine. An
`IntervalSet` represents a set of moments in time as a sorted list of
non-overlapping, non-adjacent half-open intervals `[earliest_us, latest_us)`.
Endpoints may be the sentinel values `NEG_INF`/`POS_INF` to express
unbounded extent.

All set operations (union, intersect, complement, difference,
symmetric_difference) preserve canonical form. Pure integer arithmetic;
no datetime / calendar logic at this layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Sentinel for ±∞. Chosen well above any plausible real microsecond
# timestamp (microseconds since epoch in 2024 is ~10^18; 2^62 is ~4.6e18)
# so that any real measure stays below SENTINEL_THRESHOLD, and signed
# arithmetic on the sentinel doesn't overflow Python's arbitrary
# precision ints either.
POS_INF = 2**62
NEG_INF = -(2**62)

# Threshold above which a measure is treated as "sentinel-large"
# (i.e., touches ±∞). Any real bounded measure stays below this.
SENTINEL_THRESHOLD = 2**60


@dataclass(frozen=True)
class Interval:
    """Half-open temporal interval `[earliest_us, latest_us)`.

    `earliest_us` may be `NEG_INF`; `latest_us` may be `POS_INF`.
    Invariant: `earliest_us < latest_us` (non-empty).
    """

    earliest_us: int
    latest_us: int

    def __post_init__(self) -> None:
        if self.earliest_us >= self.latest_us:
            raise ValueError(
                f"Interval must have earliest < latest; got "
                f"[{self.earliest_us}, {self.latest_us})"
            )

    @property
    def width(self) -> int:
        """Width in microseconds. Saturates near sentinel for unbounded."""
        return self.latest_us - self.earliest_us

    @property
    def left_unbounded(self) -> bool:
        return self.earliest_us <= NEG_INF + 1

    @property
    def right_unbounded(self) -> bool:
        return self.latest_us >= POS_INF - 1


@dataclass(frozen=True)
class IntervalSet:
    """A set of moments in time as a canonical list of disjoint half-open
    intervals (possibly with ±∞ endpoints).

    Canonical-form invariants (enforced by `canonicalize`):
    - Sorted by `earliest_us` ascending
    - No two intervals overlap (overlapping ones are merged)
    - No two intervals are adjacent (`a.latest_us == b.earliest_us`
      becomes one interval `[a.earliest_us, b.latest_us)`)
    - All intervals are non-empty (`earliest_us < latest_us`)
    """

    intervals: tuple[Interval, ...] = field(default_factory=tuple)

    @classmethod
    def from_intervals(cls, ivs: list[Interval] | tuple[Interval, ...]) -> "IntervalSet":
        """Build a canonical IntervalSet from arbitrary intervals."""
        return cls(intervals=tuple(canonicalize(list(ivs))))

    @classmethod
    def empty(cls) -> "IntervalSet":
        return cls(intervals=())

    @classmethod
    def universal(cls) -> "IntervalSet":
        return cls(intervals=(Interval(NEG_INF, POS_INF),))

    @classmethod
    def closed(cls, lo: int, hi: int) -> "IntervalSet":
        """Single half-open interval [lo, hi)."""
        if lo >= hi:
            return cls.empty()
        return cls(intervals=(Interval(lo, hi),))

    @classmethod
    def after(cls, t: int) -> "IntervalSet":
        """[t, +∞)."""
        return cls.closed(t, POS_INF)

    @classmethod
    def before(cls, t: int) -> "IntervalSet":
        """(-∞, t)."""
        return cls.closed(NEG_INF, t)

    def is_empty(self) -> bool:
        return not self.intervals


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def canonicalize(ivs: list[Interval]) -> list[Interval]:
    """Sort by earliest, merge any pair (a, b) where `a.latest_us >= b.earliest_us`.

    Merging on `>=` collapses both overlapping and adjacent intervals.
    Empty intervals (`earliest >= latest`) are dropped.
    """
    valid = [iv for iv in ivs if iv.earliest_us < iv.latest_us]
    if not valid:
        return []
    valid.sort(key=lambda iv: (iv.earliest_us, iv.latest_us))
    out: list[Interval] = []
    cur_lo = valid[0].earliest_us
    cur_hi = valid[0].latest_us
    for iv in valid[1:]:
        if iv.earliest_us <= cur_hi:
            if iv.latest_us > cur_hi:
                cur_hi = iv.latest_us
        else:
            out.append(Interval(cur_lo, cur_hi))
            cur_lo = iv.earliest_us
            cur_hi = iv.latest_us
    out.append(Interval(cur_lo, cur_hi))
    return out


# ---------------------------------------------------------------------------
# Core set operations
# ---------------------------------------------------------------------------


def union(A: IntervalSet, B: IntervalSet) -> IntervalSet:
    """A ∪ B."""
    return IntervalSet.from_intervals(list(A.intervals) + list(B.intervals))


def union_all(sets: list[IntervalSet]) -> IntervalSet:
    if not sets:
        return IntervalSet.empty()
    all_ivs: list[Interval] = []
    for r in sets:
        all_ivs.extend(r.intervals)
    return IntervalSet.from_intervals(all_ivs)


def intersect(A: IntervalSet, B: IntervalSet) -> IntervalSet:
    """A ∩ B via two-pointer sweep over sorted disjoint intervals.

    O(|A| + |B|). Strict `<` overlap check matches half-open semantics:
    `[a, b)` and `[b, c)` do NOT overlap.
    """
    if not A.intervals or not B.intervals:
        return IntervalSet.empty()
    out: list[Interval] = []
    i = j = 0
    a = A.intervals
    b = B.intervals
    while i < len(a) and j < len(b):
        ai, bj = a[i], b[j]
        lo = max(ai.earliest_us, bj.earliest_us)
        hi = min(ai.latest_us, bj.latest_us)
        if lo < hi:
            out.append(Interval(lo, hi))
        if ai.latest_us < bj.latest_us:
            i += 1
        else:
            j += 1
    return IntervalSet(intervals=tuple(out))


def intersect_all(sets: list[IntervalSet]) -> IntervalSet:
    """Intersect a list of IntervalSets. Empty list → universal."""
    if not sets:
        return IntervalSet.universal()
    acc = sets[0]
    for r in sets[1:]:
        acc = intersect(acc, r)
        if not acc.intervals:
            return acc
    return acc


def complement(A: IntervalSet) -> IntervalSet:
    """¬A on (-∞, +∞). Walks A's intervals and emits the gaps.

    Half-open: complement of `[lo, hi)` is `(-∞, lo) ∪ [hi, +∞)`.
    """
    if not A.intervals:
        return IntervalSet.universal()
    out: list[Interval] = []
    prev = NEG_INF
    for iv in A.intervals:
        if prev < iv.earliest_us:
            out.append(Interval(prev, iv.earliest_us))
        prev = iv.latest_us
    if prev < POS_INF:
        out.append(Interval(prev, POS_INF))
    return IntervalSet(intervals=tuple(out))


def difference(A: IntervalSet, B: IntervalSet) -> IntervalSet:
    """A \\ B = A ∩ complement(B)."""
    return intersect(A, complement(B))


def symmetric_difference(A: IntervalSet, B: IntervalSet) -> IntervalSet:
    """A △ B = (A ∪ B) \\ (A ∩ B)."""
    return difference(union(A, B), intersect(A, B))


# ---------------------------------------------------------------------------
# Predicates / measurement
# ---------------------------------------------------------------------------


def is_empty(A: IntervalSet) -> bool:
    return not A.intervals


def measure(A: IntervalSet) -> int:
    """Sum of widths over A's intervals.

    May saturate near sentinel range for unbounded intervals — callers
    check `m >= SENTINEL_THRESHOLD` to detect "infinite" measure.
    """
    total = 0
    for iv in A.intervals:
        total += iv.latest_us - iv.earliest_us
    return total


def is_infinite_measure(A: IntervalSet) -> bool:
    """True iff A has at least one ±∞ endpoint (so measure ≥ sentinel)."""
    if not A.intervals:
        return False
    first = A.intervals[0]
    last = A.intervals[-1]
    return first.earliest_us <= NEG_INF + 1 or last.latest_us >= POS_INF - 1
