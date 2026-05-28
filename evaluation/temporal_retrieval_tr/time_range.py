"""IntervalSet: list of disjoint half-open intervals on R̄ (with ±∞ sentinels).

The canonical temporal data type for the temporal-retrieval engine. An
`IntervalSet` represents a set of moments in time as a sorted list of
non-overlapping, non-adjacent half-open intervals `[earliest_us, latest_us)`.
Endpoints may be the singleton sentinel values `NEG_INF`/`POS_INF` to
express unbounded extent.

All set operations (union, intersect, complement, difference,
symmetric_difference) preserve canonical form. `_Inf` participates in
comparison / arithmetic via Python's reflected operator protocol —
mixed `int`/`_Inf` expressions just work in `max`, `min`, `<`, `<=`, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Typed sentinel: ±∞ as a singleton with sign
# ---------------------------------------------------------------------------


class _Inf:
    """An extended-real infinity. `sign = +1` is +∞; `sign = -1` is -∞.

    Implements comparison and arithmetic against both ints and other `_Inf`
    instances so that `max`/`min`/`<`/`<=` / `-` / `+` all work in mixed
    expressions without ad-hoc branching at the call sites.

    Use the module-level constants `POS_INF` / `NEG_INF` rather than
    instantiating this class directly.
    """

    __slots__ = ("sign",)

    def __init__(self, sign: int) -> None:
        if sign not in (-1, +1):
            raise ValueError(f"_Inf.sign must be ±1; got {sign}")
        self.sign = sign

    # Comparison: _Inf vs _Inf compares signs; _Inf vs anything else uses
    # sign-only heuristic (NEG_INF < any finite; POS_INF > any finite).
    def __lt__(self, other: object) -> bool:
        if isinstance(other, _Inf):
            return self.sign < other.sign
        return self.sign < 0

    def __le__(self, other: object) -> bool:
        if isinstance(other, _Inf):
            return self.sign <= other.sign
        return self.sign < 0

    def __gt__(self, other: object) -> bool:
        if isinstance(other, _Inf):
            return self.sign > other.sign
        return self.sign > 0

    def __ge__(self, other: object) -> bool:
        if isinstance(other, _Inf):
            return self.sign >= other.sign
        return self.sign > 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Inf) and self.sign == other.sign

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(("_Inf", self.sign))

    # Arithmetic: extended-real semantics.
    # ±∞ ± finite = ±∞ (sign preserved).
    # +∞ + (-∞), +∞ - (+∞), -∞ - (-∞) are undefined → raise.
    def __neg__(self) -> "_Inf":
        return _Inf(-self.sign)

    def __add__(self, other: object) -> "_Inf":
        if isinstance(other, _Inf):
            if self.sign != other.sign:
                raise ArithmeticError("∞ + (-∞) is undefined")
            return self
        return self

    def __radd__(self, other: object) -> "_Inf":
        return self.__add__(other)

    def __sub__(self, other: object) -> "_Inf":
        if isinstance(other, _Inf):
            if self.sign == other.sign:
                raise ArithmeticError("∞ − ∞ is undefined")
            return self  # +∞ − (−∞) = +∞; −∞ − (+∞) = −∞
        return self  # ±∞ − finite = ±∞

    def __rsub__(self, other: object) -> "_Inf":
        # finite − ±∞ = ∓∞
        return -self

    def __repr__(self) -> str:
        return "+∞" if self.sign > 0 else "-∞"


# Module-level singletons. Always use these; do not construct `_Inf` directly
# at call sites.
NEG_INF: _Inf = _Inf(-1)
POS_INF: _Inf = _Inf(+1)


# An interval endpoint: a microsecond-precision integer, or one of the
# two infinite sentinels. Also used as the return type of `measure` and
# arithmetic with endpoints, since both live in the same extended-real space.
Endpoint = int | _Inf


def is_inf(x: object) -> bool:
    """True iff `x` is one of the infinite sentinels."""
    return isinstance(x, _Inf)


# ---------------------------------------------------------------------------
# Interval / IntervalSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """Half-open temporal interval `[earliest_us, latest_us)`.

    `earliest_us` may be `NEG_INF`; `latest_us` may be `POS_INF`.
    Invariant: `earliest_us < latest_us` (non-empty).
    """

    earliest_us: Endpoint
    latest_us: Endpoint

    def __post_init__(self) -> None:
        if not (self.earliest_us < self.latest_us):
            raise ValueError(
                f"Interval must have earliest < latest; got "
                f"[{self.earliest_us!r}, {self.latest_us!r})"
            )

    @property
    def width(self) -> Endpoint:
        """Width in microseconds, or `POS_INF` if either endpoint is infinite."""
        return self.latest_us - self.earliest_us

    @property
    def left_unbounded(self) -> bool:
        return isinstance(self.earliest_us, _Inf)

    @property
    def right_unbounded(self) -> bool:
        return isinstance(self.latest_us, _Inf)


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
    def closed(cls, lo: Endpoint, hi: Endpoint) -> "IntervalSet":
        """Single half-open interval [lo, hi)."""
        if not (lo < hi):
            return cls.empty()
        return cls(intervals=(Interval(lo, hi),))

    @classmethod
    def after(cls, t: Endpoint) -> "IntervalSet":
        """[t, +∞)."""
        return cls.closed(t, POS_INF)

    @classmethod
    def before(cls, t: Endpoint) -> "IntervalSet":
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
    # Sort by (earliest, latest). _Inf supports tuple comparison via __lt__.
    valid.sort(key=_sort_key)
    out: list[Interval] = []
    cur_lo: Endpoint = valid[0].earliest_us
    cur_hi: Endpoint = valid[0].latest_us
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


def _sort_key(iv: Interval) -> tuple:
    """Key for sorting; maps ±∞ to comparable proxies for `tuple.__lt__`.

    Python's default tuple comparison delegates to element `<`, which calls
    `int.__lt__(_Inf)` → NotImplemented → falls through to `_Inf.__gt__(int)`.
    That's correct for direct comparison, but `list.sort` requires a stable
    key — and stability + reflected ops occasionally trip on equal keys.
    Map each endpoint to a (rank, value) pair: rank=-1 for NEG_INF,
    rank=0 for finite, rank=+1 for POS_INF.
    """
    def proxy(x: Endpoint) -> tuple[int, int]:
        if isinstance(x, _Inf):
            return (x.sign, 0)
        return (0, x)
    return (proxy(iv.earliest_us), proxy(iv.latest_us))


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
        lo = ai.earliest_us if ai.earliest_us > bj.earliest_us else bj.earliest_us
        hi = ai.latest_us if ai.latest_us < bj.latest_us else bj.latest_us
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
    prev: Endpoint = NEG_INF
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


def measure(A: IntervalSet) -> Endpoint:
    """Sum of widths over A's intervals.

    Returns `POS_INF` if any interval is unbounded (the `_Inf` value
    propagates through addition).
    """
    total: Endpoint = 0
    for iv in A.intervals:
        total = total + (iv.latest_us - iv.earliest_us)
    return total


def is_infinite_measure(A: IntervalSet) -> bool:
    """True iff A has at least one ±∞ endpoint."""
    if not A.intervals:
        return False
    first = A.intervals[0]
    last = A.intervals[-1]
    return isinstance(first.earliest_us, _Inf) or isinstance(last.latest_us, _Inf)
