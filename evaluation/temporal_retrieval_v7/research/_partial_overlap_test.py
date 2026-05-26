"""Characterization: cross-ref combiner under PARTIAL overlaps.

The mean / noisy-OR / power-mean A/Bs found the cross-ref combiner does
not move R@1 on the standard benches — because those benches' docs are
single-anchor with full-or-nothing overlaps (a doc anchor either falls
inside a query ref → 1.0, or doesn't → 0.0). The combiner only reorders
rankings when a doc PARTIALLY overlaps multiple query refs.

A partial overlap arises when a doc anchor is a SPAN (a duration) that
straddles a query-ref boundary — e.g. a project running Feb–Aug 2024
against query refs "Q1 2024" and "Q3 2024".

This test constructs exactly those cases with real TimeRanges, scores
each candidate under mean / power-mean(p) / max, and characterizes how
they rank a span-doc (touches both refs partially) against a focused-doc
(fully satisfies one ref). No LLM — pure scoring-function test.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_v7.research._partial_overlap_test
"""
from __future__ import annotations

from datetime import datetime, timezone

from temporal_retrieval_v7 import TimeRange, best_per_ref


def us(y: int, m: int = 1, d: int = 1) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1_000_000)


def q(year: int, quarter: int) -> TimeRange:
    """Calendar quarter as a TimeRange."""
    lo_m = 1 + (quarter - 1) * 3
    if quarter == 4:
        return TimeRange.closed(us(year, 10, 1), us(year + 1, 1, 1))
    return TimeRange.closed(us(year, lo_m, 1), us(year, lo_m + 3, 1))


def span(y1: int, m1: int, y2: int, m2: int) -> TimeRange:
    return TimeRange.closed(us(y1, m1, 1), us(y2, m2, 1))


# ---------------------------------------------------------------------------
# Combiners
# ---------------------------------------------------------------------------


def power_mean(vals: list[float], p: float) -> float:
    """Power mean. p<1 leans toward min (penalizes misses harder);
    p>1 leans toward max. p→0 is the geometric mean — any zero → 0."""
    if not vals:
        return 0.0
    if p == 1.0:
        return sum(vals) / len(vals)
    if p <= 0.0:  # geometric-mean limit
        prod = 1.0
        for v in vals:
            prod *= v
        return prod ** (1.0 / len(vals))
    return (sum(v ** p for v in vals) / len(vals)) ** (1.0 / p)


COMBINERS = [
    ("p=0.3", lambda v: power_mean(v, 0.3)),
    ("p=0.5", lambda v: power_mean(v, 0.5)),
    ("p=0.7", lambda v: power_mean(v, 0.7)),
    ("mean", lambda v: power_mean(v, 1.0)),
    ("p=2", lambda v: power_mean(v, 2.0)),
    ("p=3", lambda v: power_mean(v, 3.0)),
    ("max", lambda v: max(v) if v else 0.0),
]


# ---------------------------------------------------------------------------
# Scenarios — each: query refs + named candidate docs (lists of anchors)
# ---------------------------------------------------------------------------


def vec(query_refs: list[TimeRange], doc_anchors: list[TimeRange]) -> list[float]:
    """Per-ref best-overlap vector — the real pipeline's level-1 output."""
    return [round(best_per_ref(r, doc_anchors), 3) for r in query_refs]


SCENARIOS = [
    {
        "name": "S1  span vs focused  — query: active in Q1 AND Q3 2024",
        "intent": "coverage — a doc active across BOTH quarters is the "
                  "best partial answer; ranks above one that only hits Q1.",
        "refs": [q(2024, 1), q(2024, 3)],
        "docs": {
            "both_full   (Q1 + Q3 anchors)": [q(2024, 1), q(2024, 3)],
            "span_feb_aug (one Feb-Aug span)": [span(2024, 2, 2024, 9)],
            "focused_q1   (Q1 only)": [q(2024, 1)],
            "neither      (Q2 only)": [q(2024, 2)],
        },
        "expect_top": "both_full",
        "expect_2nd": "span_feb_aug",
    },
    {
        "name": "S2  weak straddle vs focused — query: Q1 AND Q3 2024",
        "intent": "a doc grazing both quarters weakly vs one fully in Q1 — "
                  "the genuinely contested case.",
        "refs": [q(2024, 1), q(2024, 3)],
        "docs": {
            "weak_span  (Mar-Aug, grazes both)": [span(2024, 3, 2024, 8)],
            "focused_q1 (Q1 only)": [q(2024, 1)],
        },
        "expect_top": "(contested)",
        "expect_2nd": "-",
    },
    {
        "name": "S3  3-ref coverage — query: Q1 AND Q2 AND Q3 2024",
        "intent": "coverage — a long span touching all three beats a doc "
                  "fully in just one.",
        "refs": [q(2024, 1), q(2024, 2), q(2024, 3)],
        "docs": {
            "long_span  (Feb-Sep, all three)": [span(2024, 2, 2024, 9)],
            "two_full   (Q1 + Q2 anchors)": [q(2024, 1), q(2024, 2)],
            "focused_q2 (Q2 only)": [q(2024, 2)],
        },
        "expect_top": "two_full",
        "expect_2nd": "long_span",
    },
    {
        "name": "S4  asymmetric straddle — query: Q1 AND Q3 2024",
        "intent": "a span heavily in Q1, barely in Q3, vs a doc fully in Q1.",
        "refs": [q(2024, 1), q(2024, 3)],
        "docs": {
            "heavy_q1_graze_q3 (Jan-Aug span)": [span(2024, 1, 2024, 8)],
            "focused_q1        (Q1 only)": [q(2024, 1)],
        },
        "expect_top": "(contested)",
        "expect_2nd": "-",
    },
]


def main() -> None:
    print("=== Cross-ref combiner under partial overlaps ===\n")

    for sc in SCENARIOS:
        print(sc["name"])
        print(f"  intent: {sc['intent']}")
        vectors = {name: vec(sc["refs"], anchors)
                   for name, anchors in sc["docs"].items()}
        # header
        hdr = f"  {'candidate':34s} {'vector':16s} " + \
            " ".join(f"{cn:>7s}" for cn, _ in COMBINERS)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        scores: dict[str, dict[str, float]] = {}
        for name, v in vectors.items():
            row = {cn: cf(v) for cn, cf in COMBINERS}
            scores[name] = row
            cells = " ".join(f"{row[cn]:>7.3f}" for cn, _ in COMBINERS)
            vstr = "(" + ",".join(f"{x:.2f}" for x in v) + ")"
            print(f"  {name:34s} {vstr:16s} {cells}")
        # ranking line per combiner
        print(f"  {'-> ranking':34s} {'':16s}")
        for cn, _ in COMBINERS:
            order = sorted(scores, key=lambda n: scores[n][cn], reverse=True)
            short = [n.split()[0] for n in order]
            print(f"  {'  ' + cn:34s} {'':16s} {' > '.join(short)}")
        print()

    # ---- parametric crossover ----------------------------------------------
    print("=" * 64)
    print("Parametric crossover: both_partial(p,p) vs focused(1,0)\n")
    print("A doc touching BOTH refs equally at fraction p scores exactly p")
    print("under every combiner (power mean of equal values = that value).")
    print("A doc fully in ONE ref scores differently per combiner. The")
    print("crossover p* is where the both-touching doc overtakes it:\n")
    print(f"  {'combiner':10s} {'focused(1,0) score':>20s} {'both wins iff p >':>20s}")
    print("  " + "-" * 52)
    for cn, cf in COMBINERS:
        focused = cf([1.0, 0.0])
        print(f"  {cn:10s} {focused:>20.3f} {focused:>20.3f}")
    print()
    print("p<1 LOWERS the bar (spread bonus — rewards touching all refs);")
    print("p>1 RAISES it (concentration bonus — rewards fully nailing one);")
    print("mean (p=1) is the neutral midpoint at 0.5 — it ranks by total")
    print("overlap mass with no distributional bias either way.")
    print()
    print("Note: p->0 (geometric mean) zeroes any doc missing a ref —")
    print("focused(1,0) collapses to 0.0, indistinguishable from (0,0),")
    print("destroying the partial-coverage signal entirely.")


if __name__ == "__main__":
    main()
