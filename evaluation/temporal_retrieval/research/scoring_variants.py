"""Alternative DNF-match scoring functions for the temporal retriever.

The shipped retriever uses binary `intersect`/`after`/`before` leaf
factors and Zadeh AND/OR aggregation (`min` / `max`). This module
provides variants for a side-by-side A/B:

    baseline           — current shipped logic
    frac_max           — intersect becomes |q∩a| / max(|q|,|a|)
    frac_min           — intersect becomes |q∩a| / min(|q|,|a|)
    prob_andor         — AND=product, OR=1 − ∏(1 − c_i)
    relation_wt        — after/before weighted at 0.7 vs intersect at 1.0
    specificity        — leaf factor × min(|q|,|a|)/max(|q|,|a|)

    doc_frac           — intersect = max(FLOOR, |q∩d|/|d|) when overlap
                         exists, 0 otherwise. (doc-fraction with floor)

    proximity_ab       — after/before are linear-decayed with floor:
                         score = max(FLOOR, 1 − gap / (K × anchor_width))
                         on the satisfying side; 0 on the wrong side.
                         K=10 anchor widths; FLOOR=0.3.

    doc_frac_prox      — both fixes combined (the recommended variant).
"""

from __future__ import annotations

from temporal_retrieval.core import (
    Interval,
    constraint_factor_for_doc,
    excluded_containment,
)
from temporal_retrieval.planner import QueryPlan


# ---------------------------------------------------------------------------
# Leaf factor variants (per-leaf, before aggregation)
# ---------------------------------------------------------------------------


def _intersect_overlap_max(d_ivs: list[Interval], a_ivs: list[Interval]) -> float:
    """|q∩a| / max(|q|,|a|), max across (doc_iv, anchor_iv) pairs."""
    best = 0.0
    for di in d_ivs:
        d_w = di.latest_us - di.earliest_us
        if d_w <= 0:
            d_w = 1
        for ai in a_ivs:
            a_w = ai.latest_us - ai.earliest_us
            if a_w <= 0:
                a_w = 1
            lo = max(di.earliest_us, ai.earliest_us)
            hi = min(di.latest_us, ai.latest_us)
            inter = max(0, hi - lo)
            denom = max(d_w, a_w)
            f = inter / denom
            if f > best:
                best = f
    return best


def _intersect_overlap_min(d_ivs: list[Interval], a_ivs: list[Interval]) -> float:
    """|q∩a| / min(|q|,|a|), capped at 1.0."""
    best = 0.0
    for di in d_ivs:
        d_w = di.latest_us - di.earliest_us
        if d_w <= 0:
            d_w = 1
        for ai in a_ivs:
            a_w = ai.latest_us - ai.earliest_us
            if a_w <= 0:
                a_w = 1
            lo = max(di.earliest_us, ai.earliest_us)
            hi = min(di.latest_us, ai.latest_us)
            inter = max(0, hi - lo)
            denom = min(d_w, a_w)
            f = min(1.0, inter / denom)
            if f > best:
                best = f
    return best


# ---------------------------------------------------------------------------
# Floor / window constants for fractional scoring.
#
# K = 10 anchor-widths defines the decay region for after/before.
# FLOOR = 0.3 is the minimum score for docs that logically satisfy the
# relation (overlap exists for intersect; right-side for after/before).
# Justification (see message thread):
#   - >0 honors the "logical match" intuition; a 1920 doc IS before 2023.
#   - <empty_doc_match (1.0) so distant temporal matches don't outrank
#     timeless docs.
#   - 0.3 ≈ exp(−1.2): linear decay's natural endpoint for ~one
#     time-constant over the chosen window.
#   - 1.0−0.3 = 0.7 gap from peak: larger than typical pool-normalized
#     cosine spread, so a perfect satisfier still beats a distant one
#     even with weaker cosine.
# ---------------------------------------------------------------------------
PROX_K = 10.0
FLOOR = 0.3


def _intersect_doc_frac(d_ivs: list[Interval], a_ivs: list[Interval]) -> float:
    """|q∩d| / |d| with FLOOR when overlap exists, 0 when no overlap.

    Rewards narrow docs inside wide anchors (July 4th doc in 'last year'
    query → 1.0). Punishes wide docs against narrow anchors ('once last
    year' doc against 'Jan 1, 2025' query → ~1/365 → floored to FLOOR).
    A doc with no overlap stays at 0 (not a logical match).
    """
    best_frac = 0.0
    found_overlap = False
    for di in d_ivs:
        d_w = di.latest_us - di.earliest_us
        if d_w <= 0:
            d_w = 1
        for ai in a_ivs:
            lo = max(di.earliest_us, ai.earliest_us)
            hi = min(di.latest_us, ai.latest_us)
            inter = max(0, hi - lo)
            if inter > 0:
                found_overlap = True
                f = inter / d_w
                if f > best_frac:
                    best_frac = f
    if not found_overlap:
        return 0.0
    return max(FLOOR, best_frac)


def _intersect_doc_frac_raw(d_ivs: list[Interval], a_ivs: list[Interval]) -> float:
    """|q∩d| / |d| with NO floor. Raw doc-fraction spans 0..1.

    Used to test whether the floor on intersect is helpful or
    flatlines too many cases (user's "if everything is floored
    there's no point" observation).
    """
    best_frac = 0.0
    for di in d_ivs:
        d_w = di.latest_us - di.earliest_us
        if d_w <= 0:
            d_w = 1
        for ai in a_ivs:
            lo = max(di.earliest_us, ai.earliest_us)
            hi = min(di.latest_us, ai.latest_us)
            inter = max(0, hi - lo)
            if inter > 0:
                f = inter / d_w
                if f > best_frac:
                    best_frac = f
    return best_frac


def _after_proximity(d_ivs: list[Interval], a_ivs: list[Interval]) -> float:
    """Distance-decayed `after` with floor for logical satisfiers.

    `after` means doc.earliest > anchor.latest. Linear decay from 1.0
    (touching the anchor's end) toward 0 over K anchor-widths, floored
    at FLOOR. A doc on the wrong side of the anchor scores 0.
    """
    if not a_ivs:
        return 1.0
    a_e = min(ai.earliest_us for ai in a_ivs)
    a_l = max(ai.latest_us for ai in a_ivs)
    a_w = max(1, a_l - a_e)
    window = PROX_K * a_w
    best = 0.0
    found_satisfying = False
    for di in d_ivs:
        if di.latest_us <= a_l:
            continue  # not strictly after — wrong side
        found_satisfying = True
        gap = max(0, di.earliest_us - a_l)
        raw = 1.0 - gap / window
        score = max(FLOOR, raw)
        if score > best:
            best = score
    return best if found_satisfying else 0.0


def _before_proximity(d_ivs: list[Interval], a_ivs: list[Interval]) -> float:
    """Distance-decayed `before` with floor for logical satisfiers.

    `before` means doc.earliest < anchor.earliest. Linear decay from 1.0
    (touching the anchor's start) toward 0 over K anchor-widths, floored
    at FLOOR. A 1920 doc against "before 2023" (gap=103y, window=10y)
    scores FLOOR (logically before, just very distant). A 2025 doc
    scores 0 (wrong side).
    """
    if not a_ivs:
        return 1.0
    a_e = min(ai.earliest_us for ai in a_ivs)
    a_l = max(ai.latest_us for ai in a_ivs)
    a_w = max(1, a_l - a_e)
    window = PROX_K * a_w
    best = 0.0
    found_satisfying = False
    for di in d_ivs:
        if di.earliest_us >= a_e:
            continue  # not strictly before — wrong side
        found_satisfying = True
        gap = max(0, a_e - di.latest_us)
        raw = 1.0 - gap / window
        score = max(FLOOR, raw)
        if score > best:
            best = score
    return best if found_satisfying else 0.0


def _specificity(d_ivs: list[Interval], a_ivs: list[Interval]) -> float:
    """min(|q|,|a|) / max(|q|,|a|) — IoU-like width-ratio, max over pairs."""
    best = 0.0
    for di in d_ivs:
        d_w = di.latest_us - di.earliest_us
        if d_w <= 0:
            d_w = 1
        for ai in a_ivs:
            a_w = ai.latest_us - ai.earliest_us
            if a_w <= 0:
                a_w = 1
            lo = max(di.earliest_us, ai.earliest_us)
            hi = min(di.latest_us, ai.latest_us)
            inter = max(0, hi - lo)
            if inter <= 0:
                continue  # no overlap → 0
            r = min(d_w, a_w) / max(d_w, a_w)
            if r > best:
                best = r
    return best


# ---------------------------------------------------------------------------
# Whole-DNF evaluators
# ---------------------------------------------------------------------------


def _aggregate_zadeh(clauses: list[list[float]]) -> float:
    """Standard Zadeh AND=min, OR=max."""
    if not clauses:
        return 1.0
    return max((min(c) if c else 1.0) for c in clauses)


def _aggregate_prob(clauses: list[list[float]]) -> float:
    """AND = ∏ ; OR = 1 − ∏(1 − c)."""
    if not clauses:
        return 1.0
    clause_scores: list[float] = []
    for c in clauses:
        if not c:
            clause_scores.append(1.0)
            continue
        prod = 1.0
        for f in c:
            prod *= f
        clause_scores.append(prod)
    prod_not = 1.0
    for cs in clause_scores:
        prod_not *= (1.0 - cs)
    return 1.0 - prod_not


def _eval_with(
    plan: QueryPlan,
    doc_ivs: list[Interval],
    resolver,
    intersect_fn,
    aggregator,
    relation_weight: dict[str, float] | None = None,
    apply_specificity: bool = False,
) -> float:
    if not plan.expr:
        return 1.0
    rel_wt = relation_weight or {}
    clause_factors: list[list[float]] = []
    for ci, clause in enumerate(plan.expr):
        leaf_factors: list[float] = []
        for li, leaf in enumerate(clause):
            anchor_ivs = resolver(ci, li, leaf)
            if not anchor_ivs:
                leaf_factors.append(1.0)  # anchor-empty default (matches shipped)
                continue
            if leaf.relation == "disjoint":
                cont = excluded_containment(doc_ivs, anchor_ivs)
                f = max(0.0, 1.0 - cont)
            elif leaf.relation == "intersect":
                f = intersect_fn(doc_ivs, anchor_ivs)
            else:  # after/before stay binary regardless of variant
                f = constraint_factor_for_doc(doc_ivs, anchor_ivs, leaf.relation)
            # Apply per-relation weight (downweight loose half-lines).
            w = rel_wt.get(leaf.relation, 1.0)
            f *= w
            # Specificity multiplier (intersect only; meaningless for half-lines).
            if apply_specificity and leaf.relation == "intersect":
                f *= _specificity(doc_ivs, anchor_ivs)
            leaf_factors.append(f)
        clause_factors.append(leaf_factors)
    return aggregator(clause_factors)


# ---------------------------------------------------------------------------
# Public variants
# ---------------------------------------------------------------------------


def baseline(plan: QueryPlan, doc_ivs, resolver) -> float:
    """Current shipped scoring: binary intersect + Zadeh min/max."""
    return _eval_with(
        plan,
        doc_ivs,
        resolver,
        intersect_fn=lambda d, a: constraint_factor_for_doc(d, a, "intersect"),
        aggregator=_aggregate_zadeh,
    )


def frac_max(plan: QueryPlan, doc_ivs, resolver) -> float:
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=_intersect_overlap_max,
        aggregator=_aggregate_zadeh,
    )


def frac_min(plan: QueryPlan, doc_ivs, resolver) -> float:
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=_intersect_overlap_min,
        aggregator=_aggregate_zadeh,
    )


def frac_min_binab(plan: QueryPlan, doc_ivs, resolver) -> float:
    """Uniform min-norm intersect (no claim_type dispatch) + binary after/before.

    Equivalent to frac_min — included by name for symmetry with the
    claim_type-dispatched variants in head-to-head benches.
    """
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=_intersect_overlap_min,
        aggregator=_aggregate_zadeh,
    )


def prob_andor(plan: QueryPlan, doc_ivs, resolver) -> float:
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=lambda d, a: constraint_factor_for_doc(d, a, "intersect"),
        aggregator=_aggregate_prob,
    )


def relation_wt(plan: QueryPlan, doc_ivs, resolver, half_line_w: float = 0.7) -> float:
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=lambda d, a: constraint_factor_for_doc(d, a, "intersect"),
        aggregator=_aggregate_zadeh,
        relation_weight={"intersect": 1.0, "after": half_line_w,
                         "before": half_line_w, "disjoint": 1.0},
    )


def specificity(plan: QueryPlan, doc_ivs, resolver) -> float:
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=lambda d, a: constraint_factor_for_doc(d, a, "intersect"),
        aggregator=_aggregate_zadeh,
        apply_specificity=True,
    )


def doc_frac(plan: QueryPlan, doc_ivs, resolver) -> float:
    """Intersect = |q∩d|/|d| (doc-fraction).  Half-lines stay binary."""
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=_intersect_doc_frac,
        aggregator=_aggregate_zadeh,
    )


def _eval_with_proximity_ab(plan, doc_ivs, resolver, intersect_fn,
                            aggregator) -> float:
    """Same as _eval_with but uses distance-decayed after/before scoring."""
    if not plan.expr:
        return 1.0
    clause_factors: list[list[float]] = []
    for ci, clause in enumerate(plan.expr):
        leaf_factors: list[float] = []
        for li, leaf in enumerate(clause):
            anchor_ivs = resolver(ci, li, leaf)
            if not anchor_ivs:
                leaf_factors.append(1.0)
                continue
            if leaf.relation == "disjoint":
                cont = excluded_containment(doc_ivs, anchor_ivs)
                f = max(0.0, 1.0 - cont)
            elif leaf.relation == "intersect":
                f = intersect_fn(doc_ivs, anchor_ivs)
            elif leaf.relation == "after":
                f = _after_proximity(doc_ivs, anchor_ivs)
            elif leaf.relation == "before":
                f = _before_proximity(doc_ivs, anchor_ivs)
            else:
                f = 0.0
            leaf_factors.append(f)
        clause_factors.append(leaf_factors)
    return aggregator(clause_factors)


def proximity_ab(plan: QueryPlan, doc_ivs, resolver) -> float:
    """Distance-decayed after/before; binary intersect (baseline)."""
    return _eval_with_proximity_ab(
        plan, doc_ivs, resolver,
        intersect_fn=lambda d, a: constraint_factor_for_doc(d, a, "intersect"),
        aggregator=_aggregate_zadeh,
    )


def doc_frac_prox(plan: QueryPlan, doc_ivs, resolver) -> float:
    """Doc-fraction intersect AND distance-decayed after/before (user-aligned)."""
    return _eval_with_proximity_ab(
        plan, doc_ivs, resolver,
        intersect_fn=_intersect_doc_frac,
        aggregator=_aggregate_zadeh,
    )


def doc_frac_raw(plan: QueryPlan, doc_ivs, resolver) -> float:
    """Raw doc-fraction intersect (no floor). Half-lines stay binary."""
    return _eval_with(
        plan, doc_ivs, resolver,
        intersect_fn=_intersect_doc_frac_raw,
        aggregator=_aggregate_zadeh,
    )


def doc_frac_raw_prox(plan: QueryPlan, doc_ivs, resolver) -> float:
    """Raw doc-fraction intersect (no floor) + after/before linear decay
    with floor. The user-aligned 'lean' variant: floor only where it
    earns its keep (binary-satisfies orientation for after/before)."""
    return _eval_with_proximity_ab(
        plan, doc_ivs, resolver,
        intersect_fn=_intersect_doc_frac_raw,
        aggregator=_aggregate_zadeh,
    )


def _intersect_by_claim_type(
    doc_refs: list[dict], a_ivs: list[Interval]
) -> float:
    """Per-ref scoring using claim_type to choose formula.

    - event: doc-fraction `|q∩d|/|d|` (rewards specificity in query window)
    - state: query-fraction `|q∩d|/|q|` (rewards coverage of query by state)

    Max across (doc_ref, anchor_iv) pairs.
    """
    if not doc_refs or not a_ivs:
        return 0.0
    best = 0.0
    for ref in doc_refs:
        d_e = ref["earliest_us"]
        d_l = ref["latest_us"]
        ctype = ref.get("claim_type", "event")
        d_w = max(1, d_l - d_e)
        for ai in a_ivs:
            a_w = max(1, ai.latest_us - ai.earliest_us)
            lo = max(d_e, ai.earliest_us)
            hi = min(d_l, ai.latest_us)
            inter = max(0, hi - lo)
            if inter <= 0:
                continue
            if ctype == "event":
                f = inter / d_w
            else:  # state
                f = inter / a_w
            if f > best:
                best = f
    return best


def _intersect_by_claim_type_eventbinary(
    doc_refs: list[dict], a_ivs: list[Interval]
) -> float:
    """Refined claim_type scoring:
    - event: BINARY intersect (preserve baseline behavior; doc-fraction
             on events broke benches with legitimate-duration events)
    - state: MIN-NORM `|q∩d|/min(|q|,|d|)` (capped at 1.0).
             Handles both containment directions correctly:
               narrow state inside wide query → q/q = 1.0 ✓
               wide state around narrow query → d/d = 1.0 ✓
             Pure query-fraction was wrongly shrinking narrow-state-in-
             wide-query cases (mixed_cue, allen, cotemporal, era).
    """
    if not doc_refs or not a_ivs:
        return 0.0
    best = 0.0
    for ref in doc_refs:
        d_e = ref["earliest_us"]
        d_l = ref["latest_us"]
        ctype = ref.get("claim_type", "event")
        d_w = max(1, d_l - d_e)
        for ai in a_ivs:
            a_w = max(1, ai.latest_us - ai.earliest_us)
            lo = max(d_e, ai.earliest_us)
            hi = min(d_l, ai.latest_us)
            inter = max(0, hi - lo)
            if inter <= 0:
                continue
            if ctype == "event":
                f = 1.0  # binary (any overlap → full match)
            else:  # state: min-norm (either-direction containment)
                f = min(1.0, inter / min(d_w, a_w))
            if f > best:
                best = f
    return best


def _intersect_by_claim_type_docfrac_event(
    doc_refs: list[dict], a_ivs: list[Interval]
) -> float:
    """Asymmetric scoring (per claim_type intent):
    - event: DOC-FRAC `|q∩d|/|d|` (rewards specificity in query window).
             Right for events: a narrow event inside wide query stays
             1.0 (full match); a wide event with partial overlap shrinks.
    - state: MIN-NORM (containment-aware, either direction). States are
             continuous conditions — partial overlap that fully contains
             one side IS a full match.
    The key bet: events SHOULD be penalized when they spill outside the
    query window (they were specific things, not spread out); states
    SHOULD NOT be penalized when they extend beyond the query window
    (they continue to be true throughout).
    """
    if not doc_refs or not a_ivs:
        return 0.0
    best = 0.0
    for ref in doc_refs:
        d_e = ref["earliest_us"]
        d_l = ref["latest_us"]
        ctype = ref.get("claim_type", "event")
        d_w = max(1, d_l - d_e)
        for ai in a_ivs:
            a_w = max(1, ai.latest_us - ai.earliest_us)
            lo = max(d_e, ai.earliest_us)
            hi = min(d_l, ai.latest_us)
            inter = max(0, hi - lo)
            if inter <= 0:
                continue
            if ctype == "event":
                f = inter / d_w  # doc-frac
            else:  # state
                f = min(1.0, inter / min(d_w, a_w))  # min-norm
            if f > best:
                best = f
    return best


def claim_type_scoring(plan: QueryPlan, doc_refs: list[dict],
                       resolver, *, event_binary: bool = False,
                       event_docfrac: bool = False,
                       binary_ab: bool = False) -> float:
    """v3.4 claim_type-dispatched scoring.

    For each leaf:
      - intersect: event → doc-frac OR binary (event_binary=True);
                   state → query-frac OR min-norm (event_binary=True)
      - after/before: linear decay with floor over K anchor widths
                      (binary_ab=True keeps binary after/before like baseline,
                      isolating the claim_type effect from proximity decay)
      - disjoint: 1 − excluded_containment over the doc intervals
    Aggregated with Zadeh min/max.
    """
    if not plan.expr:
        return 1.0

    # For after/before/disjoint we just need the doc's intervals.
    doc_ivs = [
        Interval(earliest_us=r["earliest_us"], latest_us=r["latest_us"])
        for r in doc_refs
    ]

    clause_factors: list[list[float]] = []
    for ci, clause in enumerate(plan.expr):
        leaf_factors: list[float] = []
        for li, leaf in enumerate(clause):
            anchor_ivs = resolver(ci, li, leaf)
            if not anchor_ivs:
                leaf_factors.append(1.0)
                continue
            if leaf.relation == "disjoint":
                cont = excluded_containment(doc_ivs, anchor_ivs)
                f = max(0.0, 1.0 - cont)
            elif leaf.relation == "intersect":
                if event_docfrac:
                    f = _intersect_by_claim_type_docfrac_event(doc_refs, anchor_ivs)
                elif event_binary:
                    f = _intersect_by_claim_type_eventbinary(doc_refs, anchor_ivs)
                else:
                    f = _intersect_by_claim_type(doc_refs, anchor_ivs)
            elif leaf.relation in ("after", "before"):
                if binary_ab:
                    f = constraint_factor_for_doc(doc_ivs, anchor_ivs,
                                                  leaf.relation)
                elif leaf.relation == "after":
                    f = _after_proximity(doc_ivs, anchor_ivs)
                else:
                    f = _before_proximity(doc_ivs, anchor_ivs)
            else:
                f = 0.0
            leaf_factors.append(f)
        clause_factors.append(leaf_factors)
    return _aggregate_zadeh(clause_factors)


VARIANTS: dict[str, callable] = {
    "baseline":          baseline,
    "frac_max":          frac_max,
    "frac_min":          frac_min,
    "prob_andor":        prob_andor,
    "relation_wt":       relation_wt,
    "specificity":       specificity,
    "doc_frac":          doc_frac,
    "proximity_ab":      proximity_ab,
    "doc_frac_prox":     doc_frac_prox,
    # User-refined "lean" variants: drop floor on intersect (let
    # doc-fraction be its own gradient); keep floor on after/before
    # (where binary "satisfies orientation" earns the floor).
    "doc_frac_raw":      doc_frac_raw,
    "doc_frac_raw_prox": doc_frac_raw_prox,
}
