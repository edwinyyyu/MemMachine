"""Algorithm primitives — Interval, mask, filter, pool, recency, scoring.

Deterministic helpers that compose with the LLM extractor. Each LLM
emission becomes a single `Interval(earliest_us, latest_us)` covering
the temporal envelope. All scoring (mask, filter, recency) operates on
these microsecond-precision intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import TimeEnvelope, to_us


@dataclass
class Interval:
    """Half-open temporal interval [earliest_us, latest_us) in microsecond UTC."""

    earliest_us: int
    latest_us: int


# ===========================================================================
# TimeEnvelope -> Interval
# ===========================================================================


def to_interval(e: TimeEnvelope) -> Interval:
    """Convert a TimeEnvelope to its microsecond Interval."""
    return Interval(earliest_us=to_us(e.earliest), latest_us=to_us(e.latest))


def flatten_intervals(envelopes: list[TimeEnvelope]) -> list[Interval]:
    """Convert a list of TimeEnvelopes into the intervals the retrieval
    layer reasons about. Each envelope becomes exactly one Interval —
    there is no kind-specific expansion."""
    return [to_interval(e) for e in envelopes]


# ===========================================================================
# Mask: includes (intersect/after/before), excludes (disjoint)
# ===========================================================================


def constraint_factor_for_doc(
    doc_intervals: list[Interval],
    anchor_intervals: list[Interval],
    relation: str,
) -> float:
    """Binary factor: 1.0 if doc has any TE satisfying the constraint, else 0.0.

    - "intersect": doc TE overlaps anchor (closed)
    - "after":     doc TE has any time strictly past anchor.latest
    - "before":    doc TE has any time strictly before anchor.earliest
    """
    if not anchor_intervals:
        return 1.0
    a_e = min(ai.earliest_us for ai in anchor_intervals)
    a_l = max(ai.latest_us for ai in anchor_intervals)
    for di in doc_intervals or []:
        if relation == "after":
            if di.latest_us > a_l:
                return 1.0
        elif relation == "before":
            if di.earliest_us < a_e:
                return 1.0
        else:  # "intersect"
            for ai in anchor_intervals:
                if di.earliest_us <= ai.latest_us and ai.earliest_us <= di.latest_us:
                    return 1.0
    return 0.0


def excluded_containment(d_ivs: list[Interval], excl_ivs: list[Interval]) -> float:
    """How much of the doc's anchor falls inside the excluded window.

    max over (d_iv, e_iv) of |d cap e| / |d|. Returns 0.0 if either list
    is empty.

    Strict semantics for multi-interval docs: a single doc interval that
    is fully inside the excluded window dominates (best -> 1.0), even if
    the doc has other intervals far from the exclusion. This is the
    production default. Compared empirically with `excluded_containment_aggregate`
    on a multi-interval reproducer (see notin_aggregate_validation.json):
    strict R@5 = 1.000 vs aggregate 0.750 on the bench where not_in
    actually differentiates them. Aggregate sometimes promotes mixed
    docs to top-1 (R@1 +0.125) but displaces cleanly-out-of-window gold
    from top-5. Keep strict as the production semantics.
    """
    if not d_ivs or not excl_ivs:
        return 0.0
    best = 0.0
    for di in d_ivs:
        d_dur = di.latest_us - di.earliest_us
        if d_dur <= 0:
            d_dur = 1
        for ei in excl_ivs:
            inter_lo = max(di.earliest_us, ei.earliest_us)
            inter_hi = min(di.latest_us, ei.latest_us)
            inter = max(0, inter_hi - inter_lo)
            score = inter / d_dur
            if score > best:
                best = score
                if best >= 1.0:
                    return 1.0
    return best


def excluded_containment_aggregate(
    d_ivs: list[Interval], excl_ivs: list[Interval]
) -> float:
    """Aggregate fraction of the doc's TOTAL interval-time inside the
    excluded window.

    sum over d_iv of |d ∩ union(excl)| / sum over d_iv of |d|. Treats the
    doc holistically rather than picking the worst-overlapping interval.
    A doc whose intervals are mostly outside the excluded window gets a
    fractional pass even if one interval is fully inside.

    Computes union of excluded intervals first to avoid double-counting
    when multiple excl intervals overlap each other.
    """
    if not d_ivs or not excl_ivs:
        return 0.0
    # Union the excluded intervals (sorted, merged).
    sorted_excl = sorted(
        [(e.earliest_us, e.latest_us) for e in excl_ivs], key=lambda p: p[0]
    )
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted_excl:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    total_overlap = 0
    total_doc = 0
    for di in d_ivs:
        d_dur = di.latest_us - di.earliest_us
        if d_dur <= 0:
            d_dur = 1
        total_doc += d_dur
        for lo, hi in merged:
            inter_lo = max(di.earliest_us, lo)
            inter_hi = min(di.latest_us, hi)
            inter = max(0, inter_hi - inter_lo)
            total_overlap += inter
    if total_doc <= 0:
        return 0.0
    return min(1.0, total_overlap / total_doc)


# ===========================================================================
# Filter (cheap binary predicate for hybrid pool's filter channel)
# ===========================================================================


def doc_passes_filter(
    doc_intervals: list[Interval],
    valid_includes: list[tuple[str, list[Interval]]],
    valid_excludes: list[list[Interval]],
) -> bool:
    """EXISTS-overlap filter for the hybrid pool's filter channel.

    Matches production `_v3_q1_retrieval_ablation.doc_passes_filter`:
      - Includes are OR: passes if ANY include leaf has an EXISTS match.
      - Excludes are AND-NOT: passes only if NO exclude leaf has EXISTS.

    Docs with no extracted intervals trivially satisfy NOT-EXISTS for any
    exclude clause, but cannot satisfy any include EXISTS clause; they
    pass when there are no includes (filter is exclude-only or empty).
    """
    if valid_includes:
        passed = False
        for relation, anchor_ivs in valid_includes:
            if constraint_factor_for_doc(doc_intervals, anchor_ivs, relation) >= 1.0:
                passed = True
                break
        if not passed:
            return False
    for anchor_ivs in valid_excludes:
        # disjoint's EXISTS-match uses "intersect" semantics on the same anchor.
        if constraint_factor_for_doc(doc_intervals, anchor_ivs, "intersect") >= 1.0:
            return False
    return True


# ===========================================================================
# Hybrid pool builder
# ===========================================================================


def build_pool(
    sem_scores: dict[str, float],
    all_dids: list[str],
    eligible_filt: list[str],
    pool_size: int = 10,
) -> list[str]:
    """Hybrid: top-(K/2) raw-semantic U top-(K/2) filter-survivor-semantic.

    Tops up from raw semantic when the two halves overlap heavily so the
    pool always reaches `pool_size` (e.g., when the filter is a no-op or
    when the same docs rank top-K in both channels). This matches the
    v5.1 production builder.
    """
    half = max(1, pool_size // 2)
    rs_top = _topk_in_set(sem_scores, all_dids, half)
    rsf_top = _topk_in_set(sem_scores, eligible_filt, half) if eligible_filt else []
    pool = list(dict.fromkeys(rs_top + rsf_top))
    if len(pool) < pool_size:
        extra = _topk_in_set(sem_scores, all_dids, pool_size * 2)
        for d in extra:
            if d not in pool:
                pool.append(d)
                if len(pool) >= pool_size:
                    break
    return pool[:pool_size]


def _topk_in_set(scores: dict[str, float], ids: list[str], k: int) -> list[str]:
    cand = [(d, scores.get(d, -1e9)) for d in ids if d in scores]
    cand.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in cand[:k]]


# ===========================================================================
# Recency
# ===========================================================================


def recency_scores(
    doc_bundles_for_rec: dict[str, list[dict]],
    doc_ref_us: dict[str, int],
    direction: str = "latest",
) -> dict[str, float]:
    """Per-doc recency score in [0, 1] where HIGHER = better match for
    the requested direction.

    Uses RANK-based normalization: each doc's score is its position in
    the sorted order of anchors divided by N-1. Rank-based is robust
    to outliers — a single doc with a very-distant anchor can't
    compress everyone else's scores into a tiny range. Ties (docs
    sharing an anchor value) receive the average of their tied
    positions.

    Linear `(a - lo) / span` normalization was outlier-sensitive: with
    a candidate set of 5 docs at 2020-2024 plus 1 outlier at 2090,
    the 2020-2024 docs collapsed into scores 0.000-0.057,
    indistinguishable from each other. Rank-based gives them
    0.0/0.2/0.4/0.6/0.8 (outlier 1.0), preserving the relative
    ordering of the relevant cluster.

    For each candidate, the per-doc anchor is the midpoint of the
    interval that best matches the direction: max-midpoint for
    "latest", min-midpoint for "earliest". Falls back to the doc's
    `ref_time` when the doc has no extracted intervals. Multi-
    interval docs are handled correctly in both directions — a doc
    with intervals at 2020 AND 2024 outranks a single-2023 doc on
    BOTH "latest" (max-midpoint mid-2024 wins) and "earliest"
    (min-midpoint mid-2020 wins).

    Midpoint as the per-interval anchor was empirically validated in
    project_best_us_dispensable (matches LLM-supplied point estimates
    on 6/7 benches).
    """
    if direction not in ("latest", "earliest"):
        raise ValueError(
            f"direction must be 'latest' or 'earliest'; got {direction!r}"
        )
    pick_max = direction == "latest"
    anchors: dict[str, int] = {}
    for did, ref_us in doc_ref_us.items():
        bundles = doc_bundles_for_rec.get(did, [])
        anchor: int | None = None
        for b in bundles:
            for iv in b.get("intervals", []) or []:
                cand = (iv.earliest_us + iv.latest_us) // 2
                if anchor is None:
                    anchor = cand
                elif pick_max and cand > anchor:
                    anchor = cand
                elif not pick_max and cand < anchor:
                    anchor = cand
        anchors[did] = anchor if anchor is not None else ref_us
    n = len(anchors)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(anchors)): 1.0}
    sorted_items = sorted(anchors.items(), key=lambda kv: kv[1])
    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_items[j + 1][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        score = avg_rank / (n - 1)
        if not pick_max:
            score = 1.0 - score
        for k in range(i, j + 1):
            out[sorted_items[k][0]] = score
        i = j + 1
    return out


# Back-compat alias — older callers (and the validation script in
# research/) import the old name. New code should use `recency_scores`.
linear_recency_scores = recency_scores


# ===========================================================================
# Normalization
# ===========================================================================


def normalize_dict(d: dict[str, float]) -> dict[str, float]:
    """Min-max normalize values to [0, 1]."""
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def normalize_rerank_full(
    rerank_partial: dict[str, float],
    all_doc_ids: list[str],
    tail_score: float = 0.0,
) -> dict[str, float]:
    """Normalize rerank scores to [0,1] over the partial set, fill the
    rest with `tail_score`."""
    if not rerank_partial:
        return dict.fromkeys(all_doc_ids, tail_score)
    vals = list(rerank_partial.values())
    rmin, rmax = min(vals), max(vals)
    span = (rmax - rmin) or 1.0
    out = {}
    for did in all_doc_ids:
        if did in rerank_partial:
            out[did] = (rerank_partial[did] - rmin) / span
        else:
            out[did] = tail_score
    return out


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
