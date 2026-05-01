"""F5 — Allen-relation retrieval.

Given a query with relation ``R`` and anchor interval ``A`` (resolved via
:mod:`event_resolver`), score each doc by the best match among its
AllenExpressions. A candidate doc-side expression ``d`` contributes one
of two kinds of signal:

1. **Absolute-interval signal.**  If ``d.time`` has a concrete interval
   (i.e., the doc contains an absolute/relative date that was resolved
   by pass-2), we check Allen(R, A, d.time.interval) and score by the
   relation-specific rule below.

2. **Anchor-match signal.**  If ``d`` is itself a relational
   expression (``d.relation`` + ``d.anchor`` both set) and its anchor
   resolves to an interval ``A_d``, we compare ``A_d`` to the query
   anchor ``A``.  If ``A_d`` ≈ ``A`` (same event), the doc's
   effective-interval is whatever region the doc's relation defines
   against ``A_d``, and we check whether that region satisfies the
   query's relation.

The function is given ``resolve_anchor(span)`` as a sync callable that
returns an absolute ``_Iv`` interval or ``None`` (the caller wraps an
async event-resolver behind a memoizing sync wrapper).

Relation semantics (``A`` is the query anchor interval
``[A_earliest, A_latest]``; ``d`` is a candidate interval):

- before:   d.latest <= A.earliest
- after:    d.earliest >= A.latest
- during:   d.earliest >= A.earliest AND d.latest <= A.latest
- overlaps: d.earliest < A.latest AND A.earliest < d.latest
- contains: d.earliest <= A.earliest AND d.latest >= A.latest

Ranking:
- before:   closer to A.earliest = higher.
- after:    closer to A.latest   = higher.
- during:   full containment = 1.0; partial gets 0.1 * overlap frac.
- contains: tighter d covers A = higher.
- overlaps: overlap fraction / max span.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from allen_schema import AllenExpression
from schema import TimeExpression, to_us


@dataclass
class _Iv:
    earliest: int
    latest: int


def te_interval(te: TimeExpression) -> _Iv | None:
    if te.kind == "instant" and te.instant:
        return _Iv(to_us(te.instant.earliest), to_us(te.instant.latest))
    if te.kind == "interval" and te.interval:
        return _Iv(
            to_us(te.interval.start.earliest),
            to_us(te.interval.end.latest),
        )
    if te.kind == "recurrence" and te.recurrence:
        return _Iv(
            to_us(te.recurrence.dtstart.earliest),
            to_us(te.recurrence.dtstart.latest),
        )
    return None


Relation = Literal["before", "after", "during", "overlaps", "contains"]


def _overlap(a: _Iv, b: _Iv) -> int:
    return max(0, min(a.latest, b.latest) - max(a.earliest, b.earliest))


def _span(a: _Iv) -> int:
    return max(1, a.latest - a.earliest)


def _intervals_equal(a: _Iv, b: _Iv, tol_frac: float = 0.05) -> bool:
    """Approximate equality. Two intervals are 'the same event' if
    their centers are within tol_frac of the max span."""
    sa = _span(a)
    sb = _span(b)
    max_span = max(sa, sb)
    ca = 0.5 * (a.earliest + a.latest)
    cb = 0.5 * (b.earliest + b.latest)
    # allow 5% span OR 1 day of slop
    tol = max(tol_frac * max_span, 86400 * 1_000_000)
    return abs(ca - cb) < tol and abs(sa - sb) < tol


def _allen_score(relation: Relation, anchor: _Iv, cand: _Iv) -> float:
    """Return 0 if the Allen relation does NOT hold, else a positive
    score where higher is better."""
    tol = max(1, int(0.001 * _span(anchor)))
    if relation == "before":
        if cand.latest <= anchor.earliest + tol:
            dist = max(0, anchor.earliest - cand.latest)
            norm = _span(anchor)
            return 1.0 / (1.0 + dist / norm)
        return 0.0
    if relation == "after":
        if cand.earliest + tol >= anchor.latest:
            dist = max(0, cand.earliest - anchor.latest)
            norm = _span(anchor)
            return 1.0 / (1.0 + dist / norm)
        return 0.0
    if relation == "during":
        if (
            cand.earliest + tol >= anchor.earliest
            and cand.latest <= anchor.latest + tol
        ):
            ov = _overlap(anchor, cand)
            return ov / _span(cand)
        ov = _overlap(anchor, cand)
        if ov > 0:
            return 0.1 * (ov / _span(cand))
        return 0.0
    if relation == "contains":
        if (
            cand.earliest <= anchor.earliest + tol
            and cand.latest + tol >= anchor.latest
        ):
            # Reject pure equality — that's "equals", not "contains".
            # Require the candidate to STRICTLY extend beyond the
            # anchor on at least one side.
            extends = (anchor.earliest - cand.earliest) > tol or (
                cand.latest - anchor.latest
            ) > tol
            if not extends:
                return 0.0
            return _span(anchor) / _span(cand)
        return 0.0
    if relation == "overlaps":
        ov = _overlap(anchor, cand)
        if ov > 0:
            return ov / max(_span(cand), _span(anchor))
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Relational-doc handling
# ---------------------------------------------------------------------------
def _derive_doc_interval_from_relational(
    doc_ae: AllenExpression,
    doc_anchor_iv: _Iv,
) -> _Iv | None:
    """If a doc AllenExpression is itself relational AND its own time
    field has a real (resolved) interval, prefer that. Otherwise,
    derive an effective interval from ``(relation, doc_anchor_iv)``.

    For example, "before my wedding" with no concrete date yields the
    open-left interval ``(-∞, wedding.earliest)``. We use a wide but
    finite interval so downstream scoring stays numeric.
    """
    rel = doc_ae.relation
    if rel is None:
        return None
    if rel == "before":
        # Doc events occur BEFORE the anchor. We give it a wide region
        # [anchor.earliest - 100y, anchor.earliest).
        width = 100 * 365 * 86400 * 1_000_000
        return _Iv(
            earliest=doc_anchor_iv.earliest - width,
            latest=doc_anchor_iv.earliest,
        )
    if rel == "after":
        width = 100 * 365 * 86400 * 1_000_000
        return _Iv(
            earliest=doc_anchor_iv.latest,
            latest=doc_anchor_iv.latest + width,
        )
    if rel == "during":
        return doc_anchor_iv  # doc event is somewhere INSIDE the anchor
    if rel == "overlaps":
        return doc_anchor_iv  # generous: same interval
    if rel == "contains":
        return doc_anchor_iv  # doc wraps at least the anchor
    return None


def _score_doc_ae(
    relation: Relation,
    query_anchor_iv: _Iv,
    doc_ae: AllenExpression,
    resolve_anchor: Callable[[str], _Iv | None],
    *,
    is_anchor_doc: bool = False,
) -> float:
    """Best Allen score for one doc AllenExpression against the query.

    If ``is_anchor_doc`` is True, we're scoring the doc that THIS
    resolver used to define the anchor — so an exact-equality absolute
    interval is not informative (it's just the anchor definition).
    """
    best = 0.0

    # Absolute-interval signal
    abs_iv = te_interval(doc_ae.time)
    # Skip sentinel intervals (earliest == latest, single point).
    is_real_abs = abs_iv is not None and abs_iv.latest > abs_iv.earliest
    if is_real_abs:
        # If this IS the anchor doc and the abs interval equals the
        # anchor, it's the anchor's own declaration — skip this
        # contribution (we still give the doc a chance via other exprs).
        equals_anchor = is_anchor_doc and _intervals_equal(
            abs_iv, query_anchor_iv, tol_frac=0.01
        )
        if not equals_anchor:
            s = _allen_score(relation, query_anchor_iv, abs_iv)
            if s > best:
                best = s

    # Anchor-match signal (only if doc is itself relational)
    if doc_ae.relation is not None and doc_ae.anchor is not None:
        doc_anchor_iv: _Iv | None = None
        if doc_ae.anchor.resolved is not None:
            doc_anchor_iv = te_interval(doc_ae.anchor.resolved)
        if doc_anchor_iv is None:
            doc_anchor_iv = resolve_anchor(doc_ae.anchor.span)
        if doc_anchor_iv is not None:
            # If the doc and query reference the SAME anchor event AND
            # the doc's own relation is COMPATIBLE with the query
            # relation, strong match.
            same_anchor = _intervals_equal(doc_anchor_iv, query_anchor_iv)
            if same_anchor and doc_ae.relation == relation:
                # A doc with "during X" is a direct answer to "during X"
                # even if it has no absolute date.
                if best < 0.99:
                    best = 0.99
            elif same_anchor:
                # Different Allen relation but same anchor — derive a
                # concrete region and score.
                derived = _derive_doc_interval_from_relational(doc_ae, doc_anchor_iv)
                if derived is not None:
                    s = _allen_score(relation, query_anchor_iv, derived)
                    if s > best:
                        best = s
            else:
                # Different anchor — derive a region and score.
                derived = _derive_doc_interval_from_relational(doc_ae, doc_anchor_iv)
                if derived is not None:
                    s = _allen_score(relation, query_anchor_iv, derived)
                    if s > best:
                        best = s * 0.5
    return best


def allen_retrieve(
    relation: Relation,
    anchor_te: TimeExpression,
    doc_exprs_by_doc: dict[str, list[AllenExpression]],
    *,
    resolve_anchor: Callable[[str], _Iv | None] | None = None,
    anchor_doc_id: str | None = None,
) -> dict[str, float]:
    """Rank docs by Allen relation against ``anchor_te``.

    ``resolve_anchor`` is a function that maps an anchor surface to an
    absolute ``_Iv`` interval (e.g., via EventResolver). If omitted,
    anchor-match signal is skipped (absolute-interval only).

    ``anchor_doc_id`` (optional) is the doc the resolver used to
    derive the anchor interval. For that doc, we suppress the
    equality-with-anchor signal (the doc IS the anchor's definition).
    """
    anchor_iv = te_interval(anchor_te)
    out: dict[str, float] = {}
    if anchor_iv is None:
        return out
    _resolve = resolve_anchor or (lambda _span: None)
    for doc_id, exprs in doc_exprs_by_doc.items():
        best = 0.0
        is_anchor = doc_id == anchor_doc_id
        for ae in exprs:
            s = _score_doc_ae(
                relation, anchor_iv, ae, _resolve, is_anchor_doc=is_anchor
            )
            if s > best:
                best = s
        if best > 0:
            out[doc_id] = best
    return out
