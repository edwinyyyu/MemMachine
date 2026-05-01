"""Per-axis Bhattacharyya scoring + combined multi-axis score.

Given two axis-dist dicts (query and doc), compute:

- per-axis Bhattacharyya coefficient BC(p, q) = sum_i sqrt(p_i * q_i).
- combined axis_score = geomean over axes that are informative in EITHER
  query or doc. Non-informative axes contribute a neutral 1.0 -> included
  only when at least one side is informative on that axis.
- tag_score = Jaccard over {axis}:{value} tag sets (informative only).
- interval_score = provided externally (reuse existing Jaccard composite).
- total = alpha*interval + beta*axis + gamma*tag
"""

from __future__ import annotations

import math

from axis_distributions import AXES, AxisDistribution


def bhattacharyya(p: dict, q: dict) -> float:
    if not p or not q:
        return 0.0
    s = 0.0
    for k, pv in p.items():
        qv = q.get(k, 0.0)
        if pv > 0 and qv > 0:
            s += math.sqrt(pv * qv)
    return s


def per_axis_scores(
    q_axes: dict[str, AxisDistribution],
    d_axes: dict[str, AxisDistribution],
) -> tuple[dict[str, float], list[str]]:
    """Return (per_axis_score, used_axes).

    An axis is "used" iff either side is informative on it. A used axis
    contributes BC(p_q, p_d) if BOTH sides have non-empty values; if one side
    is informative but the other is empty/uninformative, contributes 0 (no
    overlap) — this makes the score discriminative on axes that the query
    constrains but the doc does not satisfy.
    """
    per: dict[str, float] = {}
    used: list[str] = []
    for axis in AXES:
        q_ad = q_axes.get(axis)
        d_ad = d_axes.get(axis)
        q_inf = q_ad is not None and q_ad.informative
        d_inf = d_ad is not None and d_ad.informative
        if not q_inf and not d_inf:
            continue
        used.append(axis)
        if q_ad is None or d_ad is None or not q_ad.values or not d_ad.values:
            per[axis] = 0.0
            continue
        per[axis] = bhattacharyya(q_ad.values, d_ad.values)
    return per, used


def axis_score(
    q_axes: dict[str, AxisDistribution],
    d_axes: dict[str, AxisDistribution],
    skip_axis: str | None = None,
) -> float:
    per, used = per_axis_scores(q_axes, d_axes)
    used = [a for a in used if a != skip_axis]
    if not used:
        return 1.0
    # Geomean requires log. To avoid log(0), floor at 1e-6.
    s = 0.0
    for a in used:
        v = max(per.get(a, 0.0), 1e-6)
        s += math.log(v)
    return math.exp(s / len(used))


def tag_score(q_tags: set[str], d_tags: set[str]) -> float:
    if not q_tags and not d_tags:
        return 0.0
    if not q_tags or not d_tags:
        return 0.0
    inter = len(q_tags & d_tags)
    union = len(q_tags | d_tags)
    return inter / union if union else 0.0


def combined_score(
    interval_score_val: float,
    q_axes: dict[str, AxisDistribution],
    d_axes: dict[str, AxisDistribution],
    q_tags: set[str],
    d_tags: set[str],
    alpha: float,
    beta: float,
    gamma: float,
    skip_axis: str | None = None,
) -> tuple[float, float, float]:
    """Returns (total, axis_score, tag_score)."""
    a_sc = axis_score(q_axes, d_axes, skip_axis=skip_axis)
    t_sc = tag_score(q_tags, d_tags)
    total = alpha * interval_score_val + beta * a_sc + gamma * t_sc
    return total, a_sc, t_sc
