"""Co-temporal document graph.

For each document, compute edges to other documents whose time expressions
"co-mention" the same temporal ground, where co-mention is defined by:

- multi_axis_score > THRESHOLD (fuzzy axis overlap), OR
- interval overlap > 0.5 (hard Jaccard), OR
- same (year, month) axis-tag bucket

Edge weight is max over all time-expression pairs between the two docs.

Stored as sqlite adjacency: cotemporal_edges(doc_id, neighbor_id, weight,
shared_expr_id). Per-node degree capped at M (default 20) by weight.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from multi_axis_scorer import axis_score as axis_score_fn
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes
from schema import TimeExpression
from scorer import Interval, score_jaccard_composite

GRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cotemporal_edges (
    edge_id      INTEGER PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    neighbor_id  TEXT NOT NULL,
    weight       REAL NOT NULL,
    reason       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edge_doc ON cotemporal_edges(doc_id);
CREATE INDEX IF NOT EXISTS idx_edge_nb ON cotemporal_edges(neighbor_id);
"""


@dataclass
class DocTemporalBundle:
    doc_id: str
    intervals: list[Interval]
    axes_merged: dict[str, AxisDistribution]
    multi_tags: set[str]


def bundle_from_tes(
    doc_id: str, tes: list[TimeExpression], flatten_fn
) -> DocTemporalBundle:
    intervals: list[Interval] = []
    axes_per: list[dict[str, AxisDistribution]] = []
    multi_tags: set[str] = set()
    for te in tes:
        intervals.extend(flatten_fn(te))
        ax = axes_for_expression(te)
        axes_per.append(ax)
        multi_tags |= tags_for_axes(ax)
    axes_merged = (
        merge_axis_dists(axes_per)
        if axes_per
        else {a: AxisDistribution(axis=a, values={}, informative=False) for a in AXES}
    )
    return DocTemporalBundle(
        doc_id=doc_id,
        intervals=intervals,
        axes_merged=axes_merged,
        multi_tags=multi_tags,
    )


def _interval_best(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    if not q_ivs or not d_ivs:
        return 0.0
    best = 0.0
    for qi in q_ivs:
        for di in d_ivs:
            s = score_jaccard_composite(qi, di)
            if s > best:
                best = s
    return best


def edge_weight(a: DocTemporalBundle, b: DocTemporalBundle) -> tuple[float, str]:
    """Return (weight, reason)."""
    # interval best-pair Jaccard
    iv = _interval_best(a.intervals, b.intervals)
    # fuzzy axis
    ax = axis_score_fn(a.axes_merged, b.axes_merged)
    # tag Jaccard
    tg = tag_score(a.multi_tags, b.multi_tags)

    # combined fuzzy weight
    weight = 0.5 * iv + 0.35 * ax + 0.15 * tg
    # reason label
    if iv >= 0.5:
        reason = "interval"
    elif tg > 0.0:
        reason = "tag"
    elif ax > 0.3:
        reason = "axis"
    else:
        reason = "weak"
    return weight, reason


def _has_temporal_signal(bundle: DocTemporalBundle) -> bool:
    if bundle.intervals:
        return True
    for ax in bundle.axes_merged.values():
        if ax.informative:
            return True
    return False


class CotemporalGraph:
    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(GRAPH_SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        self.conn.executescript("DROP TABLE IF EXISTS cotemporal_edges;")
        self.conn.executescript(GRAPH_SCHEMA_SQL)
        self.conn.commit()

    def build(
        self,
        doc_bundles: dict[str, DocTemporalBundle],
        threshold: float = 0.3,
        cap_per_node: int = 20,
        doc_embs: dict[str, np.ndarray] | None = None,
        sem_fallback_top_k: int = 5,
        sem_fallback_min_cos: float = 0.45,
    ) -> dict[str, Any]:
        """Build edges over all doc pairs. Returns stats.

        If ``doc_embs`` is provided, ALSO emit "semantic bridge" edges:
        for any doc with no temporal signal (no intervals, no informative
        axes), connect it to its top-k most semantically-similar docs that
        DO have temporal signal, with weight = cosine. These edges are
        tagged reason="sem_bridge" so they can be audited separately.
        Without this, no-date docs are orphaned in the graph.
        """
        self.reset()
        ids = sorted(doc_bundles.keys())
        n = len(ids)
        all_edges: list[tuple[str, str, float, str]] = []
        raw_pairs_considered = 0
        raw_pairs_above = 0

        for i in range(n):
            for j in range(i + 1, n):
                a = doc_bundles[ids[i]]
                b = doc_bundles[ids[j]]
                if not a.intervals and not b.intervals:
                    continue
                raw_pairs_considered += 1
                w, reason = edge_weight(a, b)
                if w >= threshold:
                    raw_pairs_above += 1
                    all_edges.append((a.doc_id, b.doc_id, w, reason))
                    all_edges.append((b.doc_id, a.doc_id, w, reason))

        # --- Semantic bridge edges for no-temporal-signal docs ---
        sem_bridge_count = 0
        if doc_embs is not None:
            import numpy as np  # local import to avoid top-level dep surprise

            embs = {d: doc_embs[d] for d in ids if d in doc_embs}
            norms = {d: (np.linalg.norm(v) or 1e-9) for d, v in embs.items()}
            temp_docs = [
                d for d in ids if _has_temporal_signal(doc_bundles[d]) and d in embs
            ]
            no_temp = [
                d for d in ids if not _has_temporal_signal(doc_bundles[d]) and d in embs
            ]
            for d in no_temp:
                v = embs[d]
                vn = norms[d]
                sims = []
                for t in temp_docs:
                    tv = embs[t]
                    tn = norms[t]
                    cos = float(np.dot(v, tv) / (vn * tn))
                    sims.append((t, cos))
                sims.sort(key=lambda x: x[1], reverse=True)
                added = 0
                for t, cos in sims:
                    if cos < sem_fallback_min_cos:
                        break
                    if added >= sem_fallback_top_k:
                        break
                    all_edges.append((d, t, cos, "sem_bridge"))
                    all_edges.append((t, d, cos, "sem_bridge"))
                    added += 1
                    sem_bridge_count += 2

        # Cap per node to top-cap_per_node by weight
        by_doc: dict[str, list[tuple[str, float, str]]] = {}
        for d, nb, w, r in all_edges:
            by_doc.setdefault(d, []).append((nb, w, r))
        kept: list[tuple[str, str, float, str]] = []
        for d, lst in by_doc.items():
            lst.sort(key=lambda x: x[1], reverse=True)
            for nb, w, r in lst[:cap_per_node]:
                kept.append((d, nb, w, r))

        # Write
        self.conn.executemany(
            "INSERT INTO cotemporal_edges (doc_id, neighbor_id, weight, reason) "
            "VALUES (?, ?, ?, ?)",
            kept,
        )
        self.conn.commit()

        # Stats
        out_deg: dict[str, int] = {}
        for d, _, _, _ in kept:
            out_deg[d] = out_deg.get(d, 0) + 1
        avg_deg = sum(out_deg.values()) / n if n else 0.0
        max_deg = max(out_deg.values(), default=0)
        hub_docs = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:10]

        # How many kept edges were sem_bridge?
        kept_sem_bridge = sum(1 for _, _, _, r in kept if r == "sem_bridge")
        return {
            "nodes": n,
            "edges_directed": len(kept),
            "edges_undirected": len(kept) // 2,
            "raw_pairs_considered": raw_pairs_considered,
            "raw_pairs_above_threshold": raw_pairs_above,
            "avg_degree": avg_deg,
            "max_degree": max_deg,
            "hub_docs_top10": hub_docs,
            "sem_bridge_edges_kept": kept_sem_bridge,
            "sem_bridge_edges_total": sem_bridge_count,
            "threshold": threshold,
            "cap_per_node": cap_per_node,
        }

    def neighbors(self, doc_id: str, limit: int = 10) -> list[tuple[str, float, str]]:
        cur = self.conn.execute(
            "SELECT neighbor_id, weight, reason FROM cotemporal_edges "
            "WHERE doc_id = ? ORDER BY weight DESC LIMIT ?",
            (doc_id, limit),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    def all_degree(self) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT doc_id, COUNT(*) FROM cotemporal_edges GROUP BY doc_id"
        )
        return {r[0]: r[1] for r in cur.fetchall()}
