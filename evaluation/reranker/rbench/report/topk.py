"""Recall with and without reranking across top-k: the asker's metric, in its own terms.

The asker asked for "Recall with and without CE reranking across different Top-k settings".
Top-k can mean either EventMemory setting, so both are reported (the user's decision,
2026-11-09):
  L  vector_search_limit: the seeds retrieved, every one of them reranked
  k  the seeds kept after (re)ranking (max_num_segments at expand_context 0)
Without a reranker top-k is both at once (the first k seeds of the vector search). Recall is
LongMemEval's notion, per question: the share of its evidence turns found within the first k
seeds kept, read in full (share) or touched (share_touch); every-gold beside them (all turns
in full). Averaged over questions.

At k = L everything retrieved is kept, so reranking cannot change what is read, only its
order: the gain there is exactly 0 by construction. At k < L the reranker chooses which k of
the L seeds to keep. L and k are chosen once for all questions (never per question). The
setting is EventMemory's default segment length at expand_context 0, as the asker's.
"""

from __future__ import annotations

import numpy as np

from rbench.report import frontier
from rbench.report.data import Results
from rbench.report.stats import interval

READINGS = {"share": "share_full_k", "share_touch": "share_touch_k", "every": "every_ctx"}
NAMES = {"share": "evidence turns found in full", "share_touch": "evidence turns found (touched)",
         "every": "all evidence in full (every-gold)"}
KS = (1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300)
SLICE_KS = (5, 10, 20)


def available(r: Results) -> bool:
    return "share_full_k" in r.df


def recall(rows, reading: str, k: int) -> np.ndarray:
    """Per question: the share of its evidence turns found within k seeds (every: 1 if all are)."""
    column = READINGS[reading]
    if reading == "every":
        return (rows[column].to_numpy() <= k).astype(float)
    return np.asarray([float(np.mean(np.asarray(x) <= k)) if len(x) else 0.0 for x in rows[column]])


def grid(r: Results, model: str, reading: str) -> dict:
    """{L values, k values, reranked [L, k], first stage [k], gain [L, k], low and high [L, k]}
    at the core chunking and expand_context 0 (nan where k > L)."""
    chunking, _ = frontier.setting(r)
    limits = [int(d[1:]) for d in r.depths_at(chunking, model, "rerank")]
    base = r.rows(chunking, "first_stage")
    ks = [k for k in KS if k <= max(limits, default=0)]
    first = np.asarray([recall(base, reading, k).mean() for k in ks])
    shape = (len(limits), len(ks))
    reranked, gain, low, high = (np.full(shape, np.nan) for _ in range(4))
    clusters = base.cluster.to_numpy()
    for i, L in enumerate(limits):
        rows = r.rows(chunking, f"rerank:{model}@n{L}")
        for j, k in enumerate(ks):
            if k > L:
                continue
            a, b = recall(rows, reading, k), recall(base, reading, k)
            reranked[i, j], gain[i, j] = a.mean(), (a - b).mean()
            low[i, j], high[i, j] = interval(a - b, clusters) if k < L else (0.0, 0.0)
    return {"limits": limits, "ks": ks, "reranked": reranked, "first": first, "gain": gain, "low": low, "high": high}


def best_limit(g: dict) -> list[int | None]:
    """Per k, the L whose reranked recall is highest (chosen once for all questions)."""
    out = []
    for j in range(len(g["ks"])):
        col = g["reranked"][:, j]
        out.append(None if np.all(np.isnan(col)) else g["limits"][int(np.nanargmax(col))])
    return out
