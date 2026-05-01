"""E1 — Retrieval & end-to-end driver for event-time binding experiment.

For each query, extract (event, time) pairs (via event_binding.py), embed
events, resolve times. For each binding in the event_store, compute
per-pair score = alpha * cosine(q_event, d_event) + beta * temporal_overlap.
Aggregate per doc as SUM over all q-pair x d-binding combinations.

Sweeps alpha in {0.3, 0.5, 0.7}. Reports R@5/@10/MRR/NDCG@10 vs base
hybrid (T_and_S) on same 55 queries + critical top-1.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from advanced_common import (
    DATA_DIR,
    RESULTS_DIR,
    Embedder,
    LLMCaller,
    cosine,
    load_jsonl,
    mean,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from event_binding import extract_pairs, resolve_time
from event_store import EventStore
from schema import parse_iso, to_us

DB_PATH = Path(__file__).resolve().parent / "cache" / "temporal_advanced.db"


def _parse_iso_safe(s: str) -> datetime | None:
    try:
        return parse_iso(s)
    except Exception:
        return None


def _iv_from_resolution(res: dict[str, Any]) -> tuple[int, int, int | None, str] | None:
    e = _parse_iso_safe(res.get("earliest", ""))
    l = _parse_iso_safe(res.get("latest", ""))
    if e is None or l is None:
        return None
    b = _parse_iso_safe(res.get("best", "")) if res.get("best") else None
    e_us = to_us(e)
    l_us = to_us(l)
    b_us = to_us(b) if b is not None else None
    # Ensure earliest < latest
    if l_us <= e_us:
        l_us = e_us + 1
    gran = res.get("granularity", "day")
    return e_us, l_us, b_us, gran


def _overlap(q_e: int, q_l: int, d_e: int, d_l: int) -> float:
    """Jaccard overlap on two intervals in microseconds."""
    if q_e >= d_l or d_e >= q_l:
        return 0.0
    inter = min(q_l, d_l) - max(q_e, d_e)
    union = max(q_l, d_l) - min(q_e, d_e)
    if union <= 0:
        return 0.0
    return inter / union


async def _extract_and_embed(
    llm: LLMCaller,
    embedder: Embedder,
    text: str,
    ref_time: datetime,
) -> list[dict[str, Any]]:
    """Returns list of bindings for one piece of text:

    [{"event_span": str|None, "event_vec": np.ndarray|None,
      "earliest_us": int, "latest_us": int, "best_us": int|None,
      "granularity": str}, ...]
    """
    pairs = await extract_pairs(llm, text, ref_time)

    # Resolve each time + embed each event span concurrently.
    async def resolve_one(p: dict[str, Any]):
        surface = p.get("time_surface") or ""
        if not surface:
            return None
        r = await resolve_time(llm, surface, ref_time, text)
        return (p, r)

    resolutions = await asyncio.gather(*[resolve_one(p) for p in pairs])
    out: list[dict[str, Any]] = []
    # Kick off embeddings in parallel
    event_spans = []
    for rp in resolutions:
        if rp is None:
            continue
        p, r = rp
        if r is None:
            continue
        iv = _iv_from_resolution(r)
        if iv is None:
            continue
        event_spans.append(p.get("event_span"))
    # Embed non-null event spans
    emb_coros = []
    for ev in event_spans:
        if ev:
            emb_coros.append(embedder.embed(ev))
        else:
            emb_coros.append(asyncio.sleep(0, result=None))
    embeddings = await asyncio.gather(*emb_coros) if emb_coros else []

    idx = 0
    for rp in resolutions:
        if rp is None:
            continue
        p, r = rp
        if r is None:
            continue
        iv = _iv_from_resolution(r)
        if iv is None:
            continue
        e_us, l_us, b_us, gran = iv
        ev = p.get("event_span")
        vec = embeddings[idx] if idx < len(embeddings) else None
        idx += 1
        out.append(
            {
                "event_span": ev,
                "event_vec": vec,
                "earliest_us": e_us,
                "latest_us": l_us,
                "best_us": b_us,
                "granularity": gran,
            }
        )
    return out


async def build_store(
    docs: list[dict],
    llm: LLMCaller,
    embedder: Embedder,
    db_path: Path,
) -> EventStore:
    if db_path.exists():
        db_path.unlink()
    store = EventStore(db_path)

    async def one(d: dict) -> tuple[str, list[dict]]:
        bs = await _extract_and_embed(
            llm, embedder, d["text"], parse_iso(d["ref_time"])
        )
        return d["doc_id"], bs

    tasks = [one(d) for d in docs]
    results = await asyncio.gather(*tasks)
    for doc_id, bs in results:
        for b in bs:
            store.insert(
                doc_id,
                b.get("event_span"),
                b.get("event_vec"),
                b["earliest_us"],
                b["latest_us"],
                b.get("best_us"),
                b.get("granularity"),
            )
    return store


def rank_for_query(
    q_bindings: list[dict[str, Any]],
    d_bindings_rows: list[tuple],
    *,
    alpha: float,
) -> list[tuple[str, float]]:
    """d_bindings_rows: rows returned by EventStore.all_bindings()."""
    beta = 1.0 - alpha

    # Parse once
    d_parsed = []
    for (
        binding_id,
        doc_id,
        event_span,
        event_vec_str,
        e_us,
        l_us,
        b_us,
        gran,
    ) in d_bindings_rows:
        vec = EventStore.parse_vec(event_vec_str)
        d_parsed.append((doc_id, event_span, vec, e_us, l_us))

    doc_scores: dict[str, float] = {}
    for qb in q_bindings:
        q_vec: np.ndarray | None = qb.get("event_vec")
        q_e, q_l = qb["earliest_us"], qb["latest_us"]
        for doc_id, _ev, d_vec, d_e, d_l in d_parsed:
            if q_vec is not None and d_vec is not None:
                sem = cosine(q_vec, d_vec)
            else:
                sem = 0.0
            ov = _overlap(q_e, q_l, d_e, d_l)
            s = alpha * sem + beta * ov
            # Only count positive-signal pairs to avoid background noise
            # (negative cosines should not subtract from doc score).
            if s > 0.0:
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + s
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


async def main() -> None:
    docs = load_jsonl(DATA_DIR / "docs.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "gold.jsonl")
    }
    critical_pairs = json.loads((DATA_DIR / "critical_pairs.json").read_text())
    crit_map = {q_id: doc_id for (doc_id, q_id) in critical_pairs}

    llm = LLMCaller(concurrency=10)
    embedder = Embedder(concurrency=10)

    print(f"E1: extracting event bindings for {len(docs)} docs...")
    store = await build_store(docs, llm, embedder, DB_PATH)
    llm.save()
    embedder.save()

    print(f"E1: extracting event bindings for {len(queries)} queries...")

    async def q_one(q: dict) -> tuple[str, list[dict]]:
        bs = await _extract_and_embed(
            llm, embedder, q["text"], parse_iso(q["ref_time"])
        )
        return q["query_id"], bs

    q_results = await asyncio.gather(*(q_one(q) for q in queries))
    llm.save()
    embedder.save()

    q_bindings = {qid: bs for qid, bs in q_results}
    d_rows = store.all_bindings()

    report: dict[str, Any] = {"alpha_sweep": {}}
    for alpha in [0.3, 0.5, 0.7]:
        rec5s, rec10s, mrrs, ndcgs = [], [], [], []
        crit_top1 = 0
        for q in queries:
            qid = q["query_id"]
            qbs = q_bindings.get(qid, [])
            ranked_pairs = rank_for_query(qbs, d_rows, alpha=alpha)
            ranked = [d for d, _ in ranked_pairs]
            if qid in crit_map and ranked and ranked[0] == crit_map[qid]:
                crit_top1 += 1
            rel = gold.get(qid, set())
            if not rel:
                continue
            rec5s.append(recall_at_k(ranked, rel, 5))
            rec10s.append(recall_at_k(ranked, rel, 10))
            mrrs.append(mrr(ranked, rel))
            ndcgs.append(ndcg_at_k(ranked, rel, 10))
        report["alpha_sweep"][str(alpha)] = {
            "recall@5": mean(rec5s),
            "recall@10": mean(rec10s),
            "mrr": mean(mrrs),
            "ndcg@10": mean(ndcgs),
            "critical_top1": crit_top1,
            "critical_total": len(critical_pairs),
        }

    # Identify best alpha by recall@5
    best_alpha = max(
        report["alpha_sweep"].keys(),
        key=lambda a: report["alpha_sweep"][a]["recall@5"],
    )
    report["best_alpha"] = best_alpha
    report["best"] = report["alpha_sweep"][best_alpha]
    report["usage"] = llm.usage
    report["cost_usd"] = llm.cost_usd()

    # Baseline (for report)
    baseline = json.loads((RESULTS_DIR / "retrieval_results.json").read_text())
    report["baseline_T_and_S"] = baseline.get("T_and_S", {})

    out_path = RESULTS_DIR / "advanced_e1_event_binding.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"E1 wrote {out_path}")
    print(json.dumps({"best_alpha": best_alpha, **report["best"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
