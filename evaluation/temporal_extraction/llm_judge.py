"""E3 — LLM-as-relevance-judge upper bound.

For 20 queries (picked deterministically), take the union of top-20
candidates from the base semantic and base temporal rankings. For each
(query, candidate), ask gpt-5-mini to rate temporal relevance 0-1 given
each side's time expressions.

Reuses base-extractor's time expressions via cache/llm_cache.json — we
call the pipeline the same way the base eval does.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from advanced_common import (
    DATA_DIR,
    RESULTS_DIR,
    LLMCaller,
    load_jsonl,
    mean,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from baselines import embed_all, semantic_rank
from extractor import Extractor as BaseExtractor
from schema import parse_iso

JUDGE_SYSTEM = """You judge whether a document's time references are
relevant to a query's time references.

Input:
- query time references: list of surfaces or resolved windows from the
  query (what time is the user asking about?)
- document time references: list of surfaces or resolved windows from the
  document

Task: return a single float between 0.0 and 1.0 indicating how well the
document's temporal content satisfies the query's temporal scope. Strict
guide:
- 1.0 = document time window(s) match the query window exactly or are a
  direct instance of a query recurrence
- 0.7 = close overlap (same month/year as day-grain query; same quarter)
- 0.5 = partial overlap or coarse match (both refer to the same decade,
  or the doc covers the query time but is much wider)
- 0.2 = referenced but largely disjoint (different month of same year)
- 0.0 = unrelated time

Output JSON: {"score": <float>, "reason": "<short>"}.
"""

JUDGE_SCHEMA = {
    "name": "judge_score",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["score"],
    },
}


def _expr_summary(te) -> str:
    """Short human-readable summary of a TimeExpression for the judge."""
    parts = [f"surface={te.surface!r}", f"kind={te.kind}"]
    if te.kind == "instant" and te.instant:
        parts.append(
            f"window=[{te.instant.earliest.isoformat()},"
            f"{te.instant.latest.isoformat()})"
        )
        parts.append(f"granularity={te.instant.granularity}")
    elif te.kind == "interval" and te.interval:
        parts.append(
            f"window=[{te.interval.start.earliest.isoformat()},"
            f"{te.interval.end.latest.isoformat()})"
        )
    elif te.kind == "recurrence" and te.recurrence:
        parts.append(f"rrule={te.recurrence.rrule}")
    elif te.kind == "duration" and te.duration is not None:
        parts.append(f"duration_s={int(te.duration.total_seconds())}")
    return "{" + ", ".join(parts) + "}"


async def judge_pair(
    llm: LLMCaller,
    q_exprs: list,
    d_exprs: list,
    q_text: str,
    d_text: str,
) -> float:
    q_summ = "\n".join(_expr_summary(t) for t in q_exprs) or "(none)"
    d_summ = "\n".join(_expr_summary(t) for t in d_exprs) or "(none)"
    user = (
        f"Query text: {q_text}\n"
        f"Query time references:\n{q_summ}\n\n"
        f"Document text: {d_text}\n"
        f"Document time references:\n{d_summ}\n\n"
        'Return {"score": <float 0-1>, "reason": "..."}.'
    )
    raw = await llm.chat(
        JUDGE_SYSTEM,
        user,
        json_schema=JUDGE_SCHEMA,
        max_completion_tokens=400,
        cache_tag="e3_judge",
    )
    if not raw:
        return 0.0
    try:
        d = json.loads(raw)
        s = float(d.get("score", 0.0))
        return max(0.0, min(1.0, s))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


async def main() -> None:
    docs = load_jsonl(DATA_DIR / "docs.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    gold = {
        r["query_id"]: set(r["relevant_doc_ids"])
        for r in load_jsonl(DATA_DIR / "gold.jsonl")
    }
    critical_pairs = json.loads((DATA_DIR / "critical_pairs.json").read_text())
    crit_map = {q_id: doc_id for (doc_id, q_id) in critical_pairs}

    # Stratified deterministic pick: 20 queries spread across prefixes.
    by_prefix: dict[str, list[dict]] = {}
    for q in queries:
        p = q["query_id"].rsplit("_", 1)[0]
        by_prefix.setdefault(p, []).append(q)
    picked: list[dict] = []
    prefixes = sorted(by_prefix.keys())
    i = 0
    while len(picked) < 20 and prefixes:
        p = prefixes[i % len(prefixes)]
        bucket = by_prefix[p]
        if bucket:
            picked.append(bucket.pop(0))
        i += 1
        if all(len(by_prefix[p]) == 0 for p in prefixes):
            break
    picked = picked[:20]
    print(f"E3: using {len(picked)} queries")

    # Re-run base extractor (cache-reuses) to get TimeExpressions for all
    # docs and the 20 picked queries. Queries only, not full set.
    base = BaseExtractor()

    async def ext(item_id: str, text: str, ref: datetime):
        try:
            tes = await base.extract(text, ref)
        except Exception:
            tes = []
        return item_id, tes

    print("E3: extracting temporal expressions (reuses base cache)...")
    d_exprs_list = await asyncio.gather(
        *(ext(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs)
    )
    q_exprs_list = await asyncio.gather(
        *(ext(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in picked)
    )
    base.cache.save()
    doc_exprs = {i: t for i, t in d_exprs_list}
    q_exprs = {i: t for i, t in q_exprs_list}

    # Build semantic candidates per query (top-20)
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in picked]
    all_embs = await embed_all(doc_texts + q_texts)
    doc_embs = {d["doc_id"]: all_embs[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: all_embs[len(docs) + i] for i, q in enumerate(picked)}

    def top_semantic(qid: str, k: int = 20) -> list[str]:
        qe = q_embs[qid]
        ranked = semantic_rank(qe, doc_embs)
        return [d for d, _ in ranked[:k]]

    # 2) Run judge
    llm = LLMCaller(concurrency=10)
    doc_text_map = {d["doc_id"]: d["text"] for d in docs}
    q_text_map = {q["query_id"]: q["text"] for q in picked}

    pairs: list[tuple[str, str]] = []
    for q in picked:
        qid = q["query_id"]
        cands = top_semantic(qid, k=20)
        for c in cands:
            pairs.append((qid, c))

    print(f"E3: judging {len(pairs)} (query, cand) pairs via gpt-5-mini...")

    async def judge_one(qid: str, cid: str):
        s = await judge_pair(
            llm,
            q_exprs.get(qid, []),
            doc_exprs.get(cid, []),
            q_text_map[qid],
            doc_text_map[cid],
        )
        return (qid, cid, s)

    judgments = await asyncio.gather(*(judge_one(q, c) for q, c in pairs))
    llm.save()

    # 3) Rank by judge score
    judge_ranked: dict[str, list[tuple[str, float]]] = {}
    for qid, cid, s in judgments:
        judge_ranked.setdefault(qid, []).append((cid, s))
    for qid in judge_ranked:
        judge_ranked[qid].sort(key=lambda x: x[1], reverse=True)

    # 4) Metrics
    rec5s, rec10s, mrrs, ndcgs = [], [], [], []
    crit_top1 = 0
    per_query = {}
    for q in picked:
        qid = q["query_id"]
        ranked = [d for d, _ in judge_ranked.get(qid, [])]
        if qid in crit_map and ranked and ranked[0] == crit_map[qid]:
            crit_top1 += 1
        rel = gold.get(qid, set())
        per_query[qid] = {
            "judge_top10": judge_ranked.get(qid, [])[:10],
            "gold": sorted(rel),
        }
        if not rel:
            continue
        rec5s.append(recall_at_k(ranked, rel, 5))
        rec10s.append(recall_at_k(ranked, rel, 10))
        mrrs.append(mrr(ranked, rel))
        ndcgs.append(ndcg_at_k(ranked, rel, 10))

    # Compare against base on the SAME subset
    baseline = json.loads((RESULTS_DIR / "retrieval_results.json").read_text())

    # Also compute base hybrid on SAME subset — not possible without access
    # to live hybrid fn (would duplicate eval.py). Instead, compute base
    # SEMANTIC ranking on same subset as a floor, and report full baseline
    # T_and_S separately for reference.
    s_rec5s, s_rec10s, s_mrrs, s_ndcgs = [], [], [], []
    for q in picked:
        qid = q["query_id"]
        ranked = top_semantic(qid, k=len(docs))
        rel = gold.get(qid, set())
        if not rel:
            continue
        s_rec5s.append(recall_at_k(ranked, rel, 5))
        s_rec10s.append(recall_at_k(ranked, rel, 10))
        s_mrrs.append(mrr(ranked, rel))
        s_ndcgs.append(ndcg_at_k(ranked, rel, 10))

    report = {
        "n_queries": len(picked),
        "n_pairs_judged": len(pairs),
        "judge_metrics": {
            "recall@5": mean(rec5s),
            "recall@10": mean(rec10s),
            "mrr": mean(mrrs),
            "ndcg@10": mean(ndcgs),
            "critical_top1": crit_top1,
            "critical_total": sum(1 for q in picked if q["query_id"] in crit_map),
        },
        "semantic_subset_floor": {
            "recall@5": mean(s_rec5s),
            "recall@10": mean(s_rec10s),
            "mrr": mean(s_mrrs),
            "ndcg@10": mean(s_ndcgs),
        },
        "baseline_full_T_and_S": baseline.get("T_and_S", {}),
        "usage_llm": llm.usage,
        "cost_usd_llm": llm.cost_usd(),
        "picked_queries": [q["query_id"] for q in picked],
    }
    out_path = RESULTS_DIR / "advanced_e3_llm_judge.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"E3 wrote {out_path}")
    print(json.dumps(report["judge_metrics"], indent=2))
    print("semantic subset floor:")
    print(json.dumps(report["semantic_subset_floor"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
