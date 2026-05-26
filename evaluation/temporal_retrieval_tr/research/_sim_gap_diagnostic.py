"""Diagnostic: measure sim-gap between recency-winners and similarity-winners
on composition and cotemporal benches.

For each query, we extract a "competing pair":
- composition (gold IS most-recent): gold vs the doc with highest base+match
  among non-gold pool docs that are LESS recent than gold. The sim-gap is
  positive when gold needs recency to win.
- cotemporal (gold is NOT most-recent): gold vs the doc with highest recency
  rank that has lower base+match than gold. The sim-gap measures how much
  recency boost it takes to flip gold's deserved win.

If composition sim-gaps are systematically SMALLER than cotemporal sim-gaps,
a bounded recency mechanism with bonus in between can fix composition while
preserving cotemporal. If they OVERLAP, no scalar bonus can separate them.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._sim_gap_diagnostic
"""
from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_min.schema import parse_iso, to_us
from temporal_retrieval_tr.research.bench import (
    load_bench,
    make_cosine_rerank_fn,
)

setup_env()


async def gather_pool_for_query(
    vd: TemporalRetriever, query: str, ref_time: str
) -> tuple[list, dict]:
    """Run the query at W=0 and return (Results, doc_ref_us_map for pool).

    With recency_weight=0, the Result.score is base+match. Result.rerank
    and Result.match are also populated.
    """
    vd.recency_weight = 0.0
    vd.copeland_bonus = None
    results = await vd.query(query, ref_time, k=10)
    # Map doc_id -> ref_us for the returned pool docs
    ref_us = {r.doc_id: vd._doc_ref_us[r.doc_id] for r in results}
    return results, ref_us


def analyze_composition_query(
    results: list, ref_us: dict, gold_ids: list[str]
) -> dict | None:
    """For composition (gold should be MOST RECENT):
    - find the gold present in the pool
    - among non-gold docs LESS recent than gold, find highest base+match
    - sim_gap = competitor.base+match - gold.base+match
    """
    if not results:
        return None
    by_id = {r.doc_id: r for r in results}
    gold_in_pool = [g for g in gold_ids if g in by_id]
    if not gold_in_pool:
        return None
    # Use the most-recent gold (in case of multiple gold)
    gold_id = max(gold_in_pool, key=lambda g: ref_us[g])
    gold = by_id[gold_id]
    gold_sim = gold.rerank + gold.match
    gold_ref = ref_us[gold_id]
    # Competitors: NOT gold, LESS recent than gold (i.e. ref_us < gold_ref)
    # Actually composition gold should win on RECENCY — so the competing
    # threat is any doc with HIGHER base+match that's less recent.
    competitors = [r for r in results
                   if r.doc_id != gold_id
                   and ref_us[r.doc_id] < gold_ref]
    if not competitors:
        return None
    top_competitor = max(competitors, key=lambda r: r.rerank + r.match)
    comp_sim = top_competitor.rerank + top_competitor.match
    return {
        "gold_id": gold_id,
        "gold_sim": gold_sim,
        "gold_rerank": gold.rerank,
        "gold_match": gold.match,
        "competitor_id": top_competitor.doc_id,
        "competitor_sim": comp_sim,
        "competitor_rerank": top_competitor.rerank,
        "competitor_match": top_competitor.match,
        "sim_gap": comp_sim - gold_sim,  # >0 means recency needs to overcome this gap
        "rank_gap": sum(1 for r in results if ref_us[r.doc_id] > gold_ref) -
                    sum(1 for r in results if ref_us[r.doc_id] > ref_us[top_competitor.doc_id]),
    }


def analyze_cotemporal_query(
    results: list, ref_us: dict, gold_ids: list[str]
) -> dict | None:
    """For cotemporal (gold should NOT be flipped by recency):
    - find gold in pool
    - among docs MORE recent than gold (ref_us > gold_ref), find one with
      lower base+match (the doc that becomes dangerous under high W)
    - sim_gap = gold.base+match - competitor.base+match (positive = gold's
      protective advantage)
    """
    if not results:
        return None
    by_id = {r.doc_id: r for r in results}
    gold_in_pool = [g for g in gold_ids if g in by_id]
    if not gold_in_pool:
        return None
    # Use the gold with highest base+match (most likely actual winner)
    gold_id = max(gold_in_pool, key=lambda g: by_id[g].rerank + by_id[g].match)
    gold = by_id[gold_id]
    gold_sim = gold.rerank + gold.match
    gold_ref = ref_us[gold_id]
    # Competitors: MORE recent than gold, lower base+match (these threaten under W)
    competitors = [r for r in results
                   if r.doc_id != gold_id
                   and ref_us[r.doc_id] > gold_ref
                   and (r.rerank + r.match) < gold_sim]
    if not competitors:
        return None
    # The most dangerous competitor is the one closest to gold in sim
    # (smallest sim gap means smallest W needed to flip)
    threat = max(competitors, key=lambda r: r.rerank + r.match)
    threat_sim = threat.rerank + threat.match
    return {
        "gold_id": gold_id,
        "gold_sim": gold_sim,
        "gold_rerank": gold.rerank,
        "gold_match": gold.match,
        "competitor_id": threat.doc_id,
        "competitor_sim": threat_sim,
        "competitor_rerank": threat.rerank,
        "competitor_match": threat.match,
        "sim_gap": gold_sim - threat_sim,  # gold's protective advantage
        "rank_gap": sum(1 for r in results
                        if ref_us[r.doc_id] > gold_ref
                        and ref_us[r.doc_id] <= ref_us[threat.doc_id]),
    }


async def run_diagnostic(bench: str, embed_fn, rerank_fn, mode: str) -> list[dict]:
    """mode = 'composition' or 'cotemporal'."""
    loaded = load_bench(bench)
    if loaded[0] is None:
        return []
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_text = {d.id: d.text for d in docs}
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    out = []
    for q in queries:
        qid = q["query_id"]
        gold_ids = gold.get(qid, [])
        if not gold_ids:
            continue
        results, ref_us = await gather_pool_for_query(vd, q["text"], q["ref_time"])
        if mode == "composition":
            row = analyze_composition_query(results, ref_us, gold_ids)
        else:
            row = analyze_cotemporal_query(results, ref_us, gold_ids)
        if row is not None:
            row["qid"] = qid
            row["query_text"] = q["text"]
            row["gold_text"] = doc_text.get(row["gold_id"], "")[:200]
            row["competitor_text"] = doc_text.get(row["competitor_id"], "")[:200]
            out.append(row)
    return out


def summarize(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"=== {label}: no rows ===")
        return
    gaps = [r["sim_gap"] for r in rows]
    rerank_gaps = [
        (r["competitor_rerank"] - r["gold_rerank"]) if "composition" in label.lower()
        else (r["gold_rerank"] - r["competitor_rerank"])
        for r in rows
    ]
    match_gaps = [
        (r["competitor_match"] - r["gold_match"]) if "composition" in label.lower()
        else (r["gold_match"] - r["competitor_match"])
        for r in rows
    ]
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"  sim_gap (rerank+match): "
          f"min={min(gaps):+.3f} median={statistics.median(gaps):+.3f} "
          f"max={max(gaps):+.3f} mean={statistics.mean(gaps):+.3f}")
    print(f"  rerank_gap only:       "
          f"min={min(rerank_gaps):+.3f} median={statistics.median(rerank_gaps):+.3f} "
          f"max={max(rerank_gaps):+.3f}")
    print(f"  match_gap only:        "
          f"min={min(match_gaps):+.3f} median={statistics.median(match_gaps):+.3f} "
          f"max={max(match_gaps):+.3f}")
    # Bin distribution
    bins = [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
    print(f"  Distribution of sim_gap:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        c = sum(1 for g in gaps if lo <= g < hi)
        if c > 0:
            print(f"    [{lo:+.2f}, {hi:+.2f}): {c}")
    # Show a sample of the small-gap (most-actionable) rows
    sorted_rows = sorted(rows, key=lambda r: abs(r["sim_gap"]))
    print(f"\n  Sample of smallest-gap queries (where bonus matters most):")
    for i, r in enumerate(sorted_rows[:6]):
        print(f"    [{i+1}] qid={r['qid']} sim_gap={r['sim_gap']:+.3f}")
        print(f"        query: {r['query_text'][:130]}")
        print(f"        gold ({r['gold_id']}, sim={r['gold_sim']:.3f}): "
              f"{r['gold_text'][:130]}")
        print(f"        compt ({r['competitor_id']}, sim={r['competitor_sim']:.3f}): "
              f"{r['competitor_text'][:130]}")


async def main() -> None:
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print("=== sim_gap diagnostic ===", flush=True)
    print("composition: gold IS recent — sim_gap is the deficit recency must overcome",
          flush=True)
    print("cotemporal:  gold should win on sim — sim_gap is the protective margin",
          flush=True)
    comp = await run_diagnostic("composition", embed_fn, rerank_fn, "composition")
    cot  = await run_diagnostic("cotemporal", embed_fn, rerank_fn, "cotemporal")
    summarize(comp, "COMPOSITION (gold needs recency to win)")
    summarize(cot, "COTEMPORAL (gold needs protection from recency)")

    # Cross-comparison: can a single bonus cleanly separate?
    print("\n=== Separability ===")
    comp_gaps = sorted([r["sim_gap"] for r in comp])
    cot_gaps = sorted([r["sim_gap"] for r in cot])
    if comp_gaps and cot_gaps:
        # A bonus B fixes a composition query if comp_gap < B
        # A bonus B preserves a cotemporal query if cot_gap > B
        # Sweep B and count both
        print("  B (bonus)  | comp_fixed | cot_preserved | net (comp_fixed - cot_broken)")
        for B in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80, 1.0, 1.5]:
            fixed = sum(1 for g in comp_gaps if 0 < g <= B)  # gold loses by < B
            still_lost = sum(1 for g in comp_gaps if g > B)  # gap too large
            never_lost = sum(1 for g in comp_gaps if g <= 0)  # gold already winning
            preserved = sum(1 for g in cot_gaps if g > B)
            broken = sum(1 for g in cot_gaps if g <= B)
            print(f"    {B:>5.2f}    |    {fixed:>3d}/{len(comp_gaps)}     |   "
                  f"{preserved:>3d}/{len(cot_gaps)}      |  fixed={fixed} broken={broken}")


if __name__ == "__main__":
    asyncio.run(main())
