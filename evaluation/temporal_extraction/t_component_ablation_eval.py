"""T-component ablation: measure which T-score components do the work.

For each benchmark, compute T-only rankings with one of the following recipes:
  T_interval_only:  w_interval=1.0
  T_tag_only:       w_tag=1.0
  T_lattice_only:   w_lattice=1.0
  T_axis_only:      axis_score alone (Bhattacharyya geomean)
  T_lblend:         0.2 interval + 0.2 tag + 0.6 lattice (current ship)
  T_lblend_axis:    0.15 interval + 0.15 tag + 0.45 lattice + 0.25 axis
  T_eq:             0.25 each (interval, tag, lattice, axis)

No semantic, no rerank, no fusion — pure T-score ranking.
Reports R@1 and R@5 per variant per benchmark.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from multi_axis_scorer import axis_score as axis_score_fn
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    interval_pair_best,
    parse_iso,
    run_v2_extract,
    tag_score,
)

# ------------------------- per-component score tables -------------------------


def per_doc_interval_scores(q_mem_one, doc_mem):
    q_ivs = q_mem_one.get("intervals") or []
    raw = {did: interval_pair_best(q_ivs, b["intervals"]) for did, b in doc_mem.items()}
    mx = max(raw.values()) if raw else 0.0
    if mx <= 0:
        return dict.fromkeys(doc_mem, 0.0)
    return {did: raw[did] / mx for did in doc_mem}


def per_doc_tag_scores(q_mem_one, doc_mem):
    q_tags = q_mem_one.get("multi_tags") or set()
    return {did: tag_score(q_tags, b["multi_tags"]) for did, b in doc_mem.items()}


def per_doc_axis_scores(q_mem_one, doc_mem):
    qa = q_mem_one.get("axes_merged") or {}
    return {did: axis_score_fn(qa, b["axes_merged"]) for did, b in doc_mem.items()}


def blend(scores_list, weights):
    """Linear weighted sum across multiple per-doc score dicts.

    Each entry of scores_list is a dict[doc_id -> float] over the same keys.
    """
    if not scores_list:
        return {}
    keys = set()
    for s in scores_list:
        keys |= set(s.keys())
    out = {}
    for k in keys:
        v = 0.0
        for w, s in zip(weights, scores_list):
            v += w * s.get(k, 0.0)
        out[k] = v
    return out


def topk_ranked(scores, k=None):
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in items] if k is None else [d for d, _ in items[:k]]


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


# ------------------------- benchmark runner -------------------------


async def run_bench(name, docs_path, queries_path, gold_path):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    cache_label = f"v7l-{name}"
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)
    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    # Lattice store
    lat_db = ROOT / "cache" / "t_ablation" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)

    qids = [q["query_id"] for q in queries]
    per_q_lattice = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    # Variants
    variants = {
        "T_interval_only": (1.0, 0.0, 0.0, 0.0),
        "T_tag_only": (0.0, 1.0, 0.0, 0.0),
        "T_lattice_only": (0.0, 0.0, 1.0, 0.0),
        "T_axis_only": (0.0, 0.0, 0.0, 1.0),
        "T_lblend": (0.2, 0.2, 0.6, 0.0),
        "T_lblend_axis": (0.15, 0.15, 0.45, 0.25),
        "T_eq": (0.25, 0.25, 0.25, 0.25),
    }

    results_per_variant = {v: {"r1": 0, "r5": 0, "n": 0} for v in variants}

    n_total = 0
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        n_total += 1
        q_mem_one = q_mem.get(
            qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}
        )

        # Pre-compute per-component score tables for this query
        iv = per_doc_interval_scores(q_mem_one, doc_mem)
        tg = per_doc_tag_scores(q_mem_one, doc_mem)
        lt = {did: per_q_lattice.get(qid, {}).get(did, 0.0) for did in doc_mem}
        ax = per_doc_axis_scores(q_mem_one, doc_mem)

        for vname, (wi, wt, wl, wa) in variants.items():
            scored = blend([iv, tg, lt, ax], [wi, wt, wl, wa])
            ranking = topk_ranked(scored)
            hr = hit_rank(ranking, gold_set, k=10)
            if hr is not None:
                if hr <= 1:
                    results_per_variant[vname]["r1"] += 1
                if hr <= 5:
                    results_per_variant[vname]["r5"] += 1
            results_per_variant[vname]["n"] += 1

    # Print per-variant stats
    print(f"\n--- {name} per-variant ---", flush=True)
    print(f"{'variant':22} {'R@1':>10} {'R@5':>10}", flush=True)
    out = {}
    for v, st in results_per_variant.items():
        n = st["n"] or 1
        r1, r5 = st["r1"] / n, st["r5"] / n
        print(
            f"{v:22} {st['r1']}/{n} ({r1:.3f})  {st['r5']}/{n} ({r5:.3f})", flush=True
        )
        out[v] = {
            "r1": r1,
            "r5": r5,
            "r1_count": st["r1"],
            "r5_count": st["r5"],
            "n": n,
        }
    return out


async def main():
    benches = [
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
        ),
        (
            "temporal_essential",
            "temporal_essential_docs.jsonl",
            "temporal_essential_queries.jsonl",
            "temporal_essential_gold.jsonl",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
        ),
    ]
    out = {}
    for name, dp, qp, gp in benches:
        try:
            out[name] = await run_bench(name, dp, qp, gp)
        except Exception as e:
            import traceback

            traceback.print_exc()
            out[name] = {"error": str(e)}

    out_path = ROOT / "results" / "T_component_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
