"""V7L + salience hypothesis test.

Compares V7L (T+S+L) baseline vs V7L+salience-post on the 4 benchmarks.
All extractions are cached. ~5 minutes runtime, ~$0 LLM cost.

If V7L+salience matches or beats V7L baseline (where V7+salience hurt V7
baseline), the channel-routing hypothesis is correct: V7's polyglot T was
the failure cause, not salience itself.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import extractor_common

# Patch to use minimal reasoning (in case any extraction call slips through)
_orig_call = extractor_common.BaseImprovedExtractor._call


async def _patched_call(self, *args, **kwargs):
    original_create = self.client.chat.completions.create

    async def patched_create(**call_kwargs):
        model = call_kwargs.get("model", "")
        if isinstance(model, str) and model.startswith("gpt-5"):
            call_kwargs["reasoning_effort"] = "minimal"
        return await original_create(**call_kwargs)

    self.client.chat.completions.create = patched_create
    try:
        return await _orig_call(self, *args, **kwargs)
    finally:
        self.client.chat.completions.create = original_create


extractor_common.BaseImprovedExtractor._call = _patched_call

from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from rag_fusion import score_blend
from rag_fusion_salience import score_blend_with_salience_post
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    SalienceExtractor,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from salience_eval import (
    rank_t as rank_multi_axis_t,
)


def ingest_lattice(store: LatticeStore, extracted):
    for doc_id, tes in extracted.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            store.insert(doc_id, ts.absolute, ts.cyclical)


def lattice_scores_for_query(store, q_extracted, query_ids):
    out = {}
    for qid in query_ids:
        tes = q_extracted.get(qid, [])
        scores, _ = lattice_retrieve_multi(store, tes, down_levels=1)
        out[qid] = scores
    return out


def rank_with_blend(t, s, l, weights, doc_salience=None, channel_to_key=None):
    chans = {"T": t, "S": s, "L": l}
    if doc_salience is None:
        fused = score_blend(chans, weights, top_k_per=40, dispersion_cv_ref=0.20)
    else:
        fused = score_blend_with_salience_post(
            chans,
            weights,
            doc_salience,
            channel_to_key=channel_to_key or {"T": "T", "S": "S", "L": "L"},
            salience_floor=0.05,
            salience_temperature=1.0,
            top_k_per=40,
            dispersion_cv_ref=0.20,
        )
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def metrics(rankings, gold, qids):
    r1 = r3 = r5 = r10 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n = 0
    for qid in qids:
        rel = set(gold.get(qid, []))
        if not rel:
            continue
        r = rankings.get(qid, [])
        hit = None
        for i, d in enumerate(r[:10]):
            if d in rel:
                hit = i + 1
                break
        if hit:
            if hit <= 1:
                r1 += 1
            if hit <= 3:
                r3 += 1
            if hit <= 5:
                r5 += 1
            if hit <= 10:
                r10 += 1
            mrr_sum += 1.0 / hit
            dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(r[:10]) if d in rel)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), 10)))
            ndcg_sum += dcg / ideal if ideal else 0.0
        n += 1
    return {
        "n": n,
        "r@1": r1 / n if n else 0,
        "r@3": r3 / n if n else 0,
        "r@5": r5 / n if n else 0,
        "r@10": r10 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "ndcg@10": ndcg_sum / n if n else 0,
    }


async def run_bench(name, docs_path, queries_path, gold_path, cache_doc, cache_q):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===")

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_doc)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_q)

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

    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    # T scores per query
    per_q_t = {}
    per_q_s = {}
    for q in queries:
        qid = q["query_id"]
        per_q_t[qid] = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s[qid] = rank_semantic(qid, q_embs, doc_embs)

    # Lattice
    lat_db = ROOT / "cache" / "v7l_sal" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    ingest_lattice(lat, doc_ext)
    per_q_l = lattice_scores_for_query(lat, q_ext, [q["query_id"] for q in queries])

    # Salience (cached from prior run)
    sx = SalienceExtractor()
    sal_items = [(d["doc_id"], d["text"]) for d in docs]
    doc_salience = await sx.extract_many(sal_items)

    # Variants
    qids = [q["query_id"] for q in queries]
    weights = {"T": 0.4, "S": 0.4, "L": 0.2}

    variants = {}
    for vname, sal in [("V7L", None), ("V7L+sal-post", doc_salience)]:
        ranks = {}
        for qid in qids:
            ranks[qid] = rank_with_blend(
                per_q_t[qid], per_q_s[qid], per_q_l[qid], weights, sal
            )
        variants[vname] = ranks

    results = {var: metrics(ranks, gold, qids) for var, ranks in variants.items()}
    print(f"{'Variant':24} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6}")
    for var, m in results.items():
        print(
            f"{var:24} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
        )
    return results


async def main():
    benches = [
        (
            "mixed_cue",
            "mixed_cue_docs.jsonl",
            "mixed_cue_queries.jsonl",
            "mixed_cue_gold.jsonl",
            "v7l-mixed_cue",
            "v7l-mixed_cue",
        ),
        (
            "dense_cluster",
            "dense_cluster_docs.jsonl",
            "dense_cluster_queries.jsonl",
            "dense_cluster_gold.jsonl",
            "v7l-dense_cluster",
            "v7l-dense_cluster",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
            "v7l-tempreason",
            "v7l-tempreason",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
            "v7l-hard_bench",
        ),
    ]
    all_results = {}
    for name, *paths in benches:
        try:
            r = await run_bench(name, *paths)
            all_results[name] = r
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            import traceback

            traceback.print_exc()

    out_path = ROOT / "results" / "v7l_salience.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Benchmark':22} {'Variant':24} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for bname, vmap in all_results.items():
        for var, m in vmap.items():
            print(
                f"{bname:22} {var:24} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
