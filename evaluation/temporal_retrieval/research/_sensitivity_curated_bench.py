"""Build a sensitivity-curated bench for prompt optimization.

The R@5-saturated and pseudo-saturated benches don't differentiate prompts.
This script re-runs the 6 hard-bench ablation variants per-query and
identifies queries where AT LEAST ONE variant got rank ≤ K and AT LEAST
ONE variant got rank > K — those are the queries actually responsive to
extraction changes. Saves the union as a curated bench.

Re-run cost is minimal because pass-1 and pass-2 caches are already
populated by `_ablation_hard.py`; only embeddings + reranking re-execute.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._sensitivity_curated_bench
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval import extractor as ext_mod

from ._ablation_hard import BENCHES, VARIANTS
from ._common import (
    DATA_DIR,
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()


# Benches whose R@5 had ANY headroom in the ablation. mixed_cue
# saturates universally and won't have sensitivity-positive queries.
SENSITIVITY_BENCHES = ("composition", "adversarial", "realq_v2")


async def per_query_ranks(
    bench: str, variant: str, embed_fn, rerank_fn, k: int = 50
) -> dict[str, dict]:
    """Re-run a (bench, variant) and record per-query first-gold-rank."""
    docs_jsonl, queries, gold_rows = load_bench_jsonl(*BENCHES[bench])
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    cfg = VARIANTS[variant]
    extractor = ext_mod.TemporalExtractor(
        cache_subdir=f"hard_{bench}_{variant}",
        pass1_system=cfg.pass1,
        ref_context_fn=cfg.ref_ctx,
    )
    retriever = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn, extractor=extractor
    )
    await retriever.index(docs)

    out: dict[str, dict] = {}
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        results = await retriever.query(q["text"], q["ref_time"], k=k)
        ranking = [r.doc_id for r in results]
        first_gold_rank = next(
            (i + 1 for i, d in enumerate(ranking) if d in gold_set), None
        )
        out[qid] = {
            "first_gold_rank": first_gold_rank,  # None if not in top-k
            "in_top1": first_gold_rank is not None and first_gold_rank <= 1,
            "in_top5": first_gold_rank is not None and first_gold_rank <= 5,
            "in_top10": first_gold_rank is not None and first_gold_rank <= 10,
        }
    return out


def classify_sensitivity(per_variant: dict[str, dict[str, dict]]) -> dict:
    """Per query, classify into sensitivity buckets.

    Buckets:
    - irreducibly_solved: in_top5 for ALL variants → no signal
    - irreducibly_failed: NOT in_top5 for ALL variants → no signal
    - sensitive: split (some variants in_top5, others not) → signal-bearing
    """
    qids = set()
    for v_results in per_variant.values():
        qids.update(v_results.keys())

    buckets = {"irreducibly_solved": [], "irreducibly_failed": [], "sensitive": []}
    detail: dict[str, dict] = {}
    for qid in sorted(qids):
        ranks = {}
        in_top5 = {}
        for vname, vres in per_variant.items():
            r = vres.get(qid)
            if r is None:
                continue
            ranks[vname] = r["first_gold_rank"]
            in_top5[vname] = r["in_top5"]
        if not in_top5:
            continue
        all_solved = all(in_top5.values())
        none_solved = not any(in_top5.values())
        if all_solved:
            bucket = "irreducibly_solved"
        elif none_solved:
            bucket = "irreducibly_failed"
        else:
            bucket = "sensitive"
        buckets[bucket].append(qid)
        detail[qid] = {
            "bucket": bucket,
            "ranks": ranks,
            "in_top5": in_top5,
        }
    return {"buckets": buckets, "detail": detail}


def _print_bench_summary(bench: str, sens: dict) -> None:
    b = sens["buckets"]
    n_total = sum(len(v) for v in b.values())
    print(
        f"\n  {bench}: total {n_total} | "
        f"solved {len(b['irreducibly_solved'])} | "
        f"failed {len(b['irreducibly_failed'])} | "
        f"SENSITIVE {len(b['sensitive'])}",
        flush=True,
    )
    for qid in b["sensitive"]:
        ranks = sens["detail"][qid]["ranks"]
        ranks_str = " ".join(f"{v[:8]}={r}" for v, r in ranks.items() if r is not None)
        ranks_none = [v for v, r in ranks.items() if r is None]
        if ranks_none:
            ranks_str += " | none:" + ",".join(v[:8] for v in ranks_none)
        print(f"    {qid}: {ranks_str}", flush=True)


def _build_curated_for_bench(
    bench: str, sensitive_qids: set[str], curated_docs: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    docs_jsonl, all_queries, all_gold = load_bench_jsonl(*BENCHES[bench])
    docs_data = {d["doc_id"]: d for d in docs_jsonl}
    queries = [q for q in all_queries if q["query_id"] in sensitive_qids]
    for q in queries:
        q["_source_bench"] = bench
    gold = [g for g in all_gold if g["query_id"] in sensitive_qids]
    for g in gold:
        g["_source_bench"] = bench
    for did, d in docs_data.items():
        if did not in curated_docs:
            d2 = d.copy()
            d2["_source_bench"] = bench
            curated_docs[did] = d2
    return queries, gold


def _build_curated_bench(all_results: dict) -> tuple[dict, list, list]:
    curated_docs: dict[str, dict] = {}
    curated_queries: list[dict] = []
    curated_gold: list[dict] = []
    for bench in SENSITIVITY_BENCHES:
        sensitive_qids = set(all_results[bench]["buckets"]["sensitive"])
        if not sensitive_qids:
            continue
        qs, gs = _build_curated_for_bench(bench, sensitive_qids, curated_docs)
        curated_queries.extend(qs)
        curated_gold.extend(gs)
    return curated_docs, curated_queries, curated_gold


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    all_results: dict[str, dict] = {}
    for bench in SENSITIVITY_BENCHES:
        print(f"\n{'#' * 60}\n# Bench: {bench}\n{'#' * 60}", flush=True)
        per_variant: dict[str, dict[str, dict]] = {}
        for variant in VARIANTS:
            print(f"  variant: {variant}", flush=True)
            per_variant[variant] = await per_query_ranks(
                bench, variant, embed_fn, rerank_fn
            )
        sens = classify_sensitivity(per_variant)
        all_results[bench] = {"per_variant": per_variant, **sens}
        _print_bench_summary(bench, sens)

    curated_docs, curated_queries, curated_gold = _build_curated_bench(all_results)

    out_dir = DATA_DIR
    with open(out_dir / "sensitivity_curated_docs.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in curated_docs.values())
    with open(out_dir / "sensitivity_curated_queries.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in curated_queries)
    with open(out_dir / "sensitivity_curated_gold.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in curated_gold)

    summary_path = ROOT / "sensitivity_analysis_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60, flush=True)
    print(
        f"Curated bench: {len(curated_queries)} queries, "
        f"{len(curated_docs)} docs across {len(SENSITIVITY_BENCHES)} source benches",
        flush=True,
    )
    print(f"Saved to: {out_dir}/sensitivity_curated_*.jsonl", flush=True)
    print(f"Per-query analysis: {summary_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
