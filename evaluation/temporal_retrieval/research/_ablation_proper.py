"""Honest ablation harness — addresses the methodological flaws of
`_ablation_e2e.py`:

1. Caches are cleared at the start, not pre-seeded from production.
2. Baseline is run twice (variance estimate).
3. Tests harder benches with denser passages (era_refs, multi_te_doc,
   conjunctive_temporal, composition) where the prompt components are
   most likely to matter.
4. Logs per-doc extracted (surface, interval) tuples so we can compute
   extraction-quality metrics that the retrieval mask hides.
5. Honest cost: separate pass-1 cache per variant; pass-2 cache shared
   (correctly — pass-2 prompt is stable across pass-1 variants).

Run from `evaluation/`:
    uv run python -m temporal_retrieval.research._ablation_proper
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval import extractor as ext_mod

from ._common import (
    ROOT,
    Variant,
    default_variants,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    print_extraction_quality,
    print_main_summary,
    print_variance,
    print_variant_deltas,
    setup_env,
    summarize_intervals,
)

setup_env()


VARIANTS: dict[str, Variant] = default_variants()

# Harder benches than ambiguous_year + relative_time. These have denser
# passages and more diverse phrasings — exactly where the gazetteer +
# few-shots are supposed to earn their tokens.
BENCHES = {
    "era_refs": (
        "edge_era_refs_docs.jsonl",
        "edge_era_refs_queries.jsonl",
        "edge_era_refs_gold.jsonl",
    ),
    "conjunctive_temporal": (
        "edge_conjunctive_temporal_docs.jsonl",
        "edge_conjunctive_temporal_queries.jsonl",
        "edge_conjunctive_temporal_gold.jsonl",
    ),
    "multi_te_doc": (
        "edge_multi_te_doc_docs.jsonl",
        "edge_multi_te_doc_queries.jsonl",
        "edge_multi_te_doc_gold.jsonl",
    ),
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
}


async def run_variant(name: str, bench: str, embed_fn, rerank_fn) -> dict:
    cfg = VARIANTS[name]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(*BENCHES[bench])
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    # Per-variant pass-1 cache subdir; pass-2 cache shared (correct —
    # pass-2 is stable across pass-1 variants).
    extractor = ext_mod.TemporalExtractor(
        cache_subdir=f"prop_{bench}_{name}",
        pass1_system=cfg.pass1,
        ref_context_fn=cfg.ref_ctx,
    )
    retriever = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn, extractor=extractor
    )

    await retriever.index(docs)

    extraction = summarize_intervals(retriever.doc_intervals())
    n_docs = len(docs)
    total_intervals = sum(d["n_intervals"] for d in extraction.values())
    docs_with_extractions = sum(1 for d in extraction.values() if d["n_intervals"] > 0)

    K = 5
    n_queries = len(queries)
    total_recall = 0.0
    total_r1 = 0
    total_r5 = 0
    total_r10 = 0

    for q in queries:
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        gold_set = gold[q["query_id"]]
        topk = set(ranking[:K])
        n_in = len(topk & gold_set)
        total_recall += n_in / max(1, len(gold_set))
        first_gold_rank = next(
            (i + 1 for i, d in enumerate(ranking) if d in gold_set), None
        )
        if first_gold_rank is not None:
            if first_gold_rank <= 1:
                total_r1 += 1
            if first_gold_rank <= 5:
                total_r5 += 1
            if first_gold_rank <= 10:
                total_r10 += 1

    stats = retriever.stats()
    return {
        "variant": name,
        "bench": bench,
        "all_recall@5": total_recall / n_queries,
        "R@1": total_r1 / n_queries,
        "R@5": total_r5 / n_queries,
        "R@10": total_r10 / n_queries,
        "n_queries": n_queries,
        "n_docs": n_docs,
        "docs_with_extractions": docs_with_extractions,
        "total_intervals_extracted": total_intervals,
        "extractor_input_tokens": stats["extractor_usage"]["input"],
        "extractor_output_tokens": stats["extractor_usage"]["output"],
        "extraction": extraction,  # per-doc surface count + interval keys
    }


async def main():
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    all_rows = []
    for bench in BENCHES:
        print(f"\n{'#' * 60}\n# Bench: {bench}\n{'#' * 60}", flush=True)
        for name in VARIANTS:
            print(f"\n=== {bench} / variant: {name} ===", flush=True)
            try:
                r = await run_variant(name, bench, embed_fn, rerank_fn)
                all_rows.append(r)
                print(
                    f"  R@1={r['R@1']:.3f}  R@5={r['R@5']:.3f}  "
                    f"R@10={r['R@10']:.3f}  all_recall@5={r['all_recall@5']:.3f}  "
                    f"intervals/doc={r['total_intervals_extracted'] / r['n_docs']:.2f}  "
                    f"docs_with_ex={r['docs_with_extractions']}/{r['n_docs']}  "
                    f"tokens={r['extractor_input_tokens']}/{r['extractor_output_tokens']}",
                    flush=True,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                all_rows.append({"variant": name, "bench": bench, "error": str(e)})

    print_main_summary(all_rows)
    print_variance(all_rows, BENCHES)
    print_variant_deltas(all_rows, BENCHES)
    print_extraction_quality(all_rows, BENCHES)

    out_path = ROOT / "ablation_proper_results.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
