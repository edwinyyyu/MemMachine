"""Harder-bench extractor ablation.

Companion to ``_ablation_proper.py``. The proper-ablation set saturated at
R@5=1.0 on era_refs (no signal). This script targets benches where the
extractor prompt's components — gazetteer, few-shots, ref_context — are
the most plausibly load-bearing:

- composition: compositional/multi-cue queries (R@5 ~0.64 in baseline).
  Bottleneck is partly extraction quality and partly retrieval.
- adversarial: 24 sub-categories of edge-case temporal phrasings (deixis,
  embedded quotes, fuzzy eras, recurring patterns). Likely the bench that
  most stresses the extractor prompt.
- mixed_cue: 4-way mix (date, content, recur, era). Recur+era subsets are
  exactly what TRIGGER_GAZETTEER targets.
- realq_v2: real-world multi-fact passages (some with non-trivial dates).

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._ablation_hard

Caches are per-variant pass-1 (cleared by the script when missing) so we
don't pre-seed across variants.
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

# Harder benches that should NOT saturate at R@5=1.0 in the baseline.
# Selected because each stresses a specific component of the extractor
# prompt: composition (compositional cues), adversarial (deixis +
# embedded quotes + edge cases), mixed_cue (recur+era subsets directly
# match TRIGGER_GAZETTEER), realq_v2 (real-world multi-fact passages).
BENCHES = {
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
    "adversarial": (
        "adversarial_docs.jsonl",
        "adversarial_queries.jsonl",
        "adversarial_gold.jsonl",
    ),
    "mixed_cue": (
        "mixed_cue_docs.jsonl",
        "mixed_cue_queries.jsonl",
        "mixed_cue_gold.jsonl",
    ),
    "realq_v2": (
        "realq_v2_docs.jsonl",
        "realq_v2_queries.jsonl",
        "realq_v2_gold.jsonl",
    ),
}


def _gold_key(g: dict) -> str:
    return g.get("query_id") or g.get("qid") or ""


def _gold_relevant(g: dict) -> set[str]:
    """Bench-format-tolerant: composition uses 'relevant_doc_ids',
    adversarial/realq_v2 use 'gold_retrieval', mixed_cue uses one of
    those plus optional 'gold_doc_ids'."""
    for key in ("relevant_doc_ids", "gold_retrieval", "gold_doc_ids"):
        if key in g:
            return set(g[key])
    return set()


async def run_variant(name: str, bench: str, embed_fn, rerank_fn) -> dict:
    cfg = VARIANTS[name]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(*BENCHES[bench])
    gold = {_gold_key(g): _gold_relevant(g) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    extractor = ext_mod.TemporalExtractor(
        cache_subdir=f"hard_{bench}_{name}",
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
    n_skipped = 0

    for q in queries:
        qid = q.get("query_id") or q.get("qid") or ""
        gold_set = gold.get(qid, set())
        if not gold_set:
            n_skipped += 1
            continue
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
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

    n_eval = n_queries - n_skipped
    stats = retriever.stats()
    return {
        "variant": name,
        "bench": bench,
        "all_recall@5": total_recall / max(1, n_eval),
        "R@1": total_r1 / max(1, n_eval),
        "R@5": total_r5 / max(1, n_eval),
        "R@10": total_r10 / max(1, n_eval),
        "n_queries": n_queries,
        "n_eval": n_eval,
        "n_skipped_no_gold": n_skipped,
        "n_docs": n_docs,
        "docs_with_extractions": docs_with_extractions,
        "total_intervals_extracted": total_intervals,
        "extractor_input_tokens": stats["extractor_usage"]["input"],
        "extractor_output_tokens": stats["extractor_usage"]["output"],
        "extraction": extraction,
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
                    f"n_eval={r['n_eval']}/{r['n_queries']}  "
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

    out_path = ROOT / "ablation_hard_results.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
