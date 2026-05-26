"""A/B test ref_time fallback options against current production default.

Builds on the empty_doc_mask=1.0 fix: tests whether including the doc's
ref_time as a synthesized interval (point at the message timestamp) helps
or hurts retrieval. Three configurations:

  baseline_v2  - empty_doc_mask=1.0, ref_time_fallback=None
                 (current production default after the Mode A/D fix)
  rt_when_empty - empty_doc_mask=1.0, ref_time_fallback="when_empty"
                 (synth ref_time only for docs with no extracted intervals)
  rt_always    - empty_doc_mask=1.0, ref_time_fallback="always"
                 (OR-merge synth ref_time with extracted intervals)

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_ref_time_fallback
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

BENCHES = {
    "timeless_policies": (
        "timeless_policies_docs.jsonl",
        "timeless_policies_queries.jsonl",
        "timeless_policies_gold.jsonl",
    ),
    "precedents": (
        "precedents_docs.jsonl",
        "precedents_queries.jsonl",
        "precedents_gold.jsonl",
    ),
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
    "realq_v2": (
        "realq_v2_docs.jsonl",
        "realq_v2_queries.jsonl",
        "realq_v2_gold.jsonl",
    ),
    "hard_bench": (
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
    ),
    "ambiguous_year": (
        "ambiguous_year_docs.jsonl",
        "ambiguous_year_queries.jsonl",
        "ambiguous_year_gold.jsonl",
    ),
}


VARIANTS = {
    "baseline_v2": {"ref_time_fallback": None},
    "rt_when_empty": {"ref_time_fallback": "when_empty"},
    "rt_always": {"ref_time_fallback": "always"},
}


async def evaluate(
    bench: str,
    variant_name: str,
    variant_kw: dict,
    embed_fn,
    rerank_fn,
) -> dict:
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        empty_doc_mask=1.0,
        ref_time_fallback=variant_kw["ref_time_fallback"],
    )
    await retriever.index(docs)

    n_eval = 0
    n_r5 = 0
    n_r1 = 0
    all_recall_at_5 = []
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first_gold = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first_gold is not None:
            if first_gold <= 1:
                n_r1 += 1
            if first_gold <= 5:
                n_r5 += 1
        top5 = set(ranking[:5])
        if gold_set:
            all_recall_at_5.append(len(top5 & gold_set) / len(gold_set))

    return {
        "bench": bench,
        "variant": variant_name,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_recall@5": sum(all_recall_at_5) / max(1, len(all_recall_at_5)),
        "n_eval": n_eval,
    }


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    rows: list[dict] = []
    for bench in BENCHES:
        print(f"\n=== bench: {bench} ===", flush=True)
        for vname, vkw in VARIANTS.items():
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, vkw, embed_fn, rerank_fn)
            print(
                f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
                f"all_R@5={r['all_recall@5']:.3f} n={r['n_eval']}",
                flush=True,
            )
            rows.append(r)

    # Summary
    print("\n" + "=" * 100, flush=True)
    print(
        f"{'bench':22s} {'variant':16s} {'R@1':>6s} {'R@5':>6s} "
        f"{'all_R@5':>8s} {'delta_R@5':>10s} {'delta_R@1':>10s}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for bench in BENCHES:
        baseline = next(
            r for r in rows if r["bench"] == bench and r["variant"] == "baseline_v2"
        )
        for vname in VARIANTS:
            r = next(
                row for row in rows if row["bench"] == bench and row["variant"] == vname
            )
            d5 = r["R@5"] - baseline["R@5"]
            d1 = r["R@1"] - baseline["R@1"]
            tag5 = "" if vname == "baseline_v2" else f"{d5:+.3f}"
            tag1 = "" if vname == "baseline_v2" else f"{d1:+.3f}"
            print(
                f"{bench:22s} {vname:16s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} "
                f"{r['all_recall@5']:>8.3f} {tag5:>10s} {tag1:>10s}",
                flush=True,
            )
        print("-" * 100, flush=True)

    out_path = ROOT / "ref_time_fallback_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
