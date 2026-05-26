"""A/B classifier vs extractor-driven routing on representative benches.

HISTORICAL: this script was written during the classifier-removal
investigation. It references `use_classifier` and `anaphoric_fallback`
flags that were on TemporalRetriever during the A/B; those flags were
removed on ship. The script captures the methodology and numbers that
justified shipping Plan B (no_clf_pure); see `no_classifier_validation.json`
for results.

Variants:
  A) with_classifier      — production path: LLM PhraseClassifier routes
                            each leaf to calendar_pin/recurring_period/
                            anaphoric_event/personal_era/generic_skip.
  B) no_classifier_pure   — extractor-driven only. Every leaf goes through
                            v3 extractor; envelopes >= confidence_floor
                            become anchors. No anaphoric fallback.
  C) no_classifier_anaph  — extractor-driven + deterministic anaphoric
                            fallback (top-1 corpus cosine) when extractor
                            returns nothing AND direction in
                            {after,before,since,until}.

Benches chosen to exercise different failure modes:
  - realq_v2          : real query patterns, anaphoric event refs
  - ambiguous_year    : year-less month/quarter phrases ("in March")
  - composition       : multi-leaf DNF with relative directions
  - hard_bench        : date-anchored general queries
  - causal_relative   : "after X" patterns where anaphoric might matter
  - precedents        : timeless-content reproducer
  - timeless_policies : empty_doc_mask interplay

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_no_classifier
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
    "realq_v2": (
        "realq_v2_docs.jsonl",
        "realq_v2_queries.jsonl",
        "realq_v2_gold.jsonl",
    ),
    "ambiguous_year": (
        "ambiguous_year_docs.jsonl",
        "ambiguous_year_queries.jsonl",
        "ambiguous_year_gold.jsonl",
    ),
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
    "hard_bench": (
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
    ),
    "causal_relative": (
        "causal_relative_docs.jsonl",
        "causal_relative_queries.jsonl",
        "causal_relative_gold.jsonl",
    ),
    "precedents": (
        "precedents_docs.jsonl",
        "precedents_queries.jsonl",
        "precedents_gold.jsonl",
    ),
    "timeless_policies": (
        "timeless_policies_docs.jsonl",
        "timeless_policies_queries.jsonl",
        "timeless_policies_gold.jsonl",
    ),
}


VARIANTS: dict[str, dict] = {
    "with_classifier":     {"use_classifier": True,  "anaphoric_fallback": False},
    "no_clf_pure":         {"use_classifier": False, "anaphoric_fallback": False},
    "no_clf_anaph":        {"use_classifier": False, "anaphoric_fallback": True},
}


async def evaluate(
    bench: str, vname: str, vkw: dict, embed_fn, rerank_fn
) -> dict:
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
        for d in docs_jsonl
    ]

    retriever = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn, **vkw)
    await retriever.index(docs)

    n_eval = n_r1 = n_r5 = 0
    all_r5: list[float] = []
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first is not None:
            if first <= 1: n_r1 += 1
            if first <= 5: n_r5 += 1
        top5 = set(ranking[:5])
        all_r5.append(len(top5 & gold_set) / len(gold_set))

    return {
        "bench": bench,
        "variant": vname,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
        "n_eval": n_eval,
    }


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn  = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    rows: list[dict] = []
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        for vname, vkw in VARIANTS.items():
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, vkw, embed_fn, rerank_fn)
            print(f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} all_R@5={r['all_R@5']:.3f}", flush=True)
            rows.append(r)

    print("\n" + "=" * 100, flush=True)
    print(f"{'bench':22s} {'variant':18s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s} {'ΔR@1':>7s} {'ΔR@5':>7s}", flush=True)
    print("-" * 100, flush=True)

    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        base = next(r for r in rows if r["bench"] == bench and r["variant"] == "with_classifier")
        for vname in VARIANTS:
            r = next(row for row in rows if row["bench"] == bench and row["variant"] == vname)
            d1 = r["R@1"] - base["R@1"]
            d5 = r["R@5"] - base["R@5"]
            tag1 = "" if vname == "with_classifier" else f"{d1:+.3f}"
            tag5 = "" if vname == "with_classifier" else f"{d5:+.3f}"
            print(f"{bench:22s} {vname:18s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} {r['all_R@5']:>8.3f} {tag1:>7s} {tag5:>7s}", flush=True)
            macro[vname]["R@1"].append(r["R@1"])
            macro[vname]["R@5"].append(r["R@5"])
            macro[vname]["all_R@5"].append(r["all_R@5"])
        print("-" * 100, flush=True)

    print("\nMACRO across all benches:", flush=True)
    print(f"{'variant':18s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    for vname, vals in macro.items():
        m1 = sum(vals["R@1"]) / len(vals["R@1"])
        m5 = sum(vals["R@5"]) / len(vals["R@5"])
        ma = sum(vals["all_R@5"]) / len(vals["all_R@5"])
        print(f"{vname:18s} {m1:>6.3f} {m5:>6.3f} {ma:>8.3f}", flush=True)

    out_path = ROOT / "no_classifier_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
