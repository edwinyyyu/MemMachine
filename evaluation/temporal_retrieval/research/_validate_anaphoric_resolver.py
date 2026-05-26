"""A/B smart-anaphoric-resolver variants on the benches where anaphoric
event refs do real work (causal_relative, composition, precedents) plus
two control benches (realq_v2, hard_bench).

HISTORICAL: this script references `anaphoric_fallback`, `anaphoric_mode`,
and `anaphoric_topk` flags that were on TemporalRetriever during the A/B;
those flags were removed when Plan B shipped. The script and the saved
`anaphoric_resolver_validation.json` capture the methodology and numbers
that justified NOT shipping a smart resolver: topk_llm beat top-1 but
still carried a non-zero "fires when it shouldn't" cost (precedents R@1
1.000 -> 0.917), and macro R@5 / all_R@5 was best with no anaphoric path
at all.

Variants (all use classifier=False, planner v4.2):
  pure         — no anaphoric fallback (Plan B baseline)
  top1         — top-1 cosine (current band-aid)
  topk_union   — top-K cosine, union of envelopes
  topk_llm     — top-K cosine, LLM picks the referent doc

Goal: see if a smarter resolver can recover the causal_relative /
composition R@1 cost of pure WITHOUT the R@5 cost of top-1, AND
without re-introducing the precedents R@1 regression.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_anaphoric_resolver
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
    "causal_relative": (
        "causal_relative_docs.jsonl",
        "causal_relative_queries.jsonl",
        "causal_relative_gold.jsonl",
    ),
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
    "precedents": (
        "precedents_docs.jsonl",
        "precedents_queries.jsonl",
        "precedents_gold.jsonl",
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
}


VARIANTS: dict[str, dict] = {
    "pure":       {"anaphoric_fallback": False},
    "top1":       {"anaphoric_fallback": True, "anaphoric_mode": "top1"},
    "topk_union": {"anaphoric_fallback": True, "anaphoric_mode": "topk_union", "anaphoric_topk": 5},
    "topk_llm":   {"anaphoric_fallback": True, "anaphoric_mode": "topk_llm",   "anaphoric_topk": 5},
}


async def evaluate(bench: str, vname: str, vkw: dict, embed_fn, rerank_fn) -> dict:
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl]
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
    print(f"{'bench':22s} {'variant':14s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s} {'ΔR@1 vs pure':>14s}", flush=True)
    print("-" * 100, flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench in BENCHES:
        base = next(r for r in rows if r["bench"] == bench and r["variant"] == "pure")
        for vname in VARIANTS:
            r = next(row for row in rows if row["bench"] == bench and row["variant"] == vname)
            d1 = r["R@1"] - base["R@1"]
            tag = "" if vname == "pure" else f"{d1:+.3f}"
            print(f"{bench:22s} {vname:14s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} {r['all_R@5']:>8.3f} {tag:>14s}", flush=True)
            macro[vname]["R@1"].append(r["R@1"])
            macro[vname]["R@5"].append(r["R@5"])
            macro[vname]["all_R@5"].append(r["all_R@5"])
        print("-" * 100, flush=True)

    print("\nMACRO:", flush=True)
    print(f"{'variant':14s} {'R@1':>6s} {'R@5':>6s} {'all_R@5':>8s}", flush=True)
    for vname, vals in macro.items():
        m1 = sum(vals["R@1"]) / len(vals["R@1"])
        m5 = sum(vals["R@5"]) / len(vals["R@5"])
        ma = sum(vals["all_R@5"]) / len(vals["all_R@5"])
        print(f"{vname:14s} {m1:>6.3f} {m5:>6.3f} {ma:>8.3f}", flush=True)

    out_path = ROOT / "anaphoric_resolver_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
