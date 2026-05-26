"""Diff DNF vs Tree planner outputs on a target bench.

For each query, print both plans side-by-side and flag rows where the
two diverge structurally. Helps isolate which query shapes the LLM
plans differently between the two prompts.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._diff_planner_outputs edge_conjunctive_temporal
"""
from __future__ import annotations

import json
import sys

from temporal_retrieval.planner import _cache_key as dnf_cache_key
from temporal_retrieval.planner_tree import _cache_key as tree_cache_key

from ._common import ROOT, load_bench_jsonl

DNF_CACHE = ROOT / "cache" / "planner" / "llm_plan_cache.json"
TREE_CACHE = ROOT / "cache" / "planner_tree" / "llm_plan_cache.json"


def _flatten_dnf(obj: dict) -> str:
    expr = obj.get("expr") or []
    if not expr:
        return f"(none) extr={obj.get('extremum')}"
    clauses = []
    for cl in expr:
        leaves = [f"{l.get('phrase')!r}:{l.get('relation')}" for l in cl]
        clauses.append("AND[" + ", ".join(leaves) + "]")
    return " OR ".join(clauses) + f"  extr={obj.get('extremum')}"


def _flatten_tree(obj: dict) -> str:
    expr = obj.get("expr")
    if expr is None:
        return f"(none) extr={obj.get('extremum')}"

    def _r(node):
        t = node.get("type")
        if t == "leaf":
            return f"{node.get('phrase')!r}:{node.get('relation')}"
        if t in ("and", "or"):
            return f"{t.upper()}[" + ", ".join(_r(c) for c in node.get("children") or []) + "]"
        if t == "not":
            return f"NOT({_r(node.get('child'))})"
        return repr(node)

    return _r(expr) + f"  extr={obj.get('extremum')}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m temporal_retrieval.research._diff_planner_outputs <bench_name>")
        return
    bench = sys.argv[1]

    queries_file = f"{bench}_queries.jsonl"
    gold_file = f"{bench}_gold.jsonl"
    # docs not needed for plan diff
    queries, gold_rows = (
        load_bench_jsonl(f"{bench}_docs.jsonl", queries_file, gold_file)[1:]
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    dnf_cache = json.loads(DNF_CACHE.read_text())
    tree_cache = json.loads(TREE_CACHE.read_text())

    n_diff = 0
    n_total = 0
    for q in queries:
        qid = q.get("query_id", "")
        if not gold.get(qid):
            continue
        n_total += 1
        text = q["text"]
        ref = q["ref_time"]
        dk = dnf_cache_key(text, ref)
        tk = tree_cache_key(text, ref)
        dnf_obj = dnf_cache.get(dk)
        tree_obj = tree_cache.get(tk)
        dnf_repr = _flatten_dnf(dnf_obj) if dnf_obj else "(cache miss)"
        tree_repr = _flatten_tree(tree_obj) if tree_obj else "(cache miss)"

        diverges = dnf_repr != tree_repr  # cheap string compare
        marker = "DIFF" if diverges else "    "
        if diverges:
            n_diff += 1
        print(f"\n[{marker}] qid={qid}  gold={gold[qid]}")
        print(f"  Q   : {text}")
        print(f"  DNF : {dnf_repr}")
        print(f"  TREE: {tree_repr}")

    print(f"\n=== {bench}: {n_diff}/{n_total} queries diverge structurally ===")


if __name__ == "__main__":
    main()
