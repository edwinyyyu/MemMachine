"""Summarize locomo eval+search runs: c1234 / c124 accuracy + avg tokens/q.

Pairs each eval-*.json with its search-*.json sibling (same stem). Accuracy is the question-weighted mean (c1234 = cats 1-4, c124 drops cat3)
-- never the arithmetic mean of per-category rates. Tokens counted with
o200k_base on `conversation_memories` (the answerer context only).

Usage: uv run python summarize_runs.py [glob ...]
       uv run python summarize_runs.py 'eval-*v22qkey*'
"""

from __future__ import annotations
from artifacts import A, pattern, sibling  # canonical artifact names

import glob
import json
import sys

import tiktoken

ENC = tiktoken.get_encoding("o200k_base")


def acc(eval_path: str) -> tuple[float, float, dict, int]:
    """Question-weighted accuracy. c1234 = cats 1-4, c124 drops cat3.

    ALWAYS weight by question count -- never average the per-category rates.
    cat4 is ~55% of the bench and cat3 ~6%, so an arithmetic mean of category
    scores silently reweights the benchmark and is not comparable to Mem0.
    """
    d = json.load(open(eval_path))
    cat_acc = {}
    n = 0
    for cat in ("1", "2", "3", "4"):
        items = d.get(cat, [])
        if not items:
            continue
        scores = [it.get("llm_score", 0) for it in items]
        cat_acc[cat] = sum(scores) / len(scores)
        n += len(items)
    def weighted(cats):
        hit = sum(it.get("llm_score", 0) for c in cats for it in d.get(c, []))
        tot = sum(len(d.get(c, [])) for c in cats)
        return hit / tot if tot else 0.0

    c1234 = weighted(("1", "2", "3", "4"))
    c124 = weighted(("1", "2", "4"))
    return c1234, c124, cat_acc, n


def avg_tokens(search_path: str) -> float:
    d = json.load(open(search_path))
    toks, n = 0, 0
    for cat in ("1", "2", "3", "4"):
        for it in d.get(cat, []):
            ctx = it.get("conversation_memories", "")
            toks += len(ENC.encode(ctx))
            n += 1
    return toks / n if n else 0.0


def search_for_eval(eval_path: str) -> str | None:
    # canonical: {run}__eval__{judge tags}.json -> {run}__search.json
    cand = sibling(eval_path, "search", ".json")
    return cand if cand and glob.glob(cand) else None


def main() -> None:
    patterns = sys.argv[1:] or [pattern("eval", ".json")]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    rows = []
    for ep in paths:
        try:
            c1234, c124, cat_acc, n = acc(ep)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {ep}: {exc}")
            continue
        sp = search_for_eval(ep)
        tok = avg_tokens(sp) if sp else float("nan")
        rows.append((ep, c1234, c124, cat_acc, n, tok))
    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"{'c1234':>7} {'c124':>7} {'tok/q':>7}  {'c1':>5} {'c2':>5} {'c3':>5} {'c4':>5}  n     file")
    for ep, c1234, c124, cat_acc, n, tok in rows:
        cats = " ".join(
            f"{cat_acc.get(c, 0) * 100:5.1f}" for c in ("1", "2", "3", "4")
        )
        print(
            f"{c1234 * 100:7.2f} {c124 * 100:7.2f} {tok:7.1f}  {cats}  {n:4d}  {ep}"
        )


if __name__ == "__main__":
    main()
