"""Summarize locomo eval+search runs: macro c1234 accuracy + avg tokens/q.

Pairs each eval-*.json with its search-*.json sibling (same stem). Macro
accuracy = mean of per-category (1-4) accuracies. Tokens counted with
o200k_base on `conversation_memories` (the answerer context only).

Usage: uv run python summarize_runs.py [glob ...]
       uv run python summarize_runs.py 'eval-*v22qkey*'
"""

from __future__ import annotations

import glob
import json
import sys

import tiktoken

ENC = tiktoken.get_encoding("o200k_base")


def macro_acc(eval_path: str) -> tuple[float, dict, int]:
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
    macro = sum(cat_acc.values()) / len(cat_acc) if cat_acc else 0.0
    total_correct = sum(
        it.get("llm_score", 0)
        for cat in ("1", "2", "3", "4")
        for it in d.get(cat, [])
    )
    micro = total_correct / n if n else 0.0
    return macro, micro, cat_acc, n


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
    # eval-<stem>-<judgetags>.json -> search-<stem>.json
    base = eval_path[len("eval-") : -len(".json")]
    # strip trailing judge tags: -mini-mb-c14 / -gpt5-mb-c14 / -mb-c14 etc.
    for tag in ("-mini-mb-c14", "-gpt5-mb-c14", "-mb-c14", "-c14"):
        if base.endswith(tag):
            base = base[: -len(tag)]
            break
    cand = f"search-{base}.json"
    return cand if glob.glob(cand) else None


def main() -> None:
    patterns = sys.argv[1:] or ["eval-*.json"]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    rows = []
    for ep in paths:
        try:
            macro, micro, cat_acc, n = macro_acc(ep)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {ep}: {exc}")
            continue
        sp = search_for_eval(ep)
        tok = avg_tokens(sp) if sp else float("nan")
        rows.append((ep, macro, micro, cat_acc, n, tok))
    rows.sort(key=lambda r: r[2], reverse=True)
    print(f"{'MICRO':>7} {'macro':>7} {'tok/q':>7}  {'c1':>5} {'c2':>5} {'c3':>5} {'c4':>5}  n     file")
    for ep, macro, micro, cat_acc, n, tok in rows:
        cats = " ".join(
            f"{cat_acc.get(c, 0) * 100:5.1f}" for c in ("1", "2", "3", "4")
        )
        print(
            f"{micro * 100:7.2f} {macro * 100:7.2f} {tok:7.1f}  {cats}  {n:4d}  {ep[len('eval-'):]}"
        )


if __name__ == "__main__":
    main()
