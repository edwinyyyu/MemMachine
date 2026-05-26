"""Finish the MiniLM raw-text floor (text+whole) -- search+eval only.

The text-whole-minilm DB is already ingested; the embed-robust batch's
floor searches never completed. Search K=6/7 with --embedding-model
minilm, eval mini. Local model -> width 1.
"""

from __future__ import annotations

import subprocess
import sys

DATA = "../../data/locomo10.json"


def run(cmd: list[str], log: str) -> int:
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main() -> None:
    for k in (6, 7):
        tag = f"text-whole-minilm-v28-e0-rnullbmfa50-l{k}-tsshort-seg"
        search = f"search-{tag}.json"
        rc = run([
            "uv", "run", "python", "locomo_search.py",
            "--data-path", DATA, "--target-path", search,
            "--segment-db", "locomo-text-whole-minilm.sqlite",
            "--vector-db", "locomo-text-whole-minilm.vec.sqlite",
            "--vector-search-limit", "28", "--expand-context", "0",
            "--max-num-segments", str(k), "--no-reranker",
            "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
            "--timestamp-format", "short", "--embedding-model", "minilm",
        ], f"log-search-{tag}.out")
        print(f"  search K={k} rc={rc}", flush=True)
        if rc != 0:
            continue
        rc = run([
            "uv", "run", "python", "locomo_evaluate.py",
            "--data-path", search,
            "--target-path", f"eval-{tag}-mini-mb-c14.json",
            "--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
            "--skip-category-5",
        ], f"log-eval-{tag}.out")
        print(f"  eval K={k} rc={rc}", flush=True)
    print("=== MINILM FLOOR DONE ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
