"""BM25-fusion ablation: the HANDOFF production config, fusion OFF.

Current approach (HANDOFF §9): slim_v3, segmenter gpt-5.4-nano@low,
nb8, vsl=28, e0, no-reranker, tsshort, additive BM25 fusion weight 0.5,
budget point K=10/11. This re-searches the SAME slim_v3-54n-l-nb8 DBs
(reps 1-6) with --bm25-fusion none -> pure vector retrieval, everything
else identical. Judged mini + gpt-5, mem0-bench.

Baseline (fusion on, from the confirm batch):
  K=10  90.80 mini / 88.48 gpt-5 @310t
  K=11  90.76 mini / 88.74 gpt-5 @341t
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3, 4, 5, 6]
KS = [10, 11]
WIDTH = 4  # API-only but capped -- a slim_v6 batch + gemma batch are live


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def search_eval(job: tuple[int, int]) -> tuple[str, int]:
    rep, k = job
    t = f"tslimv3-54n-l-nb8-rep{rep}"
    tag = f"{t}-v28-e0-rnullbmnone-l{k}-tsshort-seg"
    search = f"search-{tag}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", search,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--vector-search-limit", "28", "--expand-context", "0",
        "--max-num-segments", str(k), "--no-reranker",
        "--bm25-fusion", "none",
        "--timestamp-format", "short",
    ], f"log-search-{tag}.out")
    if s[1] != 0:
        return s
    for judge, suffix in [("gpt-5-mini", "mini"), ("gpt-5", "gpt5")]:
        run([
            "uv", "run", "python", "locomo_evaluate.py",
            "--data-path", search,
            "--target-path", f"eval-{tag}-{suffix}-mb-c14.json",
            "--judge-model", judge, "--judge-variant", "mem0-bench",
            "--skip-category-5",
        ], f"log-eval-{tag}-{suffix}.out")
    return tag, 0


def main() -> None:
    t0 = time.time()
    jobs = [(r, k) for r in REPS for k in KS]
    print(f"=== no-BM25 ablation: {len(jobs)} search+eval (width {WIDTH}) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=WIDTH) as ex:
        for res in ex.map(search_eval, jobs):
            print(f"  {res}", flush=True)
    print(f"=== NO-BM25 ABLATION DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
