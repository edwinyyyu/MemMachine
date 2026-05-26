"""slim_v4 @ 54n-l: spend the freed token budget on more K.

slim_v4 on the production cell scores 90.85 @ 267 tok/q -- vs a 340-token
target, that is ~70 tokens of headroom (slim_v3 used 310). The K=40
finding showed accuracy rises with K and the budget was the ceiling. So
re-search the existing slim_v4-54n-l DBs at K=11/12/13 to spend the
headroom; K=13 should land near 340-350. No re-ingest -- search+eval
only. n=6 per K.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
KS = [11, 12, 13]
REPS = [1, 2, 3, 4, 5, 6]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def jobs() -> list[tuple[int, int]]:
    return [(k, rep) for k in KS for rep in REPS]


def search_and_eval(job: tuple[int, int]) -> tuple[str, int]:
    k, rep = job
    db = f"locomo-tslimv4-54n-l-nb8-rep{rep}"
    tag = f"v28-e0-rnullbmfa50-l{k}-tsshort-seg"
    t = f"tslimv4-54n-l-nb8-rep{rep}-{tag}"
    search = f"search-{t}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA,
        "--target-path", search,
        "--segment-db", f"{db}.sqlite",
        "--vector-db", f"{db}.vec.sqlite",
        "--vector-search-limit", "28",
        "--expand-context", "0",
        "--max-num-segments", str(k),
        "--no-reranker",
        "--bm25-fusion", "additive",
        "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
    ], f"log-search-{t}.out")
    if s[1] != 0:
        return s
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{t}-mini-mb-c14.json",
        "--judge-model", "gpt-5-mini",
        "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{t}.out")


def main() -> None:
    t0 = time.time()
    js = jobs()
    print(f"=== slim_v4 54n-l K-bump: {len(js)} search+eval (K={KS}) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(search_and_eval, js):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== K-BUMP DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
