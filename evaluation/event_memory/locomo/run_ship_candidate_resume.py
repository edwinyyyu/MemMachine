"""Resume ship-candidate run after sleep; switch judge to mem0-classic.

State at resume:
  - all 6 ingests complete
  - rep1-4 search complete (3.9MB JSONs)
  - rep5, rep6 search died mid-run
  - prior evals were mem0-bench; user wants mem0-classic instead

This script:
  1. Re-runs search for rep5, rep6 only.
  2. Runs mem0-classic mini + gpt-5 evals on all 6 reps in parallel.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3, 4, 5, 6]


def run(cmd, log):
    with open(log, "w") as f:
        return log, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def search(rep):
    t = f"tslimv3boverbve-54n-l-nb8-rep{rep}"
    tag = f"{t}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    out = f"search-{tag}.json"
    # large json => already complete; skip
    if os.path.exists(out) and os.path.getsize(out) > 2_000_000:
        return f"search-{tag} (skipped, complete)", 0
    return run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", out,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--vector-search-limit", "28", "--expand-context", "0",
        "--max-num-segments", "10", "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
    ], f"log-search-{tag}.out")


def eval_one(args):
    rep, judge = args
    t = f"tslimv3boverbve-54n-l-nb8-rep{rep}"
    tag = f"{t}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    search_json = f"search-{tag}.json"
    suffix = "mini" if judge == "gpt-5-mini" else "gpt5"
    # mc = mem0-classic; distinct from prior mb (mem0-bench)
    out = f"eval-{tag}-{suffix}-mc-c14.json"
    log = f"log-eval-{tag}-{suffix}-mc.out"
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search_json,
        "--target-path", out,
        "--judge-model", judge, "--judge-variant", "mem0-classic",
        "--skip-category-5",
    ], log)


def main():
    t0 = time.time()
    print(f"=== re-search rep5, rep6 (width 2) ===", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(search, REPS):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== mem0-classic eval, 6 reps x 2 judges, parallel ===", flush=True)
    tasks = [(rep, j) for rep in REPS for j in ("gpt-5-mini", "gpt-5")]
    with ThreadPoolExecutor(max_workers=12) as ex:
        for log, rc in ex.map(eval_one, tasks):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== RESUME DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
