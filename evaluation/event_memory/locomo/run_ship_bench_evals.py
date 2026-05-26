"""Run mem0-bench evals on the existing 6 ship-candidate B search files.

The 6 search files already exist (3.9MB each):
  search-tslimv3boverbve-54n-l-nb8-rep{1..6}-v28-e0-rnullbmfa50-l10-tsshort-seg.json

Run mini + gpt-5 judges, mem0-bench variant -- matches the original ship
validation protocol so this number is comparable to the existing A row
(BM25-only verbose, regex on): 91.45 mini / 89.39 gpt-5 (n=6 bench).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPS = [1, 2, 3, 4, 5, 6]


def run(cmd, log):
    with open(log, "w") as f:
        return log, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def eval_one(args):
    rep, judge = args
    tag = f"tslimv3boverbve-54n-l-nb8-rep{rep}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    search_json = f"search-{tag}.json"
    suffix = "mini" if judge == "gpt-5-mini" else "gpt5"
    out = f"eval-{tag}-{suffix}-mb-c14.json"
    log = f"log-eval-{tag}-{suffix}-mb.out"
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search_json,
        "--target-path", out,
        "--judge-model", judge, "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], log)


def main():
    t0 = time.time()
    tasks = [(rep, j) for rep in REPS for j in ("gpt-5-mini", "gpt-5")]
    print(f"=== mem0-bench ship evals, {len(tasks)} parallel ===", flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for log, rc in ex.map(eval_one, tasks):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
