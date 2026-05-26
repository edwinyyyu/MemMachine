"""6 mem0-bench evals (n=3 reps x mini+gpt-5) on the ship config.

Just enough fresh evals on the actual ship config (`tslimv3boverbve`) to
check whether it tracks the `boverb` n=3 prior (91.28 mini / 89.65 gpt-5)
and is comparable to A (n=6: 91.45 mini / 89.39 gpt-5). If yes, ship; if
divergent, expand to n=6.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPS = [1, 2, 3]


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
    print(f"=== ship-bench evals n=3 x 2 = {len(tasks)} parallel ===", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for log, rc in ex.map(eval_one, tasks):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
