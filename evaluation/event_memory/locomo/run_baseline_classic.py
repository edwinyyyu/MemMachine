"""Re-eval slim_v3 production baseline (n=6) with mem0-classic judge.

For apples-to-apples comparison against the ship-candidate
(`tslimv3boverbve`) which was just re-evaluated under mem0-classic.
Baseline search files exist from prior production run; just reuse them.
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
    tag = f"tslimv3-54n-l-nb8-rep{rep}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    search_json = f"search-{tag}.json"
    suffix = "mini" if judge == "gpt-5-mini" else "gpt5"
    out = f"eval-{tag}-{suffix}-mc-c14.json"
    log = f"log-eval-{tag}-{suffix}-mc.out"
    if not os.path.exists(search_json):
        return f"{log} (no search file)", 1
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search_json,
        "--target-path", out,
        "--judge-model", judge, "--judge-variant", "mem0-classic",
        "--skip-category-5",
    ], log)


def main():
    t0 = time.time()
    tasks = [(rep, j) for rep in REPS for j in ("gpt-5-mini", "gpt-5")]
    print(f"=== mem0-classic baseline eval, 6 reps x 2 judges, parallel ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for log, rc in ex.map(eval_one, tasks):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== BASELINE CLASSIC DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
