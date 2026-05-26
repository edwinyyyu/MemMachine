"""Mem0-comparable judge for the slim_v4-54n-l winner.

slim_v4 @ 54n-l @ nb8 on the mini judge: K=12 = 91.24 @ 320t, K=13 =
91.53 @ 346t -- both inside the <=350 budget. The Mem0-comparable number
is the gpt-5 judge (mem0-bench). Re-judge the existing K=12 and K=13
search files with gpt-5; n=6 each for a solid mean.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

KS = [12, 13]
REPS = [1, 2, 3, 4, 5, 6]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def jobs() -> list[tuple[int, int]]:
    return [(k, rep) for k in KS for rep in REPS]


def judge(job: tuple[int, int]) -> tuple[str, int]:
    k, rep = job
    t = f"tslimv4-54n-l-nb8-rep{rep}-v28-e0-rnullbmfa50-l{k}-tsshort-seg"
    search = f"search-{t}.json"
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{t}-gpt5-mb-c14.json",
        "--judge-model", "gpt-5",
        "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{t}-gpt5.out")


def main() -> None:
    t0 = time.time()
    js = jobs()
    print(f"=== gpt-5 judge: {len(js)} slim_v4 evals (K={KS}) ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(judge, js):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== GPT5 JUDGE DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
