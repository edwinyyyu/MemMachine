"""One-off: re-judge slim_v4 answers with Mem0's ORIGINAL LoCoMo evaluator.

Non-rigorous comparison point — the classic judge is gpt-4o-mini +
mem0-classic (the single-message CORRECT/WRONG rubric), the pairing
Mem0's original LoCoMo eval used. Re-judges existing slim_v4-54n-l-nb8
K=12/K=13 search files (no re-search). Lets us see our number under the
classic-style evaluator alongside the gpt-5-mini / gpt-5 mem0-bench
numbers we iterate on.
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


def judge(job: tuple[int, int]) -> tuple[str, int]:
    k, rep = job
    t = f"tslimv4-54n-l-nb8-rep{rep}-v28-e0-rnullbmfa50-l{k}-tsshort-seg"
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", f"search-{t}.json",
        "--target-path", f"eval-{t}-4omini-mc-c14.json",
        "--judge-model", "gpt-4o-mini",
        "--judge-variant", "mem0-classic",
        "--skip-category-5",
    ], f"log-eval-{t}-4omini-mc.out")


def main() -> None:
    t0 = time.time()
    js = [(k, rep) for k in KS for rep in REPS]
    print(f"=== classic judge (gpt-4o-mini + mem0-classic): {len(js)} "
          f"evals ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(judge, js):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== CLASSIC JUDGE DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
