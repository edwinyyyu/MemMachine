"""gpt-5 judge on the slim_v3 weak model cells (gpt-5-nano).

The slim_v3 LLM matrix was mini-judge only. To know whether the weakest
model still clears Mem0 (87.3) on the strict gpt-5 judge, re-judge the
gpt-5-nano cells (5n-l, 5n-m) at K=10. Search JSONs already exist.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def evaluate(job: tuple[str, int]) -> tuple[str, int]:
    cell, rep = job
    tag = f"tslimv3-{cell}-nb8-rep{rep}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    log = f"log-eval-{tag}-gpt5-mb-c14.out"
    with open(log, "w") as f:
        p = subprocess.run([
            "uv", "run", "python", "locomo_evaluate.py",
            "--data-path", f"search-{tag}.json",
            "--target-path", f"eval-{tag}-gpt5-mb-c14.json",
            "--judge-model", "gpt-5", "--judge-variant", "mem0-bench",
            "--skip-category-5",
        ], stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def main() -> None:
    jobs = [(c, r) for c in ("5n-l", "5n-m") for r in (1, 2, 3)]
    print(f"=== slim_v3 gpt-5-nano gpt-5 judge: {len(jobs)} ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for log, rc in ex.map(evaluate, jobs):
            print(f"  {log}  rc={rc}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
