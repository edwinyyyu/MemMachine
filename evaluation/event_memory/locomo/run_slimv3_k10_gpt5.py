"""K=10 gpt-5 judge for slim_v3-54n-l-nb8 (top-10 budget point headline).

K=10 search JSONs already exist (from run_slimv3_nb8, l10). Re-judge the
6 reps with the gpt-5 (Mem0-comparable) judge.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def evaluate(rep: int) -> tuple[str, int]:
    tag = f"tslimv3-54n-l-nb8-rep{rep}-v28-e0-rnullbmfa50-l10-tsshort-seg"
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
    print("=== K=10 gpt-5 judge, 6 reps ===", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for log, rc in ex.map(evaluate, [1, 2, 3, 4, 5, 6]):
            print(f"  {log}  rc={rc}", flush=True)
    print("=== K10 GPT5 DONE ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
