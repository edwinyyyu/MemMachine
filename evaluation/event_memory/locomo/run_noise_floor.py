"""Noise-floor measurement for the embed-format ablations.

The cache-reassembly variants share IDENTICAL segmentation, so the only
variance between them is answerer (gpt-5-mini QA) + judge (gpt-5-mini)
stochasticity. The embed-format deltas are all sub-1pp -- before any of
them can be called signal, the floor must be known.

Re-run search + eval on the SAME locomo-deco-cur DB N times. The spread
(max-min, stdev) of these identical-input runs IS the noise floor. Any
ablation delta below it is noise.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
SEARCH_TAG = "v28-e0-rnullbmfa50-l10-tsshort-seg"
N_REPLICATES = 4
SEARCH_ARGS = [
    "--vector-search-limit", "28",
    "--expand-context", "0",
    "--max-num-segments", "10",
    "--no-reranker",
    "--bm25-fusion", "additive",
    "--bm25-fusion-weight", "0.5",
    "--timestamp-format", "short",
]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def search_and_eval(rep: int) -> tuple[str, int]:
    search = f"search-deco-cur-rep{rep}-{SEARCH_TAG}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA,
        "--target-path", search,
        "--segment-db", "locomo-deco-cur.sqlite",
        "--vector-db", "locomo-deco-cur.vec.sqlite",
        *SEARCH_ARGS,
    ], f"log-search-deco-cur-rep{rep}.out")
    if s[1] != 0:
        return s
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path",
        f"eval-deco-cur-rep{rep}-{SEARCH_TAG}-mini-mb-c14.json",
        "--judge-model", "gpt-5-mini",
        "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-deco-cur-rep{rep}.out")


def main() -> None:
    t0 = time.time()
    reps = list(range(2, 2 + N_REPLICATES))  # rep2..rep5; rep1 == existing cur
    print(f"=== noise floor: {len(reps)} replicate runs of cur ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(reps)) as ex:
        for log, rc in ex.map(search_and_eval, reps):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== NOISE FLOOR DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
