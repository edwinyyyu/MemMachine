"""Replicate MCQ / MQC / no-dates to resolve two sub-noise questions.

All three DBs already exist (cache-reassembly) -- replication is just
re-running search+eval, capturing answerer+judge noise.

  cur        M,Q,C,D   -- MQC order; already 5 runs (noise floor), +2
  emb_mcq    M,C,Q,D   -- MCQ order; had 1 run, +4 -> 5
  emb_nodate M,Q,C     -- dates dropped from embed; had 1 run, +4 -> 5

Compare MCQ mean vs MQC mean (order effect) and emb_nodate mean vs cur
mean (dates-in-embed effect). Noise floor sigma~=0.40, so even at n=5
a <0.3pp gap won't reach significance -- this gives a direction/lean,
not proof.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
SEARCH_TAG = "v28-e0-rnullbmfa50-l10-tsshort-seg"
SEARCH_ARGS = [
    "--vector-search-limit", "28",
    "--expand-context", "0",
    "--max-num-segments", "10",
    "--no-reranker",
    "--bm25-fusion", "additive",
    "--bm25-fusion-weight", "0.5",
    "--timestamp-format", "short",
]
# (variant DB stem, rep label) -- DB = locomo-deco-<stem>.sqlite
JOBS = (
    [("cur", r) for r in (6, 7)]
    + [("emb_mcq", r) for r in (2, 3, 4, 5)]
    + [("emb_nodate", r) for r in (2, 3, 4, 5)]
)


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def search_and_eval(job: tuple[str, int]) -> tuple[str, int]:
    stem, rep = job
    t = f"deco-{stem}-rep{rep}"
    search = f"search-{t}-{SEARCH_TAG}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA,
        "--target-path", search,
        "--segment-db", f"locomo-deco-{stem}.sqlite",
        "--vector-db", f"locomo-deco-{stem}.vec.sqlite",
        *SEARCH_ARGS,
    ], f"log-search-{t}.out")
    if s[1] != 0:
        return s
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{t}-{SEARCH_TAG}-mini-mb-c14.json",
        "--judge-model", "gpt-5-mini",
        "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{t}.out")


def main() -> None:
    t0 = time.time()
    print(f"=== {len(JOBS)} replicate search+eval runs ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(JOBS)) as ex:
        for log, rc in ex.map(search_and_eval, JOBS):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== ORDER/DATE REPLICATE DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
