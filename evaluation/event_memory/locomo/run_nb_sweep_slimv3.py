"""Neighbor-window sweep for slim_v3.

How many surrounding turns does the slim_v3 segmenter actually need for
reference resolution? v22 needed 8 (nb0->nb8 was +8pp). slim_v3 rewrites
to self-contained 3rd-person prose, so the dependence should be far
weaker -- nb0 already scores 89.22 vs ~90.45 default.

Fully matched sweep: fresh slim_v3 LLM re-ingest at windows 1/2/4/8
(direction both), segmenter gpt-5.4-nano @ low (= the 54n-l cell, which
is also what the existing nb0 used). nb2 doubles as a cross-check on the
untagged default run.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
SEARCH_TAG = "v28-e0-rnullbmfa50-l10-tsshort-seg"
WINDOWS = [1, 2, 4, 8]

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


def tag(w: int) -> str:
    return f"tslimv3-54n-l-nb{w}b"


def ingest(w: int) -> tuple[str, int]:
    t = tag(w)
    cmd = [
        "uv", "run", "python", "locomo_ingest.py",
        "--data-path", DATA,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--segmenter", "terse-decoupled-slim-v3",
        "--segmenter-model", "gpt-5.4-nano",
        "--segmenter-reasoning", "low",
        "--neighbor-window", str(w),
        "--neighbor-direction", "both",
    ]
    return run(cmd, f"log-ingest-{t}.out")


def search_and_eval(w: int) -> tuple[str, int]:
    t = tag(w)
    search = f"search-{t}-{SEARCH_TAG}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA,
        "--target-path", search,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
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
    print(f"=== ingest {len(WINDOWS)} slim_v3 windows {WINDOWS} ===", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(ingest, WINDOWS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== search + eval {len(WINDOWS)} windows ===", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(search_and_eval, WINDOWS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== NB SWEEP DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
