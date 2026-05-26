"""#135: does slim_v5's disambiguated keep/drop rule close the model gap?

slim_v3 baselines (3 reps each, mini judge):
  gpt-5-nano @ medium  mean 89.93   (the weak model)
  gpt-5-mini @ medium  mean 91.04

slim_v5 reframes keep/drop as a life-fact vs conversation-move dichotomy.
Re-ingest both extremes with slim_v5, 3 replicates each, search + eval.
Win condition: nano rises toward mini WITHOUT mini regressing.
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
# (cell tag, segmenter model, reasoning)
MODELS = {
    "5n-m": ("gpt-5-nano", "medium"),
    "5m-m": ("gpt-5-mini", "medium"),
}
REPS = [4, 5, 6]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def jobs() -> list[tuple[str, str, str, int]]:
    return [
        (cell, model, effort, rep)
        for cell, (model, effort) in MODELS.items()
        for rep in REPS
    ]


def ingest(job: tuple[str, str, str, int]) -> tuple[str, int]:
    cell, model, effort, rep = job
    t = f"tslimv5-{cell}-rep{rep}"
    # Clean sqlite sidecars before a fresh build (stale WAL -> I/O error).
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True
    )
    for attempt in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            "--segmenter", "terse-decoupled-slim-v5",
            "--segmenter-model", model,
            "--segmenter-reasoning", effort,
            "--neighbor-window", "8",
            "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True
        )
    return f"log-ingest-{t}.out", rc


def search_and_eval(job: tuple[str, str, str, int]) -> tuple[str, int]:
    cell, _model, _effort, rep = job
    t = f"tslimv5-{cell}-rep{rep}"
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
    js = jobs()
    print(f"=== slim_v5 ingest {len(js)} DBs ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(ingest, js):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== slim_v5 search + eval {len(js)} ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(search_and_eval, js):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== SLIM_V4 GAP BATCH DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
