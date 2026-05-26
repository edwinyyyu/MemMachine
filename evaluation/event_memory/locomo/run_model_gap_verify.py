"""#135 step 1: is the slim_v3 cross-model gap real, or noise?

The 6-model matrix (single runs each) spanned 89.87 (gpt-5-nano @ medium,
"weakest") to 91.95 (gpt-5-mini @ medium, "strongest") = 2.08pp. But the
eval noise floor is sigma~=0.40 / range~1.1pp, and a 2pp spread across 6
single runs is partly noise inflation. Before doing any prompt-
disambiguation work to "raise the weakest model", verify the gap exists.

Replicate the two matrix extremes: 2 fresh stochastic re-ingests each
(rep2, rep3) -> with the existing matrix run that is 3 points per model.
If the means land within ~1pp, the prompt is already model-robust and
#135 is a no-op. Window 2 to match the original matrix ingests.
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
# (cell tag, segmenter model, reasoning) -- matrix extremes.
MODELS = {
    "5n-m": ("gpt-5-nano", "medium"),
    "5m-m": ("gpt-5-mini", "medium"),
}
REPS = [2, 3]


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
    t = f"tslimv3-{cell}-rep{rep}"
    cmd = [
        "uv", "run", "python", "locomo_ingest.py",
        "--data-path", DATA,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--segmenter", "terse-decoupled-slim-v3",
        "--segmenter-model", model,
        "--segmenter-reasoning", effort,
        "--neighbor-window", "2",
        "--neighbor-direction", "both",
    ]
    return run(cmd, f"log-ingest-{t}.out")


def search_and_eval(job: tuple[str, str, str, int]) -> tuple[str, int]:
    cell, _model, _effort, rep = job
    t = f"tslimv3-{cell}-rep{rep}"
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
    print(f"=== ingest {len(js)} model-replicate DBs ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(ingest, js):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== search + eval {len(js)} replicates ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(search_and_eval, js):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== MODEL-GAP VERIFY DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
