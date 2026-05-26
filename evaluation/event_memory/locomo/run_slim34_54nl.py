"""slim_v3 vs slim_v4 on the PRODUCTION segmenter cell (gpt-5.4-nano @ low).

The slim_v3-vs-slim_v4 extremes test showed a leveling trade: nano +0.33,
mini -0.51. Production uses gpt-5.4-nano @ low (54n-l) -- a nano-class
model -- so slim_v4 should be positive-to-neutral there, but the extremes
don't prove it. This is the decision-relevant matched test: both prompts,
54n-l, neighbor-window 8, n=6 each.
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
# (prompt-version tag, ingest segmenter case)
PROMPTS = {
    "v3": "terse-decoupled-slim-v3",
    "v4": "terse-decoupled-slim-v4",
}
REPS = [1, 2, 3, 4, 5, 6]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def jobs() -> list[tuple[str, str, int]]:
    return [
        (ver, seg, rep)
        for ver, seg in PROMPTS.items()
        for rep in REPS
    ]


def ingest(job: tuple[str, str, int]) -> tuple[str, int]:
    ver, seg, rep = job
    t = f"tslim{ver}-54n-l-nb8-rep{rep}"
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True
    )
    for _ in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            "--segmenter", seg,
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8",
            "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True
        )
    return f"log-ingest-{t}.out", rc


def search_and_eval(job: tuple[str, str, int]) -> tuple[str, int]:
    ver, _seg, rep = job
    t = f"tslim{ver}-54n-l-nb8-rep{rep}"
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
    print(f"=== ingest {len(js)} DBs (slim_v3/v4 x 54n-l x nb8) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(ingest, js):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== search + eval {len(js)} ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(js)) as ex:
        for log, rc in ex.map(search_and_eval, js):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== SLIM34 54NL BATCH DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
