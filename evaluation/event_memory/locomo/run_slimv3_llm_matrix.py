"""Ship condition: slim_v3 segmenter robustness across LLM model + reasoning.

slim_v3 is the proposed ship. The segmenter prompt must give the SAME
behaviour across {gpt-5-nano, gpt-5.4-nano, gpt-5-mini} x {low, medium}
at the production config (nb8). All 6 cells, n=3 reps, searched at the
budget point (K=10 and K=11) and judged with gpt-5-mini (consistent
judge -> the cross-cell spread is the robustness signal).

Existing nb8 DBs reused: 54n-l (6 reps, K=11 already done by the confirm
batch -> excluded here), 5n-m (6 reps), 5m-m (6 reps).
Ingested here: 5n-l, 54n-m, 5m-l (3 reps each).
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3]

# cell -> (model, reasoning, needs_ingest)
CELLS = {
    "5n-l": ("gpt-5-nano", "low", True),
    "54n-m": ("gpt-5.4-nano", "medium", True),
    "5m-l": ("gpt-5-mini", "low", True),
    "5n-m": ("gpt-5-nano", "medium", False),
    "5m-m": ("gpt-5-mini", "medium", False),
}
KS = [10, 11]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(job: tuple[str, int]) -> tuple[str, int]:
    cell, rep = job
    model, effort, _ = CELLS[cell]
    t = f"tslimv3-{cell}-nb8-rep{rep}"
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    for _ in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            "--segmenter", "terse-decoupled-slim-v3",
            "--segmenter-model", model,
            "--segmenter-reasoning", effort,
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(job: tuple[str, int, int]) -> tuple[str, int]:
    cell, rep, k = job
    t = f"tslimv3-{cell}-nb8-rep{rep}"
    tag = f"{t}-v28-e0-rnullbmfa50-l{k}-tsshort-seg"
    search = f"search-{tag}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", search,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--vector-search-limit", "28", "--expand-context", "0",
        "--max-num-segments", str(k), "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
    ], f"log-search-{tag}.out")
    if s[1] != 0:
        return s
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{tag}-mini-mb-c14.json",
        "--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{tag}.out")


def main() -> None:
    t0 = time.time()
    ing = [(c, r) for c, v in CELLS.items() if v[2] for r in REPS]
    print(f"=== ingest {len(ing)} new model-cell DBs ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(ing)) as ex:
        for log, rc in ex.map(ingest, ing):
            print(f"  {log}  rc={rc}", flush=True)

    se = [(c, r, k) for c in CELLS for r in REPS for k in KS]
    print(f"=== search + eval {len(se)} (cell x rep x K) ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(se)) as ex:
        for log, rc in ex.map(search_eval, se):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== LLM MATRIX DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
