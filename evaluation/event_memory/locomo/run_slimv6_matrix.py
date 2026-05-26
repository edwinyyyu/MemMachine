"""slim_v6 vs slim_v3 -- does the objective prompt close the model gap?

slim_v6 = slim_v3 + objective keep-gate (item earned only if >=1
particular) + objective anti-redundancy/deletion rules for `terse`.
Hypothesis: weak models (gpt-5-nano) stop over-segmenting and stop
restating, so segment verbosity becomes model-independent and the
350-token budget holds at K=10 across all models.

Tests the full {gpt-5-nano, gpt-5.4-nano, gpt-5-mini} x {low, medium}
matrix at nb8, n=3, plus the 54n-l ship-gate:
  - all 6 cells: search K=10 + K=11, mini judge (the spread check)
  - 54n-l: + gpt-5 judge at K=10/11, + K=40 seg & rawev (ceiling /
    faithfulness -- must not regress vs slim_v3 93.92/93.90).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Concurrency caps -- the machine OOM'd on 18-wide ingest. API-only batch,
# but each process is ~0.7 GB; keep it modest and leave RAM headroom for
# a concurrent width-1 local-embedder batch.
INGEST_WIDTH = 4
SEARCH_WIDTH = 6

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3]
CELLS = {
    "5n-l": ("gpt-5-nano", "low"),
    "5n-m": ("gpt-5-nano", "medium"),
    "54n-l": ("gpt-5.4-nano", "low"),
    "54n-m": ("gpt-5.4-nano", "medium"),
    "5m-l": ("gpt-5-mini", "low"),
    "5m-m": ("gpt-5-mini", "medium"),
}


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(job: tuple[str, int]) -> tuple[str, int]:
    cell, rep = job
    model, effort = CELLS[cell]
    t = f"tslimv6-{cell}-nb8-rep{rep}"
    # Skip DBs that already completed (resume after the OOM restart).
    if (os.path.exists(f"locomo-{t}.sqlite")
            and os.path.exists(f"locomo-{t}.vec.sqlite")):
        return f"log-ingest-{t}.out (skipped: exists)", 0
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    for _ in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            "--segmenter", "terse-decoupled-slim-v6",
            "--segmenter-model", model,
            "--segmenter-reasoning", effort,
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def evaluate(tag, search, judge, suffix):
    jm = "gpt-5" if judge == "gpt5" else "gpt-5-mini"
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{tag}-{suffix}.json",
        "--judge-model", jm, "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{tag}-{suffix}.out")


def search_eval(job: tuple) -> tuple[str, int]:
    cell, rep, k, vsl, mode = job
    t = f"tslimv6-{cell}-nb8-rep{rep}"
    tag = f"{t}-v{vsl}-e0-rnullbmfa50-l{k}-tsshort-{mode}"
    search = f"search-{tag}.json"
    cmd = [
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", search,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--vector-search-limit", str(vsl), "--expand-context", "0",
        "--max-num-segments", str(k), "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
    ]
    if mode == "rawev":
        cmd.append("--answer-with-raw-events")
    rc = run(cmd, f"log-search-{tag}.out")[1]
    if rc != 0:
        return tag, rc
    evaluate(tag, search, "mini", "mini-mb-c14")
    if cell == "54n-l" and mode == "seg":
        evaluate(tag, search, "gpt5", "gpt5-mb-c14")
    return tag, 0


def main() -> None:
    t0 = time.time()
    ing = [(c, r) for c in CELLS for r in REPS]
    print(f"=== ingest {len(ing)} slim_v6 DBs (width {INGEST_WIDTH}) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=INGEST_WIDTH) as ex:
        for log, rc in ex.map(ingest, ing):
            print(f"  {log}  rc={rc}", flush=True)

    se = [(c, r, k, 28, "seg") for c in CELLS for r in REPS
          for k in (10, 11)]
    se += [("54n-l", r, 40, 160, m) for r in REPS for m in ("seg", "rawev")]
    print(f"=== search + eval {len(se)} (width {SEARCH_WIDTH}) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=SEARCH_WIDTH) as ex:
        for res in ex.map(search_eval, se):
            print(f"  {res}", flush=True)
    print(f"=== SLIMV6 MATRIX DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
