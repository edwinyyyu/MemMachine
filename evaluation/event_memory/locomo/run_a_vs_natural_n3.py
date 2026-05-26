"""Expand A verify and bo-natural to n=3 for direct comparison.

Already have rep1 for each:
  - tslimv3datealias-bm25only-54n-l-nb8-verify (rep1) = 90.97 mini-mb @5270 seg
  - tslimv3bonatural-54n-l-nb8-rep1 = 90.71 mini-mb @5346 seg

This script adds rep2 + rep3 for each, then re-evals. Total 4 ingest+search
+ 4 evals. mini-mb only (HANDOFF iteration). Same config across all reps
(54n@low, nb8 both, K=10, vec28, no-reranker, bm25-add 0.5, ts-short).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"


def run(cmd, log):
    with open(log, "w") as f:
        return log, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def ingest_and_eval(args):
    tag_base, segmenter_case, rep = args
    if rep == 1:
        # rep1 already complete with a slightly different tag for A verify
        if tag_base == "tslimv3datealias-bm25only":
            tag = f"{tag_base}-54n-l-nb8-verify"  # rep1 was "verify"
        else:
            tag = f"{tag_base}-54n-l-nb8-rep{rep}"
        return f"rep{rep} {tag_base} (already complete)", 0

    tag = f"{tag_base}-54n-l-nb8-rep{rep}"
    db = f"locomo-{tag}.sqlite"
    vdb = f"locomo-{tag}.vec.sqlite"
    if not os.path.exists(db):
        subprocess.run(f"rm -f {db}* {vdb}*", shell=True)
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", db, "--vector-db", vdb,
            "--segmenter", segmenter_case,
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{tag}.out")
        if rc != 0:
            return f"log-ingest-{tag}.out FAILED", rc

    stag = f"{tag}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    search = f"search-{stag}.json"
    if not os.path.exists(search):
        log, rc = run([
            "uv", "run", "python", "locomo_search.py",
            "--data-path", DATA, "--target-path", search,
            "--segment-db", db, "--vector-db", vdb,
            "--vector-search-limit", "28", "--expand-context", "0",
            "--max-num-segments", "10", "--no-reranker",
            "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
            "--timestamp-format", "short",
        ], f"log-search-{stag}.out")
        if rc != 0:
            return f"log-search-{stag}.out FAILED", rc

    eval_out = f"eval-{stag}-mini-mb-c14.json"
    log, rc = run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search, "--target-path", eval_out,
        "--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{stag}-mini-mb.out")
    return f"eval-{stag} done", rc


def main():
    t0 = time.time()
    # rep2 + rep3 for each of A (datealias-bm25only) and bo-natural
    tasks = [
        ("tslimv3datealias-bm25only", "terse-decoupled-slim-v3-datealias-bm25only", 2),
        ("tslimv3datealias-bm25only", "terse-decoupled-slim-v3-datealias-bm25only", 3),
        ("tslimv3bonatural", "terse-decoupled-slim-v3-bo-natural", 2),
        ("tslimv3bonatural", "terse-decoupled-slim-v3-bo-natural", 3),
    ]
    print(f"=== {len(tasks)} parallel ingest+search+eval (API-only, width 4) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for log, rc in ex.map(ingest_and_eval, tasks):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
