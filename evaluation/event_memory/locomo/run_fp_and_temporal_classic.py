"""Parallel ingest+search+classic eval for the two new variants.

Variants:
  - bo-natural-fp: first-person memory rewrite (hypothesis: 1p makes
    temporal resolution easier).
  - bo-natural-temporal: 3p bo-natural with strengthened WRONG/RIGHT
    temporal-resolution examples in the prompt.

Methodology (matches HANDOFF iteration): 54n@low segmenter, nb8 both,
K=10, vec-28, no-reranker, BM25 add 0.5, ts-short. Judge: gpt-5-mini +
mem0-CLASSIC (user said bench is too lenient at this accuracy level).
Single rep n=1.

Compare to:
  - bo-natural rep1 mini-classic: 84.22% @309 tok
  - bo-natural raw-events ceiling: 85.91% @478 tok
"""

from __future__ import annotations
from artifacts import A  # canonical artifact names

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"

VARIANTS = [
    ("tslimv3bonaturalfp", "terse-decoupled-slim-v3-bo-natural-fp"),
    ("tslimv3bonaturaltemp", "terse-decoupled-slim-v3-bo-natural-temporal"),
]


def run(cmd, log):
    with open(log, "w") as f:
        return log, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def full_pipeline(args):
    tag_base, segmenter_case = args
    tag = f"{tag_base}-54n-l-nb8-rep1"
    db = A(f"locomo-{tag}.sqlite")
    vdb = A(f"locomo-{tag}.vec.sqlite")

    # Ingest
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
        ], A(f"log-ingest-{tag}.out"))
        if rc != 0:
            return f"INGEST FAIL {tag}", rc

    stag = f"{tag}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    search = A(f"search-{stag}.json")
    if not os.path.exists(search):
        log, rc = run([
            "uv", "run", "python", "locomo_search.py",
            "--data-path", DATA, "--target-path", search,
            "--segment-db", db, "--vector-db", vdb,
            "--vector-search-limit", "28", "--expand-context", "0",
            "--max-num-segments", "10", "--no-reranker",
            "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
            "--timestamp-format", "short",
        ], A(f"log-search-{stag}.out"))
        if rc != 0:
            return f"SEARCH FAIL {tag}", rc

    out = A(f"eval-{stag}-mini-mc-c14.json")
    log = A(f"log-eval-{stag}-mini-mc.out")
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search, "--target-path", out,
        "--judge-model", "gpt-5-mini", "--judge-variant", "mem0-classic",
        "--skip-category-5",
    ], log)


def main():
    t0 = time.time()
    print(f"=== Launching {len(VARIANTS)} variants (parallel) ===", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(full_pipeline, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
