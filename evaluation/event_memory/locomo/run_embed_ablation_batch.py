"""Orchestrate the text_to_embed ablation batch.

Phase 1: evaluate the 7 already-run deco searches (embed reorder + date
         ablation) plus the nb0 neighbors ablation.
Phase 2: run the 6 additive-lattice deco searches (e_m..e_qc).
Phase 3: evaluate those 6.

All evals: gpt-5-mini judge, mem0-bench, skip-category-5 -- the standing
iteration stack. All searches: K=10, vsl=28, e0, no-reranker,
additive bm25 w=0.5, timestamp short -- matches the `cur` baseline
(terse-decoupled-v2 K=10 mini = 91.17).
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

# Phase 1: searches already on disk -> just evaluate.
PHASE1_SEARCHES = [
    f"search-deco-{v}-{SEARCH_TAG}.json"
    for v in ["emb_mcq", "emb_qmc", "emb_qcm", "emb_cmq", "emb_cqm",
              "emb_nodate", "nodate_all"]
] + ["search-tslimv3-54n-l-nb0-v28-e0-rnullbmfa50-l10-tsshort-seg.json"]

# Phase 2: additive-lattice variants -- DBs already built, need search.
LATTICE = ["e_m", "e_q", "e_c", "e_mq", "e_mc", "e_qc"]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def eval_search(search_json: str) -> tuple[str, int]:
    stem = search_json[len("search-"):-len(".json")]
    target = f"eval-{stem}-mini-mb-c14.json"
    cmd = [
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search_json,
        "--target-path", target,
        "--judge-model", "gpt-5-mini",
        "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ]
    return run(cmd, f"log-{target}.out")


def do_search(variant: str) -> tuple[str, int]:
    target = f"search-deco-{variant}-{SEARCH_TAG}.json"
    cmd = [
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA,
        "--target-path", target,
        "--segment-db", f"locomo-deco-{variant}.sqlite",
        "--vector-db", f"locomo-deco-{variant}.vec.sqlite",
        *SEARCH_ARGS,
    ]
    return run(cmd, f"log-{target}.out")


def batched(fn, items, width):
    out = []
    with ThreadPoolExecutor(max_workers=width) as ex:
        for log, rc in ex.map(fn, items):
            print(f"  done {log}  rc={rc}", flush=True)
            out.append((log, rc))
    return out


def main() -> None:
    t0 = time.time()
    print(f"=== Phase 1: evaluate {len(PHASE1_SEARCHES)} searches ===", flush=True)
    batched(eval_search, PHASE1_SEARCHES, 4)

    print(f"=== Phase 2: run {len(LATTICE)} lattice searches ===", flush=True)
    batched(do_search, LATTICE, 3)

    print(f"=== Phase 3: evaluate {len(LATTICE)} lattice searches ===", flush=True)
    lattice_searches = [
        f"search-deco-{v}-{SEARCH_TAG}.json" for v in LATTICE
    ]
    batched(eval_search, lattice_searches, 4)

    print(f"=== ALL DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
