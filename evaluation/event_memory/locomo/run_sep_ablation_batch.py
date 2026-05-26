"""Separator ablation for text_to_embed.

M,C,Q embed order (the reorder-ablation top), dates dropped. Vary only
the string joining the three components: newline / space / blank-line.
Builds DBs via cache-reassembly, runs search + eval per variant.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
CACHE = "cache-terse-v2-raw.json"
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
VARIANTS = [
    "sep_mcq_nl", "sep_mcq_sp", "sep_mcq_nl2",
    "lab_q_off", "lab_d_off", "lab_qd_off",
]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(variant: str) -> tuple[str, int]:
    cmd = [
        "uv", "run", "python", "locomo_ingest.py",
        "--data-path", DATA,
        "--segment-db", f"locomo-deco-{variant}.sqlite",
        "--vector-db", f"locomo-deco-{variant}.vec.sqlite",
        "--segmenter", "decoupling-ablation",
        "--reassembly-variant", variant,
        "--segments-cache", CACHE,
    ]
    return run(cmd, f"log-ingest-{variant}.out")


def search_and_eval(variant: str) -> tuple[str, int]:
    search = f"search-deco-{variant}-{SEARCH_TAG}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA,
        "--target-path", search,
        "--segment-db", f"locomo-deco-{variant}.sqlite",
        "--vector-db", f"locomo-deco-{variant}.vec.sqlite",
        *SEARCH_ARGS,
    ], f"log-search-deco-{variant}.out")
    if s[1] != 0:
        return s
    target = f"eval-deco-{variant}-{SEARCH_TAG}-mini-mb-c14.json"
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", target,
        "--judge-model", "gpt-5-mini",
        "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-deco-{variant}.out")


def main() -> None:
    t0 = time.time()
    print(f"=== ingest {len(VARIANTS)} separator DBs ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for log, rc in ex.map(ingest, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== search + eval {len(VARIANTS)} variants ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for log, rc in ex.map(search_and_eval, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== SEP BATCH DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
