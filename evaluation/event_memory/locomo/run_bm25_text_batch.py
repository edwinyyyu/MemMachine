"""BM25-visible text ablation -- close the #133 gap.

Embed fixed at M,Q,C (the settled recipe). Vary only
text_to_score_bm25: raw chunk C (lexically richest -- actual
conversation words), C+dates, M+C, M+C+dates. Baseline = emb_nodate
(bm25 = M,D) = 91.30; nodate_all (bm25 = M) = 90.84.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
CACHE = "cache-terse-v2-raw.json"
SEARCH_TAG = "v28-e0-rnullbmfa50-l10-tsshort-seg"
VARIANTS = ["bm25_c", "bm25_cd", "bm25_mc", "bm25_mcd"]
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


def ingest(variant: str) -> tuple[str, int]:
    subprocess.run(
        f"rm -f locomo-deco-{variant}.sqlite* locomo-deco-{variant}.vec.sqlite*",
        shell=True,
    )
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
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-deco-{variant}-{SEARCH_TAG}-mini-mb-c14.json",
        "--judge-model", "gpt-5-mini",
        "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-deco-{variant}.out")


def main() -> None:
    t0 = time.time()
    print(f"=== ingest {len(VARIANTS)} bm25-text variants ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        for log, rc in ex.map(ingest, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== search + eval {len(VARIANTS)} variants ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        for log, rc in ex.map(search_and_eval, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== BM25-TEXT BATCH DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
