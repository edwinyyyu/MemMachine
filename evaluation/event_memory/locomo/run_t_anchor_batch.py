"""T-as-anchor ablation: can the terse field replace memory for retrieval?

T is the compressed form of M, both from one segmenter call. T already
works as block.text and as the BM25 text (bm25terse, within noise). If
it also works as the embedding anchor, the `memory` field is redundant
and the segmenter can produce only {terse, queries} -- a real
simplification.

  embed_t     embed = T,Q,C        (T replaces M)
  embed_mqct  embed = M,Q,C,T      (T as a 4th item -- redundancy check)
  all_t       embed = T,Q,C, bm25 = T,D   (M unused entirely)

Cache-reassembly on the fixed terse-v2 segmentation. Baseline =
emb_nodate (M,Q,C) 91.30; cur 7-run mean 90.85.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
CACHE = "cache-terse-v2-raw.json"
SEARCH_TAG = "v28-e0-rnullbmfa50-l10-tsshort-seg"
VARIANTS = ["embed_t", "embed_mqct", "all_t"]
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
    return run([
        "uv", "run", "python", "locomo_ingest.py",
        "--data-path", DATA,
        "--segment-db", f"locomo-deco-{variant}.sqlite",
        "--vector-db", f"locomo-deco-{variant}.vec.sqlite",
        "--segmenter", "decoupling-ablation",
        "--reassembly-variant", variant,
        "--segments-cache", CACHE,
    ], f"log-ingest-{variant}.out")


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
    print(f"=== ingest {len(VARIANTS)} T-anchor variants ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        for log, rc in ex.map(ingest, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== search + eval {len(VARIANTS)} variants ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        for log, rc in ex.map(search_and_eval, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== T-ANCHOR BATCH DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
