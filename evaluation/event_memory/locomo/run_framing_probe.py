"""4th-framing probe: does an LLM view beyond M/Q/C add embedding signal?

Phase 1: augment the fixed terse-v2 cache with atomic + topic framings.
Phase 2: ingest the reassembly variants (f_*) from the augmented cache.
Phase 3: search + eval each.

Baseline for comparison = emb_nodate (M,Q,C, no dates) = 91.30, and the
matched cur = 91.49. A 4th component is worth its complexity only if it
clears the ~0.3pp noise floor; the additive lattice predicts a paraphrase
adds <0.65pp, so atomic/topic must be genuinely orthogonal to beat that.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
AUG_CACHE = "cache-terse-v2-aug.json"
SEARCH_TAG = "v28-e0-rnullbmfa50-l10-tsshort-seg"
VARIANTS = ["f_mqca", "f_mqcp", "f_mqcap", "f_a", "f_p"]
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
    cmd = [
        "uv", "run", "python", "locomo_ingest.py",
        "--data-path", DATA,
        "--segment-db", f"locomo-deco-{variant}.sqlite",
        "--vector-db", f"locomo-deco-{variant}.vec.sqlite",
        "--segmenter", "decoupling-ablation",
        "--reassembly-variant", variant,
        "--segments-cache", AUG_CACHE,
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
    print("=== Phase 1: augment cache (atomic + topic) ===", flush=True)
    log, rc = run(
        ["uv", "run", "python", "augment_cache_framings.py"],
        "log-augment-framings.out",
    )
    if rc != 0:
        print(f"  augmentation FAILED rc={rc} -- see {log}", flush=True)
        return
    print(f"  {log}  rc={rc}", flush=True)

    # Highest API tier -- run every variant concurrently, no throttle.
    print(f"=== Phase 2: ingest {len(VARIANTS)} variants ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        for log, rc in ex.map(ingest, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== Phase 3: search + eval {len(VARIANTS)} variants ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        for log, rc in ex.map(search_and_eval, VARIANTS):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== FRAMING PROBE DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
