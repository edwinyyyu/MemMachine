"""Raw-span variant -- Phase 1: per-item raw span in the embedding.

Tests the rawspan segmenter (LLM returns a verbatim per-item raw span
alongside the rewrite, in one call) on the production cell (gpt-5.4-nano
@low, nb8). The per-item span replaces slim_v3's shared whole-message
chunk in text_to_embed. block.text stays `terse` (Phase 1).

Two field orders: spanfirst ({source,memory,terse,queries}) vs spanlast
({memory,terse,queries,source}). n=3, K=10, mini + gpt-5.

Baseline (slim_v3, shared whole-message chunk): K=10 90.80 mini /
88.48 gpt-5 @310t. If a variant clearly beats it, Phase 2 tests
block.text = raw span.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3]
VARIANTS = {
    "trawspanf": "rawspan-spanfirst",
    "trawspanl": "rawspan-spanlast",
}


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(job: tuple[str, int]) -> tuple[str, int]:
    prefix, rep = job
    t = f"{prefix}-54n-l-nb8-rep{rep}"
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
            "--segmenter", VARIANTS[prefix],
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(job: tuple[str, int]) -> tuple[str, int]:
    prefix, rep = job
    t = f"{prefix}-54n-l-nb8-rep{rep}"
    tag = f"{t}-v28-e0-rnullbmfa50-l10-tsshort-seg"
    search = f"search-{tag}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", search,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--vector-search-limit", "28", "--expand-context", "0",
        "--max-num-segments", "10", "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
    ], f"log-search-{tag}.out")
    if s[1] != 0:
        return s
    for judge, suffix in [("gpt-5-mini", "mini"), ("gpt-5", "gpt5")]:
        run([
            "uv", "run", "python", "locomo_evaluate.py",
            "--data-path", search,
            "--target-path", f"eval-{tag}-{suffix}-mb-c14.json",
            "--judge-model", judge, "--judge-variant", "mem0-bench",
            "--skip-category-5",
        ], f"log-eval-{tag}-{suffix}.out")
    return tag, 0


def main() -> None:
    t0 = time.time()
    jobs = [(p, r) for p in VARIANTS for r in REPS]
    print(f"=== ingest {len(jobs)} rawspan DBs (width 3) ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for log, rc in ex.map(ingest, jobs):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== search + eval {len(jobs)} (K=10, width 4) ===", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(search_eval, jobs):
            print(f"  {res}", flush=True)
    print(f"=== RAWSPAN PHASE1 DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
