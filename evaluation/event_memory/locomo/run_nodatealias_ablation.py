"""Ablation: slim_v3 with the programmatic `_date_aliases` line disabled.

The `Dates: <Month YYYY; Month D, YYYY>` line that `_date_aliases`
appends to text_to_embed AND text_to_score_bm25 has never been
isolated-ablated -- it was inherited into slim_v3. This re-ingests the
production cell (slim_v3, gpt-5.4-nano@low, nb8) with
`include_date_aliases=False`: dates then survive ONLY as ISO strings the
LLM weaves into `memory` per the prompt -- no programmatic alias line in
the embedding or BM25 text.

Everything else identical to production. n=3, K=10/11, mini + gpt-5.

Baseline (aliases ON, from the confirm batch):
  K=10  90.80 mini / 88.48 gpt-5 @310t
  K=11  90.76 mini / 88.74 gpt-5 @341t
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3]
KS = [10, 11]
INGEST_WIDTH = 3
SEARCH_WIDTH = 4


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(rep: int) -> tuple[str, int]:
    t = f"tslimv3nda-54n-l-nb8-rep{rep}"
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
            "--segmenter", "terse-decoupled-slim-v3-nodatealias",
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(job: tuple[int, int]) -> tuple[str, int]:
    rep, k = job
    t = f"tslimv3nda-54n-l-nb8-rep{rep}"
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
    print(f"=== ingest {len(REPS)} nodatealias DBs (width {INGEST_WIDTH}) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=INGEST_WIDTH) as ex:
        for log, rc in ex.map(ingest, REPS):
            print(f"  {log}  rc={rc}", flush=True)

    se = [(r, k) for r in REPS for k in KS]
    print(f"=== search + eval {len(se)} (width {SEARCH_WIDTH}) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=SEARCH_WIDTH) as ex:
        for res in ex.map(search_eval, se):
            print(f"  {res}", flush=True)
    print(f"=== NODATEALIAS ABLATION DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
