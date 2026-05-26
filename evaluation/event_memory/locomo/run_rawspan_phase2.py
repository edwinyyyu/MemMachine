"""rawspan Phase 2: per-item raw span ALSO replaces block.text.

Phase 1 swapped only the raw-event portion of text_to_embed (per-item
span instead of shared whole-message). Result was ~neutral (span-last)
to slightly worse (span-first). Phase 2 also makes block.text -- the
answerer-visible text counted against the token budget -- the per-item
raw span instead of the terse rewrite.

Tests span-last + block_uses_span on the production cell (54n-l, nb8).
n=3 K=10, mini + gpt-5.

Baseline: slim_v3 K=10 90.80 mini / 88.48 gpt-5 @310t (block.text =
terse). Phase 2 changes both the embedding anchor and the answerer's
text -- token cost depends on per-item raw-span length.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(rep: int) -> tuple[str, int]:
    t = f"trawspanlbs-54n-l-nb8-rep{rep}"
    if (os.path.exists(f"locomo-{t}.sqlite")
            and os.path.exists(f"locomo-{t}.vec.sqlite")):
        return f"log-ingest-{t}.out (skipped)", 0
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    for _ in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            "--segmenter", "rawspan-spanlast-blockspan",
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(rep: int) -> tuple[str, int]:
    t = f"trawspanlbs-54n-l-nb8-rep{rep}"
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
    print(f"=== ingest {len(REPS)} rawspan-blockspan DBs (width 3) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for log, rc in ex.map(ingest, REPS):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== search + eval {len(REPS)} (K=10, width 3) ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for res in ex.map(search_eval, REPS):
            print(f"  {res}", flush=True)
    print(f"=== RAWSPAN PHASE2 DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
