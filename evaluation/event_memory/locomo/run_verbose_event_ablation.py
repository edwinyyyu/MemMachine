"""Verbose-event alias: isolate the regex contribution in 2-form mode.

In RESOLVE mode 97.8% of segments have only the event date in their
alias line; the regex extracts an extra date in just 2.2%. This tests
2-form alias for the event date ONLY (no regex) against BM25-only
verbose-with-regex (91.45 mini / 89.39 gpt-5 n=6).

If tied -> the regex is dead code; the event timestamp does all the
work. If lower -> the regex's 2.2% extra dates contribute real signal.
n=3 K=10, mini + gpt-5.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
REPS = [1, 2, 3]


def run(cmd, log):
    with open(log, "w") as f:
        return log, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def ingest(rep):
    t = f"tslimv3bove-54n-l-nb8-rep{rep}"
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
            "--segmenter", "terse-decoupled-slim-v3-bo-verbose-event",
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(rep):
    t = f"tslimv3bove-54n-l-nb8-rep{rep}"
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


def main():
    t0 = time.time()
    print(f"=== ingest {len(REPS)} verbose-event DBs (width 3) ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for log, rc in ex.map(ingest, REPS):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== search + eval {len(REPS)} (K=10, width 3) ===", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for res in ex.map(search_eval, REPS):
            print(f"  {res}", flush=True)
    print(f"=== VERBOSE-EVENT ABLATION DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
