"""Re-run the gemma arm of slim_v3 embedding-robustness (bug fix).

run_slimv3_embed_robust.py passed `--embedding-model gemma`, but the CLI
choice is `embeddinggemma` -- the whole gemma arm failed rc=2. minilm
succeeded and is collected. This redoes gemma only, correctly.

Arms (embedding model = embeddinggemma-300m, DB tag short = "gemma"):
  A. slim_v3-54n-l-nb8  -- 3 reps, search K=10 + K=11.
  B. text+whole floor   -- deterministic, 1 rep, search K=6 + K=7.
ST embedding is LOCAL -> width 2.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
EMB = "embeddinggemma"  # CLI value; DB tag uses short "gemma"


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(job: tuple[str, int]) -> tuple[str, int]:
    arm, rep = job
    if arm == "slimv3":
        t = f"tslimv3-54n-l-nb8-gemma-rep{rep}"
        seg_args = [
            "--segmenter", "terse-decoupled-slim-v3",
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ]
    else:
        t = "text-whole-gemma"
        seg_args = ["--segmenter", "text", "--deriver", "whole"]
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    for _ in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            *seg_args,
            "--embedding-model", EMB,
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(job: tuple) -> tuple[str, int]:
    arm, rep, k = job
    t = (f"tslimv3-54n-l-nb8-gemma-rep{rep}" if arm == "slimv3"
         else "text-whole-gemma")
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
        "--embedding-model", EMB,
    ], f"log-search-{tag}.out")
    if s[1] != 0:
        return s
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{tag}-mini-mb-c14.json",
        "--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{tag}.out")


def main() -> None:
    t0 = time.time()
    ing = [("slimv3", r) for r in (1, 2, 3)] + [("floor", 1)]
    print(f"=== ingest {len(ing)} gemma DBs (width 1, local model serial) ===", flush=True)
    with ThreadPoolExecutor(max_workers=1) as ex:
        for log, rc in ex.map(ingest, ing):
            print(f"  {log}  rc={rc}", flush=True)

    se = ([("slimv3", r, k) for r in (1, 2, 3) for k in (10, 11)]
          + [("floor", 1, k) for k in (6, 7)])
    print(f"=== search + eval {len(se)} (width 1, local model serial) ===", flush=True)
    with ThreadPoolExecutor(max_workers=1) as ex:
        for log, rc in ex.map(search_eval, se):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== GEMMA FIX DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
