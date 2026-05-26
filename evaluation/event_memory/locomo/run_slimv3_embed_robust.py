"""Ship condition: slim_v3 robustness to the embedding model.

slim_v3 (the ship) must not degrade much when text-embedding-3-small is
swapped for a local sentence-transformers model, AND must still beat the
non-generative floor (raw recursive text segmenter + whole deriver) on
those same embedders.

Two arms, both with embeddinggemma-300m and all-MiniLM-L6-v2:
  A. slim_v3-54n-l-nb8  -- 3 reps, search K=10 + K=11.
  B. text+whole floor   -- deterministic, 1 rep, search K=6 + K=7.
Reference (3-small, already measured): slim_v3 K=11 90.76 mini /
88.74 gpt-5 @341t; text+whole K=7 82.14 mini @319t.

ST embedding is LOCAL compute -> ingest/search width capped at 2.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
EMBEDDERS = ["gemma", "minilm"]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def ingest(job: tuple[str, str, int]) -> tuple[str, int]:
    arm, emb, rep = job
    if arm == "slimv3":
        t = f"tslimv3-54n-l-nb8-{emb}-rep{rep}"
        seg_args = [
            "--segmenter", "terse-decoupled-slim-v3",
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ]
    else:  # text-whole floor: deterministic, no LLM
        t = f"text-whole-{emb}"
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
            "--embedding-model", emb,
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(job: tuple[str, str, int, int]) -> tuple[str, int]:
    arm, emb, rep, k = job
    t = (f"tslimv3-54n-l-nb8-{emb}-rep{rep}" if arm == "slimv3"
         else f"text-whole-{emb}")
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
        "--embedding-model", emb,
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
    ing = (
        [("slimv3", e, r) for e in EMBEDDERS for r in (1, 2, 3)]
        + [("floor", e, 1) for e in EMBEDDERS]
    )
    print(f"=== ingest {len(ing)} ST-embedder DBs (width 2, local) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(ingest, ing):
            print(f"  {log}  rc={rc}", flush=True)

    se = (
        [("slimv3", e, r, k) for e in EMBEDDERS for r in (1, 2, 3)
         for k in (10, 11)]
        + [("floor", e, 1, k) for e in EMBEDDERS for k in (6, 7)]
    )
    print(f"=== search + eval {len(se)} (width 2, local) ===", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(search_eval, se):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== EMBED ROBUST DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
