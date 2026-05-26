"""Embedding-model robustness: slim_v4 with local sentence-transformers.

Is the approach over-fit to text-embedding-3-small? Re-ingest slim_v4
(54n-l, nb8) with embeddinggemma-300m and all-MiniLM-L6-v2, search at
K=12, compare to the 3-small slim_v4-54n-l-nb8 K=12 baseline (91.24 mini,
n=6). Doesn't need to match 3-small -- just not be much worse.

ST embedding is LOCAL compute -> ingest/search width limited to 2 (the
embed phase contends on CPU/MPS, unlike the API case).
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
TAG = "v28-e0-rnullbmfa50-l12-tsshort-seg"
EMBEDDERS = {"gemma": "embeddinggemma", "minilm": "minilm"}
REPS = [1, 2, 3]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def jobs() -> list[tuple[str, str, int]]:
    return [(s, m, r) for s, m in EMBEDDERS.items() for r in REPS]


def ingest(job: tuple[str, str, int]) -> tuple[str, int]:
    short, emb, rep = job
    t = f"tslimv4-54n-l-nb8-{short}-rep{rep}"
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    for _ in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            "--segmenter", "terse-decoupled-slim-v4",
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
            "--neighbor-window", "8", "--neighbor-direction", "both",
            "--embedding-model", emb,
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def search_eval(job: tuple[str, str, int]) -> tuple[str, int]:
    short, emb, rep = job
    t = f"tslimv4-54n-l-nb8-{short}-rep{rep}"
    search = f"search-{t}-{TAG}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", search,
        "--segment-db", f"locomo-{t}.sqlite",
        "--vector-db", f"locomo-{t}.vec.sqlite",
        "--vector-search-limit", "28", "--expand-context", "0",
        "--max-num-segments", "12", "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
        "--embedding-model", emb,
    ], f"log-search-{t}.out")
    if s[1] != 0:
        return s
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{t}-{TAG}-mini-mb-c14.json",
        "--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{t}.out")


def main() -> None:
    t0 = time.time()
    js = jobs()
    print(f"=== ingest {len(js)} ST-embedder DBs (width 2, local) ===",
          flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(ingest, js):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== search + eval {len(js)} (width 2) ===", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        for log, rc in ex.map(search_eval, js):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== EMBEDDER ROBUST DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
