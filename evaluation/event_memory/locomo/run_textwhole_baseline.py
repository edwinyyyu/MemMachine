"""Pre-LLM floor: raw-text segments + whole-text deriver.

No generative step anywhere -- TextSegmenter (deterministic raw chunking)
+ WholeTextDeriver (one derivative per whole segment, pass-through). The
honest baseline the LLM-rewrite pipeline is measured against. Same
methodology: vsl=28, e0, no-reranker, additive bm25 0.5, tsshort, mini
judge + mem0-bench. Search at K=5/6/7 to bracket the ~340-token budget.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
DB = "locomo-text-whole"
KS = [5, 6, 7]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def search_eval(k: int) -> tuple[str, int]:
    tag = f"text-whole-v28-e0-rnullbmfa50-l{k}-tsshort"
    search = f"search-{tag}.json"
    s = run([
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", search,
        "--segment-db", f"{DB}.sqlite", "--vector-db", f"{DB}.vec.sqlite",
        "--vector-search-limit", "28", "--expand-context", "0",
        "--max-num-segments", str(k), "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
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
    subprocess.run(f"rm -f {DB}.sqlite* {DB}.vec.sqlite*", shell=True)
    print("=== ingest: text segmenter + whole deriver ===", flush=True)
    log, rc = run([
        "uv", "run", "python", "locomo_ingest.py",
        "--data-path", DATA,
        "--segment-db", f"{DB}.sqlite", "--vector-db", f"{DB}.vec.sqlite",
        "--segmenter", "text", "--deriver", "whole",
    ], f"log-ingest-{DB}.out")
    print(f"  {log}  rc={rc}", flush=True)
    if rc != 0:
        return

    print(f"=== search + eval K={KS} ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(KS)) as ex:
        for log, rc in ex.map(search_eval, KS):
            print(f"  {log}  rc={rc}", flush=True)
    print(f"=== TEXT-WHOLE BASELINE DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
