"""embeddinggemma query-prompt ablation: QA-task prompt vs default.

embeddinggemma's packaged "query" prompt is "task: search result |
query: " (option 1 -- the current default; our gemma 91.56% K=10 used
it). This re-searches the SAME 3 gemma DBs with the QA-task query
prompt "task: question answering | query: " (option 2) -- search-only,
the document embeddings are untouched, so no re-ingest.

Local model -> width 1. K=10 (budget point), eval mini, n=3.
Baseline (option 1, already measured): 91.56% mini @ ~313t.
"""

from __future__ import annotations

import subprocess
import sys
import time

DATA = "../../data/locomo10.json"


def run(cmd: list[str], log: str) -> int:
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main() -> None:
    t0 = time.time()
    # Gate: never run two local embedding models at once. The MiniLM
    # floor batch may still be searching -- wait for its done marker.
    for _ in range(160):  # up to ~40 min
        try:
            if "MINILM FLOOR DONE" in open("log-minilm-floor.out").read():
                break
        except FileNotFoundError:
            break
        time.sleep(15)
    print(f"  minilm floor clear after {time.time() - t0:.0f}s", flush=True)

    for rep in (1, 2, 3):
        db = f"locomo-tslimv3-54n-l-nb8-gemma-rep{rep}"
        tag = f"tslimv3-54n-l-nb8-gemmaqa-rep{rep}-v28-e0-rnullbmfa50-l10-tsshort-seg"
        search = f"search-{tag}.json"
        rc = run([
            "uv", "run", "python", "locomo_search.py",
            "--data-path", DATA, "--target-path", search,
            "--segment-db", f"{db}.sqlite",
            "--vector-db", f"{db}.vec.sqlite",
            "--vector-search-limit", "28", "--expand-context", "0",
            "--max-num-segments", "10", "--no-reranker",
            "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
            "--timestamp-format", "short",
            "--embedding-model", "embeddinggemma-qa",
        ], f"log-search-{tag}.out")
        print(f"  rep{rep} search rc={rc}", flush=True)
        if rc != 0:
            continue
        rc = run([
            "uv", "run", "python", "locomo_evaluate.py",
            "--data-path", search,
            "--target-path", f"eval-{tag}-mini-mb-c14.json",
            "--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
            "--skip-category-5",
        ], f"log-eval-{tag}.out")
        print(f"  rep{rep} eval rc={rc}", flush=True)
    print(f"=== GEMMA QA-PROMPT DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
