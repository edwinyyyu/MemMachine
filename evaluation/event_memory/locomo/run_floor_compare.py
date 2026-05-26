"""Pre-LLM floor vs slim_v3, matched token budget, both judges.

The non-generative floor (text segmenter + whole deriver) ran at K=5/6/7
(~230/274/319 tok/q -- raw whole-turn segments are large). slim_v3 ran
at K=10/11 (~310/341t). No shared K, so compare at matched TOKEN budget.

This fills the gaps:
  - floor K=5/6/7: add the gpt-5 judge (only mini exists).
  - slim_v3 nb8 (54n-l) rep1-3: search K=7/8/9 (~217/248/279t) to
    bracket the floor's budget points; judge mini + gpt-5.
Existing slim_v3 K=10 (310t, mini 90.80 / gpt-5 88.48) pairs with the
floor's K=7 (319t).
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def floor_gpt5(k: int) -> tuple[str, int]:
    tag = f"text-whole-v28-e0-rnullbmfa50-l{k}-tsshort"
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", f"search-{tag}.json",
        "--target-path", f"eval-{tag}-gpt5-mb-c14.json",
        "--judge-model", "gpt-5", "--judge-variant", "mem0-bench",
        "--skip-category-5",
    ], f"log-eval-{tag}-gpt5.out")


def slimv3(job: tuple[int, int]) -> tuple[str, int]:
    rep, k = job
    t = f"tslimv3-54n-l-nb8-rep{rep}"
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
    jobs = [("floor", k) for k in (5, 6, 7)] + [
        ("slimv3", (rep, k)) for k in (7, 8, 9) for rep in (1, 2, 3)]
    print(f"=== floor gpt-5 + slim_v3 K7/8/9 : {len(jobs)} jobs ===",
          flush=True)

    def dispatch(j):
        kind, arg = j
        return floor_gpt5(arg) if kind == "floor" else slimv3(arg)

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for res in ex.map(dispatch, jobs):
            print(f"  {res}", flush=True)
    print(f"=== FLOOR COMPARE DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
