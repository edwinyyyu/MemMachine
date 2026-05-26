"""Directly confirm slim_v3 as the proposed ship -- no inheritance claims.

slim_v3-54n-l-nb8 DBs already exist (run_slimv3_nb8). Re-search only:
  A. K=40 seg   -- ceiling, vs terse-v2 94.09
  B. K=40 rawev -- recall diagnostic, vs terse-v2 94.03
  C. K=11       -- budget point (~340t, slim_v3 is wordier than slim_v4
                   so K=11 not K=13); mini + gpt-5 judge.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
DB = "locomo-tslimv3-54n-l-nb8-rep{r}"
EVAL_MINI = ["--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
             "--skip-category-5"]
EVAL_GPT5 = ["--judge-model", "gpt-5", "--judge-variant", "mem0-bench",
             "--skip-category-5"]


def run(cmd, log):
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def search(rep, k, vsl, mode):
    db = DB.format(r=rep)
    tag = f"tslimv3-54n-l-nb8-rep{rep}-v{vsl}-e0-rnullbmfa50-l{k}-{mode}"
    out = f"search-{tag}.json"
    cmd = [
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", out,
        "--segment-db", f"{db}.sqlite", "--vector-db", f"{db}.vec.sqlite",
        "--vector-search-limit", str(vsl), "--expand-context", "0",
        "--max-num-segments", str(k), "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
    ]
    if mode == "rawev":
        cmd.append("--answer-with-raw-events")
    rc = run(cmd, f"log-search-{tag}.out")[1]
    return tag, out, rc


def evaluate(tag, search_json, eval_flags, suffix):
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search_json,
        "--target-path", f"eval-{tag}-{suffix}.json", *eval_flags,
    ], f"log-eval-{tag}-{suffix}.out")


def job(spec):
    kind, rep = spec
    if kind == "k40seg":
        tag, sj, rc = search(rep, 40, 160, "seg")
        return evaluate(tag, sj, EVAL_MINI, "mini-mb-c14") if rc == 0 else (tag, rc)
    if kind == "k40rawev":
        tag, sj, rc = search(rep, 40, 160, "rawev")
        return evaluate(tag, sj, EVAL_MINI, "mini-mb-c14") if rc == 0 else (tag, rc)
    if kind == "k11":
        tag, sj, rc = search(rep, 11, 28, "seg")
        if rc != 0:
            return (tag, rc)
        evaluate(tag, sj, EVAL_MINI, "mini-mb-c14")
        return evaluate(tag, sj, EVAL_GPT5, "gpt5-mb-c14")
    raise ValueError(kind)


def main():
    t0 = time.time()
    specs = (
        [("k40seg", r) for r in (1, 2, 3)]
        + [("k40rawev", r) for r in (1, 2, 3)]
        + [("k11", r) for r in (1, 2, 3, 4, 5, 6)]
    )
    print(f"=== slim_v3 confirm: {len(specs)} jobs ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        for res in ex.map(job, specs):
            print(f"  {res}", flush=True)
    print(f"=== SLIMV3 CONFIRM DONE in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
