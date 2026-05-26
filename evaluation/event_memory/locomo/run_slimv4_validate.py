"""slim_v4 validation — the three checks skipped before declaring it ships.

A. K=40 ceiling: re-search slim_v4-54n-l-nb8 at K=40/vsl=160. A leaner
   index (slim_v4 halved the segments) could cap the high-K ceiling.
   Target: >= terse-v2 K=40 seg 94.09 (mini judge).
B. Raw-event diagnostic: same but --answer-with-raw-events. The proper
   retrieval-regression check across a granularity change (HANDOFF #3).
   Target: >= terse-v2 K=40 rawev 94.03.
C. 6-model robustness: slim_v4 on the 3 untested model x reasoning cells
   (5n-l, 54n-m, 5m-l). slim_v3's matrix spanned 89.9-92.0; slim_v4 must
   not have a model that falls out of band.

A/B re-search existing DBs (no ingest). C needs fresh ingests.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

DATA = "../../data/locomo10.json"
EVAL = ["--judge-model", "gpt-5-mini", "--judge-variant", "mem0-bench",
        "--skip-category-5"]

# --- A/B: K=40 seg + rawev on existing slim_v4-54n-l-nb8 DBs ---
K40_REPS = [1, 2, 3]
# --- C: 6-model missing cells (cell -> model, reasoning) ---
C_MODELS = {
    "5n-l": ("gpt-5-nano", "low"),
    "54n-m": ("gpt-5.4-nano", "medium"),
    "5m-l": ("gpt-5-mini", "low"),
}
C_REPS = [1, 2, 3]


def run(cmd: list[str], log: str) -> tuple[str, int]:
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return log, p.returncode


def k40_job(spec: tuple[str, int]) -> tuple[str, int]:
    """spec = (mode, rep); mode in {'seg','rawev'}."""
    mode, rep = spec
    db = f"locomo-tslimv4-54n-l-nb8-rep{rep}"
    tag = f"tslimv4-54n-l-nb8-rep{rep}-v160-e0-rnullbmfa50-l40-{mode}"
    search = f"search-{tag}.json"
    cmd = [
        "uv", "run", "python", "locomo_search.py",
        "--data-path", DATA, "--target-path", search,
        "--segment-db", f"{db}.sqlite", "--vector-db", f"{db}.vec.sqlite",
        "--vector-search-limit", "160", "--expand-context", "0",
        "--max-num-segments", "40", "--no-reranker",
        "--bm25-fusion", "additive", "--bm25-fusion-weight", "0.5",
        "--timestamp-format", "short",
    ]
    if mode == "rawev":
        cmd.append("--answer-with-raw-events")
    s = run(cmd, f"log-search-{tag}.out")
    if s[1] != 0:
        return s
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{tag}-mini-mb-c14.json", *EVAL,
    ], f"log-eval-{tag}.out")


def c_ingest(job: tuple[str, str, str, int]) -> tuple[str, int]:
    cell, model, effort, rep = job
    t = f"tslimv4-{cell}-nb8-rep{rep}"
    subprocess.run(
        f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    for _ in range(3):
        log, rc = run([
            "uv", "run", "python", "locomo_ingest.py",
            "--data-path", DATA,
            "--segment-db", f"locomo-{t}.sqlite",
            "--vector-db", f"locomo-{t}.vec.sqlite",
            "--segmenter", "terse-decoupled-slim-v4",
            "--segmenter-model", model, "--segmenter-reasoning", effort,
            "--neighbor-window", "8", "--neighbor-direction", "both",
        ], f"log-ingest-{t}.out")
        if rc == 0:
            return log, rc
        subprocess.run(
            f"rm -f locomo-{t}.sqlite* locomo-{t}.vec.sqlite*", shell=True)
    return f"log-ingest-{t}.out", rc


def c_search_eval(job: tuple[str, str, str, int]) -> tuple[str, int]:
    cell, _m, _e, rep = job
    t = f"tslimv4-{cell}-nb8-rep{rep}"
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
    return run([
        "uv", "run", "python", "locomo_evaluate.py",
        "--data-path", search,
        "--target-path", f"eval-{tag}-mini-mb-c14.json", *EVAL,
    ], f"log-eval-{tag}.out")


def main() -> None:
    t0 = time.time()
    k40 = [("seg", r) for r in K40_REPS] + [("rawev", r) for r in K40_REPS]
    c_jobs = [
        (cell, m, e, rep)
        for cell, (m, e) in C_MODELS.items()
        for rep in C_REPS
    ]

    print(f"=== C: ingest {len(c_jobs)} 6-model cells ===", flush=True)
    with ThreadPoolExecutor(max_workers=len(c_jobs)) as ex:
        for log, rc in ex.map(c_ingest, c_jobs):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== A/B: {len(k40)} K=40 runs + C: {len(c_jobs)} "
          f"search/eval ===", flush=True)
    allj = [("k40", s) for s in k40] + [("c", j) for j in c_jobs]

    def dispatch(item):
        kind, payload = item
        return k40_job(payload) if kind == "k40" else c_search_eval(payload)

    with ThreadPoolExecutor(max_workers=len(allj)) as ex:
        for log, rc in ex.map(dispatch, allj):
            print(f"  {log}  rc={rc}", flush=True)

    print(f"=== SLIM_V4 VALIDATE DONE in {time.time() - t0:.0f}s ===",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
