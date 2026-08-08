"""C2sub A/B: date-prepended embed format on raw text + V8K first-person rewrite.

Runs three variants on c2sub with the same downstream settings:
  - textwhole-dateprefix vs baseline text+whole
  - V8K (baseline) vs V8L (V8K + date prepend)

Stack: gpt-5.4-nano @ low segmenter, text-embedding-3-small embedder,
gpt-5-mini answer + judge, mem0-classic, vsl=28, l=10, no reranker,
bm25_fusion=additive 0.5.

C2sub for quick signal; escalate to full bench if c2sub shows the
date-prepend lift the user observed elsewhere.
"""
from __future__ import annotations
from artifacts import A  # canonical artifact names

import os
import subprocess
import sys
from pathlib import Path

LOCOMO_DIR = Path(__file__).resolve().parent
DATA = LOCOMO_DIR / ".." / ".." / "data" / "locomo10_c2sub.json"
DATA = str(DATA.resolve())

VARIANTS = [
    # (name, segmenter, deriver, llm_needed)
    ("textwhole", "text", "whole", False),
    ("textwhole-dp", "textwhole-dateprefix", "whole", False),
    ("v8k", "deictic-resolved-verbatim-v8k", "whole", True),
    ("v8l", "deictic-resolved-verbatim-v8l", "whole", True),
]


def run(argv: list[str], log_path: Path) -> int:
    """Run a subprocess, tee output to log_path; return rc."""
    print(f"\n>>> {' '.join(argv)}", flush=True)
    print(f"    log: {log_path}", flush=True)
    with log_path.open("w") as f:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            f.write(line)
            f.flush()
        rc = proc.wait()
    return rc


def ingest(name: str, segmenter: str, deriver: str, llm: bool) -> int:
    db = LOCOMO_DIR / A(f"locomo-{name}-c2sub.sqlite")
    vec = LOCOMO_DIR / A(f"locomo-{name}-c2sub.vec.sqlite")
    if db.exists():
        print(f"=== ingest {name}: SKIPPED (DB exists) ===", flush=True)
        return 0
    argv = [
        "uv", "run", "python", str(LOCOMO_DIR / "locomo_ingest.py"),
        "--data-path", DATA,
        "--segment-db", str(db),
        "--vector-db", str(vec),
        "--segmenter", segmenter,
        "--deriver", deriver,
        "--embedding-model", "text-embedding-3-small",
    ]
    if llm:
        argv += [
            "--segmenter-model", "gpt-5.4-nano",
            "--segmenter-reasoning", "low",
        ]
    log = LOCOMO_DIR / A(f"log-ingest-{name}-c2sub.out")
    rc = run(argv, log)
    if rc != 0:
        print(f"!!! ingest {name} failed rc={rc} — see {log}", flush=True)
    return rc


def search(name: str) -> int:
    db = LOCOMO_DIR / A(f"locomo-{name}-c2sub.sqlite")
    vec = LOCOMO_DIR / A(f"locomo-{name}-c2sub.vec.sqlite")
    out = LOCOMO_DIR / A(f"search-{name}-c2sub-l10.json")
    if out.exists():
        print(f"=== search {name}: SKIPPED (output exists) ===", flush=True)
        return 0
    argv = [
        "uv", "run", "python", str(LOCOMO_DIR / "locomo_search.py"),
        "--data-path", DATA,
        "--target-path", str(out),
        "--segment-db", str(db),
        "--vector-db", str(vec),
        "--model", "gpt-5-mini",
        "--embedding-model", "text-embedding-3-small",
        "--vector-search-limit", "28",
        "--expand-context", "0",
        "--max-num-segments", "10",
        "--no-reranker",
        "--bm25-fusion", "additive",
        "--bm25-fusion-weight", "0.5",
    ]
    log = LOCOMO_DIR / A(f"log-search-{name}-c2sub-l10.out")
    rc = run(argv, log)
    if rc != 0:
        print(f"!!! search {name} failed rc={rc}", flush=True)
    return rc


def evaluate(name: str) -> int:
    search_out = LOCOMO_DIR / A(f"search-{name}-c2sub-l10.json")
    out = LOCOMO_DIR / A(f"eval-{name}-c2sub-l10-mini-mc-c14.json")
    if out.exists():
        print(f"=== eval {name}: SKIPPED (output exists) ===", flush=True)
        return 0
    argv = [
        "uv", "run", "python", str(LOCOMO_DIR / "locomo_evaluate.py"),
        "--data-path", str(search_out),
        "--target-path", str(out),
        "--judge-model", "gpt-5-mini",
        "--judge-variant", "mem0-classic",
        "--skip-category-5",
    ]
    log = LOCOMO_DIR / A(f"log-eval-{name}-c2sub-l10-mini-mc.out")
    rc = run(argv, log)
    if rc != 0:
        print(f"!!! eval {name} failed rc={rc}", flush=True)
    return rc


def main() -> None:
    os.chdir(LOCOMO_DIR)

    print("=" * 70)
    print("Stage 1: ingest", flush=True)
    print("=" * 70)
    for name, seg, der, llm in VARIANTS:
        rc = ingest(name, seg, der, llm)
        if rc != 0:
            sys.exit(rc)

    print("=" * 70)
    print("Stage 2: search", flush=True)
    print("=" * 70)
    for name, _, _, _ in VARIANTS:
        rc = search(name)
        if rc != 0:
            sys.exit(rc)

    print("=" * 70)
    print("Stage 3: evaluate", flush=True)
    print("=" * 70)
    for name, _, _, _ in VARIANTS:
        rc = evaluate(name)
        if rc != 0:
            sys.exit(rc)

    print("\n=== DONE; eval files:", flush=True)
    for name, _, _, _ in VARIANTS:
        print(f"  eval-{name}-c2sub-l10-mini-mc-c14.json", flush=True)


if __name__ == "__main__":
    main()
