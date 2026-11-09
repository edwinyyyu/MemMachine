"""Every path the benchmark reads or writes. Set RBENCH_DATA to keep data elsewhere."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RBENCH_DATA", ROOT / "data"))
RAW = DATA / "raw"  # downloads, untrusted: parsed as data only
PREPARED = DATA / "prepared"  # <dataset>/{docs.jsonl, queries.jsonl, stats.json}
CHUNKS = DATA / "chunks"  # <dataset>/<chunking>.npz, fit.json
VECTORS = DATA / "vectors"  # <embedder>/<dataset>/<chunking>.npy, query_<variant>.npy
POOLS = DATA / "pools"  # <embedder>/<dataset>/<scope>/<chunking>__<variant>.npz
SCORES = DATA / "scores"  # <reranker>.sqlite
RESULTS = ROOT / "results"
SIM = RESULTS / "sim"  # tidy per-query results, parquet
SUMMARY = RESULTS / "summary"  # markdown
FIGURES = RESULTS / "figures"  # png
SECRETS = ROOT / ".secrets"  # deepinfra_api_key
