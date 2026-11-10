"""Every path the benchmark reads or writes. Set RBENCH_DATA to keep data elsewhere."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RBENCH_DATA", ROOT / "data"))
RAW = DATA / "raw"  # downloads, untrusted: parsed as data only
PREPARED = DATA / "prepared"  # <dataset>/{docs.jsonl, queries.jsonl, stats.json}
VECTORS = DATA / "vectors"  # the embedder checks' vectors (compare-embedders, query_<variant>.npy)
SCORES = DATA / "scores"  # <reranker>.sqlite
EMBEDDINGS = DATA / "embeddings"  # document vectors by text, until their partition is stored (memory.EmbeddingCache)
# data/memory/<embedder>/<dataset>/<chunking>/: EventMemory's stores (memory.MEMORY). data/chunks and
# data/pools hold the retired flat pipeline's caches; nothing here reads them.
RESULTS = ROOT / "results"
SIM = RESULTS / "sim"  # tidy per-query results, parquet
SUMMARY = RESULTS / "summary"  # markdown
FIGURES = RESULTS / "figures"  # png
SECRETS = ROOT / ".secrets"  # deepinfra_api_key
