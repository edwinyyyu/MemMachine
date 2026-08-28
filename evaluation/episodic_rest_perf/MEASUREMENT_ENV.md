# Measurement environment

Environment behind the numbers reported from this harness (2026-08-28).
Absolute numbers are machine-specific; compare within a campaign.

## Machine

- Apple M3 Pro (11 cores: 5 performance + 6 efficiency), macOS 26.5.2.
- Everything runs on this one machine: server, mock embedder, clients,
  and both stores (each in a Docker container capped at 2 CPUs).
  Efficiency-core scheduling adds noise to per-call CPU probes.

## Software

- Python 3.14.3 (uv-managed).
- uvicorn 0.51.0 with uvloop 0.22.1 + httptools 0.8.0 (auto-detected,
  applies to both builds), fastapi 0.139.0, pydantic 2.13.4,
  httpx 0.28.1, openai 2.45.0, qdrant-client 1.18.0, sqlalchemy 2.0.51,
  asyncpg 0.31.0, orjson 3.11.9.
- Stores: pgvector/pgvector:pg16 (`--cpus 2`, port 5442),
  qdrant/qdrant:v1.12.4 (`--cpus 2`, ports 6343/6344, REST,
  prefer_grpc=false). Note qdrant-client 1.18 warns about the 1.12
  server skew; harmless for the operations used here.

## Builds

- Optimized: this branch (`perf-hot-path`), overlaid via
  `PYTHONPATH=<worktree>/packages/server/src`.
- Stock: the branch merge-base 231ce171 ("Fix a deadlock condition in
  short term memory (#1517)"), as the venv's installed package.
- One process per uvicorn worker; `MEMMACHINE_WORKERS` sets W.

## Workload

- Project `benchorg/isolated1`: 2,000 short synthetic chat messages
  ingested through the REST API (one segment each, passthrough
  segmenter, WholeTextDeriver), 1536-dim vectors.
- The shared qdrant collection also holds other bench tenants
  (~193k points total at measurement time); tenant selectivity ~1%,
  which is why filtered ANN can return fewer than the requested
  vector_search_limit=50 -- both builds see the identical store.
- Search: `top_k=10`, `types=["episodic"]` (long-term episodic only;
  short-term and semantic disabled in config.yml).
- Mock embedder returns a constant vector (zero variance across
  queries); every request exercises the same ANN + hydration work.
  `DELAY_MS` adds fixed embedding latency where stated.

## Headline numbers (this environment)

- DELAY_MS=0, W=1, c16: stock ~159 req/s vs optimized 653/570 req/s
  (ABAB pair), ~6.2 vs ~1.4-1.6 worker core-ms per request, p50 ~10 ms.
- Ingest (DELAY_MS=0, c8, 2k messages): stock ~202-212 vs optimized
  ~221-303 events/s across rounds; qdrant HNSW upsert (~7 core-ms/event
  in a 2-CPU container) is the wall.
- DELAY_MS=200 sweep (workers x concurrency, ABAB): results recorded in
  the campaign log (`results/sweep200.log` locally; summarized in the
  report accompanying this branch).
