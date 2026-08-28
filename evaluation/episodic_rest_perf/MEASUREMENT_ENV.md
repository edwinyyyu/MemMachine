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
- DELAY_MS=200 sweep (workers x concurrency, ABAB + ramped peak probe,
  2026-08-28): with a 200 ms embed wait, throughput is bounded by
  in-flight concurrency (c / 0.2s), so c16 caps near 70/s for any build
  (optimized 69.7, stock 56.3). At c256 the optimized build sustains
  **~980 req/s steady with W=1** (window 897-908/s, p50 ~250 ms,
  p99 <440 ms); stock peaks at ~245/s (W=4, c128) and anti-scales as
  queues deepen (W=1: 104/s at c64 falling to ~63/s at c128). Mock
  embedder and qdrant were both exonerated as ceilings (mock burst to
  1844 embeds/s at ~200 ms implied service; qdrant peaked at 0.73 of
  its 2 cores).
- Known artifacts on this host: (1) kern.ipc.somaxconn=128 -- opening
  >128 connections at once overflows the listen backlog and macOS SYN
  retransmits stall those clients for seconds (hence --ramp);
  (2) W>=2 arms show residual multi-second stalls with workers
  <60% CPU and connections roughly balanced across workers -- a macOS
  multi-process loopback artifact, not a capacity limit; worker scaling
  must be evaluated on Linux. (3) Stock W=4 threw 239 HTTP 500s:
  parameterized queries against the parent segment_store_sg table lock
  ALL 316 child partitions per query, exhausting Postgres's default
  lock table (64 x 104 slots) -- the optimized serving path is immune
  because it does not touch the segment store on search.
