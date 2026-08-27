# Provenance of the 2026-08 measurements

The numbers quoted in the declarative-to-event proposal documents were
produced with this harness under the following exact configuration.
Everything not listed here is encoded in the scripts themselves
(argparse defaults) and in run_all.sh (the matrix invocations, every
knob spelled out per run); each results/*.json additionally embeds the
full settings of its own run.

## Code measured

- memmachine at commit 5cb36259 ("config: default episodic long-term
  memory to the event backend"), later rebased as 6c5b4bd7. This bench
  branch is based on current main, whose neo4j and postgres compose
  service definitions are identical to the measured state (verified);
  running the harness from this branch measures the code at its HEAD,
  which is fine for comparisons but not byte-exact archaeology -- for
  that, check out the measured commit and run this directory against
  it.

## Services (all measured on one machine, client colocated)

- neo4j:5.23-community, the repo compose settings: heap 512m initial /
  1g max, bolt thread pool 2000, APOC + graph-data-science plugins
  loaded. Vector quantization at the 5.23 default (on, scalar).
- qdrant/qdrant:v1.19.0, default configuration (the measured compose
  added only a volume and healthcheck), REST transport
  (prefer_grpc=false).
- pgvector/pgvector:pg16, default configuration.
- Docker Desktop VM: 11 CPUs, ~12.5 GB RAM, macOS (Darwin 25.5.0)
  host, Apple Silicon. Benchmark client runs on the host, sharing the
  physical cores with the VM -- single-machine numbers throughout.

## Libraries (repo venv at measurement time)

- Python 3.14.3; neo4j driver 6.2.0; qdrant-client 1.18.0;
  SQLAlchemy 2.0.51; asyncpg 0.31.0; numpy 2.5.1; openai 2.45.0
  (client-overhead probe only).

## Invocations beyond run_all.sh

The attribution runs quoted in the writeups that are not part of the
matrix script, with their exact commands:

- driver-pool variation:
  `bench_declarative.py --mode query --session decl_50k --queries 300
  --warmup 20 --k 10 --concurrency 16 --threshold 1 --wait-indexes
  --pool 1000`
- multi-process aggregates (client-vs-server attribution): 2 and 4
  simultaneous `bench_declarative.py --mode query ... --concurrency 8`
  processes with distinct `--seed` values; same pattern with
  `bench_event.py` (2 x c16, distinct seeds); and
  `probe_decl_fixed_qps.py --shape full --concurrency 16` run twice
  concurrently with distinct `--seed-offset`.
- CPU attribution: sample `docker stats --no-stream` (service CPU) and
  `ps -o %cpu` of the client process mid-run during a sustained load;
  the client's process_time-per-call (probe_openai_client_overhead.py
  prints it directly) isolates client CPU where applicable.

## Numbers this environment does NOT reproduce

- Anything quoted from the memlite Rust benchmarks (fake wire-protocol
  databases; that code lives in the memlite repo). Corpus size is
  inert in those runs by construction.
- The claude-memory storage-layer numbers (private package; probe kept
  outside the repo).
- Absolute figures on different hardware: treat all absolute numbers
  as machine-specific; the harness is for same-machine, same-run
  comparisons.
