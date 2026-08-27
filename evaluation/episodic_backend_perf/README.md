# Episodic backend performance harness

Benchmarks and attribution probes comparing the two episodic long-term
memory backends as a library, driven directly (no HTTP server in the
loop):

- declarative: DeclarativeMemory over Neo4jVectorGraphStore (Neo4j)
- event: EventMemory over QdrantVectorStore + SQLAlchemySegmentStore
  (Postgres)

Both arms use a deterministic zero-latency 768-dim fake embedder
(hash-seeded gaussian, cosine), so identical vectors reach both sides and
the storage layer is isolated. Reranker: identity (declarative requires
one) vs none (event reuses embedding scores) -- neither does real work.
Knob parity: declarative `max_num_episodes=10` internally overfetches
min(5*10, 200) = 50 vectors, mirrored as the event side's
`vector_search_limit=50`; `expand_context` maps 1:1 (one message = one
segment with the passthrough segmenter).

Numbers produced by this harness (2026-08) and their interpretation live
in the proposal documents outside this repo; treat absolute numbers as
machine-specific and compare within a run.

## Prerequisites

1. Repo deps: `uv sync --extra qdrant` (neo4j and postgres drivers are
   hard deps; qdrant-client is the extra). `numpy` comes with the lock.
2. Services (this branch is based on main, whose docker-compose has no
   qdrant service yet):

   ```bash
   docker compose up -d postgres neo4j
   docker run -d --name bench-qdrant -p 6333:6333 -p 6334:6334 \
       qdrant/qdrant:v1.19.0
   ```

3. Credentials: scripts default to the compose defaults
   (`neo4j/neo4j_password`, `memmachine/memmachine_password`). If your
   `.env` overrides them, export before running:

   ```bash
   export NEO4J_PASSWORD=... POSTGRES_PASSWORD=...
   # or override endpoints wholesale:
   export BENCH_NEO4J_URI=bolt://... BENCH_PG_DSN=postgresql+asyncpg://...
   ```

Run everything with the repo venv python (`.venv/bin/python`). Results
are written to `results/*.json` beside the scripts (gitignored), one
file per run, all knobs recorded in the filename and payload, raw
per-operation latencies included.

## The main matrix

```bash
./run_all.sh            # ~30-45 min: ingest 2k/12k/16k/50k corpora on
                        # both backends, then query/filter/expand/mixed
                        # sweeps at c1/c16/c32
```

`c` = client concurrency: requests kept in flight by the single
benchmark client process (closed loop). c1 = sequential = intrinsic
latency; c16 sits at or past the saturation knee for every config
tested. One Python client process caps near one core of protocol work
(~3.7 core-ms/query on the event stack); use multiple OS processes to
probe server-side ceilings.

Individual arms (see `--help` for all knobs):

```bash
PY=.venv/bin/python
$PY bench_event.py       --mode ingest --label my_ingest --session s1 --n 10000
$PY bench_event.py       --mode query  --label my_query  --session s1 \
    --queries 300 --vsl 50 --expand 0 --concurrency 16 [--filtered]
$PY bench_declarative.py --mode query  --label my_query  --session d1 \
    --queries 300 --k 10 --expand 0 --concurrency 16 --wait-indexes
$PY bench_*.py           --mode mixed  ...   # readers + writers, 60 s
```

Declarative notes: `--threshold` sets the Neo4j index-creation
threshold (default 10000 = shipped behavior, meaning sub-10k sessions
never get a vector index and exact-scan; use `--threshold 1` +
`--wait-indexes` for Neo4j's best case). Filtered runs use a property
with 10 uniform values (10% selectivity).

## Attribution probes (run after the matrix; they target its sessions)

| script | question it answers |
|---|---|
| `probe_decl_breakdown.py` | where DeclarativeMemory's query time goes: ANN vs embedding-payload transfer vs the 50-statement traversal fan-out vs assembly |
| `probe_decl_fixed.py` | how fast a REPAIRED Neo4j query path would be: fused ANN+traversal single statement, payload projection; c16 QPS |
| `probe_decl_expand_fixed.py` | can Neo4j expansion be fixed: shipped per-seed epochSeconds shape (PROFILEd) vs direct comparisons vs batched UNWIND+CALL (Cypher's LATERAL) |
| `probe_decl_fixed_qps.py` | QPS of the fixed shapes (`--shape fused\|full`, `--concurrency`); run two instances with different `--seed-offset` to test the server-side aggregate ceiling |
| `probe_expansion.py` | segment-store expansion in isolation: get_segment_contexts latency across seed counts and window sizes |
| `probe_event_op_sweep.py` | per-operation RPS: search-only across vsl; expand-only (1 seed) across windows; combined via expand_context (expands ALL ~vsl seeds); paired raw search+top-10-expand (the fixed-Neo4j-comparable shape) |
| `probe_openai_client_overhead.py` | client-side CPU of the OpenAI embeddings client, measured against a local mock (`server` then `client` argv) |

Interpretation guardrails learned from these runs:

- Compare within a run; day-to-day drift on a shared box is a few
  percent.
- The raw ANN engines are comparable; large gaps are query-shape
  artifacts. Distinguish claims at the FIXED existing API from claims
  about repaired/redesigned query shapes.
- Corpus-size-dependent behavior (index cliffs, HNSW growth, RAM) only
  exists with the REAL backends; anything measured against fakes or
  mocks is corpus-size-independent by construction.
- Single-process QPS ceilings mix client-side (GIL, ~1 core of protocol
  work) and server-side (shared machine) budgets; attribute with
  multi-process runs plus `docker stats` sampling before quoting a
  number as either one.

A further probe against the claude-memory personal deployment
(turbovec + SQLite) exists outside this repo; it depends on that
system's private package and is not included here.
