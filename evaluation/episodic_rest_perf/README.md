# Episodic REST performance harness

Measures the v2 REST API (`/api/v2/memories/search`, `/api/v2/memories`)
with the event long-term-memory backend, and validates that this branch's
fast paths are semantically identical to the canonical paths. Companion to
`evaluation/episodic_backend_perf` (library-level, no HTTP); this harness
keeps the full server in the loop: uvicorn, routing/validation, embedder
client, vector store, episode store.

Long-term episodic memory is measured in isolation:
`short_term_memory_enabled: false` and `semantic_memory.enabled: false`
in `config.yml` (untyped requests otherwise run the semantic search leg
even when no semantic config is usable, roughly doubling embed calls).

## Files

- `mock_openai.py` -- mock OpenAI API on :8791 (embeddings + chat).
  `DELAY_MS` env adds fixed embedding latency; `MOCK_DIM` sets dimensions.
  Prints an in-flight/peak gauge once a second.
- `rest_bench.py` -- closed-loop REST client: optional ingest arm, then
  search arms at given concurrency. `--json-out` dumps raw latencies;
  transport errors are counted and retried once, not fatal.
- `multi_client.py` -- runs one search arm across several `rest_bench`
  processes and merges (one asyncio+httpx process saturates around
  300-400 req/s of client-side CPU; driving ~1000 req/s needs several).
- `sweep_delay.sh` -- the {stock,optimized} x workers x concurrency
  campaign with a delayed embedder. Fresh server per phase, bind-failure
  check, hits assertion per boot, background CPU sampler.
- `config.yml` -- server config used by every phase (event backend,
  short-term and semantic memory disabled, bench containers' ports).
- `difftest_hotpath.py` -- fast == canonical on the READ path: qdrant
  query (filters/thresholds/vectors/missing seeds), embedder, segment
  seed fetch; exact result equality.
- `difftest_ingest.py` -- fast == canonical on the WRITE path:
  add_segments rows column-by-column, upsert points (vectors+payloads),
  canonical-reader round trips, windowed context fetch, encoders.
- `difftest_episode_add.py` -- episode store fast add/fetch equality.
- `sab_test.py` -- sabotage tests: disable the canonical path and prove
  the fast path actually serves (a fast path that silently falls back
  would pass every differential test while measuring nothing).
- `probe_hierarchy2.py` -- per-stage CPU breakdown of the serving path
  (spec validation, embed, vector query, payload walk, episode fetch,
  response dump) via `time.process_time`.

## Prerequisites

1. A venv with `memmachine-server` installed (`uv sync --extra qdrant`
   plus the postgres driver, or reuse an existing repo venv). Install
   `orjson` ad hoc for the optimized build's JSON fast path
   (`uv pip install orjson`); it is optional-by-try-import.
2. Dedicated bench containers (kept off the default ports so developer
   data is never touched):

   ```bash
   docker run -d --name bench2-pg --cpus 2 -p 5442:5432 \
       -e POSTGRES_USER=memmachine -e POSTGRES_PASSWORD=memmachine \
       -e POSTGRES_DB=memmachine postgres:16
   docker run -d --name bench2-qdrant --cpus 2 -p 6343:6333 -p 6344:6334 \
       qdrant/qdrant:v1.12.4
   ```

3. Seed the searched project (default `isolated1`) once, with the mock
   embedder at zero delay and a server up:

   ```bash
   DELAY_MS=0 python mock_openai.py &
   MEMORY_CONFIG=$PWD/config.yml MEMMACHINE_WORKERS=1 memmachine-server &
   python rest_bench.py --project isolated1 --ingest 2000 \
       --ingest-concurrency 8 --search-arms 1 --queries 5 --types episodic
   ```

## Running

Differential + sabotage tests (mock embedder must be running, zero delay;
`PYTHONPATH` selects the build under test):

```bash
PYTHONPATH=<repo>/packages/server/src python difftest_hotpath.py
PYTHONPATH=<repo>/packages/server/src python difftest_ingest.py
PYTHONPATH=<repo>/packages/server/src python difftest_episode_add.py
PYTHONPATH=<repo>/packages/server/src python sab_test.py
```

Set `BENCH_EXPECT_PATH_SUBSTR=<worktree-name>` to make each script refuse
to run against the wrong build (guards the PYTHONPATH-vs-installed foot-gun).

Throughput sweep with embedder latency (defaults: 200 ms, W in {1,2,4},
both builds, ABAB reps):

```bash
DELAY_MS=200 ./sweep_delay.sh
```

The stock baseline is the branch's merge-base. Either leave `STOCK_SRC`
empty with a venv whose installed `memmachine-server` is the merge-base
checkout, or point `STOCK_SRC` at such a checkout's `packages/server/src`.
`HOT_SRC` defaults to this repo's `packages/server/src`.

## Measurement hygiene (learned the hard way)

- Every phase boots a fresh server and greps its log for
  "address already in use": a server that fails to bind leaves the
  PREVIOUS build serving, and the phase silently measures the wrong one.
- Every boot asserts the searched project returns 10 episodes. An empty
  tenant measures only overhead and looks impressively fast.
- Ingest arms grow the shared qdrant collection, which slowly shifts
  absolute search numbers (filtered ANN over more points) and can
  under-fill `limit` at low tenant selectivity; compare builds within a
  campaign, not across campaigns, and treat fast/canonical parity (not
  absolute match counts) as the invariant.
- Closed-loop latency includes queueing; with an embedder delay D the
  throughput ceiling is `c / D` regardless of server speed. Size
  concurrency for the target rate first (1000 req/s at 200 ms needs
  >= 200 in flight), then size workers for the CPU term.
