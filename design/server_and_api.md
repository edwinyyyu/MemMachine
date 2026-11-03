# Server and API

> Retroactive design document: it describes the system as built, written
> after the decisions it covers were made.

## Server process

`memmachine-server` is an ASGI application served by uvicorn. One process runs
one event loop, and every request and background task in it shares that
loop. `MEMMACHINE_WORKERS` starts several worker processes on one host, so
one machine can use more than one core; across machines, a deployment runs
more server instances behind a load balancer (see
[deployment and scaling](deployment_and_scaling.md)).

At startup the server loads its configuration, prepares the session tables,
and starts project deletion and, when semantic memory is enabled, semantic
ingestion. It builds the other resources, creates their schemas, and starts
the segment store's and vector stores' purge loops when it first uses them.

## Configuration

The configuration is one YAML file, named by `MEMORY_CONFIG` or found at a
default path. It has three kinds of sections:

- **Resources**: databases, embedders, language models, and rerankers, each
  defined once under an id.
- **Memory sections**: the episode store, episodic memory, semantic memory,
  the retrieval agent, session management, and prompts. They name the
  resources they use by id.
- **Server and logging**: ports, workers, and log settings.

The deployment defines the models and stores. A project chooses its
long-term backend and a reranker among the deployment's resources, and turns
its episodic, short-term, and long-term memory on or off. An event project
uses the deployment's embedder, vector store, and segment store; a
declarative project also names its own embedder and graph store.
`memmachine-configure` writes a first configuration interactively, and
`sample_configs/` holds complete examples.

## REST API

The project and memory routes are POST requests under `/api/v2`, with a JSON
body that names the organization and, except for listing projects, the
project:

| Group | Operations |
|---|---|
| Projects | Create, get, list, delete, count episodes |
| Memories | Add messages; search; list; delete episodic memories and semantic features |
| Semantic configuration | Set types, sets and their configuration, categories, category templates, tags, and features |
| Episodic configuration | Turn a project's episodic, short-term, and long-term memory on or off |

`GET /api/v2/health` and `GET /api/v2/metrics` serve health and metrics.
Request and response models are defined once, in `memmachine-common`, and
shared by the server and the Python SDK. The
[API reference](../docs/api_reference/intro.mdx) documents each route.

## Identity and trust

The server acts on the organization and project that a request names. It
has no notion of a caller, so it neither authenticates callers nor checks
which organizations they may use. A deployment that serves more than one
tenant puts a gateway in front of the server that authenticates each caller,
decides which organization and projects it may use, and sets those on the
request. The same gateway terminates TLS, and it enforces quotas and rate
limits where the deployment needs them.

## MCP server

The MCP server gives agents three tools: add a memory, search memories, and
delete a memory. It runs over stdio (`memmachine-mcp-stdio`), as a standalone
HTTP server (`memmachine-mcp-http`), or inside the REST server at `/mcp`. The
organization, project, and user come from tool arguments, the server's
environment, or, over HTTP, the `org-id`, `proj-id`, and `user-id` headers,
which take precedence. A client that sets none gets defaults derived from
the server's host
([#1750](https://github.com/MemMachine/MemMachine/issues/1750)). MCP over HTTP
keeps session state in the process that opened the session
([#1751](https://github.com/MemMachine/MemMachine/issues/1751)).

## SDKs and command line

- **Python**: `memmachine-client` wraps the REST API, with a project object
  for adding and searching memories. It also installs `mem-cli`, a command
  line client that the agent skill in `packages/skills` teaches coding agents
  to use.
- **TypeScript**: `@memmachine/client` wraps the project and memory routes.

## Observability

`/api/v2/metrics` serves Prometheus metrics: request counts and latencies,
per-operation timings in the memory layers, event-loop lag, and relational
connection pool saturation. Event-loop lag and pool saturation are the
signals that a process is at capacity. `server.metrics_port` serves the same
metrics from a separate thread, which keeps answering while the event loop
is saturated. `/api/v2/health` answers liveness probes. Every request is
written to an access log.
