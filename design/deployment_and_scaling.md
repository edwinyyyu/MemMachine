# Deployment and scaling

> Retroactive design document: it describes the system as built, written
> after the decisions it covers were made.

## Deployment shapes

| Shape | What runs | Fits |
|---|---|---|
| Python package | `memmachine-server` with the stores that `memmachine-configure` writes into its configuration: SQLite relational stores, Neo4j for semantic memory, and a SQLite, Qdrant, or Milvus vector store | Local use and development |
| Docker Compose | The server, PostgreSQL with pgvector, and Qdrant (event backend) or Neo4j (declarative backend), plus a local documentation site | One host |
| Helm chart | The same services on Kubernetes, one replica each; each database can be external instead | A cluster |
| AWS CloudFormation | One EC2 instance running a Compose stack of the server, PostgreSQL, and Neo4j, behind an internal network load balancer and an API Gateway; the template is in the AWS installation guide | A quick start on AWS |

The server image comes in CPU and GPU variants. Only the GPU variant, or the
`gpu` extra of the Python package, includes sentence-transformers, which runs
embedders and cross-encoder rerankers in the server process. Language
models are always reached over the network: OpenAI, Amazon Bedrock, LiteLLM,
or any OpenAI-compatible server, such as Ollama or vLLM on the deployment's
own hosts.

The [installation guide](../docs/install_guide/introduction.mdx) covers the
package, Helm, and CloudFormation shapes, and the
[quickstart](../docs/getting_started/quickstart.mdx) covers Compose.

## What a deployment provides

The server owns memory: what to store, how to index it, and how to retrieve
and delete it. The deployment around it owns the rest:

- **Identity**: authenticating callers and mapping each to the organization
  and projects it may use (see [server and API](server_and_api.md#identity-and-trust)).
- **Network**: TLS, load balancing, quotas, and rate limits.
- **Databases**: running PostgreSQL, the vector database, and any graph
  database, with their own availability, backups, and encryption at rest.
- **Models**: endpoints and credentials for the language models,
  embedders, and rerankers the configuration names.
- **Configuration and secrets**: the configuration file and the credentials
  in it.

## Scaling model

The design goal is that any number of server processes, on any hosts, serve
any project, sharing one set of databases, with no routing of a project to a
particular process. A load balancer can send any request anywhere, and a
process that dies or restarts takes no project offline.

The stores were first built for sharding: each collection or partition
owned by one process, with lifecycle guarded by locks inside that process.
Serving every project from every process instead moves that arbitration
into the databases:

- **Lifecycle in the database.** Creating and deleting partitions are
  registry rows, decided by constraints and row locks, not by locks inside
  one process.
- **Fencing by incarnation.** A handle opened on a partition's earlier life
  cannot write into its next one, on any process.
- **Background work any process can do.** Vector store purges are claimed
  from a queue in the database, and project deletion takes a lease per
  project, so every process can run these tasks without two doing the same
  one.
- **Capacity signals.** Event-loop lag and connection pool saturation show
  when a process is full and another should be added.

[#1574](https://github.com/MemMachine/MemMachine/issues/1574) records the
requirements, the sharded model and why it was built that way, the
measurements, and the issues that carry the change.

## Scope by component

Each component serves either from any process or from a bounded scope. A
deployment runs more than one process only with components that allow it.

| Component | Served from |
|---|---|
| Relational stores on PostgreSQL | Any process |
| Relational stores on SQLite | Processes on one host |
| Event backend on PostgreSQL with Qdrant or Milvus | Any process |
| SQLite vector store | One process |
| sqlite-vec vector store | Processes on one host |
| Declarative backend | One process ([#1753](https://github.com/MemMachine/MemMachine/issues/1753)) |
| Short-term memory | One process ([#1748](https://github.com/MemMachine/MemMachine/issues/1748)) |
| Semantic memory | One process ([#1745](https://github.com/MemMachine/MemMachine/issues/1745), [#1747](https://github.com/MemMachine/MemMachine/issues/1747)) |
| Episode counts | Cached per process ([#1746](https://github.com/MemMachine/MemMachine/issues/1746)) |
| Concurrent changes to one project's configuration | One process ([#1543](https://github.com/MemMachine/MemMachine/issues/1543)) |
| MCP over HTTP | The process that opened the MCP session ([#1751](https://github.com/MemMachine/MemMachine/issues/1751)) |

The deployment artifacts run one server replica
([#1754](https://github.com/MemMachine/MemMachine/issues/1754)).
