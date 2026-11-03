# Design documents

These documents explain how MemMachine is built and why: its architecture,
the decisions that shape it, and the trade-offs those decisions carry. They
are for contributors and for anyone who needs to reason about what the
system can do. The user documentation in `docs/` (installation,
configuration, and the API reference) is published on the documentation
site; these documents link to it rather than repeat it.

## System overview

These documents are retroactive: they describe the system as built, written
after the decisions they cover were made. Read them in this order:

| Document | What it covers |
|---|---|
| [Overview](overview.md) | What MemMachine is, its components, the add and search paths, the packages, and a glossary |
| [Data model and tenancy](data_model.md) | Organizations, projects, episodes, metadata and filters, and semantic sets |
| [Episodic memory](episodic_memory.md) | Short-term memory, the event and declarative long-term backends, reranking, and the retrieval agent |
| [Semantic memory](semantic_memory.md) | Features extracted by a language model: set types, sets, categories, tags, ingestion, and search |
| [Storage](storage.md) | The stores and their backends, multi-tenancy, schemas, consistency, deletion, and encryption |
| [Server and API](server_and_api.md) | The server process, configuration, the REST API, identity, MCP, the SDKs, and observability |
| [Deployment and scaling](deployment_and_scaling.md) | Deployment shapes, what a deployment provides, and serving from many processes |
| [Integrations and examples](integrations_and_examples.md) | How integrations use the SDKs, what exists, and how it is maintained |
| [Evaluation](evaluation.md) | What the benchmarks measure and what a score depends on |

## Component designs

These documents were written with the changes they describe.

| Document | What it covers |
|---|---|
| [Vector store horizontal scaling](vector_store_horizontal_scaling.md) | The vector store contract for many processes, and the index of the vector store documents: partition registry, purge, consistency, isolation, Qdrant, and Milvus |
| [Segment store shared tables](segment_store_shared_tables.md) | The event backend's segment store: shared tables with incarnation-scoped tenant keys |
