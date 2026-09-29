# Vector store horizontal scaling

Status: in review in #1631 (2026-09-29). The documents below mark what is
accepted and what is proposed.

## Problem

The `VectorStore` contract said "a collection must be managed by at most one
process; the consumer shards names across processes." No consumer sharded
anything, and the Qdrant and Milvus stores relied on that sentence for
correctness:

- their catalogs lived inside the backend, which cannot arbitrate a create, a
  delete or a purge between processes;
- they serialized their own operations with process-local locks;
- a collection's name was the tenant discriminator on its points, so deleting
  and re-creating a name did not end the old life (#1563).

The goal is that any number of MemMachine server processes, on any hosts, can
serve any collection of a shared Qdrant or Milvus deployment, with:

- creation of a name arbitrated once;
- deletion final, not undone by a write in flight;
- a re-created name starting empty;
- reclamation of deleted data bounded in cost and safe to run from every
  process;
- reads and writes whose consistency the contract states.

The SQLite stores keep their single-process (engine-backed store) and
single-node (sqlite-vec) bounds.

## Design at a glance

| Piece | What it does | Document |
|---|---|---|
| Collection registry | A relational database arbitrates which collections exist, under which incarnation and configuration, and which dead incarnations await purge. | [collection registry](vector_store_collection_registry.md) |
| Incarnations and handles | Each collection life has a random UUID its points carry; a handle is bound to one life and fenced by a liveness check before every operation and after every write. | [collection registry](vector_store_collection_registry.md) |
| Tombstones and purge | Deletion queues a tombstone; after a retention, bounded purge rounds reclaim the points, with backoff and dead-lettering for rounds that fail. | [purge](vector_store_purge.md) |
| Record identity | Where record UUIDs come from, what reusing one does, and the isolation guarantee between collections. | [record identity](vector_store_record_identity.md) |
| Consistency | What a read sees of earlier writes on each backend, the level each Milvus call uses, and what the contract guarantees. | [consistency](vector_store_consistency.md) |
| Backend layouts | How Qdrant and Milvus lay out shared native collections, indexes and tenancy. | [backend layouts](vector_store_backend_layouts.md) |

## Lifecycle of a collection

1. **Create.** The store ensures the native collection for the namespace and
   configuration exists, indexed and loaded, then inserts a registry row under
   a freshly minted incarnation. A racing creator loses at the registry's
   primary key.
2. **Open.** The registry resolves `(namespace, name)` to the live incarnation
   and its configuration; the handle is bound to that incarnation.
3. **Use.** Every point a handle writes carries its incarnation, and every
   read and delete is scoped to it. Before each operation, and after each
   write, the handle checks that its incarnation is still live.
4. **Delete.** One registry transaction removes the live row and queues a
   tombstone. Every handle of that life is stale from then on, in every
   process.
5. **Purge.** Once the retention (a day by default) has passed, sweepers in
   every process claim tombstones oldest first and reclaim their points in
   bounded rounds, until a round finds none and the tombstone goes.

## Guarantees by store

| Store | Processes that may serve a collection | Deletion reclaims |
|---|---|---|
| SQLite (engine-backed) | one | at once |
| sqlite-vec | any on one node | at once |
| Qdrant | any, sharing the store's registry | by purge, after the retention |
| Milvus | any, sharing the store's registry | by purge, after the retention |

## Configuration

`QdrantConf` and `MilvusConf` gain:

- `collection_registry` (required): the relational database, a name under
  `resources.databases`, that holds the store's registry;
- `tombstone_retention_seconds` (86,400): how long a deleted collection's
  points stay before their purge starts;
- `request_timeout_seconds` (30): the bound on every request to the backend,
  which the retention must far exceed.

`MilvusConf` also gains `max_varchar_length` (65,535) and `purge_batch_size`
(10,000), two sizes the Milvus server's own configuration bounds. The wizard
points the registry at its SQLite database; the sample configurations and the
Helm chart point it at the relational database their other components use.

## Clients

Every backend client is the library's asynchronous client: `AsyncQdrantClient`
and pymilvus's `AsyncMilvusClient`. A synchronous client run on worker threads
holds a thread of the process's shared executor for the whole of each request,
so enough slow requests starve every other call of the process (measured in
[consistency](vector_store_consistency.md)). (Decided: a library's async
client is always used when one exists.)

## Related work

- The segment store's shared tables with incarnation-scoped keys (#1661, [its
  design](segment_store_shared_tables.md)) are the model for the registry's
  incarnation logic.
- #1627 makes a vector store one native collection with string-keyed
  partitions, on top of this.
- #1663 removes `VectorStoreCollection.get` and the semantic memory's
  read-modify-write that used it.
- #1570 tracks moving provisioning (table and collection creation) out of
  runtime startup for every store.

## Deployment consequences

- Existing Qdrant and Milvus data is orphaned, and an existing native Milvus
  collection has to be dropped. No migration; pre-GA.
- Milvus Lite is not supported; the Milvus store needs a Milvus server.
- Every Qdrant or Milvus deployment needs a relational database for its
  registry.
