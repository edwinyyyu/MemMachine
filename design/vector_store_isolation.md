# Vector store: isolation between collections, and record UUIDs

Status: accepted and implemented 2026-09-29 in #1631. Part of [vector store
horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

A record is addressed by its UUID within its collection, and an upsert
replaces a whole record. On a backend where the collections of a store share
one id space, one collection's upsert of a UUID another collection already
holds would replace that other collection's record. Qdrant was such a backend
as MemMachine laid it out: many collections share a native collection, and a
point's id was its record's UUID. The contract guarded against it with a rule
on callers, that a record's UUID is minted by the service and never a value a
caller supplied, which held only as long as every ingestion path honored it.
This document answers what keeps collections apart, what happens when a UUID
is reused (across collections, across lives of one collection, or by a caller
that chose it), and what the contract can promise, given that every backend
has to be able to keep the promise.

## Collection lives

- An incarnation is a random UUID (version 4) minted by the registry at
  registration, one per collection life (see [collection
  registry](vector_store_collection_registry.md)). Every read, write and
  delete of a handle is scoped to it.
- An incarnation is never re-minted while it is live or its tombstone is
  queued, so a new life starts empty: no dead life's points are adopted by it
  or reclaimed out from under it, whatever record UUIDs either life used.
- The backends store UUIDs in RFC 9562's hyphenated text form, `str(uuid)`, 36
  characters. (Accepted: one text form everywhere.) In the registry an
  incarnation is SQLAlchemy's `Uuid` type, whose storage (native on
  PostgreSQL, 32-character hex on SQLite) never leaves the registry.

## The guarantee

`VectorStoreCollection` states:

> All data operations are scoped to this logical collection, whatever UUIDs
> its records carry: a record's UUID names it in this collection only, and
> the same UUID in another collection names another record.

A record UUID may come from anywhere, a caller included: reusing one in
another collection stores another record, and no operation on one collection
reads, replaces or deletes another's. The rule that the service mints every
UUID is gone.

| Store | How ids are scoped to a collection |
|---|---|
| SQLite, sqlite-vec | a records table per collection |
| Milvus | the incarnation in the primary key, `"{incarnation}:{record_uuid}"` (see [Milvus](milvus_vector_store.md)) |
| Qdrant | the point id is a UUIDv5 of the record UUID under the incarnation, with the record UUID kept in the payload (see [Qdrant](qdrant_vector_store.md)) |

A UUIDv5 carries 122 bits, and deriving one from a random incarnation and any
record UUID gives a caller no way to aim at another collection's point: a
SHA-1 collision needs control of both inputs.

### Can every backend keep it?

Surveyed from vendor documentation and source (2026-09-29; none of these
backends was run). Every one can, almost always by scoping the id.

| Backend | Per-tenant unit that scopes ids | Tenant in the native id | Conditional write (outcome on collision) |
|---|---|---|---|
| Qdrant | (custom shard keys; ids "only enforced unique within a shard key", which Qdrant calls an anti-pattern) | id must be a u64 or UUID: a derived UUID | `update_filter` (skipped silently) |
| Milvus | collection (deployment caps) | composite VARCHAR key | none found |
| Pinecone | namespace (100 to 1M per index by plan) | string id, 512 characters | none |
| Chroma | collection (Chroma Cloud: 1M) | string id (Chroma Cloud: 128 bytes) | `add` of an existing id is skipped silently; no conditional replace |
| Weaviate | multi-tenancy tenant (recommended; 1M+ tenants) | id must be a UUID: UUIDv5 of the pair | a single insert of an existing id fails (422), not atomically; batch overwrites |
| turbopuffer | namespace (recommended; unlimited) | string id, 64 bytes (a 73-character pair does not fit) | `upsert_condition` (skipped silently) |
| PostgreSQL + pgvector | table, or list partition (a few thousand partitions) | composite primary key | `ON CONFLICT ... DO UPDATE ... WHERE` (row count 0) |
| Elasticsearch, OpenSearch | index (shard limits cap it) | composite `_id` (512 bytes); custom routing does not scope `_id` | `op_type=create` (409); scripted upsert (`noop`) |
| Redis Query Engine | none: one keyspace | key prefix | `JSON.SET NX` (nil); Lua |
| LanceDB | table | `merge_insert` on (tenant, id); ids are never enforced unique | `merge_insert ... where` (skipped silently) |
| MongoDB Atlas | collection (not recommended) | composite `_id` or unique compound index | a filtered upsert on a raw id raises DuplicateKey |
| Vespa | streaming-mode group (`g=` in the document id) | document id | test-and-set (412), except when every replica lacks the document |
| Azure AI Search | index (at most 3,000 per service) | composite key `{tenant}_{record}` (1,024 characters; no colon) | none |

## Alternatives considered

- **Keep the rule that UUIDs are service-minted.** It holds only while every
  ingestion path honors it, and a slip replaces another tenant's record, a
  security failure far from its cause.
- **A conditional upsert on Qdrant** (`update_filter` on the writer's
  incarnation, first write wins). It keeps collections from replacing each
  other's points, but a reused UUID's record is dropped without an error, the
  contract has to keep a uniqueness rule for callers, and single-point writes
  ran 46% slower. It is Qdrant-specific, where scoping the id is what nearly
  every backend does.
- **Check before writing** (read the id, then write if it is absent or the
  writer's). Not atomic: another collection's write can land between the read
  and the write.
- **Readable point ids on Qdrant.** Deriving hides the record UUID from the
  point id, but it stays in the payload, so an operator still finds a record
  by filtering on it, as with Milvus's composite key; the cost is a payload
  field returned by every search (measured in the
  [Qdrant](qdrant_vector_store.md) document).
- **Reversible derived ids on Qdrant** (for instance the record UUID XOR the
  incarnation), which would spare a search the payload read. Isolation would
  then rest on incarnations staying secret: someone who learns two
  incarnations and a record UUID, from logs, the registry's tables or a
  backup, could compute a colliding UUID and, through an ordinary tenant
  account, hide another tenant's record. A one-way UUIDv5 needs write access
  to a database for that. (Accepted: UUIDv5; the analysis is in the
  [Qdrant](qdrant_vector_store.md) document.)

## Consequences

- Existing Qdrant points written before #1631 are orphaned with the rest of
  its layout changes; their ids are not derived.
- A lookup by hand on Qdrant filters on `sys-record_uuid`, which scans the
  collection; a keyword index on it would make that an index read, at a small
  cost to writes, and is not added.
