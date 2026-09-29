# Vector store: record identity, UUID reuse and isolation

Status: the layout is accepted 2026-09-29, in review in #1631. The isolation
guarantee, the loosened UUID rule and Qdrant's conditional upsert are
proposed. Part of [vector store horizontal
scaling](vector_store_horizontal_scaling.md).

## Problem

Many logical collections share one native collection on Qdrant and Milvus (see
[backend layouts](vector_store_backend_layouts.md)). A record is addressed by
its UUID within its collection, and an upsert replaces a whole record: Qdrant
replaces the point, vectors and payload alike; Milvus deletes by primary key
in every partition, then inserts. So wherever the collections of a native
collection share the backend's id space, one collection's upsert of a UUID
another collection already holds replaces that other collection's record.

The questions this document answers: what identifies a collection life and a
record in each backend; what happens when a UUID is reused, across
collections, across lives of one collection, or by a caller that chose it; and
what the contract guarantees.

## Incarnations

- An incarnation is a random UUID (version 4) minted by the registry at
  registration, one per collection life (see [collection
  registry](vector_store_collection_registry.md)).
- In the backends it is written in RFC 9562's hyphenated text form,
  `str(uuid)`, 36 characters: in Qdrant's `sys-incarnation` payload field and
  in Milvus's `partition_key` field and primary key. Every read, write and
  delete of a handle is scoped to it. (Decided: one text form everywhere; an
  earlier revision mixed 32-character hex into Milvus keys.)
- In the registry it is SQLAlchemy's `Uuid` type: native `uuid` on PostgreSQL
  and 32-character hex on SQLite, which is SQLAlchemy's storage detail and
  never leaves the registry.
- An incarnation is never re-minted while it is live or its tombstone is
  queued, so a new life starts empty: no dead life's points are adopted by it
  or reclaimed out from under it.

## Record identity in each backend

**Qdrant.** The point id is the record UUID as it is. The incarnation is a
payload field with a keyword index marked `is_tenant`. Every search, retrieve
and delete is filtered on the incarnation, so a collection sees only its own
points. But the id space is the native collection's: a plain upsert of a UUID
that another collection's point already holds replaces that point, and it
moves to the writer's incarnation. The first collection's record then goes
missing (its reads filter it out) and the writer's record is stored; neither
collection ever reads the other's.

**Milvus.** The primary key is `"{incarnation}:{record_uuid}"` (73 characters,
in a VARCHAR field of 128), with the record UUID also stored in a
`record_uuid` field that reads return. The composite key is load-bearing:
Milvus's upsert deletes by primary key in every partition (`AllPartitionsID`)
before inserting, so with a bare record UUID as the key, one collection's
upsert would delete another collection's record of the same UUID. With the
composite key, two collections' records of one UUID are two entities, and the
key's first half is the same value as the partition key. Measured against bare
keys (Milvus 2.6.24, 4 CPUs / 5 GB, 1.3M rows: 1,000 tenants of 1,000 and 3
dead tenants of 100,000; one run), the composite key cost nothing measurable:

| p50 | Composite key | Bare key | Bare, filtered on incarnation |
|---|---|---|---|
| Get 10 ids | 1.29 ms | 0.87 ms | 1.22 ms |
| Delete 1 id | 1.10 ms | 1.10 ms | 200 ms (a delete by filter queries first) |
| Upsert 5 | 1.61 ms | 1.50 ms | |
| Search | 1.56 ms | 1.73 ms | |
| Memory after load | 1.51 GB | 1.44 GB | |

A later sweep (upsert by batch size, get, delete, 8 concurrent writers,
searches with and without writers) put every difference within run-to-run
variation.

**SQLite and sqlite-vec.** Each collection has its own records table with
`uuid` unique in it, so records of different collections never share an id
space.

## Where record UUIDs come from

Every record UUID in MemMachine today is minted by the service: random, or
derived (UUIDv5) only from identifiers the service minted itself. The contract
as committed (`d3231dbad`) makes that a rule of `upsert`: a record's UUID is
never a value a caller supplied or one derived from it, because the
collections of one store may share the backend's id space, so a UUID a caller
chose could name, and replace, another collection's record.

## Proposed: an isolation guarantee, and a looser UUID rule

The rule above protects isolation only as long as every ingestion path honors
it. The proposal moves the protection into the stores, so that isolation holds
whatever UUID a record carries, and relaxes the rule to what is still needed.

**Contract text.** The collection docstring already says "All data operations
are scoped to this logical collection." It becomes:

> All data operations are scoped to this logical collection, whatever UUIDs
> its records carry: none reads, replaces or deletes another collection's
> record.

and `upsert`'s paragraph becomes:

> Record UUIDs must be unique across the store, including collections
> deleted but not yet purged; what happens to a record whose UUID is not is
> implementation-defined.

**How each store meets it.**

- SQLite and sqlite-vec: per-collection tables; a reused UUID is simply
  stored.
- Milvus: the composite key; a reused UUID is simply stored.
- Qdrant: a conditional upsert. Qdrant 1.16 added `update_filter`
  (qdrant/qdrant#7006): an upsert with a filter on the writer's incarnation
  inserts new ids as before, updates the writer's own points, and leaves a
  point that exists under another incarnation as it is, skipping the writer's
  point without an error. That is first-write-wins, where the plain upsert is
  last-write-wins. Reads and deletes are already filtered on the incarnation.

**What a reused UUID costs.** Only the caller that reused it, and only on
Qdrant: its record is not stored. It cannot read, replace or delete anything
of another collection's. A caller that deliberately reuses another
collection's UUIDs gains nothing. The remaining hazard is a caller that reuses
UUIDs innocently, for instance copying a collection's records into a new
collection under the same UUIDs; on Qdrant, those records are silently not
stored while the old collection lives or its tombstone waits for purge. The
contract's uniqueness rule is for that caller.

**Threats considered.**

- *Stealing* (replacing another collection's record by upserting its UUID):
  impossible under the guarantee on every store.
- *Squatting* (claiming a UUID first so that another collection's later record
  of that UUID is skipped): requires predicting a UUID another collection will
  use. Service-minted UUIDs are random or derived from service-minted
  identifiers, so this is not a reasonable attack. (Decided.)
- *Derived UUIDs.* A UUIDv5 carries 122 bits; deriving one from another UUIDv5
  does not reduce that meaningfully, and a SHA-1 collision needs control of
  both inputs, which a caller does not have over the service's namespace.

**Cost of Qdrant's conditional upsert** (Qdrant 1.19.1, 4 CPUs / 4 GB, 300
tenants x 1,000 points in the store's index layout):

| Request | p50 plain -> scoped | Server CPU per point |
|---|---|---|
| 1 new point | 1.53 -> 2.23 ms (+46%) | x1.85 |
| 1 point overwritten | 1.90 -> 2.00 ms (+6%) | x1.23 |
| 10 points | +12-14% | x1.10-1.20 |
| 100 points | +4% new, +39% overwrite (noisy) | x1.08 / x1.60 |
| 1,000 points | +1% new, +12% overwrite | x1.23-1.28 |
| 8 clients, 1 new point each | 4,384-4,684 -> 3,440-3,554 requests/s (-24%) | x1.10 per request |

The cost stays with the writer. With 4 clients searching tenants 0-149 while 4
others upserted into tenants 150-299, the searchers ran 2,653-2,964 per second
beside conditional 1-point upserts against 2,729-2,753 beside plain ones, and
2,300-2,367 against 2,307-2,381 with 5-point upserts; the writers' own
throughput dropped 7-9%.

### Can every backend meet the guarantee?

Surveyed from vendor documentation and source (2026-09-29; none of these
backends was run). Every one can, almost always by scoping the id, not by a
conditional write.

| Backend | Per-tenant unit that scopes ids | Tenant in the native id | Conditional write (outcome on collision) |
|---|---|---|---|
| Qdrant | (custom shard keys; ids "only enforced unique within a shard key", which Qdrant calls an anti-pattern) | id must be a u64 or UUID: a derived UUID only | `update_filter` (skipped silently) |
| Milvus | collection (deployment caps) | composite VARCHAR key (this store's) | none found |
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

What happens to a colliding record differs (stored separately, refused with an
error, or skipped silently), which is why the contract leaves it
implementation-defined. Weaviate's shared-collection fallback is the one
probabilistic mechanism (a hashed id); its native tenants scope ids
structurally.

## Decisions

- **Qdrant point ids are the record UUIDs, not derived ids.** A derived id
  (UUIDv5 of incarnation and record) would scope the id space structurally,
  but the mapping from a point back to its record would no longer be visible
  in the database, and every operator query would need it. Readable ids are
  kept unless derivation becomes necessary.
- **Milvus keeps the composite key**, for the upsert semantics above, at no
  measured cost.
- **First-write-wins is preferred to last-write-wins** for a colliding UUID:
  it cannot take anything from another collection.

## Alternatives considered

- **Keep only the rule that UUIDs are service-minted** (the committed state).
  It holds only while every ingestion path honors it; an ingestion path that
  passed a caller's identifier through would break isolation on Qdrant.
- **Check before writing** (read the id, then write if it is absent or the
  writer's). Not atomic: another collection's write can land between the read
  and the write.
- **Bare keys on Milvus.** As fast, but they break isolation through the
  upsert's delete in every partition.

## Consequences

- If the proposal is taken, a Qdrant backend needs 1.16 or later; CI tests
  against 1.19.1.
- A record written under a reused UUID on Qdrant is dropped without an error.
  Callers keep minting UUIDs per record, as every caller does today.
