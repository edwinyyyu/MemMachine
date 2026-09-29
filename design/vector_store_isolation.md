# Vector store: isolation between collections, and record UUIDs

Status: incarnations and the service-minted UUID rule are accepted and
implemented in #1631. The isolation guarantee and the looser UUID rule it
allows are proposed. Part of [vector store horizontal
scaling](vector_store_horizontal_scaling.md).

## Problem

A record is addressed by its UUID within its collection, and an upsert
replaces a whole record. On a backend where the collections of a store share
one id space, one collection's upsert of a UUID another collection already
holds would replace that other collection's record. Qdrant is such a backend
as MemMachine lays it out: many collections share a native collection, and a
point id is the record UUID. This document answers what keeps collections
apart, what happens when a UUID is reused (across collections, across lives of
one collection, or by a caller that chose it), and what the contract can
promise, given that every backend has to be able to keep the promise.

## Collection lives

- An incarnation is a random UUID (version 4) minted by the registry at
  registration, one per collection life (see [collection
  registry](vector_store_collection_registry.md)). Every read, write and
  delete of a handle is scoped to it.
- An incarnation is never re-minted while it is live or its tombstone is
  queued, so a new life starts empty: no dead life's points are adopted by it
  or reclaimed out from under it, whatever record UUIDs either life used.
- The backends store it in RFC 9562's hyphenated text form, `str(uuid)`.
  (Accepted: one text form everywhere.) In the registry it is SQLAlchemy's
  `Uuid` type, whose storage (native on PostgreSQL, 32-character hex on
  SQLite) never leaves the registry.

## Where record UUIDs come from

Every record UUID in MemMachine is minted by the service: random, or derived
(UUIDv5) only from identifiers the service minted itself. The contract makes
that a rule of `upsert` (accepted, implemented): a record's UUID is never a
value a caller supplied or one derived from it, even where an ingestion path
would pass the caller's identifier through, because the collections of one
store may share the backend's id space.

## Proposed: an isolation guarantee, and a looser UUID rule

The rule above protects isolation only as long as every ingestion path honors
it. The proposal makes isolation the stores' obligation, whatever UUID a
record carries, and relaxes the callers' rule to what is still needed.

**Contract text.** The collection docstring's "All data operations are scoped
to this logical collection" becomes:

> All data operations are scoped to this logical collection, whatever UUIDs
> its records carry: none reads, replaces or deletes another collection's
> record.

and `upsert`'s rule becomes:

> Record UUIDs must be unique across the store, including collections
> deleted but not yet purged; what happens to a record whose UUID is not is
> implementation-defined.

**How each store meets it.**

| Store | Mechanism | A reused UUID's record |
|---|---|---|
| SQLite, sqlite-vec | a records table per collection | is stored |
| Milvus | the incarnation in the primary key (see [Milvus](milvus_vector_store.md)) | is stored |
| Qdrant | a conditional upsert, first write wins (see [Qdrant](qdrant_vector_store.md)) | is not stored, without an error |

**Why it is easier to use correctly.** Under the committed rule a caller that
slips (one ingestion path passing a client's identifier through as a record
UUID) can replace another tenant's record: a security failure far from its
cause. Under the guarantee the same slip can cost only the slipping caller its
own record, on one backend. Stealing another collection's record is impossible
on every store; squatting (claiming a UUID first so another collection's later
record of it is skipped) requires predicting a UUID another collection will
use, which service-minted UUIDs rule out, so it is not a reasonable attack
(accepted). The remaining hazard is an innocent reuse, such as copying a
collection's records into a new collection under the same UUIDs; the
uniqueness rule is for that caller.

**The stronger alternative.** The contract could instead say that a UUID names
a record only within its collection, so a reused UUID is always stored, as
SQLite and Milvus already behave. Every surveyed backend can scope ids that
way (below), but on Qdrant only by deriving each point id from the incarnation
and the record UUID, which makes a point's id unreadable to an operator; and
Weaviate's shared-collection layout would need the same hashing. (Accepted for
now: Qdrant point ids stay readable, unless derivation becomes necessary.)

### Can every backend meet the guarantee?

Surveyed from vendor documentation and source (2026-09-29; none of these
backends was run). Every one can, almost always by scoping the id rather than
by a conditional write.

| Backend | Per-tenant unit that scopes ids | Tenant in the native id | Conditional write (outcome on collision) |
|---|---|---|---|
| Qdrant | (custom shard keys; ids "only enforced unique within a shard key", which Qdrant calls an anti-pattern) | id must be a u64 or UUID: a derived UUID only | `update_filter` (skipped silently) |
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

What happens to a colliding record differs (stored separately, refused with an
error, or skipped silently), which is why the proposed rule leaves it
implementation-defined.

**On derived UUIDs.** A UUIDv5 carries 122 bits; deriving one from another
UUIDv5 does not reduce that meaningfully, and a SHA-1 collision needs control
of both inputs, which a caller does not have over the service's namespace.

## Alternatives considered

- **Keep only the rule that UUIDs are service-minted** (the committed state).
  It holds only while every ingestion path honors it.
- **Check before writing** (read the id, then write if it is absent or the
  writer's). Not atomic: another collection's write can land between the read
  and the write.
- **Last-write-wins on a collision.** It takes the record from whichever
  collection wrote first. First-write-wins cannot take anything from another
  collection. (Accepted: first-write-wins is preferred.)

## Consequences

- If the proposal is taken, a Qdrant backend needs 1.16 or later, for the
  conditional upsert; CI tests against 1.19.1.
- A record written under a reused UUID on Qdrant is dropped without an error.
  Callers keep minting UUIDs per record, as every caller does today.
