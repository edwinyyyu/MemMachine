# Vector store: the collection registry

Part of [vector store horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

A Qdrant or Milvus backend holds records and nothing that can arbitrate between
processes: no transactions, no unique constraints, and until Qdrant 1.16 no
conditional writes. The stores nevertheless kept their catalog inside the
backend, one `__registry` collection per namespace, and serialized their own
operations with process-local `asyncio` locks. Across processes that catalog
could not decide anything:

- **Creation raced.** Create was a read, then a write. Two processes creating
  the same name both succeeded: Qdrant's registry upsert let the last writer
  win, and Milvus's `insert` does not enforce primary-key uniqueness, so it
  kept two live entries.
- **Deletion did not end a collection.** The logical name was the tenant
  discriminator on every record, so a handle held in one process kept writing
  into a collection another process had deleted and re-created under the same
  name (#1563, reproducible sequentially, without any race).
- **Deletion raced with writes.** Delete was a filter-delete by name, and a
  write in flight during it outlived it.

The `VectorStore` contract papered over all three with "a collection must be
managed by at most one process; the consumer shards names across processes",
which no consumer did.

## Design

A collection's catalog entry lives where a primary key and a transaction can
arbitrate it: a relational database, reached through SQLAlchemy. The backend
keeps only records, each carrying the *incarnation* of the collection life that
wrote it.

### Roles

- `VectorStoreCollectionRegistry` (`common/vector_store/collection_registry/`)
  is the ABC. Its operations are `startup`, `register`, `get`, `is_live`,
  `unregister` and `claim_purgeable_incarnation`.
- `SQLAlchemyVectorStoreCollectionRegistry` is the one implementation. It
  supports PostgreSQL and SQLite, the dialects the segment store supports, and
  its params refuse any other dialect.
- Every store whose client connects to the same backend data, the same Qdrant
  server or cluster or the same Milvus database, must share one registry, in
  any process; a store connected to other data must not. A store reclaims its
  registry's tombstones through its own client: a round run through a client
  that cannot reach the records finds none, removes the tombstone, and leaves
  the records behind.
- The caller owns the registry's lifecycle. The database manager builds and
  starts each Qdrant and Milvus store's registry before it opens the store's
  client, and hands the started registry to the store in its params. The store
  never starts or stops it: a store that started a registry it was given
  would own a lifecycle it does not control.

### Tables

The registries of every vector store share two tables, defined once at module
level as the segment store's are, and keyed by `vector_store_name`: the
backend's key under `resources.databases`, which must match `[a-z0-9_]+` and
be at most 32 bytes. Two registry objects under one name are one registry.
One table pair with a key column, rather than a table pair per vector store,
keeps the schema static.

`collection_registry_ct`, the live collections:

| Column | Type | Role |
|---|---|---|
| `vector_store_name`, `namespace`, `name` | string, primary key | The collection's identity. The primary key arbitrates creation. |
| `incarnation` | UUID, unique | The collection's current life. Its records carry it. |
| `config` | JSON (JSONB on PostgreSQL) | The configuration the collection was created with. A handle is built from it; open-or-create compares against it. |

`collection_registry_gc`, the purge queue, one row per deleted incarnation
(its *tombstone*; see [purge](vector_store_purge.md)):

| Column | Type | Role |
|---|---|---|
| `incarnation` | UUID, primary key | The deleted life whose records remain. |
| `vector_store_name` | string | Whose purge claims the tombstone. |
| `namespace` | string | With `config`, names the native collection the records are in. |
| `name` | string | Kept for inspection; the purge does not read it. |
| `config` | JSON | With `namespace`, names the native collection. |
| `enqueued_at` | timestamp | When the deletion committed, on the database clock. |
| `failed_rounds` | integer | Consecutive purge rounds on the tombstone that raised. |
| `last_failed_at` | timestamp, nullable | When the last of them raised, on the database clock. |

Index `collection_registry_gc__vs_ea` on (`vector_store_name`, `enqueued_at`)
bounds the purge claim, equality before order. The index follows the
repository's naming scheme (table, two letters per column).

The queue carries `namespace` and `config` because nothing else knows where a
dead incarnation's records are once the collection is gone: the native
collection's name is derived from them, and every handle that knew them has
been discarded. It carries `name` only so that an operator looking at a
tombstone, a dead-lettered one above all, can tell which collection it was, as
the segment store's queue carries its partition key "for forensics". Dropping
it was considered; it stays, for inspection.

### Operations

**`register(namespace, name, config) -> incarnation`.** Mints a random UUID
(version 4) and inserts the live row. In the same transaction, *after* the
insert, a locking read checks that the incarnation is not waiting in the purge
queue. After the insert, a concurrent deletion that moved a colliding row to
the queue has already committed, because the insert waited on it. After the
check no queue row for this incarnation can appear before commit, because the
only live row carrying it is this uncommitted one. A primary-key violation
with a live row under the key raises
`VectorStoreCollectionAlreadyExistsError`. A rejected incarnation (unique
violation, or queued) is re-minted, up to 10 attempts, then
`VectorStoreAttemptsExhaustedError`. The loop and its bound are the segment
store's (#1661).

**`get(namespace, name)`** returns the live collection's incarnation and
configuration, or `None`. It is how a handle is opened: callers address
collections by name, and the name is resolved to an incarnation once, at open.

**`is_live(incarnation)`** is how handles are fenced (below).

**`unregister(namespace, name)`** is one transaction: `DELETE ... RETURNING`
the live row, then insert its tombstone with `enqueued_at = now()`. The
collection is unreachable when it commits. Racing deleters serialize on the
row's write lock and the loser deletes nothing, on PostgreSQL and SQLite
alike, so a deletion is idempotent and queues one tombstone. (The segment
store pins its row with the write fence its writes use before its queue
insert; the registry has no such fence, so the `DELETE` goes first.)

**`claim_purgeable_incarnation()`** is described in
[purge](vector_store_purge.md).

### Creation

Creation is *native first, registry last*. The store first makes sure the
native collection its configuration names exists, is indexed and is loaded,
then registers:

- A crash between the two leaves an empty native collection that the next
  creation of the same configuration adopts, never a live row whose records
  have nowhere to go.
- The registry's primary key is the one arbiter: a racing creator on any
  process loses at the insert, never in the backend.
- Native collections are shared by every logical collection of one namespace
  and configuration, so an empty one left by a failed creation is the one the
  next creation would have made, not per-collection garbage.
- Native creation converges: each step runs only when missing, so a creation
  that failed part way is completed by the next one as if it had never been
  attempted. How each backend does it is in the
  [Qdrant](qdrant_vector_store.md) and [Milvus](milvus_vector_store.md)
  documents.

**`open_or_create_collection`** is read-then-create, retried: a live row is
opened (or refused with `VectorStoreCollectionConfigMismatchError` if its
configuration differs); no row means create; losing the create means a racing
creator won, so the loop reads the winner's row and opens it; finding no row
after losing means a racing deleter removed the winner, so the loop creates
again. After 10 lost races it raises `VectorStoreAttemptsExhaustedError`. The
event backend's service locator creates a session's collection strictly when
it finds none and, on losing that create, opens the winner's.

The contract tests (`collection_lifecycle_contract.py`) pin both outcomes of a
lost race on Qdrant and Milvus: the loser opens the winner's collection, or
refuses it when its configuration differs, and exactly one registration is
lost.

### Handles and fencing

A handle is bound to the incarnation it was opened under. After that life is
deleted, every operation on the handle raises
`VectorStoreCollectionHandleStaleError`, and a collection created again under
the same name is a new life the old handle cannot reach.

- Every operation reads `is_live` before its remote call, to refuse a handle
  known to be dead.
- Every write reads it again after the remote call, so a write that completed
  under an incarnation that died meanwhile raises instead of reporting
  success.
- A read is not checked afterwards. A deleted collection's records stay until a
  purge round claims its tombstone, so a read in flight when the deletion
  commits returns a snapshot from before it, never a state halfway through a
  deletion: the answer it would have given had it run a moment earlier.
- No lock spans a remote call, and neither store holds a process-local lock.

A write can still land under a dead incarnation: between the two checks, or
after a check that never ran because the process died. Nothing can refuse it
at the backend, so the purge reclaims it. That is why a tombstone waits out a
retention before its purge starts.

### Differences from the segment store

The registry reuses the segment store's incarnation logic (#1661) wherever it
can: the bounded mint loop, the in-transaction locking re-check of the queue,
the idempotent deletion that queues a tombstone, the claim under `FOR UPDATE
SKIP LOCKED`, and the retried read-then-create of open-or-create. It differs
where the backend is remote:

- deletion is `DELETE ... RETURNING` then the queue insert, since there is no
  write fence to pin the row with;
- a handle is fenced by a liveness read before and after, not by a row lock
  held across the write, since the write is not in the database;
- the purge is one tombstone per call, each backend deleting the way it
  measured best;
- a tombstone waits out a retention, since a remote write can land after the
  deletion.

## Decisions

- **`unregister` is keyed by name.** Callers delete collections by name, and
  the name is resolved to the live incarnation inside the deleting
  transaction. Keying it by incarnation would make every caller resolve the
  name first, in a separate transaction, and would still need the name-keyed
  path for a caller that holds no handle.
- **The registry keeps the namespace** in the queue, because a dead
  incarnation's records can be found only through the native collection its
  namespace and configuration name. In #1627, where a store is one native
  collection, the queue no longer needs it.
- **`startup` keeps its name.** It creates the tables when missing, which is
  provisioning. Renaming it to `provision`, and taking provisioning out of
  runtime startup, is a change for every store at once, tracked in #1570.

## Alternatives considered

- **Keep the catalog in the backend** (the previous design). Rejected: neither
  backend can arbitrate a create, a delete or a claim.
- **Process-local locks.** They serialize one process only; the goal is any
  process serving any collection.
- **A lock or coordination service** (etcd, ZooKeeper). It would arbitrate,
  but it would add a stateful dependency for a problem the deployment's
  relational database, which every SQL-backed component already requires,
  solves with a primary key.
- **The incarnation as the only key**, with names resolved elsewhere. Callers
  address collections by name, so the name lookup has to be arbitrated
  somewhere; the primary key on `(namespace, name)` is that place.

## Consequences

- Every Qdrant or Milvus store needs a relational database: `QdrantConf` and
  `MilvusConf` name one in `collection_registry`, a required key.
- Existing Qdrant and Milvus data is orphaned: the per-namespace registry
  collections are no longer read, and existing records carry name-keyed values
  no incarnation resolves. No migration; pre-GA.
- A liveness read costs every operation one indexed lookup on the registry's
  database (two for a write).
