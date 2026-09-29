# Vector store: consistency

Status: the backend models and measurements are facts as of 2026-09-29. The
async clients are accepted and implemented in #1631. The contract text,
Milvus's levels, and removing `get` in #1631 are proposed. Part of [vector
store horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

Once any process may serve any collection, what a read sees of earlier writes,
including writes from other processes, is part of the contract, and the
contract says nothing about it. Two defects were found under it on Milvus:

- The store created every collection at Session consistency
  (`MilvusConf.consistency_level` defaulted to `"Session"` since the Milvus
  backend's introduction in #1471, with no stated reason; Milvus's own default
  is Bounded) and passed no level on its calls. A Session search waits until
  the searching process's last write is visible, so under steady ingestion
  every search in that process stalls.
- The store ran pymilvus's synchronous client under `asyncio.to_thread`, so
  every waiting call held a thread of the process's shared executor.

## What each backend guarantees

### SQLite (engine-backed) and sqlite-vec

The engine-backed store commits a write to SQLite and applies it to its
in-process search index before returning; its collections are served by one
process. sqlite-vec writes the vector in the same SQL transaction as the row.
On both, a write that has returned is visible to every later query, and writes
take effect in the order they return.

### Qdrant

- **Single node** (what MemMachine's tests run): upserts and deletes pass
  `wait=True`, qdrant-client's default, and return only once applied. Every later read, from any process or host, sees them. The writer
  waits; readers never do.
- **Replicated** (untested by MemMachine). From Qdrant's consistency
  documentation:
  - a write succeeds once `write_consistency_factor` replicas apply it, 1 by
    default;
  - a read goes to one replica by default ("read consistency 1"), so it can
    miss an acknowledged write, and successive reads can "blink" between
    replicas;
  - with the default `weak` write ordering, "write operations can be freely
    reordered", and "concurrent updates on one point can result in an
    inconsistent state": replicas can keep different versions of a point.
    `medium` and `strong` ordering serialize writes through a leader.
  - The store sets none of these, and creates collections with the server's
    default replication factor, 1.

### Milvus

- **Writes are totally ordered per shard.** Each write is appended to its
  shard's write-ahead log, which stamps it with a timestamp at append
  (`internal/streamingnode/server/wal/interceptors/timetick/timetick_interceptor.go`,
  v2.6.24). Every replica applies the log in that order. A record's writes all
  go to one shard, so they take effect in the order they were appended,
  identically everywhere: a write sent after another returned, from any
  process or host, lands after it.
- **Reads see a prefix of that order.** A query node searches at a snapshot
  timestamp, its *tsafe*: the point up to which it has applied the log. A read
  never sees a later write to a shard without the earlier ones.
- **A consistency level picks how fresh that snapshot must be.** The proxy
  computes a guarantee timestamp and the query node waits until its tsafe
  reaches it (`waitTSafe`, `internal/querynodev2/delegator/delegator.go`),
  then searches at its tsafe, which may be fresher:
  - `Strong`: the request's own start time, so every write that returned
    before the read began;
  - `Bounded`: the start time less `common.gracefulTime` (5,000 ms by
    default);
  - `Session`: the searching process's last write (pymilvus caches it per
    process, per collection, in `GlobalCache.collection_ts`); a process that
    wrote nothing gets no wait;
  - `Eventually`: no wait.
- **A level is per request.** A request that names one overrides the
  collection's (`internal/proxy/task_query.go`); one that names none gets the
  collection's, fixed at creation.
- **Bounded holds by waiting, not by speed.** When the node keeps up, which is
  the normal case, a Bounded search waits for nothing and reads the node's
  latest state, much fresher than 5 s. When the node falls behind by more than
  5 s, the search waits until it catches up: a slow node makes searches
  slower, never staler. If the node's tsafe does not advance for 3 s
  (`queryNode.waitTsafeStallTimeout`, from 2.6.15), the request fails with an
  error the proxy can retry on another replica; `request_timeout_seconds`
  bounds the rest. Strong and Session use the same wait with a tighter target.
- **Session means nothing across processes.** Its guarantee is the searching
  process's own writes. A deployment whose requests land on any process gives
  a client no read-your-writes through it.
- **The wait is per shard, not per tenant.** A query node's tsafe belongs to
  a shard of the native collection (`shardDelegator.latestTsafe`), and
  pymilvus's Session timestamp to the whole collection (keyed by endpoint,
  database and collection). Every tenant of a native collection shares its
  shards' logs, so a Strong or Session search waits for every tenant's
  writes, not only its own.

## Measurements

All on Milvus 2.6.24 (4 CPUs / 5 GB), one native collection in the store's
layout (composite key, partition key = incarnation with isolation, HNSW_SQ
4-bit + FP16), 300 tenants x 1,000 rows of 128 dimensions; 8 tasks searching
tenants 0-149 (top 10); 4 writers upserting 5 new points at a time into
tenants 150-299, which the searches never read; 15 s per phase, two rounds.

**Search level against where the writes come from, on the async client**
(collection created at Session, as the store creates it today; "default"
passes no level, as the store does):

| Writers | Search level | Searches/s | p50 | p99 |
|---|---|---|---|---|
| none | default (Session) | 1,272-4,008 | 1.8-4.3 ms | 4.1-36 ms |
| none | Bounded | 1,256-4,106 | 1.8-4.1 ms | 3.9-33 ms |
| same process | default (Session) | 4.3-5.5 | 649-850 ms | 5.3-5.8 s |
| same process | Strong | 7.6-8.4 | 154-188 ms | 6.7-6.9 s |
| same process | Bounded | 718-1,403 | 4.2-7.1 ms | 39-58 ms |
| same process | Eventually | 654-1,263 | 4.6-7.9 ms | 37-58 ms |
| other process | default (Session) | 715-874 | 6.3-7.2 ms | 45-55 ms |
| other process | Bounded | 769-891 | 6.1-6.7 ms | 45-57 ms |
| other process | Strong | 40-41 | 5.8-6.3 ms | 6.5 s |

The second round ran the phases in reverse order; writes grow the collection,
so later phases run slower, and the ranges cover both rounds. Session stalls
whenever the searching process also writes, which is MemMachine's normal case
(one server process ingests and retrieves), and Strong stalls whoever writes.
Bounded serves as many searches as Eventually, so no looser level is needed.
The stall is the server-side wait; the client does not change it.

**Waiting calls and the client's threads** (collection Bounded; writers in
another process; the measured process runs 8 Bounded searches plus G gets by
id at level L; on the sync client every call runs under `asyncio.to_thread` on
the default executor, 15 threads here):

| Beside the 8 searches | Client | Searches/s | Search p99 | Get p99 |
|---|---|---|---|---|
| nothing | sync | 1,128-1,627 | 20-43 ms | |
| 4 Strong gets | sync | 806-1,452 | 31-53 ms | 2.9-3.9 s |
| 32 Bounded gets | sync | 201-293 | 49-96 ms | 49-95 ms |
| 32 Strong gets | sync | 12-21 | 9.3-10.0 s | 13-15 s |
| 4 Strong gets | async | 1,055-1,434 | 32-44 ms | 3.0-3.5 s |
| 32 Strong gets | async | 670-752 | 43-45 ms | 0.4-2.5 s |

On the sync client, once more calls wait than the executor has threads, every
other call of the process queues behind them. On the async client a waiting
call holds no thread. The server-side wait itself is the same on both.

**Before the fix, on the sync client** (collection Session; 4 searcher and 4
writer threads in one process): searches that passed no level ran 2.4-3.1 per
second at a p50 of 0.4-0.7 s and a p99 of 6.2-7.5 s, against 604-990 per
second and a p99 of 23-43 ms at Bounded.

## Decisions and proposals

**Clients are asynchronous** (accepted, implemented): `AsyncQdrantClient` and
`AsyncMilvusClient`. pymilvus's docstring still calls `AsyncMilvusClient`
experimental and partial; it has every call the store makes, and the store's
tests pass on it.

**Milvus searches run at Bounded** (proposed). Every search passes the
configured level per request, and `MilvusConf.consistency_level` defaults to
`Bounded`, Milvus's own default, instead of `Session`. Passing it per request
makes collections created at Session before follow the setting too. Tests set
`Strong`.

**The Milvus purge listing runs at Strong** (proposed). Bounded would also be
correct: every write under a due tombstone's incarnation is older than the
retention, far older than the bound, so a round never finds nothing while
points remain. Strong also lets a round see the deletions of the round before
it, which the sweeper runs 1 s later, so no round re-lists keys already
deleted. It is one background call at a time, so its wait costs nothing else.

**`get` is removed in #1631, not left to #1663** (proposed). `get` has one
production caller: the semantic storage's `update_feature`, which reads a
feature's stored vector back to write it again with fresh properties, because
`upsert` replaces a whole record.

- That read feeds a write, so a stale read becomes a wrong write that stays.
  At Bounded it can miss a feature created moments before and fail the update,
  or return an embedding older than a concurrent update's and write the old
  one back. The second race exists at every level on every backend, because
  nothing serializes two updates.
- Keeping `get` means stating its consistency. The useful statement, that it
  reflects every write that returned before it began, costs Strong on Milvus
  (a p99 of 3.0-3.5 s under writes, measured above) and read consistency
  settings the Qdrant store does not make on a replicated deployment.
- #1663 removes `get` and the read-modify-write anyway. The minimal version
  here: the semantic storage's vector records carry only `feature_id`, which
  never changes (the other ten properties are copies of feature-row columns
  nothing filters on); `update_feature` writes the vector store only when
  given a new embedding, and writes the whole record; `get` leaves the ABC,
  the four stores and the in-memory test collection; the tests that observe
  records through `get` move to `query`, as #1663's port already did. About 4
  hours here, plus merging it into #1627 and #1663.

**Contract text** (proposed), on `VectorStoreCollection`, once `get` is gone:

> A write (`upsert` or `delete`) that has returned is durable, and every
> later query eventually sees it; how soon is implementation-defined, and
> each implementation states it. Until then a query may miss a record
> upserted, or return a record deleted, shortly before it. Writes to one
> record take effect in the order they were made, when each begins after the
> one before it returned.

Each store's own docstring states its bound: at once for both SQLite stores
and for Qdrant on a single node; at most `common.gracefulTime` (5 s by
default) for Milvus at Bounded, a query waiting rather than reading staler
when the server falls behind.

**Open: replicated Qdrant.** The ordering sentence holds for SQLite,
sqlite-vec, Milvus and single-node Qdrant. A replicated Qdrant deployment with
the default `weak` write ordering can apply a record's writes in different
orders on different replicas, a delete before its upsert for instance, which
would leave the record on that replica (inferred from Qdrant's documentation,
not reproduced). Meeting the contract there needs `medium` or `strong` write
ordering on the store's writes, which costs nothing on a single node and has
not been measured on a cluster.

## Alternatives considered

- **Keep Session.** It stalls searches behind the process's own writes
  (measured above), and its guarantee does not extend past one process.
- **Strong for searches.** Every search waits for every write, from any
  process.
- **Eventually.** No faster than Bounded when the node keeps up (measured
  above), and no bound when it does not.
- **Per-handle guarantee timestamps**: each handle remembers its last write's
  timestamp and searches up to it, so a tenant waits only for its own writes.
  pymilvus's `upsert` does not return the timestamp and overwrites a caller's
  `guarantee_timestamp` unless the request passes the undocumented
  `Customized` level; and the timestamp lives in one process, so a request
  served by another process gets no guarantee from it.
- **Carrying write timestamps across processes**, in the registry (a SQL write
  per upsert) or as a session token returned to API clients (an API change).
  Neither is justified while no caller needs read-your-writes on search.
- **Keep `get` with Strong until #1663.** Cheaper now (the level on one call),
  but it leaves the lost-update race and a contract statement that the stores
  meet only at Strong's cost.
