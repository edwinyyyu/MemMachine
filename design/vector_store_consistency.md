# Vector store: consistency

Status: accepted and implemented 2026-09-29 in #1631, except the write
ordering of replicated Qdrant deployments, which is open. Part of [vector
store horizontal scaling](vector_store_horizontal_scaling.md).

## Problem

Once any process may serve any collection, what a read sees of earlier writes,
including writes from other processes, is part of the contract, and the
contract said nothing about it. Under that silence:

- the Milvus store read at Session consistency, which stalls every search in a
  process behind that process's own writes, and means nothing to another
  process;
- `VectorStoreCollection.get` fed a read-modify-write in semantic memory whose
  correctness depended on reading every earlier write (#1721);
- the stores' tests assumed every write is visible to the next read.

## The contract

`VectorStoreCollection` states:

> An `upsert` or `delete` is durable once it returns. Queries reflect it as
> soon as it returns, or after a delay a store states in its own contract; a
> query within that delay may miss a record upserted, or return a record
> deleted, shortly before it. A write to a record that begins after another
> write to it returned takes effect after that one; writes to one record
> that overlap in time take effect in some order, the same for every query.

"Reflect" is about what a query can return; an approximate search may still
rank a reflected record out of its results. A store with no delay says
nothing; a store with one states it. The contract promises no read-your-writes
beyond that delay, and nothing about a query's view across processes except
through it.

## What each store states

| Store | Delay before queries reflect a write | Overlapping writes to one record |
|---|---|---|
| SQLite (engine-backed) | none | a known bug: the table and the search engine can apply them in different orders (#1468; fixed by #1469 and #1673) |
| sqlite-vec | none | one SQL transaction each |
| Qdrant, one node | none: a write returns once applied | applied in one order |
| Qdrant, replicated | unbounded: a query reads one replica | diverge across replicas under Qdrant's default `weak` ordering (measured; open, below) |
| Milvus | at most `common.gracefulTime` (5 s by default) at Bounded, the default level | one write-ahead log order |

The measurements and mechanisms are in the [Qdrant](qdrant_vector_store.md)
and [Milvus](milvus_vector_store.md) documents.

## Read-your-writes on Qdrant and Milvus

Qdrant and Milvus put the cost of a read reflecting every earlier write on
opposite sides. A Qdrant write returns once applied, so on one node every
later read reflects it without waiting: 4 clients searched 2,307-2,381 times
per second (p99 4.0-4.7 ms) beside 4 clients upserting 5 points at a time
1,000-1,044 times per second, each upsert taking a p50 of 2.2 ms (Qdrant
1.19.1, 4 CPUs / 4 GB, 300 tenants x 1,000 points of 128 dimensions). A Milvus
write returns once logged, and a Strong read waits until the query node has
applied the log up to the read's start: 40-41 searches per second at a p99 of
6.5 s beside writers in another process (Milvus 2.6.24, 4 CPUs / 5 GB, the
same tenants and points).

Qdrant's write ordering is not the counterpart of Milvus's Strong. It is a
write setting: it orders one point's writes the same way on every replica, and
what it costs, measured in the [Qdrant](qdrant_vector_store.md) document, is
write throughput. Milvus orders every write through its log with no setting. A
replicated Qdrant would need a read setting as well for reads to reflect every
earlier write, read consistency `all` with a write consistency factor of 1,
which is unmeasured.

## Decisions

**`get` is gone** (accepted, implemented). It had one production caller,
semantic memory's `update_feature`, which read a feature's stored vector back
to write it again with fresh properties, because `upsert` replaces a whole
record.

- A read that feeds a write turns a stale read into a lasting wrong write.
  Concurrent updates could write an old embedding back over a new one on every
  backend at every consistency level, and a read that missed the record failed
  the update after the row had committed (#1721).
- A `get` safe to write from must reflect every write that returned before it
  began, from any process. Qdrant gives that on one node, replicated Qdrant
  only with read-consistency settings the store does not make, and Milvus only
  at Strong, whose wait under steady writes reached a p99 of 3.0-3.5 s.
- Nothing replaces it. The feature row is the authority for everything but the
  embedding; the vector record carries only the `feature_id` a search hit is
  resolved through, and an update writes the vector store only when given a
  new embedding. #1663 goes further with a `vector_uuid` column.

**Every read of a store runs at the store's one configured level** (accepted,
implemented). On Milvus that is Bounded by default, for searches and for the
purge's listing alike; the purge is correct at it because the retention
exceeds the delay (see [purge](vector_store_purge.md)).

**Tests do not require a store to read at Strong** (accepted, implemented). A
test that checks what the backend holds reads it past the store, at Strong
where the backend has levels (the lifecycle contract's `count_stored` and
`stored_uuids`). A test whose subject is what the store's own read returns
first settles: `settle(collection)` returns once the store's reads reflect
every earlier write, a no-op on Qdrant and one Strong read on Milvus.

**Clients are asynchronous** (accepted, implemented). A library's async client
is used whenever one exists; a synchronous client on worker threads holds a
thread of the process's shared executor for every request's duration, and slow
requests then starve every other call (measured in the
[Milvus](milvus_vector_store.md) document).

## Open: overlapping writes on replicated Qdrant

On a three-node Qdrant 1.19.1 cluster with every shard replicated three times,
two clients upserting different values of one point at once, through different
nodes, left the replicas holding different values for 115 of 400 points under
the default `weak` ordering, and for none under `medium` or `strong`.
Sequential writes, each sent after the previous returned, stayed in order
under all three (0 of 1,200). So a replicated deployment meets the contract's
last clause only with `medium` or `strong` ordering, which the store does not
set. Passing it costs even a single node: single-point upserts ran 3,533-4,425
per second under `medium` and 3,684-4,321 under `strong`, against 5,570-5,727
under `weak` (p50 1.7-1.9 ms against 1.2 ms); 10-point upserts ran the same
under all three. The options:

- pass `medium` on every write, and pay that on every deployment;
- make the ordering a Qdrant setting, `weak` by default, which a replicated
  deployment sets;
- pass `strong`, which costs the same as `medium` but makes writes unavailable
  while a shard's permanent leader is down, where `medium` re-elects a leader
  and may diverge briefly around the change.

## Alternatives considered

- **Leave consistency unstated.** Every caller would have to assume the
  weakest backend, and nothing would tell a store's author what to meet.
- **Promise read-your-writes.** Strong on Milvus stalls every read behind
  every tenant's writes (measured in the [Milvus](milvus_vector_store.md)
  document), and replicated Qdrant would need stronger read consistency on
  every query.
- **Keep `get` with Strong until #1663.** Cheaper to write, but it keeps the
  lost-update race and a contract clause the stores meet only at Strong's
  cost.
- **Carry write timestamps across processes**, in the registry (a SQL write
  per upsert) or as session tokens returned to API clients (an API change),
  for read-your-writes. No caller needs read-your-writes on search.
- **Per-handle guarantee timestamps on Milvus**, so a tenant waits only for
  its own writes: pymilvus does not return a write's timestamp from its client
  API and overwrites a caller's guarantee timestamp unless the request passes
  the undocumented `Customized` level, and the timestamp lives in one process.
