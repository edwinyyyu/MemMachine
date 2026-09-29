# Vector store: tombstones and purge

Status: accepted 2026-09-29, in review in #1631; the purge listing's
consistency level is proposed (see
[consistency](vector_store_consistency.md)). Part of [vector store horizontal
scaling](vector_store_horizontal_scaling.md).

## Problem

Deleting a Qdrant or Milvus collection used to be a filter-delete by name,
issued when the collection was deleted. It had three defects once more than
one process serves a backend:

- A write in flight during the deletion, from a handle in another process,
  lands after it and outlives it.
- One delete of a large tenant is one burst of work the backend applies at
  once, stalling other tenants (measured below).
- Nothing records that a deletion is incomplete, so a crash part way through
  leaves points no one will ever reclaim.

## Design

Deletion is logical and immediate; reclamation is physical, deferred, bounded
and retried.

### Tombstones

`unregister` removes the live row and queues the incarnation's *tombstone* in
one transaction (see [collection
registry](vector_store_collection_registry.md)). The collection is unreachable
when it commits. Its points stay in the backend until purge rounds reclaim
them.

A tombstone becomes *due* once `tombstone_retention_seconds` (per store;
86,400, one day, unless configured) has passed since its deletion.

- **Why wait.** A write that passed its liveness check before the deletion
  committed can land after it, and the backend cannot refuse it. Every such
  write has landed once the longest a request can be in flight has passed;
  `request_timeout_seconds` (30 unless configured) bounds that. The retention
  exceeds it by orders of magnitude, so a round that runs after it and finds
  nothing proves the incarnation empty for good.
- **The retention decides nothing about validity.** A stale write is refused
  by the handle's check after it; the retention only has to outlast any write
  in flight.
- **Only the database's clock is used.** The queue stores the deletion's time,
  `enqueued_at`, written by the database's `now()`. The retention is applied
  by the database's own arithmetic when a claim is decided (`now() - interval`
  on PostgreSQL; `datetime('now', '-N seconds')` on SQLite, which yields the
  same text form as `CURRENT_TIMESTAMP` so stamps and cutoff compare in time
  order). A changed retention therefore reaches every tombstone already
  queued, and no client clock enters a decision.
- An incarnation is never re-minted while its tombstone exists, so no new
  collection can adopt, or have reclaimed out from under it, a dead life's
  points.

### The purge round

`VectorStore.purge_deleted_collections()` is part of the contract: one bounded
round per call, on the tombstone that came due first; it returns whether it
reclaimed anything. The stores whose deletion reclaims physically (both SQLite
stores) return `False`.

A round runs inside `claim_purgeable_incarnation()`:

1. **Claim.** One range on the `enqueued_at` index: the oldest due tombstone
   that is neither backing off nor dead-lettered, `LIMIT 1`, under `FOR UPDATE
   SKIP LOCKED`. On PostgreSQL a concurrent purger skips a locked tombstone
   and takes the next, so purgers on every process split a backlog without
   coordinating. SQLite holds no row lock, so two purgers can claim one
   tombstone; that costs a repeated round and nothing more, since by the time
   any round runs no write can land, so a round finding nothing still proves
   the incarnation empty.
2. **Round.** The store looks for points under the incarnation in the native
   collection the tombstone's `namespace` and `config` name, deletes what it
   finds (per backend, below), and reports whether it found any.
3. **Record.** In the claim's transaction: a round that found nothing removes
   the tombstone, which frees the incarnation; a round that found points keeps
   it due and clears its failed rounds.

### Failed rounds: backoff and dead-lettering

A round that raises rolls back, then counts against its tombstone in a
transaction of its own: `failed_rounds + 1`, and `last_failed_at = now()` on
the database clock.

- **Backoff.** After its f-th consecutive failure, a tombstone is claimed
  again only once `min(purge_retry_backoff * 2^(f-1),
  max_purge_retry_backoff)` has passed since `last_failed_at`: 30 s, 60 s, 120
  s and so on, at most 1 h. The tombstones behind it are claimed meanwhile, so
  one failing tombstone does not hold the queue.
- **Dead-lettering.** After 10 consecutive failures, about 3 hours of retries,
  the tombstone is dead-lettered: kept, never re-minted, no longer claimed,
  and reported by an error log naming the incarnation, the last error, and the
  table. Setting its `failed_rounds` back to 0 returns it to the purge.
- The backoff is computed from recorded facts (the count and the time of the
  last failure), not stored as a time to retry at. Both durations are registry
  parameters in seconds with those defaults, so they can become configuration
  without changing the schema.

(Decided: a dead-letter bound rather than retrying forever, because a
tombstone that never purges must be a visible problem, not garbage that is
quietly retried; a counter on the queue row rather than a separate table;
recorded facts rather than a scheduled-time column; seconds, with a base and a
maximum.)

### Per-backend rounds

Chosen by measurement (Qdrant 1.18.3, 1.19.0 and 1.19.1; Milvus 3.0.2; 2.11M
points in this layout, dead incarnations of 10k to 1M among live tenants under
search, upsert and scroll traffic; containers capped at 2 CPUs and 4 GB):

- **Qdrant: one filter-delete of the whole incarnation per round.** The round
  scrolls for one point under the incarnation and, when it finds one, deletes
  by filter with `wait=True`. The delete stalls writes and scrolls on the
  shard, never searches, for its duration: about 1.3 s per 1M points on 1.19.1
  at 3 segments per shard (7.3 s at 101), with no errors. Deleting in batches
  instead collapsed once a vacuum rebuilt the incarnation's segments
  mid-purge: every batched 1M purge (7 of 7 runs, on all three versions)
  slowed to minutes-long deletes and 60-120 s stalls for every tenant.
- **Milvus: bounded batches.** A round lists up to `purge_batch_size` (10,000
  unless configured) of the incarnation's primary keys, by a query on the
  incarnation field, and deletes them by key. Rounds stayed flat at about 100
  ms to the end of a 1M purge, with at most a 0.3 s stall for other tenants.
  One filter-delete of 1M points instead stalled every tenant's reads and
  writes for 2.7-9 s under Session consistency. The batch must stay within the
  server's `quotaAndLimits.limits.maxQueryResultWindow` (16,384 by default),
  so the size is a store parameter, not a constant. Listing by the incarnation
  field took 57-63 ms per round of 10,000 against 70-75 ms by primary-key
  range (1.4M rows, Milvus 2.6.24).

A native collection that no longer exists holds nothing: the round finds
nothing, and the tombstone goes.

### The sweeper

The store never schedules its own purge. The resource manager starts one
sweeper task per vector store the first time it hands the store out, and
`close()` cancels them. A sweeper calls `purge_deleted_collections()` again
after 1 s while rounds reclaim something and after 60 s when one finds nothing
due; a round that raises is logged and retried a tick later. Sweepers on other
processes need no coordination: the claim arbitrates.

### Measured cost of the claim

- Backoff (PostgreSQL 18.6 and SQLite 3.50.4; 1,000,000 tombstones not yet
  due; median of 30 claims): the claim reads each due tombstone that is
  backing off and none that is not yet due. It took 0.3 / 1.3 / 11 ms on
  PostgreSQL and 0.3 / 2.1 / 21 ms on SQLite with 1k / 10k / 100k tombstones
  backing off, and 0.15-0.3 ms with none.
- Interference (PostgreSQL 16, the claim's earlier two-statement form; 20,000
  live collections and 20,000 tombstones): with two sweepers running rounds
  back to back, about 78 per second, beside 16 interactive workers,
  interactive throughput and `is_live` p99 were unchanged within run-to-run
  noise on PostgreSQL. On SQLite, where the sweepers write in the same
  process, throughput dropped 2.5-14% at that rate, as much as with sweepers
  that only commit a one-row write per round, and not measurably at the
  resource manager's pace.

## Alternatives considered

- **Delete immediately, by filter** (the previous design). Rejected for the
  three defects above.
- **Order the claim by failures first**, sinking a failing tombstone behind
  untried ones. Rejected: it retains a failing tombstone's points indefinitely
  while the queue has other work, and hides the failure. The backoff gives the
  tombstones behind it their turn without reordering the queue.
- **A `retry_at` column.** Rejected in favor of computing from recorded facts,
  so a changed backoff reaches tombstones already failing.
- **Count the backoff from when the tombstone came due**, with no new column.
  Measured: a tombstone due for days retries at once, its backoff spent before
  it first fails, so after an outage it fails as fast as before.
- **An index on `(enqueued_at, failed_rounds)`.** It helps only skipping
  dead-lettered tombstones, so the claim keeps `(enqueued_at)`.
- **A separate dead-letter table.** A counter on the queue row does the same
  with no move between tables.
- **Batched deletes on Qdrant; one filter-delete on Milvus; listing Milvus
  keys by primary-key range.** Measured worse, above.

## Consequences

- Deleted records stay in the backend at least the retention: storage is
  reclaimed a day after deletion by default.
- An operator watches for the dead-letter error log. A dead-lettered
  tombstone's points stay until someone resets its `failed_rounds`.
- The purge loads the backend in bounded rounds from every process's sweeper;
  on PostgreSQL the rounds of different tombstones proceed in parallel.
