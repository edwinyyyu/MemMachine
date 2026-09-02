# Segment store: shared tables with incarnation-scoped tenant keys

Status: accepted 2026-08-31. Supersedes the per-tenant partitioned layout of
the SQLAlchemy segment store on all dialects.

## Problem

The PostgreSQL segment store gave every partition (tenant) its own pair of
child tables under LIST-partitioned parents, created and dropped by
per-tenant DDL. That architecture produced a family of defects, each
diagnosed and measured on the way here:

- Partition deletion with `DROP ... CASCADE` destroyed the store-wide
  parent-level foreign key, silently ending referential integrity for every
  remaining tenant (#1544).
- Any statement carrying the partition key as a bind parameter flipped to a
  cached generic plan after five executions and then locked every child
  partition per execution, exhausting the lock table under load (#1546):
  measured 9 locks per read with a custom plan, 319 with the generic plan at
  316 partitions, HTTP 500 `out of shared memory` under 128-way concurrency.
- PostgreSQL's own foreign-key integrity triggers address the partitioned
  parents internally, so writes and deletes kept a one-shot per-backend lock
  spike proportional to partition count even after client statements were
  fixed (measured 10 -> 166 -> 10 locks across the generic-plan attempt).
- Create/delete DDL against the shared parents deadlocked with writers and
  with each other (measured on upstream and on every intermediate fix;
  PostgreSQL resolves each by aborting a transaction).
- Every fix for the above added machinery: detach-before-drop, a
  store-wide management lock, per-partition ORM entities with a memoized
  cache, statement-cache tuning.

Two scaling requirements then ruled out the whole per-tenant-table family:
tenant creation must stay cheap up to hundreds of thousands to millions of
tenants (any per-tenant table costs ~5 catalog relations and disk files,
which does not survive those counts on PostgreSQL), and individual tenants
range from tens of thousands to tens of millions of rows.

## Design

One physical schema on every dialect: the ORM models are the tables.

- `segment_store_pt` -- the tenant registry. One row per live tenant:
  logical partition key (primary key), an `incarnation` (a random UUID
  generated at creation, unique-constrained), and the payload codec config.
- `segment_store_sg` and `segment_store_dv_ln` -- shared segment and
  derivative-link tables, not partitioned. Rows carry **no logical key at
  all**: they are keyed by the incarnation UUID alone, so a data query
  cannot even be constructed without resolving the registry first --
  referencing the wrong tenant is structurally impossible, not merely
  guarded. A 16-byte UUID (vs a composite key string) also narrows every
  index entry, and random UUIDs are globally unique across nodes without
  coordination, so moving a tenant between databases carries its rows
  verbatim. Within one database, a colliding mint is rejected rather than
  left to probability: a collision with a live incarnation violates the
  registry's unique constraint, and one whose predecessor still has
  garbage awaiting purge is caught by an in-transaction re-check of the
  purge queue at mint time -- race-free because the check runs after the
  registry insert (a concurrent deletion moving the colliding row to the
  queue is already visible once the insert's unique-index wait resolves),
  and no new queue entry for the minted value can appear before commit,
  since the only registry row carrying it is uncommitted. Double
  enqueueing is rejected by the queue's primary key. An incarnation value
  therefore cannot be reused while any trace of it remains; across
  databases, uniqueness rests on random-uuid collision resistance.
  "Incarnation" is the
  established term for exactly this construct (a per-lifetime random
  identifier; cf. Kafka's broker incarnation ID) -- deliberately not
  "generation" or "epoch", which would imply ordered, comparable values.
- `segment_store_gc` -- the purge queue: one row per dead incarnation,
  carrying the logical key purely for forensics.

The link table keeps its foreign key to the segment table (`ON DELETE
CASCADE`), both sides keyed by incarnation. The segment table's foreign key
to the registry is removed: registry rows and data rows are decoupled so
that registry deletion is O(1); orphaned data rows are exactly what the
purge queue tracks.

### Tenant lifecycle

- **Create**: validate the logical key, generate an incarnation, insert the
  registry row and re-check the purge queue in the same transaction -- no
  DDL, no management lock, ~microseconds.
  A duplicate key fails on the primary key (partition-exists error); a
  minted incarnation that collides with one still leaving traces is
  rejected (constraint or queue re-check) and re-minted.
- **Open**: read the registry row; the handle captures the logical key,
  the incarnation, and the codec.
- **Delete**: one transaction -- `SELECT ... FOR UPDATE` on the registry row
  (waits out in-flight writers, who hold `FOR SHARE` pins on it), insert the
  incarnation into the purge queue, delete the registry row. O(1)
  regardless of tenant size. The tenant is immediately unreachable: every
  subsequent operation resolves the registry first.
- **Re-create**: a new registry row with a fresh incarnation. Safe at any
  time -- the old incarnation's rows are invisible to the successor, even
  while the purger is still sweeping them.
- **Purge**: `purge_deleted_partitions() -> bool` on the `SegmentStore`
  ABC. The caller's whole protocol is "call until False"; how much a
  call does is implementation policy, not a call argument -- in this
  store, a single transaction that deletes up to
  `SQLAlchemySegmentStoreParams.purge_max_segments` rows
  (`uuid IN (SELECT ... LIMIT n)` sub-selects, portable across dialects),
  lets the link-table cascade follow (measured faster than deleting the
  link rows manually, at one and at four links per segment), and retires
  queue rows whose incarnations are drained. Link fan-out is a
  property of the ingestion pipeline -- the deriver produces a small,
  design-bounded number of derivatives per segment -- not something the
  store can meaningfully reject after derivation: by then the caller
  cannot act on a dedicated error, and only a deployment's choice of
  segmenter/deriver can actually change the shape. The caller's one
  promise is that a purge call does not noticeably degrade concurrent
  request serving; every bound above exists to keep that promise. Measured, cascade
  deletion saturates around 3M link rows/s as density grows (380k
  segments/s at one link per segment, 46k at 64), so even
  heavily-linked partitions purge in sub-second bounded calls.
  Integrity-escaped link rows (the retire-path guard) draw
  count-for-count on the same per-call budget: measured, a link row
  deletes about 3x cheaper than a segment row (1.0 vs 3.3 us/row,
  batched deletes on the benchmark box), so one budget calibrated on
  segment rows upper-bounds the call without assuming any ratio -- a
  separate link limit could only be safe by guessing that ratio. The
  measured direction is the precondition, not just support: the shared
  budget stays conservative only while a link row deletes cheaper than
  a segment row, so widening the link table or adding indexes to it
  revisits this choice.
  Entries are claimed oldest-first (FIFO by
  enqueue time, stamped by the database clock so every server's entries
  order on one clock, at its resolution) and carry their own per-call bound
  (`SQLAlchemySegmentStoreParams.purge_max_partitions`), since their
  cost is round trips rather than row deletions: empty partitions are
  cheap to create and delete, so a large backlog of empty entries is
  easy to accumulate and must not turn one bounded call into an
  unbounded transaction. A deployment sets engine-appropriate transaction sizing once at
  construction -- different limits may suit PostgreSQL, SQLite, or other
  dialects -- and callers never see sizing at all (the server currently
  constructs its stores with the defaults; wiring the bounds into
  server configuration is future work); True means another
  call may reclaim more, so draining a backlog is the caller's loop and
  each call's committed progress survives interruption. Queue
  entries are claimed one at a time with `FOR UPDATE SKIP LOCKED` as the
  call processes them, so a bounded call neither materializes nor locks
  the rest of the backlog, and concurrent purgers -- including from other
  processes -- share the queue instead of contending: only the claiming
  call touches a dead incarnation's rows (writers cannot; the fence pins
  live incarnations only), making reclamation deadlock-free by
  construction rather than by lock ordering. On SQLite, which drops
  locking clauses and defers BEGIN to the first DML, the claim is a plain
  read and purgers serialize on the database write lock at the DELETE
  instead: two purgers may claim the same entry, and the second deletes
  whatever rows the first left (none, once the first retired the entry)
  and re-retires it -- duplicated round trips, never duplicated or missed
  reclamation, since an entry is retired only by a purger whose own
  DELETEs (segments, then any integrity-escaped link rows -- reclaimed
  in batches under the same budget and logged as a warning) found
  fewer rows than its remaining budget. The
  store never schedules purging itself: when and how often is the
  caller's policy, and implementations whose deletes reclaim physically
  implement it as a no-op returning False. In the server, the resource
  manager is that scheduler: a background task per store loops bounded
  calls a short pause apart while the store reports more work -- the
  pause yields the database to request serving -- and sleeps a full
  tick when it does not (failures log and retry a tick later;
  cancelled on close),
  alongside LongTermMemory's inline drain on session drop -- so a
  backlog left by an interrupted drain is reclaimed promptly rather
  than waiting for the next deletion, and racing purgers need no
  coordination because claiming already handles them. Deletion latency contracts are
  "unreachable immediately, physically erased asynchronously". Measured
  deletion throughput ~147k rows/s on the benchmark box, so a
  ten-million-row tenant erases in about a minute of background work.

### Fencing (resolves #1549)

Every write pins the registry row with
`SELECT ... WHERE incarnation = :incarnation FOR SHARE` -- the incarnation
alone resolves the row, exactly like the data queries -- and raises a
stale-handle error when no row matches. Reads conjoin the same predicate
to their data statements as an `EXISTS`, so one statement (one snapshot)
both checks liveness and reads -- a stale handle reads nothing, at no
extra round trip -- and issue the check as its own statement only when a
read returns no rows, to tell an empty partition from a stale handle.
On SQLite, whose driver defers BEGIN until the first
write, a SELECT-only fence would run outside the write transaction; the
proper primitive, BEGIN IMMEDIATE, requires taking over the engine's
transaction management in SQLAlchemy (isolation_level=None plus a
begin-event hook), which the store cannot do to a caller-owned, possibly
shared engine -- so the write fence is a self-checking registry-row
UPDATE instead: it acquires the same write lock scoped to the transaction,
and its match count is the staleness check (deletion opens its
transaction the same way, so racing deletions serialize). A handle held
across delete, or across
delete and re-create, fails loudly instead of operating on the successor
tenant -- on every dialect, including SQLite, which previously had no
mechanism for this at all.

### Locking model

Row locks only. Writers hold `FOR SHARE` on their registry row for the
duration of a write transaction; deletion takes `FOR UPDATE` on the same
row; the purger claims queue rows with `FOR UPDATE SKIP LOCKED`, so it
never waits; the mint's queue re-check takes `FOR SHARE` on a queue row
only in the collision case. There is no table-level management lock, no
DDL, and no lock upgrade anywhere in the store, which removes the
measured deadlock classes at the root: writers and tenant lifecycle
operations share no lockable object except the one registry row of the
same tenant, where blocking is the intended semantics, and reclamation
touches only rows no other operation can reach -- claimed queue entries
and dead incarnations' data.

## Alternatives considered

Measured 2026-08-31 (raw asyncpg, 40 tenants x 2000 row-pairs; scaled runs
with a 500k-pair tenant among 50 small tenants; pgvector:pg16, 2 CPUs):

| layout | ingest | seed read | tenant create | tenant delete | catalog cost |
|---|---|---|---|---|---|
| `PARTITION OF` children | 8.7-9.2k pairs/s | 0.43-0.46 ms | ~12 ms DDL | O(1) + detach | ~5 relations/tenant |
| standalone child tables | 8.0-8.4k pairs/s | 0.38-0.41 ms | ~6 ms DDL | O(1) | ~5 relations/tenant |
| shared tables | 15k pairs/s | 0.21 ms | 0.006 ms | O(1) logical | none |

At scale (1M pairs, big-tenant reads): shared 0.41 ms seed / 0.23 ms window
vs per-tenant 0.46 / 0.36 -- the serving path is point lookups and short
index ranges, which do not suffer from tenants interleaving in a shared
heap. Batched deletion measured faster on the shared layout (147k vs 93k
rows/s). Per-tenant tables win only small-tenant removal latency (9 ms drop
vs 58 ms row delete), which does not justify their catalog cost at the
required tenant counts; both per-tenant options also retain the RI-trigger
lock fan-out (parent-level FK) or require dialect-conditional constraints
on the shared models to avoid doubled foreign keys.

The industry norm for this shape (pool model) matches: O(1) logical
deletion via a registry plus asynchronous batched reclamation; O(1)
physical drops exist only in per-tenant-resource (silo) architectures,
which the tenant-count requirement excludes.

## Consequences

- Removed: partitioned parents, per-tenant DDL, detach machinery, the
  store-wide management lock, per-partition ORM entities and their cache,
  the compiled-statement-cache tuning. The generic-plan behavior is benign
  on unpartitioned tables (a plan references one table), so no plan-mode
  tuning is needed.
- One architecture on every dialect; the remaining dialect splits are the
  LATERAL-vs-loop read strategy, the PostgreSQL-only ordered row locks in
  `delete_segments` (SQLAlchemy drops locking clauses on SQLite), and
  SQLite's foreign-key pragma. The pragma is per-connection state that
  the engine's owner registers at engine creation; the store requires it
  and does not verify it, matching the reference practice for SQLite
  foreign keys (SQLAlchemy's dialect docs, Django's and Rails'
  adapters). A listener added later by the store would miss connections
  already in the pool. An engine without the pragma leaves link rows
  that outlive their segments until the partition is dropped, when the
  purge reclaims them with a warning.
- Replication and sharding: four ordinary tables; logical replication
  covers new tenants automatically (they are rows, not relations); moving a
  tenant between nodes is an indexed row copy plus a registry insert whose
  fresh incarnation fences all stale handles.
- Tenant deletion is O(rows) physically, in the background; deployments
  that require synchronous physical erasure must run the purge inline,
  accepting that an entry a concurrent purger has already claimed
  completes in that purger's bounded call moments after the drain
  returns.
- No migration from the partitioned layout is provided (the event backend
  is opt-in and pre-GA); existing databases recreate their schema.
- Datetime convention: filter nodes normalize to UTC-aware instants at
  construction (`FilterExpr`'s contract), for every consumer -- the Neo4j
  compiler's `datetime.timestamp()` therefore sees instants (ISO-string
  filter values are parsed and normalized there under the same rule),
  where a naive value used to be read in the server's local zone; that backend
  now agrees with the naive-means-UTC rule the rest of the codebase
  applies, a deliberate alignment; the in-memory short-term evaluator
  tags a naive stored metadata datetime under the same rule at
  comparison time, since stored user data is not rewritten; storage
  write paths spell
  `ensure_tz_aware(...).astimezone(UTC)` as two explicit steps at each
  site. A composed `to_utc` helper was rejected on two grounds: the
  name pins only the conversion, not the naive-means-UTC tagging
  decision; and it could not structurally prevent half-applied
  normalization anyway, because read paths legitimately need the
  tagging step alone (segment reads reapply the stored original
  offset; cluster and episode reads only tag naive database values),
  so the first step stays independently available no matter what. The
  per-site repetition is deliberate, not debt to consolidate.
