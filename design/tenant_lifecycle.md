# Tenant lifecycle for the MemMachine server

Status: proposal, under review in its own PR, stacked on #1545 so the
segment store contract it refers to is in the tree. Shared reference for
the sessions working on the segment store, the vector store registry
stack, and the server. Where it disagrees with shipped code, the "What
changes" section says so. Companion: `design/segment_store_shared_tables.md`.
Tracking: #1574.

## Problem

A tenant's data lives in several systems: the segment store partition and
the vector store collection under `EventMemory`, the episode store rows,
short-term memory state, optionally semantic memory, and the session row.
Today each system manages its own lifecycle, the session row carries a
two-state status, and a background worker deletes by calling each system
once with no retry (#1575, #1577). Deletion is acknowledged before it runs,
can fail permanently, can leave a tenant half-deleted, and a stray read
recreates a deleted tenant (#1576). Creation is a side effect of the first
request. Nothing reports progress. The segment store's inline purge tried
to promise prompt physical erasure and cannot keep the promise on SQLite.

## Principles

1. Unreachable immediately, reclaimed eventually, provably. A delete flips
   one row; every data path fences on it; reclamation is a durable job
   that retries until every piece reports done; completion is observable.
2. Idempotent steps, atomic only within one backend. A step repeated after
   any partial state completes it. Atomicity is used where a backend gives
   it away (one SQL insert, one SQL transaction) and never simulated across
   backends. No distributed transactions.
3. Atomicity for callers comes from visibility. A tenant is reachable only
   while Active. Half-created and half-deleted tenants are invisible, not
   inconsistent, and the reconciler converges them.
4. Vector operations stay outside SQL transactions. SQL records intent,
   the vector step runs idempotently, SQL records completion. Orphans are
   invisible (keyed by the store's generation) and swept by the store.
5. The resource set is open. A composition declares which resources a
   tenant has; the lifecycle layer knows steps, not resource types.
6. Dependency direction: stores know nothing of tenants. The tenant layer
   is the only component that knows all the stores, through the step
   interface, and it speaks keys only: the one key it minted for the
   tenant, used as the partition key and the collection name in every
   store. Generation identity (incarnations), reclaim queues
   and fences are each store's private concern and never cross the
   boundary. Library use is unchanged.
7. Prompt purge is best-effort and not a contract. The delete path nudges
   the reconciler; it does not drain anything itself.

## Terms

- Tenant: the unit of lifecycle. Application identity (org, project, user,
  session) is data on the tenant record, never encoded into a store key.
- Tenant key: the application identity (org, project, session) a caller
  addresses the tenant by. Lives only on the tenant record.
- Key: a `UUID` the tenant layer mints once per tenant lifetime and
  hands to every store as its partition key or collection name. Stores
  take a `UUID` and nothing else; a backend that wants a string renders
  the 32 hex characters, which every backend's name limit fits. Never
  reused: a recreated tenant gets a new key. Each store privately maps a
  live key to its current generation
  (incarnation) and keeps dead generations invisible and queued for
  reclamation; the tenant layer never sees generation values.
- Resource step: a named set of idempotent operations on a key,
  registered by the composition: `ensure(key)`, `delete(key)` (logical,
  O(1)), `reclaim_some(key)` (bounded), and `reclaim_unknown(live_keys)`
  (the store's own orphan sweep, driven by the reconciler but implemented
  by the store, since only it knows its tables and backend). Examples:
  `segment_partition`, `vector_collection`, `episode_rows`,
  `short_term_state`, `semantic_features`.
- Reconciler: the per-process loop that drives tenants through their
  states by running steps, bounded per pass, with per-tenant backoff.
- Tenant handle: what the application holds. Holds the resource handles,
  each bound to its store's current generation, is opened only on an
  Active tenant, and is invalidated locally on delete.

## Library composition

The tenant layer is a library object, not a server feature, so that going
through it is the easy path for every composition:

- Constructed from the SQL engine the caller already has for the segment
  store, plus the step implementations, which are the store objects the
  caller would otherwise wire by hand.
- Steps are hooks. The server registers segment partition, vector
  collection, episode rows, short-term state and semantic features; a
  memory system that uses only `EventMemory` registers two. The tenant
  layer knows no resource type.
- The reconciler is a call (`reconcile_once`) and optionally a task, as
  the store sweeper is today: the caller drives, nothing schedules
  itself.
- Human naming is the caller's: the tenant record holds whatever
  identity fields the composition declares, and lookup by them is a
  registry query. Stores never see them.
- Bypassing is possible, not forbidden: a caller can mint `uuid4()` and
  use the stores directly. The tenant layer is the safer path, not a
  gate.

## Tenant record

Extends the existing `sessions` row rather than adding a table beside it.

- `key` (primary key, minted per lifetime), `tenant_key` (unique among
  live rows), `status` (`provisioning`, `active`, `deleting`), `config`,
  `created_at`, `updated_at`, plus the application identity fields.
  The key is the cross-store correlation value; a store never sees the
  tenant key.
- Step rows: `(key, step_name)` primary key, `phase` (`ensure` or
  `reclaim`), `status` (`pending`, `done`, `failed`), `attempts`,
  `last_error`, `next_attempt_at`, `updated_at`. One row per declared step
  per phase. This is the job table: progress is the count of done steps,
  a specific tenant's retry is resetting `next_attempt_at`.

All transitions are single SQL transactions on these rows.

## State machine

- Create: insert the tenant row as `provisioning` with its `ensure` step
  rows in one transaction. The unique key is the create arbiter: a
  concurrent create loses on the insert. Run the steps (inline for the
  request, since creation is small and the caller wants the tenant). When
  every `ensure` step is done, flip to `active` in one transaction.
- A tenant that never reaches `active` within a deadline, or whose create
  request fails, is flipped to `deleting`. Partial creation is reclaimed by
  the same path as deletion. There is no "nothing happened" outcome.
- Delete: flip `active` (or `provisioning`) to `deleting` and insert the
  `reclaim` step rows in one transaction. Return. Nudge the local
  reconciler so the common case completes within milliseconds.
- Reclaim: the reconciler claims step rows (`FOR UPDATE SKIP LOCKED` on
  PostgreSQL; a plain read on SQLite, where purgers serialize at the
  write), runs the step, marks it `done` or `failed` with the error and
  a backoff. When every `reclaim` step is done, delete the tenant row and
  its step rows in one transaction.
- Recreate under the same tenant key while `deleting`: allowed. The
  recreate mints a new key, so it touches nothing the old one holds;
  the old row's reclaim steps continue under the old key. This is what
  "tenant key unique among live rows only" means, and it is why the
  minted key, not the tenant key, is the primary key.
- A failed step never terminates the job. It retries with backoff forever
  and is visible as failed with its last error. An operator resolves a
  stuck step by fixing the cause; the reconciler picks it up.

## Handles and fencing

- The tenant handle is opened against an `active` row and holds the
  resource handles, each bound to its store's current generation. On
  delete it is invalidated in-process. Every stale
  condition, from any resource, surfaces through it as one error type,
  mapped at the API boundary (#1571).
- Final-check rule: an operation that spans several statements or several
  resources ends by checking the tenant is still `active`; a read that
  ends dead is rejected, a write that ends dead is reported as failed to
  the caller, and its rows are garbage the reconciler removes.
- Resource-level fences stay where the backend can afford one, on the
  resource's own registry row, which the tenant layer deletes as a step
  strictly after the tenant is `deleting`. The segment store's write pin
  and in-statement read predicate are kept: they cost one statement per
  write and a few microseconds per read, and they order writes against
  deletion, which is what lets its purge retire an entry exactly. No store
  reads the tenant table.
- Resources that cannot fence (every vector backend) rely on records
  keyed by their private generation plus their own sweep. The residual is
  an in-flight write landing in a dead generation, which is wasted, not
  harmful. Lease
  validation against the tenant row would close it (#1564) and is not
  planned unless a wasted write becomes unacceptable.

## Resource contracts

What each resource-level operation promises. These are the contracts the
stores expose regardless of whether a tenant layer sits above them.

### SegmentStore (shipped in #1545; adjustments marked)

- `create_partition(key, config)`: one registry insert; raises if the key
  exists. The store mints the incarnation; nothing outside sees it.
- `open_partition(key)`: read the registry row; the handle is bound to
  that incarnation for its lifetime.
- `delete_partition(key)`: one transaction, O(1) regardless of size: lock
  the registry row (waiting out pinned writers), enqueue the incarnation,
  delete the row. Idempotent. Data rows become unreachable at once.
- `purge_deleted_partitions()`: the sweeper. Reclaims a bounded amount
  across all keys, oldest deletion first at the clock's resolution,
  commits, returns True if more may remain (conservative on the entry
  cap), False if nothing was claimable; entries a concurrent purger holds
  are that purger's. Safe from any process. A deployment must run it.
- `purge_partition(key)`: the reconciler's bounded step for one key.
  Reclaims this key's dead generations, oldest first at the clock's
  resolution. On PostgreSQL never waits on another purger's claim and
  False means exactly "no garbage under this key". On SQLite the DELETE
  waits on the write lock up to the driver's busy timeout and raises past
  it; the caller treats the error as "retry later". Adjustment: no longer
  called from the delete path; the held-entry pause and the slow-purge
  warning go with the inline drain.
- Handle operations: writes pin the registry row for the transaction and
  raise the stale-handle error if it is gone; reads carry the liveness
  predicate in the statement and raise only when a read finds nothing and
  the row is gone; windowed reads end with a registry read. Empty input
  may return without checking.
- Data rows carry no logical key; the registry is the only map from key
  to incarnation. No foreign key from data to registry. Link rows cascade
  from segments; an engine without foreign keys enforced leaves link rows
  until the partition is reclaimed, when the purge removes them with a
  warning. SQLite foreign-key enforcement is required of the engine's
  owner and not verified.

### VectorStore (registry stack, #1524 to #1533, #1537, #1562 to #1565, #1572, #1573)

- `create_collection(namespace, name)`: one registry insert recording the
  store's own generation value and the native container the records live
  in. Native containers are provisioned before serving (#1572), so create
  is a row write and never a dual write. Collection shape is per
  deployment, one schema per container (#1573).
- `open_collection(namespace, name)`: read the registry row; the handle is
  bound to its incarnation.
- `delete_collection(namespace, name)`: logical, O(1): the registry row
  goes and a reclaim entry is recorded. Records under the dead incarnation
  are unreachable at once because every query filters by incarnation.
- Reclaim: a bounded step that deletes records of dead generations
  (#1565), safe to repeat, from any process, plus `reclaim_unknown` for
  records whose collection the tenant layer no longer names. Native
  containers are not dropped per tenant.
- Handle operations: writes and reads carry the generation; there is no
  fence. A write through a stale handle lands under a dead generation.
  The final-check rule at the tenant handle turns that into an error for
  the caller.
- Data-plane contracts: no read-your-writes guarantee; stored properties
  are the collection's own copy with no authority over data governed in
  SQL (#1531).

### Episode store, short-term memory, semantic memory

- Episode rows and short-term state are keyed by the session's string
  identity today; they become steps keyed by the minted key: `delete`
  marks, `reclaim_some` deletes in batches, idempotent.
- Semantic memory is optional and declared only by compositions that use
  it; its steps run only when declared (#1575 is this rule missing).

### Session data manager

- Becomes the tenant record above. Concurrent create and update are
  arbitrated by the database, not by read-then-write (#1543).

## Server API

- `POST /projects`: creates the tenant, runs `ensure` steps inline, returns
  when `active`; 409 if the key is live; 5xx with the tenant left
  `provisioning` (and therefore reclaimable) if a step fails.
- `POST /projects/delete`: flips to `deleting`, nudges, returns at once.
  Idempotent; a second call while `deleting` is a no-op with the same
  status, never a "does not exist".
- `POST /projects/get`: returns status, and while `deleting` the step
  progress (done, pending, failed with last error).
- Reads and writes on a tenant that is not `active`: 404 for absent, 409
  for `provisioning` or `deleting`. No implicit creation on any read or
  write path (#1576).
- Stale-handle errors from in-flight requests that lose a race with a
  delete map to 409 (#1571).

## Reconciler

- One task per process, started with the server. Bounded work per pass
  (per-step budgets are the stores' own bounds). Idle tick tens of
  seconds; a nudge from the delete path wakes it immediately. Racing
  reconcilers across processes share the step table by claiming rows;
  none waits on another.
- Orphan sweeps on a slow schedule: the reconciler hands each store the
  set of live keys and the store reclaims whatever it holds under other
  keys (`reclaim_unknown`). The driver is shared; the work is per store
  by necessity, since only a store knows its tables and its backend's
  listing. This closes the vector side's residual and any straggler
  after a crash.
- Observability: step counts by status, oldest pending step age, per-step
  failure counters, and a log line naming the tenant and the elapsed time
  when a tenant has been `deleting` past a threshold, re-fired as the wait
  doubles.

## What changes relative to shipped and in-flight work

- #1545: kept as is at the store level. The inline drain, the held-entry
  pause and the slow-purge warning in `drop_session_partition` are
  removed in favor of the nudge; `purge_partition` stays as the step.
  Erasure language changes from "normally before returning" to
  "reclaimed by the reconciler, normally within milliseconds".
- Registry stack (#1526 to #1533): the tenant row arbitrates which
  tenants exist; the collection registry remains the store's private map
  from name to generation and native container.
- #1537, #1563: generation-scoped resources are the mechanism this design
  relies on; unchanged.
- #1565, #1572, #1573, #1564: unchanged direction; #1565 becomes a step.
- #1571: the tenant handle is the single place; the mapping above.
- #1575, #1576, #1577: resolved by construction (declared steps, no
  implicit creation, retry with backoff and visible failure).
- #1570: unchanged; the tenant and step tables are provisioned like any
  other schema.
- #1542: the SQLite busy timeout bounds how long a step can block; WAL
  does not change writer-writer contention.
- #1574: gains a "tenant lifecycle" row and this document as the target.

## Open questions

- Keys, to confirm: the text adopts `UUID`-only store keys minted per
  tenant lifetime. The string key contract (charset, 32 bytes, validator,
  `partition_key_for_session` hashing, the vector stores' name
  constraints from #1201) is retired; this is a breaking change to the
  store ABCs and belongs to the tenant-layer series, not #1545. The
  segment store's incarnation is nearly redundant under single-use keys
  and is kept anyway: one column and one mint per create, and it closes
  the one case no queue check can, a UUID reused after its garbage is
  gone, which would otherwise resurrect a stale handle onto the second
  life.
- Synchronous create versus 202 with polling. Inline `ensure` is proposed
  because creation is small; large provisioning (a native container) is
  already moved off this path by #1572.
- Recreate-while-deleting: unique key among live rows only, or rename the
  deleting row. Affects the schema.
- Hierarchical tenancy (org owning projects owning sessions): registry
  data with cascading delete as a step that enqueues children, versus one
  tenant per leaf with the hierarchy held elsewhere.
- Library composition shape: how a library user (for example a memory
  system that uses no semantic memory) declares its steps and gets the
  reconciler without the server.
- Whether the segment store's own purge queue remains as its internal
  reclaim list or is replaced by the step rows entirely.
