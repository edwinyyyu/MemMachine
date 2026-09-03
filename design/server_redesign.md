# MemMachine server redesign

Status: proposal, under review in its own PR against `speedkick`. It
replaces the tenant lifecycle draft in this file's history: the lifecycle
cannot be fixed inside the current API, configuration and wiring, so the
document covers the server. Companion:
`design/segment_store_shared_tables.md` (the segment store as shipped in
#1548). Tracking: #1574. Line references are to `speedkick` at 7752e4cb,
paths under `packages/server/src/memmachine_server` unless given in full.

## Scope

In: the data model (tenants and events), the HTTP API, configuration,
startup and wiring, tenant lifecycle, the event store, the episodic
memory subsystem with its two derived stores, and schema management
(initial setup and migrations). Breaking changes to the API and to the
store ABCs are accepted.

Out, and not wired into the new server: short-term memory (callers manage
their own near-term context), semantic memory (cost high, benefit
unproven), declarative memory and the graph stores behind it (Neo4j,
NebulaGraph), the retrieval agent. The MCP surface gets one note.

Requirements that shape everything below:

- Horizontal scaling without sharding: any process serves any tenant, no
  process owns a tenant, nothing is coordinated in process memory.
  Twenty thousand tenants now; millions at the next levels of scale.
- DDL that is not tenant-specific runs only in a dedicated setup step
  that serves no requests and cannot race. Tenant-specific DDL is the
  only DDL allowed anywhere else, and it is avoided where it would be
  expensive in a large deployment. The SQLite stores are not for large
  deployments: they are fine as long as they work and obey the
  contracts.
- Garbage that is never collected is unacceptable. Wasted writes are
  acceptable.
- Rejection of operations on a deleted tenant is structural, by database
  locks and rows, never by comparing clocks.
- One client per provider per process, shared by every tenant.
- The tenant layer neither routes data operations nor knows the stores
  or the options a component takes.
- Configuration is derived from the code that consumes it, not written
  beside it.
- No component of the current server is carried over unless named under
  "What is reused".

## What is wrong with the current server

Named so the redesign can be checked against it.

- Identity. The scope of every request is `(org_id, project_id)`, joined
  into `session_key = "org/project"` (`server/api_v2/service.py:38`),
  split back by string (`server/api_v2/router.py:251`), and defaulting
  to `universal/universal` when omitted
  (`packages/common/src/memmachine_common/api/spec.py:58`). "Session"
  names a memory scope, not a conversation; the documentation says
  "project"; store keys are a hash of the string
  (`episodic_memory/long_term_memory/service_locator.py:166`).
- Layers. A request passes `MemMachine` (`main/memmachine.py`, 1726
  lines), `EpisodicMemoryManager`, `EpisodicMemory` and `LongTermMemory`
  (a `match` over two backends) before reaching `EventMemory`, with a
  model translation at each layer: `MemoryMessage` to `EpisodeEntry` to
  `Episode` to `Event`.
- Lifecycle. A search or add on an unknown scope creates it
  (`main/memmachine.py:767`, `:826`; #1576). Deletion returns 204 and an
  in-process queue with no persistence or retry does the work
  (`main/memmachine.py:343`; #1575, #1577).
- Vector stores. Each keeps its collection registry inside its own
  backend (a registry collection on Qdrant and Milvus, a table in the
  store's SQLite file), and a collection handle keeps writing after
  `delete_collection`. Both follow from #1201's model: one process owned
  a collection, so a stale handle could only be that process's own, and
  a store depending on one backend only was worth more than a fence
  nobody needed. Under any-process-serves-any-tenant the handle is
  stale across processes and nothing rejects it (#1537, #1563).
- Configuration. Resources are built lazily behind double-checked locks
  (`common/resource_manager/`), `DatabaseManager` repeats the pattern
  seven times, locator functions assemble params (`*/service_locator.py`),
  `ResourceManagerImpl.build()` is never called by the server, a
  misconfigured provider silently disables a subsystem
  (`common/configuration/__init__.py:384`), every YAML key is lowercased
  (`:586`), an opt-in HTTP API mutates the running process
  (`server/api_v2/config_router.py`), and the configuration models are a
  second description of the components, hand-mapped onto their
  constructors.
- Errors. Status mapping is hand-written per handler, two handlers have
  none, an unknown exception serializes its traceback into the response
  (`server/api_v2/exceptions.py:49`), and an unhandled exception drops the
  connection instead of answering.
- Storage. Eight `create_all` sites in serving processes and one Alembic
  chain (#1570). Every ingested message is stored twice, in the
  `episodestore` table and as segments, linked by an `_episode_uid`
  property with no transaction between the two
  (`episodic_memory/long_term_memory/long_term_memory.py:386`).
  Re-ingesting an id duplicates segment rows.

## Vocabulary

- Tenant: an isolated memory with its own configuration and lifecycle. It
  is not a user, a conversation or an agent run; an application maps
  those onto tenants by naming them. "Tenant" is chosen over "memory",
  "space" and "namespace" because it says isolation and lifecycle and
  nothing about what the application puts inside.
- Tenant name: the application's label. Any string up to 1024 bytes,
  unique among tenants that are not deleting, renamable.
- Tenant id: a UUID minted at creation, permanent, never reused.
- Event: what a caller ingests: an id, a timestamp, a producer, one or
  more content blocks, properties. Events are the caller's data; the
  event store records them and memory subsystems process them.
- Event store: the component that records a tenant's events, in the
  order they were ingested. The system of record.
- Memory subsystem: a component that processes a tenant's events into
  derived data and serves queries over it. This design has one,
  episodic memory (the current `EventMemory`, renamed: "event" is the
  ingestion type, not the memory). Segment and derivative keep their
  meanings from `event_memory/data_types.py`.
- Component with per-tenant resources: the event store and each memory
  subsystem. Both register with the tenant service the same way.
- Key: a UUID a store takes as the identity of a partition or
  collection. The event store's key for a tenant is the tenant id. A
  memory subsystem mints the keys of its derived resources, one per
  generation, and records them in its own per-tenant row. Keys are
  never reused; a store does not detect reuse.
- Provider: a process-wide client declared in configuration and shared
  by every tenant: a database, an embedder, a language model, a
  reranker.
- Template: a named block of per-component options in configuration,
  copied into a tenant's configuration at creation. Nothing is built
  from a template.
- Tenant configuration: the resolved options recorded on the tenant
  row, one section per component, applied to the component by a job.
- Job: a row describing one action for one component on one tenant.
  `ensure` and `delete` are the tenant service's; a component may
  define others (`catch_up`, `rebuild`).
- Reconciler: the role that claims and executes jobs. A process runs it
  when configured to; a deployment runs as many as it needs.
- Instance: a component's per-tenant object inside one process, holding
  that tenant's store handles. A cache entry, never visible to callers.

## Principles

1. Control plane and data plane are separate objects with separate
   endpoints. The tenant service creates, renames, configures and
   deletes tenants and records jobs, and is the only reader of the
   tenant table. Components serve data operations and own their
   per-tenant state, learning of tenants through their hooks.
2. A data operation names its tenant by id in the request. The API has
   no handle. Inside a process, per-tenant instances are bounded cache
   entries that any process can open from the databases.
3. The tenant service knows a component only through its registration: a
   name, a tenant configuration model to validate against, and hooks
   (`ensure`, `delete`, `maintain`, and any job kinds the component
   defines). It never sees a store or an option.
4. Stores take a UUID key and nothing else, and fence on that key.
5. Every store rejects operations on a deleted key by itself and
   structurally: a registry row in SQL keyed by the caller's UUID that
   writes lock in share mode, deletes lock exclusively, and reads verify.
   Nothing is rejected by comparing clocks.
6. Lifecycle steps are idempotent. There is no transaction across
   stores, so a step interrupted midway is completed by repeating it,
   and whatever a failed creation left behind is removed by the same
   path as a deletion.
7. Deletion records jobs and returns. Reclamation is the reconciler's,
   finished when every component reports done, observable throughout.
8. Configuration is the components' own parameter models, rendered as
   one document. Everything process-wide is built from it at startup,
   in dependency order, by constructor injection of resolved objects.
   Nothing is looked up lazily, nothing is mutated at runtime, a bad
   document fails startup.
9. Schema that is not tenant-specific is versioned migrations applied
   by an operator's command that serves nothing and races with nothing;
   serving and reconciler processes verify it and never run it.
   Tenant-specific DDL is the only DDL a job may run, and the backends
   of large deployments avoid it by holding tenants as rows or values.

## Architecture

Every process runs the same binary and builds the same objects from the
configuration document, in this order:

1. The document, parsed into the components' parameter models and
   validated, references included.
2. Providers: one engine per database, one client per embedder, language
   model and reranker.
3. Schema verification: every component's version table is at head, or
   startup fails (see "Schema management").
4. Stores, each constructed with its resolved database engine or client.
5. The event store, then each memory subsystem, each constructed with
   its stores and with the providers the document lists for it. Each
   registers with the tenant service.
6. The ingest service, constructed with the event store and the ordered
   list of memory subsystems the document declares.
7. The tenant service over the tenant and job tables.
8. Roles, from `server.roles`: `api` binds the HTTP routers; `reconciler`
   starts the job loop. A single-node deployment runs both in one
   process; a cluster runs many `api` processes and as many `reconciler`
   processes as its job volume needs, one at least.

Who knows what:

- Tenant service: the tenant table, the job table, the registrations.
- Event store: its tables, keyed by tenant id.
- Memory subsystem: its derived stores, its own per-tenant table (keys of
  the current generation, watermark, applied configuration), the
  providers it was given, its instance cache, and the event store as a
  reader.
- Ingest service: the event store and the list of subsystems.
- Stores: their backend and their keys.
- Routers: the tenant service, the ingest service, the event store, the
  subsystems.

Dependency direction: the registration interface is defined by the
tenant package; component packages import it and nothing else from the
tenant package. The tenant package imports no component.

Horizontal scaling: all shared state is in the databases and the vector
backend; per-process state is caches any process can rebuild; concurrent
creates are arbitrated by a unique index, concurrent jobs by row claims,
concurrent data operations and deletes by store fences.

## Tenant registry

Two tables in the tenant database. Every transition is one transaction
on them. Only the tenant service reads or writes them.

`tenants`:

- `id UUID PK`.
- `name TEXT NULL`, unique index. NULL while deleting: the name is
  released when deletion starts, so a new tenant can take it at once and
  gets a new id.
- `former_name TEXT NULL`: the name at deletion time, for operators.
- `state`: `provisioning`, `active`, `deleting`.
- `configuration JSON`: one object per component name. The record of
  what was requested; each component holds its own applied copy.
- `configuration_version INTEGER`: incremented by every configuration
  update.
- `created_at`, `updated_at`.

`tenant_jobs`:

- `id PK`, `tenant_id`, `component`, `action`, `payload JSON`; unique
  on `(tenant_id, component, action)`.
- `state`: `pending`, `done`.
- `configuration_version`: for `ensure`, the version the job applies.
- `attempts`, `last_error`, `next_run_at`, `lease_until`, `created_at`,
  `updated_at`.

Rename is an update of `name`. No store key contains the name, which is
why names can be arbitrary strings while store keys are the 32 hex
characters every backend accepts.

## Tenant lifecycle

Create, `POST /v1/tenants`:

1. Resolve the configuration: the named template (default `default`),
   overlaid with the request's per-component overrides, each section
   validated by its component's tenant configuration model. An unknown
   component name or an invalid option is 422.
2. Insert the tenant row as `provisioning` and one `ensure` job per
   registered component, in one transaction. A duplicate name fails on
   the unique index: 409, with the existing tenant's id and state in the
   body. With `if_exists: "return"` the existing tenant is returned
   instead, with 200, without comparing configuration.
3. Run this tenant's jobs inline in the request, through the claim path
   the reconciler uses; an `api` process does this whether or not it
   runs the reconciler role. The transaction that marks the last job
   done also sets the row `active`. Return 201 with the tenant.
4. If a job raises, return 503 with the tenant in `provisioning`. A
   reconciler retries the job; the tenant becomes `active` without
   further requests, which `GET` shows.

Delete, `DELETE /v1/tenants/{id}`:

1. One transaction: `state = deleting`, `former_name = name`,
   `name = NULL`, and one `delete` job per registered component. Allowed
   from `provisioning` and `active`; a repeat while `deleting` returns
   the same 202.
2. Return 202 with the tenant in `deleting`. A reconciler in the same
   process is woken; every other reconciler sees the jobs at its next
   poll.
3. A reconciler executes the delete jobs. A component's first delete
   call makes the tenant unreachable in every one of its stores; each
   call reclaims a bounded amount; the job is done when the component
   reports nothing remains.
4. When every delete job is done, one transaction removes the job rows
   and the tenant row. `GET` then returns 404.

Configuration update, `PATCH /v1/tenants/{id}` with `configuration`:
each component validates its section's change against its model, in
which every option is mutable or immutable; an immutable option in the
patch is 422. One transaction writes the document, increments
`configuration_version`, and inserts (or resets) an `ensure` job per
changed component carrying the new version. The job runs inline in the
request as in create; on failure it is left for a reconciler and the
response is 503 with the tenant unchanged from the components' view.

Rename, `PATCH /v1/tenants/{id}` with `name`: one update; 409 on a
duplicate.

States: `provisioning -> active -> deleting -> row removed`, and
`provisioning -> deleting`. There is no failed state. A job that raises
is rescheduled with exponential backoff (1 s doubling to 5 min) and keeps
`attempts` and `last_error` on its row for as long as it takes. An
operator fixes the cause and the next attempt succeeds. A tenant in
`provisioning` or `deleting` past a threshold is logged with its jobs'
last errors, and logged again each time the age doubles.

Reconciler role:

- One loop per process that has the role. It polls every
  `tenant_registry.reconciler.poll_interval` (default 5 s) and when
  woken locally.
- Claim: one `UPDATE ... SET lease_until = now + lease` over the rows
  selected by `state = 'pending' AND next_run_at <= now AND (lease_until
  IS NULL OR lease_until < now) ORDER BY next_run_at LIMIT n FOR UPDATE
  SKIP LOCKED`, returning the claimed rows. On SQLite the same statement
  without `SKIP LOCKED` under `BEGIN IMMEDIATE`; concurrent reconcilers
  serialize there. A crashed process's claim expires with its lease
  (default 60 s) and the job is claimable again. Hooks are idempotent,
  so a job executed twice, or by two processes across a lease expiry, is
  harmless; the lease governs liveness of the loop, never correctness.
- Execute: the component's hook for the job's action, with the payload
  (for `ensure`, the tenant's configuration section at the job's
  version). `DONE` marks the job done; `MORE` reschedules it at
  `now + reclaim_interval` (default 1 s); an exception reschedules with
  backoff. The transaction that marks an `ensure` or `delete` job done
  checks the tenant's remaining jobs of that action and applies the
  state transition if none remain.
- Enqueue: a component inserts its own jobs (`catch_up`, `rebuild`)
  through the tenant service's `enqueue(tenant_id, component, action,
  payload)`; the tenant service records them without reading the
  payload.
- Maintenance: each component's `maintain` hook runs on a slow schedule
  (default every 10 min), bounded per call, from reconciler processes
  only.
- Bounded per pass: at most `n` jobs, each hook bounded by its own
  budget.

## Component contract

A component with per-tenant resources registers with the tenant service
at startup:

- `name`: its section in tenant configuration and, for a memory
  subsystem, its path segment in the API (`episodic_memory`,
  `/episodic-memory`).
- `tenant_configuration`: a Pydantic model for its section, every field
  mutable or immutable, with defaults. Provider references in it are
  typed (`Ref[Embedder]`) and validated against the providers the
  component was constructed with. The tenant service calls
  `validate(section)` and `validate_update(old, new)` and never reads a
  field.
- Hooks, each `(tenant_id, payload) -> DONE | MORE`, idempotent, bounded
  per call, allowed to raise (the reconciler retries):
  - `ensure`: create the component's resources for the tenant if absent,
    verify the immutable options if present, apply the mutable ones,
    record the section in the component's per-tenant row.
  - `delete`: the first call makes the tenant unreachable in every one
    of the component's stores and marks its per-tenant row `deleting`;
    each call reclaims a bounded amount; `DONE` when nothing remains,
    at which point the row is removed. The row outlives the first call
    because it holds the keys reclamation needs.
  - `maintain` (no tenant): what the component's stores need done on a
    schedule.
  - Any further action the component defines.

A memory subsystem additionally exposes to the ingest service and the
routers: `process(tenant_id, events)`, `forget(tenant_id, event_ids)`,
and its queries.

Data operations:

- A component keeps instances in a process-local LRU (size and idle TTL
  configurable, defaults 1000 and 60 s). Opening an instance reads the
  component's own per-tenant row (absent: the tenant is unknown to this
  component), resolves the section's provider references against the
  providers the component holds, and opens store handles on the keys the
  row names. Two indexed reads, and where a backend has a per-tenant
  client object, that object, built from the row and the shared client
  (see "Vector store"). At millions of tenants the LRU bounds what a
  process holds and any process opens any tenant.
- An instance older than the TTL is reopened on its next use. That is
  how a configuration update reaches every process: within one TTL, with
  no coordination.
- After open, store fences reject every operation on a deleted key. A
  stale-handle error from any store evicts the instance and surfaces as
  one error type.
- On an unknown tenant or a stale-handle error, the router asks the
  tenant service for the tenant's state and answers 404 (no row) or 409
  (`provisioning` or `deleting`). No component reads the tenant table.

## Event store

The system of record. Shared tables in a SQL database, keyed by the
tenant id, with the fence under "Store contracts".

- Registry row per key.
- `events`: key, event id, position, timestamp, context, properties,
  blocks (codec-encoded). Unique on `(key, event id)`, which is what
  makes ingest idempotent per event id. `position` is assigned at
  insert from one sequence, so within a key positions are strictly
  increasing in commit order; it is what subsystems process by.
- `create_partition(key)`, `open_partition(key)`, `delete_partition(key)`
  (logical, O(1), idempotent), `reclaim_partition(key) -> DONE | MORE`,
  `purge_deleted_partitions()` for library users.
- Handle: `add_events(events) -> (stored, skipped)`, `get_events(ids)`,
  `list_events(filter, cursor)`, `read_after(position, limit)`,
  `delete_events(ids)`.

Ingest, `POST /v1/tenants/{id}/events`, in the ingest service:

1. `add_events` on the event store; ids already present are skipped and
   reported.
2. For each memory subsystem in the configured order, `process` with the
   stored events. Each subsystem advances its per-tenant watermark to
   the last position it processed.
3. Respond 200 with stored ids, skipped ids, and per-subsystem
   processing status: `done`, or `deferred` if the subsystem raised, in
   which case it has enqueued a `catch_up` job for the tenant and will
   process the events from its watermark.

Delete events, `POST /v1/tenants/{id}/events/delete`: `forget` on each
subsystem, then `delete_events` on the event store. A crash between the
two leaves an event whose derived data is gone; a rebuild recreates it.

## Episodic memory

The one memory subsystem: the current `EventMemory` renamed, processing
events into segments and derivative embeddings and answering searches.

Derived stores:

- Segment store: segments and their derivative links, as shipped in
  #1548. Rebuildable from the event store.
- Vector store: derivative embeddings, an index. Rebuildable from the
  event store.

Per-tenant row (the subsystem's own table): tenant id, current segment
partition key, current vector collection key, watermark (the last event
position processed), applied configuration and its version, state
(`live`, `rebuilding`, `deleting`). The keys are minted by the subsystem
(uuid4), one pair per generation.

Identifiers: segment and derivative ids are minted (uuid4). Idempotency
is per event, not per derived row: `process` for an event first forgets
that event's derived rows, so repeating it after a crash, or from a
`catch_up`, leaves one copy.

Operations, in the order the stores are touched:

- `process`: segment, derive, embed; write segments; upsert derivatives;
  advance the watermark. A crash between stores leaves an event with
  partial derived data behind the watermark; `catch_up` reprocesses it.
- `catch_up` (job): `read_after(watermark)` in batches, `process`,
  until the event store has nothing newer.
- Search: embed the query; vector query (verified against the ledger
  after the query, inside the vector store); segment contexts; on
  request, the full events from the event store.
- `forget`: look up segments and derivatives; delete vector records;
  delete segments.
- `rebuild` (job): mint a new pair of keys, create the partition and the
  collection, process from position zero into them, and when the new
  generation has reached the live watermark, swap the keys in the
  per-tenant row under its row lock (ingests for the tenant wait on
  that lock for the swap), then delete the old keys through the stores'
  ordinary delete and reclaim. Re-embedding after a change of embedder
  is a rebuild.

Tenant configuration section `episodic_memory`, with mutability:

- `embedder: Ref[Embedder]`: immutable through `PATCH`; changing it is a
  `rebuild` with the new embedder in its payload.
- `reranker: Ref[Reranker] | None`: mutable.
- `segmenter`, `deriver`: their options; mutable, applying to events
  processed after the change.
- `search`: default `limit`, `expand_context`, score threshold; mutable.

Episodic memory uses no language model today (both segmenters and both
derivers are deterministic; the embedder is the only model call). The
section gains a `language_model: Ref[LanguageModel]` when a deriver
needs one.

Hooks:

- `ensure`: if the per-tenant row is absent, mint keys, create the
  partition and the collection (exists is success), insert the row;
  otherwise verify immutable options and apply mutable ones.
- `delete`: first call, mark the row `deleting` and logically delete
  the current keys in both stores (and any older generation the row
  still lists); every call, `reclaim` on each; `DONE` when all report
  done, then remove the row.
- `maintain`: the segment store's `purge_deleted_partitions` as a safety
  net for queue entries no job covers; the vector store's tombstone
  sweep.

## Store contracts

Every store takes a UUID key and nothing else. The string key contract
(charset, 32 bytes, validators, hashing in `partition_key_for_session`)
is retired, and so is the segment store's incarnation: with keys never
reused, the registry row keyed by the caller's UUID is the whole fence,
and replacement is the subsystem's, by a new key.

### The fence every store implements

A store keeps one registry row per key in a SQL database it is given,
and every operation on the key goes through that row:

- Write: the write's transaction executes `SELECT 1 FROM <registry>
  WHERE key = ? FOR SHARE`. No row: raise the stale-handle error. The
  shared row lock is held until the transaction ends; the write's own
  statements run inside it (SQL stores), or the remote write runs while
  it is open and is acknowledged as applied before it commits (vector
  stores).
- Delete (logical): one transaction executes `... FOR UPDATE` on the
  row, which waits for every shared lock to be released, then removes
  or marks the row. After it commits no write can take the shared lock,
  so no write under the key can start, and every write that started has
  finished.
- Read: the read carries `EXISTS (SELECT 1 FROM <registry> WHERE key =
  ?)` in its statement where the data is in the same database, and
  verifies the row after the read where it is not. A read that
  completes and then finds the row gone raises. A read is answered only
  if the key was live after the read finished.
- Locks are released by the database when the transaction ends: commit,
  rollback, or the session dying. A crashed process cannot leave a row
  locked. A live session that never ends its transaction can, and two
  settings bound it: on the writer's database session,
  `idle_in_transaction_session_timeout`, set above the longest remote
  write the store makes, since the session is idle-in-transaction
  during it; on the delete, `lock_timeout`, past which the delete
  raises and the reconciler retries.
- SQLite has no row locks. The write transaction opens with `BEGIN
  IMMEDIATE`, which takes the file's write lock, and checks the row
  inside it; the delete does the same and so serializes behind the
  writer, waiting up to `busy_timeout` and raising past it. For a store
  whose data lives outside that SQLite file (a remote vector backend
  with a SQLite ledger) the write lock is held for the duration of the
  remote write, serializing every writer on that file; such a pairing
  is supported and documented as slow, and PostgreSQL is the ledger for
  anything larger.

Nothing above reads a clock. The segment store ships this fence
(`design/segment_store_shared_tables.md`); the event store and the
vector store adopt it.

### Event store

Above. Its rows are keyed by the tenant id directly, its registry row is
the fence, and its purge queue is keyed by the key.

### Segment store

As shipped in #1548, with these changes:

- Key type `UUID`; the `incarnation` column of every table becomes the
  key, and the purge queue is keyed by the key.
- `reclaim_partition(key) -> DONE | MORE`: reclaims this key's dead rows,
  bounded per call; `DONE` when no garbage remains under the key. On
  SQLite the DELETE waits on the write lock up to the driver's busy
  timeout and raises past it; the reconciler retries.
- `purge_deleted_partitions()`: kept for library users; the server runs
  it from `maintain`.

### Vector store

The collection registry leaves the vector backend and becomes a ledger
in a SQL database given to the store. The ledger is the fence, and it is
what makes every record reclaimable on a backend that cannot list or
reject keys.

`vector_collections`: `key UUID PK`, `container`, `state` (`live`,
`dropping`, `dropped`), `created_at`, `updated_at`.

- Native containers are deployment configuration: one per embedder
  provider per vector store, created by the schema command, never by a
  request. A container's dimensions and metric are the embedder's; its
  indexed properties are the store's, one schema for every container
  (#1573, #1572). Inside a container a tenant is a value or a native
  tenant object, per the table below. The two SQLite stores keep a
  table (and, for the usearch store, an index file) per collection:
  `sqlite_vec_vector_store.py:4` records that partition keys were
  avoided because a future sqlite-vec ANN index may not support them.
  Both stores are single-process by contract and not for large
  deployments, so a table per tenant costs nothing that matters there;
  what is asked of them is the contracts, not scale.
- `create_collection(key, container)`: one ledger insert; on the SQLite
  stores, followed by the collection's table, created after the ledger
  row and inside the `ensure` job.
- `open_collection(key)`: read the ledger row; `live` required. A handle
  is obtainable no other way, so no write creates a collection.
- Handle write: the fence's write step, with the remote upsert
  acknowledged as applied (Qdrant `wait=True`; Milvus with strong
  consistency) before the transaction commits.
- Handle read: query, then verify the ledger row is `live`; raise stale
  otherwise.
- `delete_collection(key)`: the fence's delete step, setting `dropping`.
  O(1), idempotent.
- `reclaim_collection(key) -> DONE | MORE`: delete records under the key
  in bounded steps, by the backend's means in the table below; `MORE`
  while records remain; when nothing remains, set `dropped` and return
  `DONE`. The row stays.
- Tombstone sweep, from `maintain`: re-run the reclaim step for
  `dropped` rows at a bounded rate (default 10 keys per call), oldest
  `updated_at` first, and stamp `updated_at`. With a million dead keys
  and one call a minute that is one pass in ten weeks, at a background
  rate no serving path notices. `memmachine tenants prune-tombstones
  --older-than` removes rows an operator no longer wants swept. Where
  the backend can list tenants, `maintain` also compares that list with
  the ledger and reclaims what the ledger does not know.

Why nothing escapes. Every record carries a key that a ledger row
recorded before the record could exist. For a live writer the fence
orders its write before the delete, and after the delete no write can
start. The only write that can land after `dropping` comes from a
writer whose database session ended while its vector request was in
flight, so the lock was released before the backend applied the write.
That write lands under a key whose row is still in the ledger, in
`dropping` or `dropped`, and the reclaim or the tombstone sweep deletes
it. No path compares timestamps, and no key is forgotten while a record
under it could exist.

Backends at the tier that scales to the stated tenant counts. "Per-tenant
object" is what the instance holds beyond the shared client; none
requires a server-side object to stay open across processes.

| Backend | Tenant inside the container | Per-tenant object | Rejects a write to a dead tenant | Lists tenants | Reclaim |
| --- | --- | --- | --- | --- | --- |
| Qdrant | payload value (#1564) | none | no | no (`facet` is approximate) | filter delete |
| Milvus | partition-key value | none; the container is loaded once | no | no | filter delete |
| pgvector | column value | none | yes, in-statement (ledger in the same database) | yes | keyed delete |
| Pinecone | namespace, a call parameter; created implicitly on first upsert | none | no | yes | delete all in the namespace, O(1) |
| S3 Vectors | filterable metadata value (indexes per bucket are capped, so not one per tenant) | none | no | no | no delete by filter: filtered query, delete returned keys, repeat |
| Weaviate | native tenant, one shard each, activity tiers | `with_tenant` wrapper, built client-side | yes (tenant not found) | yes | remove tenant, O(1) |
| Chroma | metadata value (a collection per tenant costs an index each) | none | no | no | delete by `where` |
| SQLite stores | table per collection | table name | yes (dropped table) | yes | drop table |

Pinecone's implicit namespace creation and any backend's inability to
reject are covered the same way: the ledger row exists before the first
record, and reclaim deletes by the tenant's value. A backend whose
per-tenant object must be opened by a server call is admissible: the
instance cache holds it, opening costs one call on a miss, and the LRU
bounds what a process holds. Limits quoted for S3 Vectors are as
documented at its preview and are verified before an implementation.

Reads through a stale handle raise on the post-query verification, so
no dead tenant's records reach a caller.

## Configuration

The configuration is the components' parameter models. Each component
class (a database engine, an embedder, a store, the event store, a
memory subsystem, the tenant service, the server) declares a Pydantic
`Params` model; the document is a tree of those models, one section per
component family, and the loader builds objects by calling each class
with its validated params. There is no second set of configuration
classes, no mapping code from document to constructor, and no key that
is not a field.

- `kind` selects the class within a family (`embedders: {kind: openai}`
  resolves to `OpenAIEmbedder.Params`). Each family has one table from
  kind to class, in code.
- A dependency is a typed reference field, `Ref[Database]`,
  `Ref[Embedder]`, holding the name of an entry in that family's
  section. The loader validates every reference, orders construction by
  them, and passes the constructed object to the constructor. A
  component receives its dependencies, never a catalog to look them up
  in.
- Where a component offers tenants a choice among providers, its params
  list them (`embedders: [openai-large, local-gemma]`, a
  `list[Ref[Embedder]]`); the loader passes exactly those objects, and
  a tenant configuration naming another embedder is rejected at
  creation. The list is a constructor argument holding a closed, typed
  set that the document declares, which is what distinguishes it from a
  locator: nothing is looked up after startup, and nothing outside the
  list is reachable.
- `memmachine config schema` prints the JSON Schema of the whole
  document from the models; `memmachine config example` prints a
  documented example. Documentation is generated, not maintained.
- Keys are case-sensitive. Secrets are written as `${ENV_VAR}` and
  resolved at load. A validation failure, an unknown key, or a reference
  to an undeclared name fails startup; nothing is auto-disabled.
- There is no runtime configuration API. Changing the document is a
  restart, which a scaled deployment does as a rolling replacement.

Example, each key a field of the named class's params:

```yaml
databases:
  main:
    kind: postgres
    url: ${MEMMACHINE_DATABASE_URL}
    pool_size: 20
  vectors:
    kind: qdrant
    url: http://qdrant:6333

embedders:
  openai-large:
    kind: openai
    model: text-embedding-3-large
    dimensions: 1024
    api_key: ${OPENAI_API_KEY}

rerankers:
  bm25:
    kind: bm25

tenant_registry:
  database: main
  reconciler:
    poll_interval: 5s
    lease: 60s

event_store:
  kind: sqlalchemy
  database: main

episodic_memory:
  segment_store:
    kind: sqlalchemy
    database: main
  vector_store:
    kind: qdrant
    client: vectors
    ledger_database: main
    indexed_properties: [producer, kind]
  embedders: [openai-large]
  rerankers: [bm25]

ingest:
  subsystems: [episodic_memory]

tenant_templates:            # data: copied into new tenants, never built
  default:
    episodic_memory:
      embedder: openai-large
      reranker: bm25
      search:
        limit: 10
        expand_context: 4

server:
  bind: 0.0.0.0:8080
  roles: [api, reconciler]
  request_timeout: 60s
```

Tenant templates are validated at startup against each component's
tenant configuration model, including their provider references against
the component's lists, and nothing is built from them. A template edit
changes future tenants only; existing tenants keep their recorded
configuration, which an operator changes with `PATCH`.

Provider names are stable identities. A provider's model or dimensions
are not changed under a name; a new model is a new name. Removing a name
from a component's list fails startup while any tenant of that component
references it, which the component checks from its own per-tenant table
at construction.

## Startup and wiring

`memmachine serve --config PATH` builds the objects in the order under
"Architecture", each by constructor from already-built objects, in the
dependency order the references give. Startup fails on the first
component that cannot be built, naming the component and the cause,
before the socket is bound. Shutdown runs in reverse: routers stop
accepting, a reconciler finishes its current job, components close
their instances, stores and providers close.

Instances and store handles are the only objects created after startup,
on demand, bounded by the LRU.

## Server API

Prefix `/v1`. Tenant ids in paths are UUIDs; names are looked up
explicitly. Bodies are JSON.

Tenants:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST /v1/tenants` | create; body `name`, `template`, `configuration`, `if_exists` | 201; 200 with `if_exists: return`; 409 duplicate; 422; 503 provisioning stalled |
| `GET /v1/tenants?name=` | look up by name | 200; 404 |
| `GET /v1/tenants?prefix=&cursor=` | list, paged | 200 |
| `GET /v1/tenants/{id}` | record, state, jobs with attempts and last error | 200; 404 |
| `PATCH /v1/tenants/{id}` | rename and/or configuration update | 200; 409; 422; 503 |
| `DELETE /v1/tenants/{id}` | start deletion | 202; 404 |

Events, under `/v1/tenants/{id}`:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST .../events` | ingest a batch | 200 with stored ids, skipped ids, per-subsystem status; 404; 409; 422 |
| `GET .../events/{event_id}` | one event | 200; 404 |
| `GET .../events?filter=&cursor=` | list events | 200 |
| `POST .../events/delete` | body `ids` | 200 |

Episodic memory, under `/v1/tenants/{id}/episodic-memory`:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST .../search` | body `query`, `limit`, `filter`, `expand_context`, `include_events` | 200 with scored hits |
| `GET ...` | watermark, lag behind the event store, generation, state | 200 |
| `POST .../rebuild` | body: optional new `embedder` | 202 with the job |

Event body: `id` (optional UUID), `timestamp` (optional; server time if
absent), `producer` (optional string), `blocks` (list of `{type: text,
text}`), `properties` (string keys, scalar values; what `filter` sees).
Search hit: `score`, `segments` (each with `event_id`, `index`,
`timestamp`, `producer`, `text`, `properties`) and, with
`include_events`, the events.

Errors: one handler for the domain error hierarchy maps to a status and
a body `{error: {code, message}}` with a closed set of codes:
`tenant_not_found`, `tenant_not_active`, `tenant_exists`,
`invalid_request`, `provider_unavailable`, `internal`. No traceback
leaves the process. Everything unmapped is 500 `internal` with the
traceback logged. Every request is answered; no path drops the
connection.

MCP: rebuilt over the same component objects, with the tenant id taken
from a header; not designed here.

Clients: the Python and TypeScript clients are generated from the OpenAPI
document, not mirrored by hand.

## Schema management

Two kinds of schema:

1. Component schema: the tenant registry tables, each store's registry
   and data tables, each component's per-tenant table, the vector
   ledger, and the native containers of the vector backends. Static,
   versioned, migrated by an operator's command.
2. Per-tenant resources. Tenant-specific DDL is the only DDL allowed
   outside `memmachine schema upgrade`: a job's `ensure` or `reclaim`
   step may create or drop a tenant's own table, after the registry row
   that records the key. It is avoided where it would be expensive in a
   large deployment, so every store a large deployment uses holds
   tenants as rows or values. The two SQLite vector stores create a
   table per collection, which is fine at their scale, in the one
   process that owns the file.

Component schema:

- Each SQL-backed component owns an Alembic script directory beside its
  code and its own version table (`schema_version_<component>`), so a
  library user composing some components migrates only those.
  Migrations are written from Alembic autogenerate diffs against the
  component's metadata; the metadata is never applied with
  `create_all`.
- Each vector store owns `provision_containers(config)`, which
  idempotently creates the containers its configuration declares.
- `memmachine schema upgrade --config PATH` is the only thing that runs
  component DDL: per configured database, under
  `pg_advisory_xact_lock` on PostgreSQL or `BEGIN IMMEDIATE` on SQLite,
  it upgrades every component assigned to that database to head, then
  provisions containers. Initial setup is an upgrade from an empty
  database; there is no separate path. It runs from a deploy job, an
  init container, or a shell before `serve`.
- `memmachine serve` verifies at startup that every component's version
  table is at the head its code carries and fails otherwise, naming the
  component and both versions. `memmachine schema status --config PATH`
  prints the same comparison.
- Rolling deployments: a migration must keep the previous release's
  code working (expand and contract: add before the code that reads,
  remove after the code that writes is gone), because during a rollout
  processes of both releases run against one schema. A migration that
  cannot is a release note that requires a stop.

## What is reused and what is removed

Reused, with the change named:

- `episodic_memory/event_memory/`: `EventMemory` renamed to episodic
  memory, the segmenters, the derivers, the data types, and the segment
  store (UUID keys in place of incarnations, `reclaim_partition`).
- `common/vector_store/`: the four implementations, with the registry
  replaced by the ledger and its fence, the `config` parameter removed
  from `create_collection`, and containers provisioned by the schema
  command.
- `common/embedder/`, `common/language_model/`, `common/reranker/`,
  each class gaining its `Params` model in place of the separate
  configuration model.
- `common/filter/`, `common/metrics_factory/`, `common/payload_codec/`.
- `enable_sqlite_foreign_keys` and the engine construction in
  `common/resource_manager/database_manager.py`, as the SQLite and
  PostgreSQL database components.

New: the event store, the ingest service, the tenant service and
reconciler, the configuration loader, the schema command, the routers.

Removed: `main/memmachine.py`; `episodic_memory/episodic_memory.py`,
`episodic_memory_manager.py`, `instance_lru_cache.py`,
`long_term_memory/`, `declarative_memory/`, `short_term_memory/`;
`common/session_manager/`, `common/episode_store/`,
`common/vector_graph_store/`, `common/neo4j_utils.py`;
`common/resource_manager/` (the managers, the locator functions, the
`CommonResourceManager` protocol); `common/configuration/`;
`server/api_v2/` including the config router and the traceback-carrying
error model; `semantic_memory/`; `retrieval_agent/`; `installation/`;
`memmachine_common/api/spec.py`.

No data migration from the current server's tables is planned. The
current server keeps running on its branch; moving data is an export
through its API and an ingest through the new one.

## Relation to open issues

- #1574: this document is the target for every row.
- #1548: kept; UUID keys replace incarnations and `reclaim_partition`
  is added.
- #1571: the router's 404/409 mapping and instance eviction under
  "Component contract".
- #1575, #1576, #1577: resolved by construction: declared components, no
  implicit creation, durable jobs with retry.
- #1572, #1573, #1564, #1565, #1537, #1563: the vector store contract
  above: containers from configuration, the ledger as fence, tenants as
  values, single-use keys, reclamation plus tombstone sweep. In-flight
  registry PRs are measured against that section.
- #1570: "Schema management".
- #1542: SQLite pragmas become fields of the SQLite database component's
  params (`busy_timeout`, `journal_mode`); the reconciler's retry covers
  the busy-timeout raise.

## Open questions

- Hierarchy: flat tenants with prefix listing, proposed, or a parent
  column with cascading delete as jobs.
- Whether ingest processes subsystems synchronously in the request
  (proposed, with `catch_up` as the repair) or always through a queue.
- Event size limits, and block types beyond text (the data types admit
  others).
- The library composition surface: the constructors a user calls to get
  the event store, episodic memory and the tenant service without the
  HTTP layer.
- Retention: deleting events by age or by producer, as a job kind.
