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

- Horizontal scaling without sharding, at cluster scope: any process on
  any machine serves any tenant, no process owns a tenant, nothing is
  coordinated in process memory. Twenty thousand tenants now; millions
  at the next levels of scale.
- Every component declares its concurrency scope, the widest deployment
  boundary within which concurrent instances may manage the same
  resources (`process`, `machine`, `cluster`). A deployment declares
  the scope it runs at, and startup refuses any component narrower than
  that. The SQLite stores declare `process` or `machine`;
  within their scope they obey every contract, and nothing about scale
  is asked of them.
- DDL that is not tenant-specific runs only in a dedicated setup step
  that serves no requests and cannot race. Tenant-specific DDL is the
  only DDL allowed anywhere else, and it is avoided where it would be
  expensive at cluster scope.
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
  collection. It is the tenant id, in every store. Keys are never
  reused; a store does not detect reuse.
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
  define others (`catch_up`).
- Reconciler: the role that claims and executes jobs. A process runs it
  when configured to; a deployment runs as many as it needs.
- Concurrency scope: `process` < `machine` < `cluster`, the widest
  deployment boundary within which concurrent instances of a component
  may safely manage the same resources. Computed from a component's
  params at construction; a composition's scope is the minimum of its
  parts'. Declared and validated at startup, never enforced at runtime.

## Principles

1. Control plane and data plane are separate objects with separate
   endpoints. The tenant service creates, renames, configures and
   deletes tenants and records jobs, and is the only reader of the
   tenant table. Components serve data operations and own their
   per-tenant state, learning of tenants through their hooks.
2. A data operation names its tenant by id in the request, and every
   store operation names its key. There are no handles at the API and
   none in the stores: nothing is opened or closed per tenant, and a
   process holds no per-tenant state. A store may keep a private
   per-key cache for a backend that needs one (the usearch store's
   loaded index files), bounded by the store and invisible above it.
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
   Tenant-specific DDL is the only DDL a job may run, and components
   declaring `cluster` scope avoid it by holding tenants as rows or
   values.

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
8. The scope check: the minimum over every component's concurrency
   scope must reach `server.concurrency_scope`, or startup fails naming
   the narrowest component.
9. Roles, from `server.roles`: `api` binds the HTTP routers; `reconciler`
   starts the job loop. A single-node deployment runs both in one
   process; a cluster runs many `api` processes and as many `reconciler`
   processes as its job volume needs, one at least.

Who knows what:

- Tenant service: the tenant table, the job table, the registrations.
- Event store: its tables, keyed by tenant id.
- Memory subsystem: its derived stores, its own per-tenant table
  (watermark, applied configuration), the providers it was given, and
  the event store as a reader.
- Ingest service: the event store and the list of subsystems.
- Stores: their backend and their keys.
- Routers: the tenant service, the ingest service, the event store, the
  subsystems.

Dependency direction: the registration interface is defined by the
tenant package; component packages import it and nothing else from the
tenant package. The tenant package imports no component.

Horizontal scaling, at `cluster` scope: all shared state is in the
databases and the vector backend; per-process state is caches any
process can rebuild; concurrent creates are arbitrated by a unique
index, concurrent jobs by row claims, concurrent data operations and
deletes by store fences. A `machine` deployment is several processes on
one host sharing SQLite files; a `process` deployment is one process.
The contracts are the same at every scope; the scope says only which
components may share resources with how many others.

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
- Enqueue: a component inserts its own jobs (`catch_up`) through the
  tenant service's `enqueue(tenant_id, component, action,
  payload)`; the tenant service records them without reading the
  payload.
- Maintenance: each component's `maintain` hook runs on a slow schedule
  (default every 10 min), bounded per call, from reconciler processes
  only, in every one of them without exclusion. A hook claims what it
  works on where duplicated work would cost, and is idempotent
  everywhere, so two processes at once waste at most effort.
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
    of the component's stores and removes its per-tenant row; each call
    reclaims a bounded amount; `DONE` when nothing remains.
  - `maintain` (no tenant): what the component's stores need done on a
    schedule.
  - Any further action the component defines.

A memory subsystem additionally exposes to the ingest service and the
routers: `process(tenant_id, events)`, `forget(tenant_id, event_ids)`,
and its queries.

Data operations:

- Every request reads the component's own per-tenant row (absent: the
  tenant is unknown to this component), resolves the section's provider
  references against the providers the component holds (a lookup in the
  injected set), and calls its stores with the tenant id. One indexed
  read; no per-tenant object survives the request. A configuration
  update therefore takes effect on the next request, on every process,
  with no coordination.
- Every store operation is fenced by its own registry row, and the same
  row read supplies what the operation needs to address the tenant (the
  codec configuration, the container, the collection UUID). A deleted
  key raises one error type.
- On an unknown tenant or a deleted-key error, the router asks the
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
- `create_partition(key)` (strict; see "Create is strict"),
  `delete_partition(key)` (logical, O(1), idempotent),
  `reclaim_partition(key) -> DONE | MORE`, `purge_deleted_partitions()`
  for library users.
- Data operations, each taking the key: `add_events(key, events) ->
  (stored, skipped)`, `get_events(key, ids)`, `list_events(key, filter,
  cursor)`, `read_after(key, position, limit)`, `delete_events(key,
  ids)`.

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
two leaves an event whose derived data is gone; the caller's retry of
the delete finishes it.

## Episodic memory

The one memory subsystem: the current `EventMemory` renamed, processing
events into segments and derivative embeddings and answering searches.

Derived stores:

- Segment store: segments and their derivative links, as shipped in
  #1548.
- Vector store: derivative embeddings, an index.

Both are keyed by the tenant id. There is no rebuild of derived data: a
full reprocessing costs what an ingestion costs, so it is one, into a
new tenant. What the event store gives the subsystem is repair of
partial processing (`catch_up`) and processing of history for a
subsystem enabled on an existing tenant, whose watermark starts at zero.

Per-tenant row (the subsystem's own table): tenant id, watermark (the
last event position processed), applied configuration and its version.

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

Tenant configuration section `episodic_memory`, with mutability:

- `embedder: Ref[Embedder]`: immutable; a different embedder is a new
  tenant and a new ingestion.
- `reranker: Ref[Reranker] | None`: mutable.
- `segmenter`, `deriver`: their options; mutable, applying to events
  processed after the change.
- `search`: default `limit`, `expand_context`, score threshold; mutable.

Episodic memory uses no language model today (both segmenters and both
derivers are deterministic; the embedder is the only model call). The
section gains a `language_model: Ref[LanguageModel]` when a deriver
needs one.

Hooks:

- `ensure`: create the partition and the collection under the tenant
  id; an existing `live` row is this component's own earlier attempt
  and is success, a row in any other state is a reused key and raises
  (see "Create is strict"); insert the per-tenant row if absent, else
  verify immutable options and apply mutable ones.
- `delete`: first call, logically delete the tenant id in both stores
  and remove the row; every call, `reclaim` on each; `DONE` when both
  report done.
- `maintain`: the segment store's `purge_deleted_partitions` as a safety
  net for queue entries no job covers; the vector store's tombstone
  sweep.

## Store contracts

Every store takes a UUID key and nothing else. The string key contract
(charset, 32 bytes, validators, hashing in `partition_key_for_session`)
is retired, and so is the segment store's incarnation: with keys never
reused, the registry row keyed by the caller's UUID is the whole fence.

### The fence every store implements

A store keeps one registry row per key in a SQL database it is given,
and every operation on the key goes through that row. The row also
holds whatever the store needs to address the key on its backend, so the
fence read is the lookup and no handle is opened:

- Write: the write's transaction executes `SELECT ... FROM <registry>
  WHERE key = ? FOR SHARE`. No row: raise the deleted-key error. The
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
  has `machine` scope and is documented as slow, and PostgreSQL is the
  ledger at `cluster` scope. The file lock is what gives a SQLite-backed
  store `machine` scope: it reaches every process on the host and no
  further.

Cost, so a deployment can be sized for it: a write holds one pooled
connection, idle in an open transaction, for the duration of the remote
write, milliseconds on Qdrant and Milvus, so the pool bounds the vector
writes a process can have in flight. `pool_size` is sized for that plus
the process's other SQL work, or the ledger is given a pool of its own.
`idle_in_transaction_session_timeout` on that pool has to exceed the
longest remote write, the opposite of the low value one would otherwise
choose. A read costs one indexed row read after the query.

Nothing above reads a clock. The segment store ships this fence
(`design/segment_store_shared_tables.md`); the event store and the
vector store adopt it.

### Create is strict

`create_partition(key)` and `create_collection(key, container)` insert
the registry row and raise `KeyExistsError` if any row exists under the
key, in any state. They are not idempotent, on purpose: the primary key
is the one place a violated never-reuse invariant can be detected, and
an idempotent create would turn a reused key into a successful create
that inherits whatever records were never reclaimed under it. There is
no open-or-create at the store: a store cannot tell a retry of its
caller's own create from a foreign key, and only the caller can.

Idempotency lives in the component's `ensure`, which knows the key's
provenance: the tenant service inserted the tenant row with this id
before any job ran, so a `live` row under the key can only be this
component's own earlier attempt, and `ensure` treats it as success; a
`creating` row is an interrupted attempt of its own, and `ensure`
resumes it. A row in `dropping` or `dropped` means the key had a
previous life, and `ensure` raises `KeyReusedError`; the job records it
and an operator resolves it, since no retry can. A restored backup of the tenant registry that
is not paired with the stores' state at the same point is the realistic
way to get there, and this is what catches it.

What every store operation does with a key whose row is present but
not `live` (the vector ledger's `creating`, `dropping` and `dropped`; a
SQL store's row while its purge is pending), and with no row at all.
Only `ensure` proceeds on `creating`, as above:

| Operation | present, not live | no row |
| --- | --- | --- |
| create | `KeyExistsError` | creates |
| write | deleted-key error | deleted-key error |
| read | deleted-key error | deleted-key error |
| logical delete | returns; idempotent | returns; idempotent |
| reclaim | proceeds; `dropping` becomes `dropped` once nothing remains, `dropped` is swept once more | `DONE` |

A SQL store's purge is exact and fenced, so after it nothing can remain
under the key and its row can go; a reused key there creates cleanly,
and the vector ledger's tombstone is what still detects the reuse.

Scope declarations, computed from params:

| Component | Scope |
| --- | --- |
| Tenant registry, event store, segment store | `cluster` on PostgreSQL; `machine` on a SQLite file; `process` on in-memory SQLite |
| Vector store | the minimum of its backend's and its ledger's: networked Qdrant, Milvus, Pinecone, S3 Vectors, Weaviate, Chroma are `cluster`; local-mode Qdrant and Milvus are `process`; sqlite-vec is `machine` (its bookkeeping is the ledger, in the same file); the usearch store is `process` (index state in process memory) |
| Reconciler, ingest service, routers | any; they hold no shared state of their own |

### Event store

Above. Its rows are keyed by the tenant id directly, its registry row is
the fence, and its purge queue is keyed by the key.

### Segment store

As shipped in #1548, with these changes:

- Key type `UUID`; the `incarnation` column of every table becomes the
  key, and the purge queue is keyed by the key.
- The partition handle goes: every data operation takes the key, and
  the registry read that fences it returns the codec configuration.
  Codec objects are cached process-wide by configuration, not per key.
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

`vector_collections`: `key UUID PK`, `container`, `address` (what the
backend needs beyond the container, such as Chroma's collection UUID),
`state` (`creating`, `live`, `dropping`, `dropped`), `created_at`,
`updated_at`.

- Native containers are deployment configuration: one per embedder
  provider per vector store, created by the schema command, never by a
  request. A container's dimensions and metric are the embedder's; its
  indexed properties are the store's, one schema for every container
  (#1573, #1572). A container is retired by the schema command when
  the configuration no longer declares its embedder and no ledger row
  references it in any state; `memmachine schema status` shows, per
  container, the ledger rows referencing it by state, which is how an
  operator sees an old embedder's container drain. Until then it stays
  and serves the tenants pinned to it. Inside a container a tenant is a
  value or a native tenant object, per the table below. The two SQLite
  stores keep a
  table (and, for the usearch store, an index file) per collection:
  `sqlite_vec_vector_store.py:4` records that partition keys were
  avoided because a future sqlite-vec ANN index may not support them.
  Within their `process` and `machine` scopes a table per tenant costs
  nothing that matters, and what is asked of them is the contracts.
- `create_collection(key, container)`: strict (see "Create is strict").
  Where the tenant is a value inside the container, one ledger insert
  straight to `live`. Where the tenant is a native object (a Chroma
  collection, a Weaviate tenant, a SQLite table), the row is inserted
  as `creating`, the object is created, and the row is set `live` with
  its address; a crash between the two leaves `creating`, which
  `ensure` resumes by creating the object if absent and setting `live`.
  Telling "already exists" from other failures is per backend; on
  Chroma it is by message, since its duplicate-create error is untyped
  (`InternalError` 500 locally, `ChromaError` 400 over HTTP, never the
  `UniqueConstraintError` the module exports; chromadb 1.5.9).
- Write (`upsert(key, records)`, `delete(key, ids)`): the fence's write
  step, whose row read returns the container (and, per backend, the
  collection UUID or tenant name) the operation addresses; the remote
  write is acknowledged as applied (Qdrant `wait=True`; Milvus with
  strong consistency) before the transaction commits. No row: the
  deleted-key error, and no write creates a collection.
- Read (`query(key, ...)`, `get(key, ids)`): read the row for the
  address, query, then verify the row is still `live`; raise the
  deleted-key error otherwise.
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
  rate no serving path notices. The sweep claims its rows with `FOR
  UPDATE SKIP LOCKED`, so concurrent reconciler processes take different
  rows; correctness never depends on exclusion between them. Where the
  backend can list tenants, `maintain` also compares that list with the
  ledger and reclaims what the ledger does not know; that comparison is
  idempotent and duplicated across processes at worst.
- Tombstones are kept by default: they are what makes a reused key
  detectable and a late write collectable, at one row per dead key.
  `memmachine tenants prune-tombstones` is an operator's trade of that
  for space. It removes only rows whose last sweep after `dropped` found
  nothing, and its help text says what pruning gives up: a key presented
  again after pruning creates, and inherits any record that landed
  after that last sweep. `memmachine schema status` and the prune
  command's dry run report how many tombstones are prunable and how many
  still await a clean sweep, so the trade is made against a number, not
  an age.

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

Backends at the tier that scales to the stated tenant counts. "Addressed
by" is what an operation needs beyond the shared client, all of it held
in the ledger row or equal to the key; no backend requires an object
opened per tenant and kept across operations.

| Backend | Tenant inside the container | Addressed by | Rejects a write to a dead tenant | Lists tenants | Reclaim |
| --- | --- | --- | --- | --- | --- |
| Qdrant | payload value (#1564) | container name; the key as the payload filter | no | no (`facet` is approximate) | filter delete |
| Milvus | partition-key value | container name, loaded once; the key as the partition-key value | no | no | filter delete |
| pgvector | column value | table name; the key as the column value | yes, in-statement (ledger in the same database) | yes | keyed delete |
| Pinecone | namespace, a call parameter; created implicitly on first upsert | index host; the key as `namespace` | no | yes | delete all in the namespace, O(1) |
| S3 Vectors | filterable metadata value (10,000 indexes per bucket, so not one per tenant) | bucket and index names; the key as the metadata filter (`$eq`; filters are evaluated during the search) | no | no | no delete by filter: filtered query (top-K up to 10,000), `DeleteVectors` by key (500 per call), repeat |
| Weaviate | native tenant, one shard each, activity tiers | collection name; the key as the tenant name (the client's `with_tenant` wrapper is built per call, no request) | yes (tenant not found) | yes | remove tenant, O(1) |
| Chroma | collection per tenant (Chroma's own write-up warns that metadata filtering "can become slow" as users and documents grow) | the collection's UUID, recorded in the ledger row at creation; operations go to the HTTP API by that UUID | yes: operations route by the UUID, and a stale one raises `NotFoundError` instead of reaching a replacement (chromadb 1.5.9) | yes (`list_collections`, paged) | `delete_collection`, O(1) |
| SQLite stores | table per collection | table name | yes (dropped table) | yes | drop table |

Pinecone's implicit namespace creation and any backend's inability to
reject are covered the same way: the ledger row exists before the first
record, and reclaim deletes by the tenant's value. Chroma's Python
client only offers a `Collection` object obtained by `get_collection`,
one round trip resolving a name to the collection's UUID, after which
every operation addresses the UUID (`chromadb/api/fastapi.py`). The
store records that UUID in the ledger row at creation and calls the
HTTP API by it, so no operation makes the lookup and no object is
held. Where a client library only offers per-tenant objects, the store
uses the backend's HTTP API directly, as here, or builds the object per
call where that is free, as for Weaviate. Verified against chromadb
1.5.9 (comment on #1579): a stale UUID raises `NotFoundError` rather
than writing into a replacement, so the UUID is the fence on this
backend; eight clients racing `create_collection` on one name produce
one winner every round, the constraint living in the persisted sysdb;
metadata values must be scalars, which the only thing stored on a
collection, its container reference, is. The per-collection cost on a
single Chroma node (an index each) and the collection count Chroma
Cloud supports remain unverified and are checked before an
implementation.
S3 Vectors' limits are from its documentation as of this writing:
filterable metadata is 2 KB per vector, and timestamps must be stored
as numbers to be range-filtered, since comparisons apply to numbers
only.

A read on a deleted key raises on the post-query verification, so no
dead tenant's records reach a caller.

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
  concurrency_scope: cluster
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
accepting, a reconciler finishes its current job, stores and providers
close.

Nothing per tenant is created after startup: a request reads rows and
calls stores with the tenant id, and a process holds no per-tenant
state beyond a store's private cache where a backend needs one.

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
| `GET ...` | watermark and lag behind the event store | 200 |

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
   that records the key. It is avoided where it would be expensive at
   `cluster` scope, so every store declaring `cluster` holds tenants as
   rows or values. The two SQLite vector stores create a table per
   collection, which is fine within `process` and `machine` scope, where
   the file lock serializes it.

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
- #1548: kept; UUID keys replace incarnations, `reclaim_partition` is
  added, the partition handle gives way to key-parameterized
  operations, and `open_or_create_partition` goes.
- #1530: agrees. The store ABCs keep only the strict create; the
  idempotent form is the component's `ensure`, for the reason under
  "Create is strict": only the caller knows why an existing row is
  acceptable.
- #1571: the router's 404/409 mapping under "Component contract"; with
  no handles there is nothing to evict.
- #1575, #1576, #1577: resolved by construction: declared components, no
  implicit creation, durable jobs with retry.
- #1572, #1573, #1564, #1565, #1537, #1563: the vector store contract
  above: containers from configuration, the ledger as fence, tenants as
  values, single-use keys, reclamation plus tombstone sweep. In-flight
  registry PRs are measured against that section.
- #1570: "Schema management".
- #1531: the reference for the concurrency scope idea. The levels,
  the declarations and the deployment-side check above are this
  design's own; they differ from #1531 where the ledger changes what a
  store can promise (Qdrant, Milvus and sqlite-vec widen).
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
