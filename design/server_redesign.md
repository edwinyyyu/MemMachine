# MemMachine server redesign

Status: proposal, under review in its own PR against `speedkick`. It
replaces the tenant lifecycle draft in this file's history: the lifecycle
cannot be fixed inside the current API, configuration and wiring, so the
document covers the server. Companion:
`design/segment_store_shared_tables.md` (the segment store as shipped in
#1548). Tracking: #1574. Line references are to `speedkick` at 7752e4cb,
paths under `packages/server/src/memmachine_server` unless given in full.
Settings are named, never given numeric defaults: none has been measured.

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
  any host serves any tenant, no process owns a tenant, nothing is
  coordinated in process memory. Twenty thousand tenants now; millions
  at the next levels of scale.
- Every component declares its concurrency scope, the widest deployment
  boundary within which concurrent instances may manage the same
  resources (`process`, `host`, `cluster`). A deployment declares the
  scope it runs at, and startup refuses any component narrower than
  that. The SQLite stores declare `process` or `host`; within their
  scope they obey every contract, and nothing about scale is asked of
  them.
- DDL that is not tenant-specific runs only in a dedicated setup step
  that serves no requests and cannot race. Tenant-specific DDL is the
  only DDL allowed anywhere else, and it is avoided where it would be
  expensive at cluster scope. No store in this design creates a table
  per tenant.
- Garbage that is never collected is unacceptable. Wasted writes are
  acceptable.
- Rejection of operations on a deleted tenant is structural, by database
  locks and rows, never by comparing clocks. Time is used for
  scheduling, never for deciding whether an effect is valid.
- One client per provider per process, shared by every tenant.
- The tenant layer neither routes data operations nor knows the stores
  or the options a component takes.
- Resources are plain classes, wireable by hand, unaware of any
  configuration or injection system. Configuration is derived from
  their constructors, not written beside them.
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
  unique among tenants that are not deleting or deleted, renamable.
- Tenant id: a UUID minted at creation, permanent, never reused. The
  tenant row outlives the tenant as a tombstone, which is what makes
  "never reused" enforced rather than hoped.
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
  collection. It is the tenant id, in every store.
- Resource: any object the server builds from configuration: a database
  engine, a provider client, a store, a component, the tenant service.
  A plain class with a typed constructor.
- Provider: a resource that is a process-wide model or backend client
  shared by every tenant: an embedder, a language model, a reranker, a
  database engine, a vector backend client.
- Template: a named block of per-component tenant options in
  configuration, copied into a tenant's configuration at creation.
  Nothing is built from a template.
- Tenant configuration: the resolved options recorded on the tenant
  row, one section per component, applied to the component by a job.
- Job: a row describing one action for one component on one tenant.
  `ensure` and `delete` are the tenant service's; a component may
  define others (`catch_up`).
- Reconciler: the role that claims and executes jobs. A process runs it
  when configured to; a deployment runs as many as it needs.
- Concurrency scope: `process` < `host` < `cluster`, the widest
  deployment boundary within which concurrent instances of a component
  may safely manage the same resources. Computed from a resource's
  constructor arguments at construction; a composition's scope is the
  minimum of its parts'. Declared and validated at startup, never
  enforced at runtime. "Host" is one operating system instance, which
  is what a file lock reaches; "node" is avoided as overloaded.

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
   per-key cache for a backend that needs one, bounded by the store and
   invisible above it.
3. The tenant service knows a component only through its registration: a
   name, a tenant configuration model to validate against, and hooks
   (`ensure`, `delete`, `reclaim`, `maintain`, and any job kinds the
   component defines). It never sees a store or an option.
4. Stores take a UUID key and nothing else, and fence on that key.
5. Every store rejects operations on a deleted key by itself and
   structurally: a registry row in SQL keyed by the caller's UUID that
   writes lock in share mode, deletes lock exclusively, and reads verify.
   Locks, not leases: a lock is released by the database when its
   holder's session ends, and nothing compares clocks.
6. Lifecycle steps are idempotent. There is no transaction across
   stores, so a step interrupted midway is completed by repeating it,
   and whatever a failed creation left behind is removed by the same
   path as a deletion.
7. Every lifecycle change is accepted, recorded as jobs, and reported
   on; none is refused for a failure the reconciler will retry. A caller
   polls or waits; it is never told "try again later".
8. Resources are plain classes with typed constructors. Configuration
   is the set of constructor arguments, rendered as one document; the
   loader reflects on constructors and builds the object graph in
   dependency order. Nothing is looked up lazily, nothing is mutated at
   runtime, a bad document fails startup, and a resource can be wired
   by hand without the loader.
9. Schema that is not tenant-specific is versioned migrations applied
   by an operator's command that serves nothing and races with nothing;
   serving and reconciler processes verify it and never run it.
   Tenant-specific DDL is the only DDL a job may run, and no store in
   this design runs any.

## Architecture

Every process runs the same binary and builds the same objects from the
configuration document, in this order:

1. The document, parsed into constructor arguments and validated,
   references included.
2. Resources, in the dependency order the references give: database
   engines and backend clients, then embedders, language models and
   rerankers, then stores, then the event store and each memory
   subsystem (each registering with the tenant service), then the
   ingest service, then the tenant service.
3. Schema verification: every component's version table is at head, or
   startup fails (see "Schema management").
4. The scope check: the minimum over every resource's concurrency scope
   must reach `server.concurrency_scope`, or startup fails naming the
   narrowest resource.
5. Roles, from `server.roles`: `api` binds the HTTP routers; `reconciler`
   starts the job loop. A single-host deployment runs both in one
   process; a cluster runs many `api` processes and as many `reconciler`
   processes as its job volume needs, one at least.

Who knows what:

- Tenant service: the tenant table, the job table, the registrations.
- Event store: its tables, keyed by tenant id.
- Memory subsystem: its derived stores, its own per-tenant table
  (watermark, applied configuration and its version), the providers it
  was given, and the event store as a reader.
- Ingest service: the event store and the list of subsystems.
- Stores: their backend and their keys.
- Routers: the tenant service, the ingest service, the event store, the
  subsystems.

Dependency direction: the registration interface is defined by the
tenant package; component packages import it and nothing else from the
tenant package. The tenant package imports no component. No resource
imports the loader.

Horizontal scaling, at `cluster` scope: all shared state is in the
databases and the vector backend; per-process state is caches any
process can rebuild; concurrent creates are arbitrated by a unique
index, concurrent jobs by row locks, concurrent data operations and
deletes by store fences. A `host` deployment is several processes on
one host sharing SQLite files; a `process` deployment is one process.
The contracts are the same at every scope; the scope says only which
resources may share state with how many others.

## Tenant registry

Two tables in the tenant database. Every transition is one transaction
on them. Only the tenant service reads or writes them.

`tenants`:

- `id UUID PK`.
- `name TEXT NULL`, unique index. NULL from the moment deletion starts,
  so a new tenant can take the name at once and gets a new id.
- `former_name TEXT NULL`: the name at deletion time, for operators.
- `state`: `provisioning`, `active`, `deleting`, `deleted`.
- `configuration JSON`: one object per component name. The record of
  what was requested; each component holds its own applied copy.
- `configuration_version INTEGER`: incremented by every configuration
  update.
- `created_at`, `updated_at`, `deleted_at`, `swept_at`.

`tenant_jobs`:

- `id PK`, `tenant_id`, `component`, `action`, `payload JSON`; unique
  on `(tenant_id, component, action)`.
- `state`: `pending`, `done`.
- `configuration_version`: for `ensure`, the version the job applies.
- `attempts`, `last_error`, `next_run_at`, `created_at`, `updated_at`.

A `deleted` row is a tombstone: id, former name, `deleted_at`,
`swept_at`, nothing else. It is what enforces "never reused": minting
is an insert on the primary key, so a duplicate id, whether from a
collision, a replayed id, or a registry restored from a backup the
stores were not restored to, fails at the tenant service before any
store is touched. It is also what drives the periodic re-sweep under
"Tenant lifecycle". Tombstones are kept; `memmachine tenants
prune-tombstones` is an operator's trade of the reuse detector and the
re-sweep for space, gated on every component having reported `DONE` for
the tenant and a later re-sweep having found nothing, and it reports how
many rows are prunable and how many are not, so the trade is made
against a number.

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
   the unique index: 409 `tenant_exists`, and nothing else. A caller
   that wants the existing tenant looks it up by name; an error response
   does not carry another tenant's record, and there is no get-or-create
   flag.
3. Respond 202 with the tenant in `provisioning`. With `?wait=` the
   request instead blocks until the tenant is `active` or the wait
   elapses, and responds 201 or 202 accordingly. While waiting, the
   `api` process executes this tenant's jobs itself, through the same
   claim as a reconciler, so a single-process deployment completes the
   create inside the request; a failed step is left for a reconciler,
   never surfaced as an error to the creator.
4. `GET /v1/tenants/{id}` shows the state and each job's attempts and
   last error. The tenant becomes `active` when the last `ensure` job's
   transaction marks it done.

Delete, `DELETE /v1/tenants/{id}`:

1. One transaction: `state = deleting`, `former_name = name`,
   `name = NULL`, `deleted_at = now()`, and one `delete` job per
   registered component. Allowed from `provisioning` and `active`; a
   repeat while `deleting` responds the same 202.
2. Respond 202 with the tenant in `deleting`; `?wait=` as for create. A
   reconciler in the same process is woken; every other reconciler sees
   the jobs at its next poll.
3. A reconciler executes the delete jobs. A component's first `delete`
   call makes the tenant unreachable in every one of its stores; each
   call reclaims a bounded amount; the job is done when the component
   reports nothing remains.
4. When every delete job is done, one transaction removes the job rows
   and sets the tenant row `deleted`. `GET` then returns 404, and the
   row stays as the tombstone.

Re-sweep: a `maintain` duty of the tenant service, run by reconcilers,
takes `deleted` rows oldest `swept_at` first, bounded per call, and
calls every registered component's `reclaim(tenant_id)`, then stamps
`swept_at`. This is what collects the one write that can land after
reclamation (see "Vector store"); components whose stores cannot hold
such a write return `DONE` at once.

Configuration update, `PATCH /v1/tenants/{id}` with `configuration`:

- The tenant stays `active` throughout. Each component validates its
  section's change against its model, in which every option is mutable
  or immutable. An option is immutable exactly when changing it would
  require touching existing data (the embedder; anything that reshapes
  stored rows); mutable options apply to events processed after the
  change or to reads (a reranker, search defaults, segmenter options).
  An immutable option in the patch is 422, and there is no "expensive
  but allowed" class.
- One transaction writes the document, increments
  `configuration_version`, and inserts (or resets) an `ensure` job per
  changed component carrying the new version. Respond 202; `?wait=`
  blocks until every such job is done.
- How it reaches the server processes: the `ensure` job calls the
  component's hook with the new section, and the hook writes it, with
  its version, into the component's own per-tenant row. Every request
  reads that row, so the next request on any process uses the new
  options. No process is notified; the row is the channel. `GET` shows
  the requested version and, per component, the applied version.

Rename, `PATCH /v1/tenants/{id}` with `name`: one update; 409 on a
duplicate.

States: `provisioning -> active -> deleting -> deleted`, and
`provisioning -> deleting`. There is no failed state. A job that raises
is rescheduled with exponential backoff (`reconciler.backoff`) and keeps
`attempts` and `last_error` on its row for as long as it takes. An
operator fixes the cause and the next attempt succeeds. A tenant in
`provisioning` or `deleting` past `reconciler.stuck_after` is logged
with its jobs' last errors, and logged again each time the age doubles.

Reconciler role:

- One loop per process that has the role. It polls every
  `reconciler.poll_interval` and when woken locally.
- Claim: `SELECT ... FROM tenant_jobs WHERE state = 'pending' AND
  next_run_at <= now() ORDER BY next_run_at LIMIT n FOR UPDATE SKIP
  LOCKED`, and the row lock is held for the duration of the step: the
  hook runs, and the same transaction marks the outcome and commits. A
  crashed process's lock is released by the database and the job is
  claimable at once. There is no lease and no clock comparison; the one
  `now()` is the database's and decides only when a backed-off job is
  due, never whether an effect is valid. On SQLite the same statement
  without `SKIP LOCKED` under `BEGIN IMMEDIATE`; concurrent reconcilers
  serialize there.
- Execute: the component's hook for the job's action, with the payload
  (for `ensure`, the tenant's configuration section at the job's
  version). `DONE` marks the job done; `MORE` reschedules it at
  `now() + reconciler.reclaim_interval`; an exception reschedules with
  backoff. The transaction that marks an `ensure` or `delete` job done
  checks the tenant's remaining jobs of that action and applies the
  state transition if none remain. Hooks are idempotent, so a step
  repeated after a crash is harmless.
- Enqueue: a component inserts its own jobs (`catch_up`) through the
  tenant service's `enqueue(tenant_id, component, action, payload)`;
  the tenant service records them without reading the payload.
- Maintenance: each component's `maintain` hook and the tenant service's
  re-sweep run on `reconciler.maintenance_interval`, bounded per call,
  from reconciler processes only, in every one of them without
  exclusion. A duty claims what it works on with `FOR UPDATE SKIP
  LOCKED` where duplicated work would cost, and is idempotent
  everywhere, so two processes at once waste at most effort.
- Cost: a reconciler holds one database connection per job it is
  executing, for the step's duration; steps are bounded per call by
  their hooks, and `reconciler.jobs_per_pass` bounds the connections.

## Component contract

A component with per-tenant resources registers with the tenant service
at startup:

- `name`: its section in tenant configuration and, for a memory
  subsystem, its path segment in the API (`episodic_memory`,
  `/episodic-memory`).
- `tenant_configuration`: a Pydantic model for its section, every field
  mutable or immutable, with defaults. Provider references in it are
  ids that the component validates against the providers it was
  constructed with. The tenant service calls `validate(section)` and
  `validate_update(old, new)` and never reads a field.
- Hooks, each `(tenant_id, payload) -> DONE | MORE`, idempotent, bounded
  per call, allowed to raise (the reconciler retries):
  - `ensure`: create the component's resources for the tenant if absent,
    verify the immutable options if present, apply the mutable ones,
    record the section and its version in the component's per-tenant
    row.
  - `delete`: the first call makes the tenant unreachable in every one
    of the component's stores and removes its per-tenant row; each call
    reclaims a bounded amount; `DONE` when nothing remains.
  - `reclaim`: reclaim anything under the tenant id in every one of the
    component's stores, bounded; `DONE` when nothing is found. Called by
    the re-sweep after `delete` has reported `DONE`.
  - `maintain` (no tenant): what the component's stores need done on a
    schedule.
  - Any further action the component defines.

A memory subsystem additionally exposes to the ingest service and the
routers: `process(tenant_id, events)`, `forget(tenant_id, event_ids)`,
and its queries.

Data operations:

- Every request reads the component's own per-tenant row (absent: the
  tenant is unknown to this component), resolves the section's provider
  ids against the providers the component holds (a lookup in the
  mapping it was constructed with), and calls its stores with the tenant
  id. One indexed read; no per-tenant object survives the request.
- Every store operation is fenced by its own registry row, and the same
  row read supplies what the operation needs to address the tenant (the
  codec configuration, the container, the collection UUID). A key that
  is not live raises one error type.
- On an unknown tenant or a not-live-key error, the router asks the
  tenant service for the tenant's state and answers 404 (no row, or
  `deleted`) or 409 (`provisioning` or `deleting`). No component reads
  the tenant table.

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
- Search: embed the query; split the filter and choose the plan under
  "Properties and filtering"; vector query (verified against the ledger
  after the query, inside the vector store); segment contexts; on
  request, the full events from the event store.
- `forget`: look up segments and derivatives; delete vector records;
  delete segments.

Tenant configuration section `episodic_memory`, with mutability:

- `embedder` (provider id): immutable; a different embedder is a new
  tenant and a new ingestion.
- `reranker` (provider id or null): mutable.
- `segmenter`, `deriver`: their options; mutable, applying to events
  processed after the change.
- `search`: default `limit`, `expand_context`, minimum score; mutable.

Episodic memory uses no language model today (both segmenters and both
derivers are deterministic; the embedder is the only model call). The
section gains a `language_model` id when a deriver needs one.

Hooks:

- `ensure`: create the partition and the collection under the tenant
  id; an existing `live` row is this component's own earlier attempt
  and is success, a `creating` row is resumed, a `dropping` row is a
  reused key and raises (see "Create is strict"); insert the per-tenant
  row if absent, else verify immutable options and apply mutable ones.
- `delete`: first call, logically delete the tenant id in both stores
  and remove the row; every call, reclaim on each; `DONE` when both
  report done.
- `reclaim`: reclaim under the tenant id in both stores; `DONE` when
  neither finds anything.
- `maintain`: the segment store's `purge_deleted_partitions` as a safety
  net for queue entries no job covers.

## Properties and filtering

Two tiers of fields, one mechanism underneath. The reference is the
`default` branch of edwinyyyu/MemMachine (commits 27b3279b, 822ccb6b,
2d5dc2b5), adjusted where noted.

System fields. Defined by the server, first-class in the API, typed: for
an event, `id`, `timestamp` and `producer`. Search takes them as named
parameters, `since` and `before` (inclusive and exclusive, so ranges
meet without overlap) and `producers` (a list). They are never spelled
inside the user filter, so no caller and no model decides between
`timestamp` and some prefixed form of it. Underneath, each system field
is stored as a reserved property key, `memmachine_<system>_<field>`,
built by one function that validates the key against the stores' naming
contract at import time; the prefix is the distribution name, so its
uniqueness is the package registry's. Stores therefore index and filter
system fields with the same machinery as user properties, and a caller
key beginning with the prefix is rejected on the way in.

User properties. `properties` on an event: keys `[a-z0-9_]`, bounded by
the stores' naming contract, not reserved; values scalar only: string
(bounded by `properties.max_string_bytes`), integer, float, boolean,
datetime; no lists, no nesting, no nulls (absence is the only way a
field holds nothing); at most `properties.max_keys` keys per event.
Scalar-only is what every backend accepts (Chroma rejects nested values;
S3 Vectors caps filterable metadata at 2 KB per vector). Long text is
content, not a property: it goes in a block. Metadata that is not meant
to be filtered goes in `payload`, an opaque JSON document on the event,
bounded by `payload.max_bytes`, stored in the event store only, returned
with the event, never propagated to derived data and never filterable.
`payload` is a proposal (see "Open questions"); without it, the answer to
"where does unfilterable metadata go" is "in the caller's own store,
keyed by event id".

Propagation. Properties are set at ingest and are immutable; changing an
event's properties is a delete and a re-ingest. The event store holds
them. Segments and derivatives carry a verbatim copy of their event's
properties; that is a clause of the segmenter and deriver contracts,
because filtering in the segment store depends on it. Vector records
carry only the declared subset, below.

Declared, not dynamic, indexes. The deployment declares, per vector
store, which user property keys are indexed inside its containers and
with which types (`indexed_properties`), one schema for every container
of that store, created by the schema command (#1573, #1535). Nothing
creates a filter index at runtime, per tenant, or from a request; the
current per-collection `indexed_properties_schema` goes. System fields
are always declared. A user property that is not declared is still
filterable, through the segment store, where a deployment adds an
expression index online (`CREATE INDEX CONCURRENTLY` on PostgreSQL)
without touching the vector store or any co-tenant.

Filter representation. A filter is a tree, constructed, never parsed:
the closed union from the reference, `Equals`, `NotEquals`, `Ordering`
over numbers and datetimes only (ordering strings depends on how a
store encodes them; ordering booleans is meaningless), `In` over a
homogeneous list, `IsMissing`, n-ary `And` and `Or`, `Not`, compiled by
each store with an exhaustive `match`, so a node a store does not handle
is a type error rather than a query-time one. At the API and in MCP a
filter is a JSON object validated by the schema generated from that
union, for example
`{"and": [{"eq": {"field": "kind", "value": "note"}},
{"gte": {"field": "score", "value": 3}}]}`. There is no string language,
nothing to learn beyond the schema, and nothing that parses successfully
into a different filter than intended; an MCP tool exposes the schema as
its parameter, which a model fills more reliably than a grammar. A field
in a filter must be a legal user key; system fields have their own
parameters. Semantics: a predicate matches only a record holding a value
of the compared type; `NotEquals` keeps records holding a differing
comparable value, and `Not(Equals)` also keeps records holding none.

Routing. Where a predicate is evaluated depends on what the backend can
do and on how selective the predicate is; the caller never chooses.

- Declared keys, system fields included, are evaluated inside the vector
  search on every backend that filters during the search (Qdrant,
  Milvus, Weaviate, S3 Vectors, sqlite-vec, pgvector), which is what
  keeps a filtered search returning enough results instead of filtering
  away the ones it found.
- Undeclared keys never reach the vector store. Episodic memory splits
  the filter: the declared part goes to the vector query; the rest is
  resolved in the segment store by a bounded probe
  (`filter.selective_limit`). If the matching segments fit under it,
  their derivative ids become an allowlist the vector store scores
  directly (`get_cosine_similarity`, or the backend's id-restricted
  search). If not, the vector query runs with the declared part alone,
  over-fetches with bounded widening up to `filter.max_overfetch`, and
  the segment store drops the seeds that do not match. At the cap the
  search returns what survived, which can be fewer than `limit`.
- A backend that cannot filter during the search (the usearch engine)
  is handed an allowlist by its store, computed the same way over the
  store's own records table.

Limits are maximums. Every count a caller passes is a maximum: `limit`
on search is the most hits returned, and a filtered search may return
fewer. Nothing is called "top k", which promises exactly k.

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
  WHERE key = ? FOR SHARE`. No row, or a row not `live`: raise the
  not-live error. The shared row lock is held until the transaction
  ends; the write's own statements run inside it (SQL stores), or the
  remote write runs while it is open and is acknowledged as applied
  before it commits (vector stores).
- Delete (logical): one transaction executes `... FOR UPDATE` on the
  row, which waits for every shared lock to be released, then removes
  or marks the row. After it commits no write can take the shared lock,
  so no write under the key can start, and every write that started has
  finished.
- Read: the read carries `EXISTS (SELECT 1 FROM <registry> WHERE key =
  ? AND live)` in its statement where the data is in the same database,
  and verifies the row after the read where it is not. A read that
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
  has `host` scope and is documented as slow, and PostgreSQL is the
  ledger at `cluster` scope. The file lock is what gives a SQLite-backed
  store `host` scope: it reaches every process on the host and no
  further.

Cost, so a deployment can be sized for it: a write holds one pooled
connection, idle in an open transaction, for the duration of the remote
write, so the pool bounds the vector writes a process can have in
flight. The pool is sized for that plus the process's other SQL work, or
the ledger is given a pool of its own. `idle_in_transaction_session_timeout`
on that pool has to exceed the longest remote write, the opposite of the
low value one would otherwise choose. A read costs one indexed row read
after the query.

Locks, not leases. A lease is a row saying "holder H owns this until
T"; the holder is expected to finish or renew before T, and others take
over after it. Three things follow, none of which this design wants.
The holder cannot know it still holds the lease when it acts without
comparing a clock, and between that comparison and the act the lease
can expire and be taken over, so a late write is prevented only if the
resource itself rejects a stale holder, which needs a fencing token
checked by the backend, which the vector backends cannot do. The clock
compared must be the database's, never the process's, so every check is
a round trip anyway. And a lease is a row write per acquire, renew and
release, with no queue: after expiry whoever asks first wins, so
starvation needs its own mitigation. A database row lock has none of
this: it is held in the lock manager's memory, it is released the
instant the holder's session ends, waiters are queued, and the database
does the rejecting. A lease is the right tool only where a takeover
must be delayed rather than immediate after the holder's death, which
nothing here needs. The one lease this design had, on reconciler jobs,
is now a held row lock. The lease once considered for vector writes was
not a mutual-exclusion lease at all: every writer held a shared,
time-limited validation, no delete waited for it, and reclamation was
delayed past the longest validation instead; the `FOR SHARE` /
`FOR UPDATE` pair is that same shared-versus-exclusive shape with the
database enforcing it and no clock.

Nothing above reads a clock. The segment store ships this fence
(`design/segment_store_shared_tables.md`); the event store and the
vector store adopt it.

Scope declarations, computed from constructor arguments:

| Resource | Scope |
| --- | --- |
| Tenant registry, event store, segment store | `cluster` on PostgreSQL; `host` on a SQLite file; `process` on in-memory SQLite |
| Vector store | the minimum of its backend's and its ledger's: networked Qdrant, Milvus, Pinecone, S3 Vectors, Weaviate, Chroma are `cluster`; local-mode Qdrant and Milvus are `process`; sqlite-vec is `host` (ledger and data in one file, under the file lock); the usearch store is `process` (index state in process memory) |
| Reconciler, ingest service, routers | any; they hold no shared state of their own |

### Create is strict

`create_partition(key)` and `create_collection(key, container)` insert
the registry row and raise `KeyExistsError` if any row exists under the
key, in any state. They are not idempotent, on purpose: a store cannot
tell a retry of its caller's own create from a create under a key
someone else chose, and only the caller can, so there is no
open-or-create at the store.

Idempotency lives in the component's `ensure`, which knows the key's
provenance: the tenant service minted the id on the tenant table's
primary key before any job ran, so a `live` row under the key can only
be this component's own earlier attempt, and `ensure` treats it as
success; a `creating` row is an interrupted attempt of its own, and
`ensure` resumes it; a `dropping` row means the key had a previous life,
which the tenant service's tombstone makes impossible for the server and
which a library user reusing keys can cause, and `ensure` raises
`KeyReusedError` for an operator.

What every store operation does with a key whose row is present but
not `live` (`creating` and `dropping`; a SQL store's row while its purge
is pending), and with no row at all. Only `ensure` proceeds on
`creating`, as above:

| Operation | present, not live | no row |
| --- | --- | --- |
| create | `KeyExistsError` | creates |
| write | not-live error | not-live error |
| read | not-live error | not-live error |
| logical delete | returns; idempotent | returns; idempotent |
| reclaim | proceeds; the row goes when nothing remains | deletes by key in every container; `DONE` when nothing is found |

### Segment store

As shipped in #1548, with these changes:

- Key type `UUID`; the `incarnation` column of every table becomes the
  key, and the purge queue is keyed by the key. `open_or_create_partition`
  goes.
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
`state` (`creating`, `live`, `dropping`), `created_at`, `updated_at`.

- Native containers are deployment configuration: one per embedder
  provider per vector store, created by the schema command, never by a
  request. A container's dimensions and metric are the embedder's; its
  indexed properties are the store's, one schema for every container
  (#1573, #1572). A container is retired by the schema command when
  the configuration no longer declares its embedder and no ledger row
  references it; `memmachine schema status` shows, per container, the
  ledger rows referencing it by state, which is how an operator sees an
  old embedder's container drain. Until then it stays and serves the
  tenants pinned to it. Inside a container a tenant is a value or a
  native tenant object, per the table below.
- `create_collection(key, container)`: strict (see "Create is strict").
  Where the tenant is a value inside the container, one ledger insert
  straight to `live`. Where the tenant is a native object (a Chroma
  collection, a Weaviate tenant), the row is inserted as `creating`,
  the object is created, and the row is set `live` with its address; a
  crash between the two leaves `creating`, which `ensure` resumes by
  creating the object if absent and setting `live`. Telling "already
  exists" from other failures is per backend; on Chroma it is by
  message, since its duplicate-create error is untyped (`InternalError`
  500 locally, `ChromaError` 400 over HTTP, never the
  `UniqueConstraintError` the module exports; chromadb 1.5.9).
- Write (`upsert(key, records)`, `delete(key, ids)`): the fence's write
  step, whose row read returns the container and address the operation
  uses; the remote write is acknowledged as applied (Qdrant
  `wait=True`; Milvus with strong consistency) before the transaction
  commits. No live row: the not-live error, and no write creates a
  collection.
- Read (`query(key, vectors, limit, filter, allowed_ids)`,
  `get_cosine_similarity(key, vector, ids)`): read the row for the
  address, query, then verify the row is still `live`; raise the
  not-live error otherwise. `filter` names declared keys only and is
  evaluated during the search where the backend can; `allowed_ids`
  restricts the search to given records; queries return record ids and
  scores, never properties.
- `delete_collection(key)`: the fence's delete step, setting `dropping`.
  O(1), idempotent.
- `reclaim_collection(key) -> DONE | MORE`: with a `dropping` row,
  delete records under the key in bounded steps by the backend's means
  in the table below, `MORE` while records remain, and remove the row
  when nothing remains. With no row, delete by key in every container
  the store has, which is how the tenant service's re-sweep reaches a
  record that landed after the row went; `DONE` when nothing is found.
  Containers are few (one per embedder), so a no-row reclaim is a
  bounded number of filter deletes that mostly find nothing.

Why nothing escapes. Every record carries a key that a ledger row
recorded before the record could exist. For a live writer the fence
orders its write before the delete, and after the delete no write can
start. The only write that can land after `dropping` comes from a
writer whose database session ended while its vector request was in
flight, so the lock was released before the backend applied the write.
If it lands before reclaim finishes, reclaim deletes it. If it lands
after, the tenant service's tombstone brings the re-sweep back to the
key, and the store's no-row reclaim deletes it. No path compares
timestamps, and no key is forgotten while a record under it could
exist, because the tenant tombstone is kept. Alternatives considered:
accepting bounded leakage (rejected: garbage that is never collected
is unacceptable); delaying reclamation past the longest possible write
(rejected: a clock comparison); a fencing token checked by the backend
(unavailable where the tenant is a payload value). Keeping tombstones
in the stores instead of the tenant service was rejected because a SQL
store's purge is exact and needs none, so the stores would disagree
about a reused key: a duplicate id would be accepted by the SQL stores
and refused by the vector store, mid-`ensure`, after rows were created.
Held by the tenant service, one tombstone refuses the id before any
store is touched.

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
| sqlite-vec | partition-key value in one `vec0` table per container | table name; the key as the partition key | yes, in-statement (ledger in the same file) | yes | keyed delete |
| usearch store | rows in a shared table; one index file per tenant | table name; the key as the column value; the index file path | yes, in-statement (ledger in the same file) | yes | keyed delete; unlink the file |

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
implementation. S3 Vectors' limits are from its documentation as of
this writing: filterable metadata is 2 KB per vector, and timestamps
must be stored as numbers to be range-filtered, since comparisons apply
to numbers only.

The SQLite stores. Both are rewritten away from a table per collection,
like the segment store: the sqlite-vec store keeps one `vec0` table per
container with the tenant key as its partition key, the usearch store
keeps records in one shared table and
one index file per tenant (a file it owns, not schema; index state
lives in process memory, which is why it stays at `process` scope), and
both keep their ledger in the same file, so their fence is in-statement
and the sqlite-vec store reaches `host` scope. The O(1) drop that a
table per collection gave becomes a keyed delete run by the reclaim
job, as in the segment store; unreachability is immediate through the
ledger either way. A `vec0` partition key is not a filter over a shared
scan: vec0 stores each partition value's vectors in chunks of its own,
and a KNN constrained on the key reads only those chunks. Measured on
the pinned 0.1.9 with 400 tenants of 25 vectors each at 1024
dimensions, the constrained query cost two orders of magnitude less
than an unconstrained one over the same table. The cost of a partition
is the chunk allocation: every partition value holds at least one
chunk of `chunk_size` vectors, `chunk_size` times dimensions times four
bytes, which a table per collection paid identically (measured: the
same size per tenant either way). `chunk_size` is a per-table setting
the store exposes, and a deployment with many small tenants sets it
low. `sqlite_vec_vector_store.py:4` records that partition keys were
avoided in case a future sqlite-vec ANN index does not support them;
if that comes to pass, the store stays brute-force, which is what it is
today and what its scope is for.

A read on a deleted key raises on the post-query verification, so no
dead tenant's records reach a caller.

## Configuration

Resources are plain classes with typed constructors. Configuration is
their constructor arguments and nothing else, so a resource can be
built by hand, in a script with no configuration at all, or by a
different configuration system, and never learns which. The loader is
the only code that reflects.

- A kind names a callable: a class, or a factory function, ours or a
  third party's. Its parameters are what the loader reflects.
  Dependencies are parameters annotated with resource types (`engine:
  AsyncEngine`, `embedder: Embedder`, `embedders: Mapping[str,
  Embedder]`); options are scalar parameters, grouped into a Pydantic
  `Params` parameter where that reads better and left as plain keyword
  parameters where it does not. A third-party constructor with typed
  keyword parameters is registered as it is (`AsyncQdrantClient`); one
  whose signature is `**kwargs` or untyped gets a factory function of a
  few lines with explicit typed parameters, and no `Params` model.
  Nothing in a resource refers to ids, references, or the loader.
- The document is one flat map, `resources`, from id to
  `{kind: ..., <constructor arguments>}`. Ids are unique across all
  resources, so a database and an embedder cannot share one. `kind`
  selects a class from a table of kind to class, in code; a library
  user registers a class under a kind in that table, which is the one
  registration point. A dependency argument holds the id of another
  resource; a `Mapping[str, T]` argument holds a list of ids; `Params`
  fields are inline.
- The loader validates each entry against the callable's signature
  (`inspect.signature` with `typing.get_type_hints`): every argument
  named exists, scalar arguments validate against their annotations,
  and every dependency id names a built resource that satisfies the
  annotated type. It orders construction by dependencies (a topological
  sort, cycles an error) and calls each callable with exactly what it
  declares. The dependency check is a runtime instance check of the
  resolved object against the annotation, which Pydantic performs from
  the signature (`validate_call` with arbitrary types allowed):
  `isinstance` for classes and abstract base classes, method presence
  for runtime-checkable protocols, element-wise for `Mapping[str, T]`
  and unions. It catches a wrong id or a wrongly typed resource at
  startup with the argument named; it does not prove more than
  `isinstance` can, and hand wiring is checked by the type checker
  like any other call, since a resource is called the same way with or
  without the loader.
- Third-party clients are resources like any other: a `postgres` kind
  is a factory function producing a SQLAlchemy `AsyncEngine` from `url`
  and the pool settings it names, since `create_async_engine` takes
  `**kwargs`; a `qdrant` kind is `AsyncQdrantClient` itself. That is
  what the `databases` entries of the earlier draft were.
- Where a component offers tenants a choice among providers, its
  constructor takes `Mapping[str, Embedder]` and the entry lists ids;
  the loader passes exactly those objects keyed by id, and a tenant
  configuration naming another id is rejected by the component's own
  validation. The mapping is a constructor argument holding a closed,
  typed set the document declares; nothing is looked up after startup.
- `memmachine config schema` prints the JSON Schema of the document from
  the kind table and the constructors; `memmachine config example`
  prints a documented example. Documentation is generated, not
  maintained.
- Keys are case-sensitive. Secrets are written as `${ENV_VAR}` and
  resolved at load. A validation failure, an unknown key, or a reference
  to an undeclared id fails startup; nothing is auto-disabled.
- There is no runtime configuration API. Changing the document is a
  restart, which a scaled deployment does as a rolling replacement.

Relation to the earlier `resource_initializer.py` proposal (an
edwinyyyu/MemMachine commit, e134c531): the topological build order and
the table from type to builder are kept. Its three problems are solved
by reflection: dependencies come from constructor annotations, not a
`get_dependency_ids` written per builder; the build is the constructor
itself, not a `build` written per builder; and ids are checked against
annotated types at the injection point rather than passed as strings.
Dynamic registration is the kind table.

Example, each key a constructor argument of the named class:

```yaml
resources:
  main:
    kind: postgres
    url: ${MEMMACHINE_DATABASE_URL}
  vectors:
    kind: qdrant
    url: http://qdrant:6333
  openai-large:
    kind: openai_embedder
    model: text-embedding-3-large
    dimensions: 1024
    api_key: ${OPENAI_API_KEY}
  bm25:
    kind: bm25_reranker
  events:
    kind: sqlalchemy_event_store
    engine: main
  segments:
    kind: sqlalchemy_segment_store
    engine: main
  vector-store:
    kind: qdrant_vector_store
    client: vectors
    ledger_engine: main
    indexed_properties:      # once per store; system fields implicit
      kind: string
      score: integer
  episodic_memory:
    kind: episodic_memory
    event_store: events
    segment_store: segments
    vector_store: vector-store
    embedders: [openai-large]
    rerankers: [bm25]
  ingest:
    kind: ingest_service
    event_store: events
    subsystems: [episodic_memory]
  tenants:
    kind: tenant_service
    engine: main
    components: [events, episodic_memory]

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
  tenant_service: tenants
  ingest_service: ingest
```

Tenant templates are validated at startup against each component's
tenant configuration model, including their provider ids against the
component's mapping, and nothing is built from them. A template edit
changes future tenants only; existing tenants keep their recorded
configuration, which an operator changes with `PATCH`.

Provider ids are stable identities. A provider's model or dimensions
are not changed under an id; a new model is a new id. Removing an id
from a component's mapping fails startup while any tenant of that
component references it, which the component checks from its own
per-tenant table at construction.

## Startup and wiring

`memmachine serve --config PATH` builds the resources in the order the
references give, each by constructor from already-built objects, then
verifies schema and scope, then starts the roles. Startup fails on the
first resource that cannot be built, naming the resource and the cause,
before the socket is bound. Shutdown runs in reverse: routers stop
accepting, a reconciler finishes its current job, resources close.

Nothing per tenant is created after startup: a request reads rows and
calls stores with the tenant id, and a process holds no per-tenant
state beyond a store's private cache where a backend needs one.

## Server API

Prefix `/v1`. Tenant ids in paths are UUIDs; names are looked up
explicitly. Bodies are JSON. `?wait=` on the three lifecycle requests
blocks until the change has applied or the wait elapses.

Tenants:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST /v1/tenants` | create; body `name`, `template`, `configuration` | 202 provisioning, or 201 with `wait` once active; 409 `tenant_exists`; 422 |
| `GET /v1/tenants?name=` | look up by name | 200; 404 |
| `GET /v1/tenants?prefix=&cursor=` | list, paged | 200 |
| `GET /v1/tenants/{id}` | record, state, requested and applied configuration versions, jobs with attempts and last error | 200; 404 |
| `PATCH /v1/tenants/{id}` | rename and/or configuration update | 202, or 200 with `wait` once applied; 409; 422 |
| `DELETE /v1/tenants/{id}` | start deletion | 202, or 204 with `wait` once deleted; 404 |

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
| `POST .../search` | body `query`, `limit`, `since`, `before`, `producers`, `filter` (JSON tree), `expand_context`, `include_events` | 200 with up to `limit` scored hits |
| `GET ...` | watermark and lag behind the event store | 200 |

Event body: `id` (optional UUID), `timestamp` (optional; server time if
absent), `producer` (optional string), `blocks` (list of `{type: text,
text}`), `properties` (scalar values under legal keys; what `filter`
sees), `payload` (opaque JSON, not filterable; proposed).
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
2. Per-tenant resources: rows, in every store. Tenant-specific DDL is
   the only DDL allowed outside `memmachine schema upgrade`, and no
   store in this design uses it; the usearch store's index file per
   tenant is a file, not schema.

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
  prints the same comparison, the per-container ledger counts, and the
  tombstone counts.
- Rolling deployments: a migration must keep the previous release's
  code working (expand and contract: add before the code that reads,
  remove after the code that writes is gone), because during a rollout
  processes of both releases run against one schema. A migration that
  cannot is a release note that requires a stop.

## What is reused and what is removed

Reused, with the change named:

- `episodic_memory/event_memory/`: `EventMemory` renamed to episodic
  memory, the segmenters, the derivers, the data types, and the segment
  store (UUID keys in place of incarnations, `reclaim_partition`, no
  handle).
- `common/vector_store/`: the four implementations, with the registry
  replaced by the ledger and its fence, the `config` parameter removed
  from `create_collection`, containers provisioned by the schema
  command, and the two SQLite stores on shared tables.
- `common/embedder/`, `common/language_model/`, `common/reranker/`,
  each class gaining a typed constructor and a `Params` model in place
  of the separate configuration model.
- `common/filter/`, `common/metrics_factory/`, `common/payload_codec/`.
- `enable_sqlite_foreign_keys` and the engine construction in
  `common/resource_manager/database_manager.py`, as the `sqlite` and
  `postgres` factory kinds.

New: the event store, the ingest service, the tenant service and
reconciler, the loader, the schema command, the routers.

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
- #1530: agrees on the outcome, the store ABCs keeping only the strict
  create, and on the reason: only the caller knows why an existing row
  is acceptable. The two prove different things at the same signature.
  #1530's names are reused by design and successive lives are separated
  by incarnations, so its create raising means "this exists"; here keys
  are never reused, the tenant service's tombstone enforces it, and a
  row under a key in a store is evidence of an interrupted attempt or
  a library caller's reuse.
- #1531: the reference for the concurrency scope idea. The levels,
  the declarations and the deployment-side check above are this
  design's own; they differ from #1531 where the ledger changes what a
  store can promise (Qdrant, Milvus and sqlite-vec widen).
- #1571: the router's 404/409 mapping under "Component contract"; with
  no handles there is nothing to evict.
- #1575, #1576, #1577: resolved by construction: declared components, no
  implicit creation, durable jobs with retry.
- #1572, #1573, #1564, #1565, #1537, #1563: the vector store contract
  above: containers from configuration, the ledger as fence, tenants as
  values, single-use keys, reclamation plus the tenant service's
  re-sweep. In-flight registry PRs are measured against that section.
- #1535: one declared, typed `indexed_properties` schema per vector
  store, in configuration, under "Properties and filtering".
- #1570: "Schema management".
- #1542: SQLite pragmas become `Params` fields of the `sqlite` factory
  kind (`busy_timeout`, `journal_mode`); the reconciler's retry covers
  the busy-timeout raise.

## Open questions

- Hierarchy: flat tenants with prefix listing, proposed, or a parent
  column with cascading delete as jobs.
- Whether ingest processes subsystems synchronously in the request
  (proposed, with `catch_up` as the repair) or always through a queue.
- Event size limits, and block types beyond text (the data types admit
  others).
- `payload`: whether an opaque, unfilterable, event-store-only document
  belongs on the event, or whether unfilterable metadata is the caller's
  to keep.
- The library composition surface: the constructors a user calls to get
  the event store, episodic memory and the tenant service without the
  HTTP layer, which the constructor-argument model above makes the
  same call the loader makes.
- Retention: deleting events by age or by producer, as a job kind.
