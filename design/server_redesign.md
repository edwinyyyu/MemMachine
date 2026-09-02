# MemMachine server redesign

Status: proposal, under review in its own PR against `speedkick`. It
replaces the tenant lifecycle draft in this file's history: that design
stands, and this document widens it to the whole server, because the
lifecycle cannot be fixed inside the current API, configuration and
wiring. Companion: `design/segment_store_shared_tables.md` (the segment
store as shipped in #1548). Tracking: #1574. Line references are to
`speedkick` at 7752e4cb, paths under `packages/server/src/memmachine_server`
unless given in full.

## Scope

In: the data model (tenants), the HTTP API, configuration, startup and
wiring, tenant lifecycle, the event memory subsystem with its two stores,
and schema management (initial setup and migrations). Breaking changes to
the API and to the store ABCs are accepted.

Out, and not wired into the new server: short-term memory (callers manage
their own near-term context), semantic memory (cost high, benefit
unproven), declarative memory and the graph stores behind it (Neo4j,
NebulaGraph), the retrieval agent. The MCP surface gets one note.

Requirements that shape everything below:

- Horizontal scaling without sharding: any process serves any tenant, no
  process owns a tenant, nothing is coordinated in process memory.
- Garbage that is never collected is unacceptable. Wasted writes are
  acceptable.
- One client per provider per process, shared by every tenant.
- The tenant layer knows neither the stores nor the options a subsystem
  takes, and does not sit in the path of data operations.
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
  (`main/memmachine.py:343`; #1575, #1577). Vector collection handles are
  not invalidated by deletion; segment partition handles are.
- Configuration. Resources are built lazily behind double-checked locks
  (`common/resource_manager/`), `DatabaseManager` repeats the pattern
  seven times, locator functions assemble params (`*/service_locator.py`),
  `ResourceManagerImpl.build()` is never called by the server, a
  misconfigured provider silently disables a subsystem
  (`common/configuration/__init__.py:384`), every YAML key is lowercased
  (`:586`), and an opt-in HTTP API mutates the running process's
  configuration (`server/api_v2/config_router.py`).
- Errors. Status mapping is hand-written per handler, two handlers have
  none, an unknown exception serializes its traceback into the response
  (`server/api_v2/exceptions.py:49`), and an unhandled exception drops the
  connection instead of answering.
- Storage. Eight `create_all` sites and one Alembic chain (#1570). Every
  ingested message is stored twice, in the `episodestore` table and as
  segments, linked by an `_episode_uid` property with no transaction
  between the two (`episodic_memory/long_term_memory/long_term_memory.py:386`).
  Re-ingesting an id duplicates segment rows.

## Vocabulary

- Tenant: an isolated memory with its own configuration and lifecycle. It
  is not a user, a conversation or an agent run; an application maps
  those onto tenants by naming them. "Tenant" is chosen over "memory",
  "space" and "namespace" because it says isolation and lifecycle and
  nothing about what the application puts inside.
- Tenant name: the application's label. Any string up to 1024 bytes,
  unique among tenants that are not deleting, renamable.
- Tenant id: a UUID minted at creation, permanent, never reused. The key
  under which every store holds the tenant's data, rendered as 32 hex
  characters where a backend wants a string.
- Subsystem: a memory implementation exposing data operations per
  tenant. This design has one, event memory.
- Resource: what a subsystem holds per tenant inside a store: the segment
  partition, the vector collection.
- Provider: a process-wide client declared in server configuration and
  shared by every tenant: a database, an embedder, a language model, a
  reranker.
- Template: a named block of per-subsystem options in server
  configuration, copied into a tenant's configuration at creation.
  Nothing is instantiated from a template.
- Tenant configuration: the resolved options stored on the tenant record,
  one section per subsystem, read whenever the tenant is opened.
- Event: the unit of ingestion into event memory: an id, a timestamp, a
  producer, one or more content blocks, properties. Segment and
  derivative keep their meanings from `event_memory/data_types.py`.
- Job: a row describing one lifecycle action (provision or delete) for one
  subsystem on one tenant. The reconciler executes jobs.
- Reconciler: the loop, one per process, that claims and executes jobs.

## Principles

1. Control plane and data plane are separate objects with separate
   endpoints. The tenant service creates, renames, configures and
   deletes tenants and runs jobs. A subsystem serves data operations.
   Neither calls the other's operations; both read the tenant table.
2. A data operation names its tenant by id in the request. There is no
   server-side handle object and no routing through the tenant layer.
3. The tenant layer knows a subsystem only through its registration: a
   name, a configuration schema to validate against, and two hooks,
   provision and delete. It never sees a store or an option.
4. Stores take a UUID key and nothing else. They do not read the tenant
   table. Generations, queues and fences are store-private.
5. Every step is idempotent. A step is atomic only where its backend
   gives that away (one insert, one transaction); there is no transaction
   across backends. Callers still never observe a partial tenant, because
   the tenant row gates reachability: a tenant is served only while
   `active`, and one that is `provisioning` or `deleting` is refused
   until the reconciler finishes it.
6. Deletion stages jobs and returns. The tenant is unreachable at once,
   reclaimed by the reconciler, finished when every subsystem reports
   done, and observable throughout.
7. A store that cannot reject a stale write makes every such write
   reclaimable: the write lands under a key recorded for reclamation, and
   reclamation runs after the last such write can land.
8. Configuration is a declarative document. Everything process-wide is
   built eagerly at startup from it, in dependency order, by constructor
   injection. Nothing is looked up lazily, nothing is mutated at runtime,
   and a bad document fails startup.
9. Schema lives in versioned migrations run by one process at a time.
   Serving code runs DDL in exactly one place, named under "Schema
   management".

## Architecture

Per process, built at startup in this order:

1. The configuration document, parsed and validated.
2. Providers: one engine per database, one client per embedder, language
   model and reranker.
3. The schema step: migrations and native containers under a lock, or
   verification only (see "Schema management").
4. Stores: the segment store and the vector store, over their databases.
5. The tenant registry: the tenant and job tables over the main database,
   and a read-only lookup object for subsystems.
6. Subsystems: event memory, constructed with the registry lookup, the
   provider catalog and its stores. It registers with the tenant service.
7. The tenant service and the reconciler.
8. HTTP routers bound to the objects above.

Who knows what:

- Tenant service: the registry, the job table, the registrations.
- Subsystem: the registry lookup, the provider catalog, its stores, its
  per-tenant instances.
- Stores: their backend and their keys.
- Routers: the tenant service and the subsystems, nothing below them.

The registration interface and the registry lookup are defined by the
tenant package; the subsystem package depends on it and not the reverse.

Horizontal scaling: every process holds the same objects; all shared
state is in the databases and the vector backend; per-process state is
caches any process can rebuild; concurrent creates are arbitrated by a
unique index, concurrent jobs by row claims, concurrent data operations
and deletes by store fences.

## Tenant registry

Two tables in the main database. Every transition is one transaction on
them.

`tenants`:

- `id UUID PK`.
- `name TEXT NULL`, unique index. NULL while deleting: the name is
  released when deletion starts, so a new tenant can take it at once and
  gets a new id.
- `former_name TEXT NULL`: the name at deletion time, for operators.
- `state`: `provisioning`, `active`, `deleting`.
- `configuration JSON`: one object per subsystem name.
- `configuration_version INTEGER`: incremented by every configuration
  update.
- `created_at`, `updated_at`.

`tenant_jobs`:

- `id PK`, `tenant_id`, `subsystem`, `action` (`provision`, `delete`);
  unique on `(tenant_id, subsystem, action)`.
- `state`: `pending`, `done`.
- `attempts`, `last_error`, `next_run_at`, `lease_until`, `created_at`,
  `updated_at`.

Rename is an update of `name`. No store key contains the name, which is
why names can be arbitrary strings while store keys are the 32 hex
characters every backend accepts.

## Tenant lifecycle

Create, `POST /v1/tenants`:

1. Resolve the configuration: the named template (default `default`),
   overlaid with the request's per-subsystem overrides, each section
   validated by its subsystem's schema. An unknown subsystem name or an
   invalid option is 422.
2. Insert the tenant row as `provisioning` and one `provision` job per
   registered subsystem, in one transaction. A duplicate name fails on
   the unique index: 409, with the existing tenant's id and state in the
   body. With `if_exists: "return"` the existing tenant is returned
   instead, with 200, without comparing configuration.
3. Run this tenant's jobs inline through the claim path the reconciler
   uses. The transaction that marks the last job done also sets the row
   `active`. Return 201 with the tenant.
4. If a job raises, return 503 with the tenant in `provisioning`. The
   reconciler retries the job; the tenant becomes `active` without
   further requests, which `GET` shows.

Delete, `DELETE /v1/tenants/{id}`:

1. One transaction: `state = deleting`, `former_name = name`,
   `name = NULL`, and one `delete` job per registered subsystem. Allowed
   from `provisioning` and `active`; a repeat while `deleting` returns
   the same 202.
2. Wake the local reconciler and return 202 with the tenant in
   `deleting`. Other processes see the jobs at their next poll.
3. The reconciler executes the delete jobs. A subsystem's first delete
   call makes the tenant unreachable in that subsystem's stores; every
   call reclaims a bounded amount; the job is done when the subsystem
   reports nothing remains.
4. When every delete job is done, one transaction removes the job rows
   and the tenant row. `GET` then returns 404.

Configuration update, `PATCH /v1/tenants/{id}` with `configuration`:
each subsystem validates its section's change against its schema, in
which every option is marked mutable or immutable; an immutable option
in the patch is 422. One transaction writes the document and increments
`configuration_version`. Subsystems apply it when the tenant's instance
is next reopened (see "Subsystem contract").

Rename, `PATCH /v1/tenants/{id}` with `name`: one update; 409 on a
duplicate.

States: `provisioning -> active -> deleting -> row removed`, and
`provisioning -> deleting`. There is no failed state. A job that raises
is rescheduled with exponential backoff (1 s doubling to 5 min) and keeps
`attempts` and `last_error` on its row for as long as it takes. An
operator fixes the cause and the next attempt succeeds. A tenant in
`provisioning` or `deleting` past a threshold is logged with its jobs'
last errors, and logged again each time the age doubles.

Reconciler:

- One task per process, started with the server. It polls every
  `tenants.reconciler.poll_interval` (default 5 s) and when woken
  locally.
- Claim: one `UPDATE ... SET lease_until = now + lease` over the rows
  selected by `state = 'pending' AND next_run_at <= now AND (lease_until
  IS NULL OR lease_until < now) ORDER BY next_run_at LIMIT n FOR UPDATE
  SKIP LOCKED`, returning the claimed rows. On SQLite the same statement
  without `SKIP LOCKED` under `BEGIN IMMEDIATE`; concurrent reconcilers
  serialize there. A crashed process's claim expires with its lease
  (default 60 s) and the job is claimable again; hooks are idempotent,
  so a job executed twice is harmless.
- Execute: `provision` calls the hook and marks the job done. `delete`
  calls the hook; `DONE` marks the job done; `MORE` reschedules it at
  `now + reclaim_interval` (default 1 s); an exception reschedules with
  backoff. The transaction that marks a job done checks the tenant's
  remaining jobs and applies the state transition if none remain.
- Bounded per pass: at most `n` jobs, each hook bounded by its own
  budget.

## Subsystem contract

A subsystem registers with the tenant service at startup:

- `name`: its section in tenant configuration and its path segment in
  the API (`event_memory`, `/event-memory`).
- `configuration_schema`: a Pydantic model for its section, every field
  marked mutable or immutable, with defaults. The tenant service calls
  `validate(section)` and `validate_update(old, new)` and never reads a
  field.
- `provision(tenant_id, section) -> None`: idempotent; creates the
  subsystem's resources or verifies they exist; complete on return.
- `delete(tenant_id) -> DONE | MORE`: idempotent; the first call makes
  the tenant unreachable in every resource; every call reclaims a
  bounded amount; `DONE` when nothing of the tenant remains. May raise;
  the reconciler retries.

Data operations:

- The subsystem keeps per-tenant instances in a process-local LRU (size
  and idle TTL configurable, defaults 1000 and 60 s). Opening an instance
  reads the tenant row: absent is 404; not `active` is 409; otherwise the
  stored section and `configuration_version` are read, provider names are
  resolved through the catalog (a missing provider is 503 and a log
  line), and store handles are opened on the tenant id.
- An instance older than the TTL is reopened on its next use. That is
  how a configuration update reaches every process: within one TTL, with
  no coordination.
- After open, store fences protect data operations, not the tenant row.
  A stale-handle error from any store evicts the instance and maps to
  409 if the tenant row still exists and 404 if it does not.
- Every operation's final store call goes to a store that fences. The
  subsystem's code keeps this rule; the tenant layer cannot enforce it.
  Event memory's ordering below satisfies it.

## Event memory

What is stored. The segment store is the system of record. A segment row
carries the event's id, timestamp, context and properties, its block
index and chunk offset within the event, and its chunk. Segmenters are
required to split so that concatenating a block's chunks in offset order
rebuilds the block exactly; the passthrough segmenter does, and the text
segmenter needs `strip_whitespace=False` added to its splitter (it already
uses `chunk_overlap=0` and `keep_separator="end"`,
`segmenter/text_segmenter.py:31`). With that, the segment store
reconstructs any event. The `episodestore` table and the `Episode` model
are not carried over: they duplicated the segments' content, were linked
to them by a property with no transaction, and made every search end
with a second lookup. The vector store holds derivative embeddings only, is
rebuildable from the segment store, and is treated as an index.

Identifiers. The event id is supplied by the caller or minted by the
server (uuid4). Segment and derivative ids are derived from the event id
and position (uuid5), so a retried ingest rewrites the same rows instead
of duplicating them. Ingest is idempotent per event id: an event whose id
is already stored is skipped and reported as such.

Per-tenant resources: one segment partition and one vector collection,
both keyed by the tenant id. The vector collection's container is chosen
by the tenant's embedder (see "Vector store").

Operation ordering. The vector store cannot fence; the segment store can.
Every operation ends at the segment store:

- Ingest: segment, derive, embed; upsert derivatives into the vector
  collection; then write segments. An ingest that straddles a deletion
  fails at the segment write with the stale error (409); its vector
  records lie under the dead key and are reclaimed with it. This reverses
  the current order (`event_memory.py:274`, `:292`).
- Search: embed the query; vector query; then `get_segment_contexts`.
- Delete events: look up segments and derivatives; delete vector records;
  then delete segments.
- Reconstruct events: segment store only.

Tenant configuration section `event_memory`, with mutability:

- `embedder` (provider name): immutable; it fixes the vector container.
- `reranker` (provider name or null): mutable.
- `segmenter`, `deriver`: their options; mutable, applying to events
  ingested after the change.
- `search`: default `limit`, `expand_context`, score threshold; mutable.

Event memory uses no language model today (both segmenters and both
derivers are deterministic; the embedder is the only model call). The
schema gains a `language_model` option when a deriver needs one; the
provider catalog already carries language models for that.

Hooks:

- `provision`: `segment_store.create_partition(id, codec config)` and
  `vector_store.create_collection(id, container for the embedder)`;
  "already exists" is success on repeat.
- `delete`: the first call `segment_store.delete_partition(id)` and
  `vector_store.delete_collection(id)`; every call
  `segment_store.reclaim_partition(id)` and
  `vector_store.reclaim_collection(id)`; `DONE` when both report done.

## Store contracts

Both stores take a UUID key. The string key contract (charset, 32 bytes,
validators, hashing in `partition_key_for_session`) is retired.

### Segment store

As shipped in #1548, with the key type changed and one method added:

- `create_partition(key, config)`, `open_partition(key)`,
  `delete_partition(key)` (logical, O(1), idempotent; handles opened on
  the partition raise from then on): as today.
- `reclaim_partition(key) -> DONE | MORE`: reclaims this key's dead rows,
  bounded per call; `DONE` when no garbage remains under the key. On
  SQLite the DELETE waits on the write lock up to the driver's busy
  timeout and raises past it; the reconciler retries.
- `purge_deleted_partitions()`: kept for library users without a
  reconciler. The server runs it on a slow schedule as a safety net for
  queue entries no job covers.
- Fencing, incarnations and the purge queue stay as designed in
  `design/segment_store_shared_tables.md`. The incarnation is redundant
  under single-use keys and is kept: one column and one mint per create.

### Vector store

The store's collection registry moves out of the vector backend into a
SQL ledger, in a database given to the store (the main database by
default), because the ledger has to be readable without enumerating the
backend and transactional on its own:

`vector_collections`: `key UUID PK`, `container`, `state` (`live`,
`dropping`), `created_at`, `deleted_at`.

- Native containers are deployment configuration: one per embedder
  provider per vector store. A container's dimensions and metric are the
  embedder's; its indexed properties are the store's, one schema for
  every container (#1573). The schema step creates them (#1572); no
  request does.
- `create_collection(key, container)`: one ledger insert. On a backend
  whose container is a table per collection (both SQLite stores) the
  store also creates the key's table, after the ledger row and inside the
  provision job.
- `open_collection(key)`: read the ledger row; `live` required. A handle
  is obtainable no other way, so no write creates a collection.
- `delete_collection(key)`: `state = dropping`, `deleted_at = now`. O(1),
  idempotent. Opens fail from then on.
- `reclaim_collection(key) -> DONE | MORE`: `MORE` until
  `deleted_at + reclaim_grace` has passed; then delete the records under
  the key (a filter delete on Qdrant and Milvus, `DROP TABLE` on the
  SQLite stores); `MORE` while records remain; when none remain, delete
  the ledger row and return `DONE`.
- Writes are acknowledged only once applied (Qdrant `wait=True`; Milvus
  with strong consistency on the count that decides "none remain"), so a
  write acknowledged before reclamation is removed by it.

Why every write is collected. A backend that partitions by a payload
value (#1564) cannot reject a write carrying a dead key and cannot list
the keys it holds (#1565). So every key is written to the ledger before
any record can carry it, and a key leaves the ledger only after its
records are gone. A stale write lands under a key that is `live` (a
wasted write, reclaimed with the tenant) or `dropping` (reclaimed by the
job). The remaining case is a write landing after reclamation finished,
and it is excluded by bounding how late a stale write can be. A vector
write is issued only by an operation on a per-tenant instance. An
instance is opened against an `active` row, is used by operations that
start at most `instance_ttl` after that open (later uses reopen and see
`deleting`), and every operation is bounded by `server.request_timeout`
through a server middleware. So the last stale write lands no later than
`instance_ttl + request_timeout` after the tenant's deletion, on every
process alike. `reclaim_grace` is that sum plus a margin for clock skew
between the processes and the database (defaults: 60 s + 60 s + 10 s),
and reclamation never precedes the last possible stale write. The
segment-store fence on the operation's final call is not what gives this
bound; it makes the straddling operation fail visibly and evicts the
instance early. A caller using the stores directly, outside a subsystem,
is outside this guarantee, and the store's docstring says so. The earlier
draft's `reclaim_unknown(live_keys)` sweep is dropped: it needed an enumeration
the backends do not have, and the ledger makes it unnecessary.

Both SQLite stores reject stale writes on their own, because a write to
a dropped table fails. They use the same ledger for uniformity.

Reads through a stale vector handle return a dead tenant's records until
reclamation; every such read is followed by the segment store call, which
raises, so no result reaches a caller.

## Configuration

One YAML document, validated into a Pydantic tree at startup. Keys are
case-sensitive. Secrets are written as `${ENV_VAR}` and resolved at load.
A validation failure, an unknown key or a reference to an undeclared name
fails startup; nothing is auto-disabled.

```yaml
server:
  bind: 0.0.0.0:8080
  request_timeout: 60s
  schema: migrate          # migrate | verify | skip

databases:
  main:
    kind: postgres
    url: ${MEMMACHINE_DATABASE_URL}
    pool_size: 20
  vectors:
    kind: qdrant
    url: http://qdrant:6333

providers:
  embedders:
    openai-large:
      kind: openai
      model: text-embedding-3-large
      dimensions: 1024
      api_key: ${OPENAI_API_KEY}
  language_models:
    fast:
      kind: openai_responses
      model: gpt-5-mini
      api_key: ${OPENAI_API_KEY}
  rerankers:
    bm25:
      kind: bm25

tenants:
  database: main
  reconciler:
    poll_interval: 5s
    lease: 60s

subsystems:
  event_memory:
    segment_store:
      database: main
    vector_store:
      database: vectors
      indexed_properties: [producer, kind]

tenant_templates:            # data: copied into new tenants, never built
  default:
    event_memory:
      embedder: openai-large
      reranker: bm25
      search:
        limit: 10
        expand_context: 4
```

Rules:

- `databases` and `providers` declare process-wide clients, each built
  once at startup and shared. Concurrency limits belong to the client.
- `subsystems` declares which subsystems the server runs and their
  stores. A subsystem absent here is not registered, has no endpoints,
  and no tenant can carry a section for it.
- `tenant_templates` is data. Startup validates every template against
  the registered schemas and builds nothing from it. A template edit
  changes future tenants only; existing tenants keep their stored
  configuration, and an operator changes them with `PATCH`.
- Provider names are stable identities. A tenant references a provider
  by name; removing the name leaves those tenants unopenable (503 naming
  the provider) until it returns or the tenant is patched. A provider's
  model or dimensions are not changed under a name; a new model is a new
  name.
- There is no runtime configuration API. Changing the document is a
  restart, which a horizontally scaled deployment does as a rolling
  replacement.

Tenant configuration is resolved at creation: the template section, then
the request's section, field by field, validated by the subsystem's
schema with its defaults, and the resolved document is stored. Reading a
tenant returns the stored document, not the template.

## Startup and wiring

`memmachine serve --config PATH` builds the objects in the order under
"Architecture", each by constructor from already-built objects. There is
no service locator and no lazy singleton: a component that needs a
provider receives the catalog (an immutable mapping) or the provider
itself. Startup fails on the first component that cannot be built, naming
the component and the cause, before the socket is bound. Shutdown runs in
reverse: routers stop accepting, the reconciler finishes its current job,
subsystems close their instances, stores and providers close.

Per-tenant objects (instances and store handles) are the only things
created after startup, on demand, bounded by the LRU.

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
| `PATCH /v1/tenants/{id}` | rename and/or configuration update | 200; 409; 422 |
| `DELETE /v1/tenants/{id}` | start deletion | 202; 404 |

Event memory, under `/v1/tenants/{id}/event-memory`:

| Method and path | Effect | Status |
| --- | --- | --- |
| `POST .../events` | ingest a batch | 200 with ids and skipped ids; 409 tenant not active or deleted mid-request; 422 |
| `POST .../search` | body `query`, `limit`, `filter`, `expand_context` | 200 with scored hits |
| `GET .../events/{event_id}` | reconstructed event | 200; 404 |
| `POST .../events/delete` | body `ids` | 200 |

Event body: `id` (optional UUID), `timestamp` (optional; server time if
absent), `producer` (optional string), `blocks` (list of `{type: text,
text}`), `properties` (string keys, scalar values; what `filter` sees).
Search hit: `score` and `segments`, each with `event_id`, `index`,
`timestamp`, `producer`, `text`, `properties`.

Errors: one handler for the domain error hierarchy maps to a status and
a body `{error: {code, message}}` with a closed set of codes:
`tenant_not_found`, `tenant_not_active`, `tenant_exists`,
`invalid_request`, `provider_unavailable`, `internal`. No traceback
leaves the process. Everything unmapped is 500 `internal` with the
traceback logged. Every request is answered; no path drops the
connection.

MCP: rebuilt over the same subsystem objects, with the tenant id taken
from a header; not designed here.

Clients: the Python and TypeScript clients are generated from the OpenAPI
document, not mirrored by hand.

## Schema management

Two kinds of schema, handled differently:

1. Component schema: the tenant registry tables, the segment store
   tables, the vector ledger, the SQLite vector stores' base tables, and
   the native containers of Qdrant and Milvus. Static, versioned,
   migrated.
2. Per-tenant resources: a segment partition (a registry row) and a
   vector collection (a ledger row, plus a table on the SQLite stores).
   Created and removed by jobs, never by the schema step and never by a
   data operation. The SQLite stores' per-key tables are the one place
   serving code runs DDL, inside a provision or reclaim job, after the
   ledger row that records the key.

Component schema:

- Each SQL-backed component owns an Alembic script directory beside its
  code and its own version table (`schema_version_<component>`), so a
  library user composing some components migrates only those. Migrations
  are written from Alembic autogenerate diffs against the component's
  metadata; the metadata is never applied with `create_all`.
- Each vector store owns `provision_containers(config)`, which
  idempotently creates the containers the configuration declares.
- The schema step runs per configured database under a lock:
  `pg_advisory_xact_lock` on PostgreSQL, `BEGIN IMMEDIATE` on SQLite.
  Processes starting concurrently serialize, and all but one find the
  schema current. Containers are created after the database step,
  idempotently, without a lock.
- `server.schema` selects the mode. `migrate` (default; development and
  single-image deployments) runs the step at startup. `verify`
  (production) compares each component's version with its head and
  fails startup on a mismatch; a deploy job runs `memmachine schema
  upgrade --config PATH` first. `skip` does neither.
- Initial setup is an upgrade from an empty database. There is no
  separate path.
- Data migrations over tenant data (re-embedding after changing a
  tenant's embedder, reshaping segments) are jobs of a further action
  kind, one per tenant, run by the reconciler, added when needed.

## What is reused and what is removed

Reused, with the change named:

- `episodic_memory/event_memory/`: `EventMemory`, the segmenters, the
  derivers, the data types, the segment store (UUID keys,
  `reclaim_partition`, ingest order).
- `common/vector_store/`: the four implementations, with the registry
  replaced by the ledger, the `config` parameter removed from
  `create_collection`, and containers provisioned from configuration.
- `common/embedder/`, `common/language_model/`, `common/reranker/` and
  their configuration models, behind the provider catalog.
- `common/filter/`, `common/metrics_factory/`, `common/payload_codec/`.
- `enable_sqlite_foreign_keys` and the engine construction in
  `common/resource_manager/database_manager.py`, as a function of a
  `databases` entry.

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
- #1548: kept; UUID keys, `reclaim_partition` and the ingest order are
  added here.
- #1571: the 409/404 mapping and instance eviction under "Subsystem
  contract".
- #1575, #1576, #1577: resolved by construction: declared subsystems, no
  implicit creation, durable jobs with retry.
- #1572, #1573, #1564, #1565, #1537, #1563: the vector store contract
  above (containers from configuration, the ledger, payload partitioning,
  single-use keys, reclamation after a grace period). In-flight registry
  PRs are measured against that section.
- #1570: "Schema management".
- #1542: SQLite pragmas become fields of a `databases` entry
  (`busy_timeout`, `journal_mode`); the reconciler's retry covers the
  busy-timeout raise.

## Open questions

- Hierarchy: flat tenants with prefix listing, proposed, or a parent
  column with cascading delete as jobs.
- Whether `POST /v1/tenants` also needs a 202 form for deployments whose
  provisioning is slow (the SQLite stores create a table per tenant).
- Event size limits, and block types beyond text (the data types admit
  others).
- The library composition surface: the constructors a user calls to get
  event memory and the tenant service without the HTTP layer.
- Retention: deleting events by age or by producer, as a further job
  kind.
