# Tenant service

New component. Control plane: creates, renames, configures and deletes
tenants; records jobs; runs the reconciler role and the tombstone sweep;
knows components only through their registrations. The only reader and
writer of the tenant tables.

## Constructed with

- `engine: AsyncEngine`, the tenant database.
- `components: Sequence[TenantComponent]`, the registrations (below).
- `templates: Mapping[str, TenantTemplate]`, validated at construction
  against every component's `tenant_configuration` model.
- `settings: TenantServiceSettings`: `reconciler.poll_interval`,
  `reconciler.jobs_per_pass`, `reconciler.reclaim_interval`,
  `reconciler.backoff`, `reconciler.stuck_after`,
  `reconciler.sweep_interval`, `tombstones.retention`.

## Component registration protocol

```python
class TenantComponent(Protocol):
    name: str
    tenant_configuration: type[BaseModel]   # fields marked mutable or immutable

    async def provision(self, tenant_id: UUID, section: BaseModel) -> None: ...
    async def delete(self, tenant_id: UUID) -> None: ...
    async def reclaim(self, tenant_id: UUID) -> Progress: ...
    def validate_update(self, old: BaseModel, new: BaseModel) -> None: ...
    def job_kinds(self) -> Mapping[str, JobHandler]: ...   # component-defined
```

- `provision(tenant_id, section)`: idempotent by provenance. Create the
  component's resources under the key if absent (a `live` row is its
  own earlier attempt; a `creating` row is resumed; a `dropping` row
  raises `KeyReusedError`), verify immutable options if present, apply
  mutable ones, record the section and its version in the component's
  own per-tenant table. Complete on return.
- `delete(tenant_id)`: idempotent; make the tenant unreachable in every
  one of the component's stores (each store's logical delete) and
  remove its per-tenant row. Fast; no reclamation.
- `reclaim(tenant_id) -> Progress`: remove a bounded amount of what the
  component's stores hold under the key; `DONE` when nothing is found.
  Called by the delete job after `delete`, and by the tombstone sweep.
  Never called on a live key (see "Serialization").
- `validate_update(old, new)`: raise `InvalidTenantConfigurationError`
  if `new` changes an immutable field.
- `job_kinds()`: component-defined actions the reconciler executes,
  each `JobHandler = Callable[[UUID, dict], Awaitable[Progress]]`;
  `catch_up` for episodic memory.

The tenant service imports nothing from a component; a component
imports this protocol and nothing else from the tenant package.

## Storage

`tenants`:

| column | type | note |
| --- | --- | --- |
| `id` | `UUID PK` | minted `uuid4` at create; never reused |
| `name` | `TEXT NULL`, unique | NULL from the moment deletion starts |
| `former_name` | `TEXT NULL` | the name at deletion, for operators |
| `state` | enum | `provisioning`, `active`, `deleting`, `deleted` |
| `configuration` | `JSON` | one object per component name; the record of what was requested |
| `configuration_version` | `INTEGER` | incremented by every update |
| `created_at`, `updated_at`, `deleted_at`, `swept_at` | timestamps | database clock |

`tenant_jobs`:

| column | type | note |
| --- | --- | --- |
| `id` | `PK` | |
| `tenant_id` | `UUID` | |
| `component` | `TEXT` | registration name |
| `action` | `TEXT` | `provision`, `delete`, or a component-defined kind |
| `payload` | `JSON` | for `provision`, the configuration version |
| `state` | enum | `pending`, `done` |
| `attempts`, `last_error`, `next_run_at`, `created_at`, `updated_at` | | |

Unique on `(tenant_id, component, action)`.

## API

```python
class TenantService:
    async def create(self, name: str, template: str,
                     overrides: Mapping[str, Mapping]) -> Tenant
    async def get(self, tenant_id: UUID) -> Tenant
    async def get_by_name(self, name: str) -> Tenant
    async def list(self, prefix: str | None, cursor: Cursor | None,
                   limit: int) -> Page[Tenant]
    async def rename(self, tenant_id: UUID, name: str) -> Tenant
    async def update_configuration(self, tenant_id: UUID,
                                   overrides: Mapping[str, Mapping]) -> Tenant
    async def delete(self, tenant_id: UUID) -> Tenant
    async def wait(self, tenant_id: UUID, until: TenantState,
                   timeout: timedelta) -> Tenant
    async def enqueue(self, tenant_id: UUID, component: str,
                      action: str, payload: Mapping) -> None
    async def state_of(self, tenant_id: UUID) -> TenantState | None
    async def run_reconciler(self) -> None      # the role's loop
    async def reconcile_tenant(self, tenant_id: UUID) -> None
```

`Tenant` carries the row plus, per component, the applied configuration
version reported by the component, and the jobs with attempts and last
error. `state_of` is what a router calls on a subsystem miss to answer
404 or 409; it is the one method of the service a router calls on the
data path.

Semantics:

- `create`: resolve the template overlaid with overrides, validate each
  section with its component's model (unknown component or invalid
  option: `InvalidTenantConfigurationError`), insert the row as
  `provisioning` and one `provision` job per component in one
  transaction; a duplicate name raises `TenantExistsError`, nothing
  more. Then `reconcile_tenant` inline, so a single process finishes the
  create in the request; a failing step is left pending.
- `update_configuration`: validate with `validate_update` per changed
  section; one transaction writes the document, increments the version,
  inserts or resets a `provision` job per changed component with the
  version in its payload; the tenant stays `active`.
- `delete`: one transaction, under the tenant row's lock: `state =
  deleting`, `former_name = name`, `name = NULL`, `deleted_at = now()`;
  mark every pending `provision` and component-defined job done; insert
  one `delete` job per component. Allowed from `provisioning` and
  `active`; idempotent while `deleting`.
- `wait`: poll the row until the state is reached or the timeout
  elapses; used by `?wait=`.

## Reconciler role

- Claim: `SELECT ... FROM tenant_jobs WHERE state = 'pending' AND
  next_run_at <= now() ORDER BY next_run_at LIMIT n FOR UPDATE SKIP
  LOCKED`, then `SELECT ... FROM tenants WHERE id = ? FOR UPDATE`, both
  held for the step. On SQLite, `BEGIN IMMEDIATE` and no `SKIP LOCKED`.
- Serialization: every lifecycle transition and every step holds the
  tenant row's lock, so transitions and steps for one tenant are totally
  ordered. A step re-reads the state under the lock: a `provision` or
  component-defined step on a `deleting` tenant marks itself done and
  calls nothing. A delete request waits for a running step, then
  cancels pending `provision` jobs. No `provision` hook runs after a
  `delete` hook; `reclaim` never sees a live key.
- Execute: `provision` calls the hook with the section at the payload's
  version and marks done; `delete` calls `delete` then `reclaim`, `DONE`
  marking the job done and `MORE` rescheduling at `now() +
  reclaim_interval`; a component-defined kind calls its handler the
  same way. An exception reschedules with backoff and records
  `attempts` and `last_error`. The transaction that marks the last
  `provision` job done sets `active`; the one that marks the last
  `delete` job done removes the job rows and sets `deleted`.
- Tombstone sweep, every `sweep_interval`, in every reconciler process
  without exclusion: claim `deleted` rows with `FOR UPDATE SKIP LOCKED`,
  oldest `swept_at` first, bounded per call; call every component's
  `reclaim`; stamp `swept_at`; when every component returned `DONE` and
  `deleted_at < now() - tombstones.retention` on the database clock,
  delete the row. The retention assumes every remote client has a
  request timeout; the composition refuses one without.
- Cost: one connection per executing job, bounded by `jobs_per_pass`.

## Concurrency scope

`cluster` on PostgreSQL; `host` on a SQLite file; `process` on in-memory
SQLite.

## Changes to existing code

Replaces `common/session_manager/` (`SessionDataManager`, the `sessions`
and `short_term_memory_data` tables), `MemMachine.delete_session` and
`_delete_session_worker` (`main/memmachine.py:343`, `:635`), and the
implicit creation in `add_episodes` and `_search_episodic_memory`
(`:767`, `:826`). Nothing is carried over.

## Race matrix

Every concurrent pair on one tenant, and the defined outcome. "Serialized"
means the tenant row's lock orders the two and the second re-reads state.

| First | Concurrent | Outcome |
| --- | --- | --- |
| create, name N | create, name N | unique index: the second raises `TenantExistsError`; no store touched |
| create (id from a restored registry or a library caller) | any | primary key: `TenantExistsError`; no store touched |
| provision step running | delete request | delete waits for the step (tenant row lock), then marks pending `provision` jobs done and inserts `delete` jobs; later `provision` steps see `deleting` and mark themselves done; created resources are reclaimed by the delete jobs |
| delete request | delete request | serialized; the second finds `deleting`, inserts nothing, responds the same 202 |
| provision step | delete step, same tenant | cannot overlap: serialized, and the delete request already cancelled pending `provision` jobs; no `provision` hook runs after a `delete` hook |
| configuration update | delete request | serialized; update after delete raises `TenantNotActiveError`; update before delete leaves a `provision` job that sees `deleting` and marks itself done |
| configuration update | configuration update | serialized; both versions recorded in order; the `provision` job row is reset to the later version (a running step finishes first, then the reset lands); applied version is the latest |
| rename | create with that name | unique index; the loser raises `TenantExistsError` |
| rename | delete | serialized; rename on `deleting` raises `TenantNotActiveError` |
| any step | reconciler crash mid-step | the database releases the job and tenant row locks; the job stays `pending` and is claimable at once; hooks are idempotent; a half-done `provision` resumes from `creating` rows |
| create request executing steps inline | api process crash | same as above; a reconciler finishes; the caller's `?wait=` fails and it polls |
| claim | claim, other reconciler | `FOR UPDATE SKIP LOCKED`: a job has one executor; on SQLite the file lock serializes |
| delete step | tombstone sweep | never overlap: a row is `deleted` only after every `delete` job is done |
| tombstone sweep | tombstone sweep, other reconciler | `SKIP LOCKED` on tombstone rows; a row has one sweeper per pass |
| `catch_up` step | delete request | serialized; a `catch_up` step on `deleting` marks itself done |
| `catch_up` step | tenant still `provisioning` | rescheduled with backoff; `catch_up` runs only on `active` |
| data operation | delete commit | store fences: in-statement for SQL stores, check-after for the rest; the operation raises `KeyNotLiveError`, the router answers 409 while the row exists and 404 after; a remote write already sent is reclaimed by the delete job or the sweep |
| data operation | tenant `provisioning` | the component's per-tenant row is written at the end of `provision`, so the operation finds no row and the router answers 409 `tenant_not_active` |
| data operation | configuration update | the operation uses the row it read at its start; the next request uses the new configuration |
| `schema upgrade` | `serve` starting | `serve` verifies and fails if not at head; a deployment orders upgrade before serve; during a rollout, expand/contract keeps both releases valid |
