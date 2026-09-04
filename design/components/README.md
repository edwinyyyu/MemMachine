# Component specifications

One file per component of `design/server_redesign.md`. Each states the
component's purpose, what it is constructed with, its types and API, its
storage and fencing where it has any, its settings, and, for a component
that exists today, the changes required of it with file references to
`speedkick` at 7752e4cb under `packages/server/src/memmachine_server`.

Conventions shared by every specification:

- Identifiers are `uuid.UUID` values in every signature, row and model,
  never strings. A backend that wants a string receives the 32 lowercase
  hex characters of the UUID, rendered at the store's boundary and
  nowhere else.
- A key is a tenant id. Stores take it and nothing else, fence on it,
  and never mint an identity of their own.
- `Progress` is an enum with two members, `DONE` and `MORE`, returned by
  every bounded, repeatable step. `MORE` means "call again"; `DONE`
  means "nothing remains". A step may raise instead, and its caller
  retries.
- Errors are one hierarchy under `MemMachineError`, named for the
  condition, never for a driver: `KeyExistsError`, `KeyNotLiveError`,
  `KeyReusedError`, `TenantNotFoundError`, `TenantNotActiveError`,
  `TenantExistsError`, `InvalidTenantConfigurationError`,
  `UndeclaredPropertyKeyError`, `ProviderUnavailableError`,
  `AttemptsExhaustedError`. The HTTP layer maps the hierarchy once.
- Every method that is a hook or a bounded step is idempotent: calling
  it again after any partial outcome completes it.
- Settings are Pydantic models, one per component, named `<Component>
  Settings`; no numeric default is given here, since none has been
  measured.
- Nothing per tenant is opened, closed or held: an operation takes the
  key, reads what it needs, and returns.

Files:

- `tenant_service.md`: the tenant registry, jobs, the reconciler role,
  the tombstone sweep, the component registration protocol.
- `key_registry.md`: per-key bookkeeping for stores whose data is not
  in SQL, and the scoped view a store receives.
- `event_store.md`: the system of record for events.
- `segment_store.md`: the segment store as shipped in #1548 and what
  changes.
- `vector_store.md`: the vector store contract, its registry rows,
  containers, declared properties, and per-backend notes.
- `episodic_memory.md`: `EpisodicMemory`, the current `EventMemory`.
- `episodic_memory_manager.md`: the resource that builds and dispatches
  to `EpisodicMemory` objects and registers with the tenant service.
- `ingest_service.md`: the write path from the API to the event store
  and the subsystems.
- `filters_and_properties.md`: the filter expression tree, property
  keys and values, and the reserved namespace.
- `server_and_settings.md`: the `Server` object, roles, routers, error
  mapping, and the settings models.
