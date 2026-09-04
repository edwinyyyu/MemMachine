# EpisodicMemoryManager

New component; the name is repurposed from the current
`episodic_memory/episodic_memory_manager.py`, which is not carried over.
The resource that stands in for a family of `EpisodicMemory` objects:
builds one per structural configuration, dispatches requests to the
right one, fills per-request defaults, validates tenant configuration,
and registers with the tenant service as the `episodic_memory`
component.

## Constructed with

```python
EpisodicMemoryManager(
    event_store: EventStore,
    segment_store: SegmentStore,
    vector_store: VectorStore,
    embedders: Mapping[str, Embedder],     # resources, all the deployment built
    rerankers: Mapping[str, Reranker],
    engine: AsyncEngine,                   # its per-tenant table
    settings: EpisodicMemorySettings,      # offered subsets, filter bounds, cache size
    metrics_factory: MetricsFactory | None,
)
```

The mappings hold resources; products of the manager never appear in
settings. `settings.embedders` and `settings.rerankers`, when given,
restrict the offered ids to a subset of the mappings.

## Tenant configuration model

```python
class EpisodicMemoryTenantConfiguration(BaseModel):
    embedder: str                       # immutable; an offered id
    segmenter: SegmenterOptions         # mutable; later events
    deriver: DeriverOptions             # mutable; later events
    reranker: str | None                # mutable; default for query
    search: SearchDefaults              # mutable; defaults for query
```

`SearchDefaults` has exactly the fields of `query`'s per-request
parameters (`limit`, `expand_context`, `min_score`, `include_events`),
checked at import time against `EpisodicMemory.query`'s signature. The
structural key of a configuration is `(embedder, segmenter, deriver)`.

## Schema

`episodic_memory_tenants`:

| column | type | constraint |
| --- | --- | --- |
| `tenant_id` | `Uuid` | primary key |
| `watermark` | `BigInteger` | not null, default 0; the last log position processed |
| `configuration` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `configuration_version` | `Integer` | not null |
| `updated_at` | `DateTime(timezone=True)` | not null, `func.now()` |

No other index: every access is by primary key. The watermark is
written with `SET watermark = GREATEST(watermark, ?)` (`MAX` on
SQLite), so it moves only forward.

## API

Toward the tenant service (the `TenantComponent` protocol):

- `provision(tenant_id, section)`: `segment_store.create_partition` and
  `vector_store.create_collection(key, container=section.embedder)`,
  each treating its own `live` row as success and resuming `creating`;
  insert or update the per-tenant row with the section and version.
- `delete(tenant_id)`: `segment_store.delete_partition`,
  `vector_store.delete_collection`, remove the per-tenant row.
- `reclaim(tenant_id)`: `reclaim_partition` and `reclaim_collection`;
  `DONE` when both are.
- `validate_update`: `embedder` changed raises.
- `job_kinds()`: `catch_up`.

Toward the ingest service and the routers:

```python
    async def process(self, tenant_id: UUID, events: Sequence[StoredEvent]) -> ProcessingStatus
    async def forget(self, tenant_id: UUID, event_uuids: Iterable[UUID]) -> None
    async def search(self, tenant_id: UUID, request: SearchRequest) -> SearchResponse
    async def status(self, tenant_id: UUID) -> SubsystemStatus   # watermark, lag
```

Each reads the per-tenant row (absent: `TenantNotFoundError`, which the
router turns into 404 or 409 by asking the tenant service), takes the
`EpisodicMemory` for the row's structural key from the cache, building
it on a miss with `embedders[e]`, `vector_store.for_container(e)` and
segmenter and deriver objects from the options, and makes one call.
`search` fills each per-request field the request omits from the row's
defaults, resolves `request.reranker` or the default to an object
(`InvalidTenantConfigurationError` for an id not offered), and calls
`query`. `process` advances the watermark to the last position in the
same transaction as its last segment write where the engines are the
same, and after it otherwise; on an exception it enqueues `catch_up`
and reports `deferred`.

`catch_up(tenant_id, payload) -> Progress`: replay the event store's
log from `min(payload.from_position, watermark)`: `read_log(key,
after, batch)`, then for each entry `process` the event of an `added`
entry that still has one, or `forget` the uuid of a `deleted` entry;
advance the watermark past the batch; `MORE` while `head(key)` is past
the watermark. Both kinds of entry are handled, so a deletion a client
was acknowledged for is applied to derived data at least once without
the client retrying.

## Cache

Keyed by structural configuration, never by tenant; bounded by
`settings.cache_size`; an entry is a few references, rebuilt in
microseconds on a miss.

## What it does not do

No search or ingest logic, no model translation, no per-tenant state
beyond its table, no reading of the tenant table.

## Changes to existing code

Replaces `EpisodicMemoryManager` and `MemoryInstanceCache`
(`episodic_memory/episodic_memory_manager.py:63`,
`instance_lru_cache.py:32`), `EpisodicMemory` and `LongTermMemory` as
facades (`episodic_memory/episodic_memory.py:94`,
`long_term_memory/long_term_memory.py:177`), and
`long_term_memory/service_locator.py`. Nothing is carried over.

## Data-path races

The watermark and `catch_up` are the parts that need a rule. Two
processes may ingest into one tenant at once, positions come from one
sequence, and a batch may fail after a later batch succeeded.

- Watermark semantics: every position at or below the watermark has
  been processed at least once, except positions named by a pending
  `catch_up` job. The watermark only moves forward: it is written with
  `SET watermark = GREATEST(watermark, ?)`.
- On a `process` failure the manager enqueues `catch_up` with
  `from_position` = the lowest position of the failed batch; if a
  `catch_up` job is already pending, the row is reset to the lower of
  the two. The handler processes from `from_position` to the head,
  idempotently per event, and returns `MORE` while newer events exist.
- Read-your-writes: an acknowledged ingest is durable in the event
  store; it is visible to search after the vector backend has indexed
  it, which is the backend's consistency, not this design's.

| First | Concurrent | Outcome |
| --- | --- | --- |
| ingest batch A (positions 1..10) | ingest batch B (11..20) on another process | the event store serializes the two under the key's exclusive lock, so positions are contiguous and commit-ordered; each batch is processed by its own process; the watermark ends at 20 by `GREATEST` whatever the processing order |
| batch A fails after batch B advanced the watermark to 20 | | `catch_up(from_position=1)` reprocesses 1..20 idempotently; nothing is skipped |
| ingest of event uuid U | ingest of U on another process | unique `(key, uuid)`: one stores, the other reports U skipped; only the storing process processes it |
| delete of event U | `process` that already read U | the delete appends a `deleted` log entry after U's `added` entry; a derived write that lands after the request path's `forget` is removed when the `deleted` entry is replayed, by `catch_up` or by the next request path that reaches it; nothing depends on the client retrying |
| search | ingest | the search sees what the backends have indexed; no guarantee about the batch in flight |
| `process` | `catch_up` on the same tenant | serialized by the tenant row lock for the step; `process` in a request does not take that lock, so the two may both write an event's derived rows; `process` is idempotent per event (forget first), so the later write leaves one copy |
