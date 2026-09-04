# Event store

New component. The system of record: a tenant's events as ingested, in
position order. Read by memory subsystems to process and to repair;
never derived from anything.

## Constructed with

- `SqlAlchemyEventStore(engine: AsyncEngine, settings: EventStoreSettings)`;
  settings: `blocks.max_bytes`, `properties.max_keys`,
  `properties.max_string_bytes`, `payload_codec` (the codec settings for
  blocks and context).

## Types

`Event`, `Block`, `Context` from `event_memory/data_types.py`, with
`Event.uuid: UUID` and `position: int` added on the stored form:

```python
class StoredEvent(Event):
    position: int          # strictly increasing within a key, in commit order
```

## Storage

Registry row per key, beside the data (the fence):

`event_store_pt`: `key UUID PK`, `state` (`live`, `dropping`),
`codec_config JSON`, `created_at`.

`event_store_ev`: `key UUID`, `uuid UUID`, `position BIGINT` (from one
sequence), `timestamp`, `context BLOB` (codec-encoded), `properties
JSON`, `blocks BLOB` (codec-encoded), `ingested_at`. Primary key
`(key, uuid)`; unique `(key, position)`.

`event_store_gc`: `key UUID PK`, `enqueued_at` (database clock).

## API

```python
class EventStore(ABC):
    async def create_partition(self, key: UUID, config: EventPartitionConfig) -> None
    async def delete_partition(self, key: UUID) -> None
    async def reclaim_partition(self, key: UUID) -> Progress
    async def purge_deleted_partitions(self) -> bool     # library use only

    async def add_events(self, key: UUID, events: Iterable[Event]) -> AddResult
    async def get_events(self, key: UUID, uuids: Iterable[UUID]) -> list[StoredEvent]
    async def list_events(self, key: UUID, filter: FilterExpr | None,
                          cursor: Cursor | None, limit: int) -> Page[StoredEvent]
    async def read_after(self, key: UUID, position: int, limit: int) -> list[StoredEvent]
    async def delete_events(self, key: UUID, uuids: Iterable[UUID]) -> None
```

- `create_partition`: strict; `KeyExistsError` on any row under the key,
  including one awaiting purge.
- `delete_partition`: one transaction: lock the row `FOR UPDATE`, enqueue
  the key, remove the row. O(1), idempotent.
- `reclaim_partition`: delete a bounded number of the key's rows; remove
  the queue entry when none remain; `DONE` then. On SQLite the delete
  waits up to `busy_timeout` and raises past it.
- `add_events`: validates every event (block sizes, property keys and
  values, reserved keys), assigns positions, inserts; events whose uuid
  exists under the key are skipped and returned in `AddResult.skipped`.
  One transaction; the fence's `FOR SHARE` on the registry row.
- Reads carry `EXISTS (registry row live)`; a read on a dead key raises
  `KeyNotLiveError`.

## Concurrency scope

`cluster` on PostgreSQL; `host` on a SQLite file; `process` in memory.

## Changes to existing code

Replaces `common/episode_store/` (`EpisodeStorage`, the `episodestore`
table, `CountCachingEpisodeStorage`) and the `Episode` model. Not a
rename: the event's shape is `event_memory/data_types.py`'s `Event`,
blocks and context are codec-encoded, and positions exist. Nothing is
carried over.
