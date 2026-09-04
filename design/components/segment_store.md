# Segment store

Existing component, `episodic_memory/event_memory/segment_store/`, as
shipped in #1548 (`design/segment_store_shared_tables.md`). Derived
data: segments and their derivative links, rebuildable from the event
store. This file lists what changes; everything not listed stays.

## Constructed with

- `SQLAlchemySegmentStore(engine: AsyncEngine,
  settings: SegmentStoreSettings)`;
  settings: `purge_max_segments`, `purge_max_partitions`, `payload_codec`
  defaults.

## Storage, after the changes

`segment_store_pt`: `key UUID PK`, `payload_codec_config JSON`,
`created_at`. `segment_store_sg`: `key UUID`, `uuid UUID`, `event_uuid
UUID`, `index`, `offset`, `timestamp`, `timestamp_timezone_offset`,
`context BLOB`, `block BLOB`, `properties JSON`; primary key `(key,
uuid)`. `segment_store_dv_ln`: `key UUID`, `uuid UUID`, `segment_uuid
UUID`, foreign key to the segment row with cascade. `segment_store_gc`:
`key UUID PK`, `enqueued_at`.

## API, after the changes

Two ABCs behind one implementation, exposed as the manager and
`manager.store`, so data callers cannot reach lifecycle and lifecycle
callers cannot reach data.

```python
class SegmentStoreManager(ABC):
    async def create_partition(self, key: UUID, config: SegmentStorePartitionConfig) -> None
    async def delete_partition(self, key: UUID) -> None
    async def reclaim_partition(self, key: UUID) -> Progress
    async def purge_deleted_partitions(self) -> bool     # library use only
    @property
    def concurrency_scope(self) -> ConcurrencyScope

class SegmentStore(ABC):
    async def add_segments(self, key: UUID,
                           segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]]) -> None
    async def get_segment_contexts(self, key: UUID, seed_segment_uuids: Iterable[UUID], *,
                                   max_backward_segments: int, max_forward_segments: int,
                                   since: datetime | None, before: datetime | None,
                                   property_filter: FilterExpr | None) -> dict[UUID, list[Segment]]
    async def get_neighbours(self, key: UUID, anchor: UUID, *,
                             before: int, after: int,
                             producers: Iterable[str] | None) -> list[Segment]
    async def get_segment_uuids_by_event_uuids(self, key: UUID,
                                               event_uuids: Iterable[UUID]) -> dict[UUID, list[UUID]]
    async def get_derivative_uuids_by_segment_uuids(self, key: UUID,
                                                    segment_uuids: Iterable[UUID]) -> dict[UUID, list[UUID]]
    async def find_segments(self, key: UUID, property_filter: FilterExpr,
                            limit: int) -> list[UUID]
    async def delete_segments(self, key: UUID, segment_uuids: Iterable[UUID]) -> None
```

`get_neighbours` serves expansion (`episodic_memory.md`): the segments
ordered by `(timestamp, event_uuid, index, offset)` within the key, the
`before` segments preceding the anchor and the `after` following it,
optionally restricted to producers; the anchor itself is included. The
order is total and stable, so a caller can walk by repeating the call
from the last segment returned.

## Changes required

- Key type `UUID` (`sqlalchemy_segment_store.py:145`, `String(255)`),
  and `validate_partition_key`, `PARTITION_KEY_MAX_BYTES` and
  `partition_key_for_session` (`long_term_memory/service_locator.py:166`)
  go.
- The incarnation goes: the `incarnation` column of every table
  (`:146`, `:158`, `:201`, `:234`) becomes the key, the registry row's
  unique incarnation goes, the purge queue is keyed by the key, the
  physical-key helper in `utils.py` goes, and the store mints nothing.
  Rationale in `server_redesign.md`, "Segment store".
- `SegmentStorePartition` (`segment_store.py:20`) and `open_partition`,
  `open_or_create_partition`, `close_partition` (`:176`, `:191`, `:220`)
  go; every data operation takes the key and the registry read that
  fences it returns the codec configuration; codec objects are cached
  process-wide by configuration.
- `create_partition` stays strict and also raises on a key whose purge
  is pending (a queue entry under the key).
- `reclaim_partition(key) -> Progress` is added: this key's dead rows,
  bounded by `purge_max_segments`; `DONE` when none remain. It is what
  the delete job and the tombstone sweep call; `purge_deleted_partitions`
  stays for library users and the server does not run it.
- `get_segment_contexts` gains `since` and `before` on the real
  `timestamp` column, as on the reference branch (commit 27b3279b), and
  the reserved timestamp property key goes from the segment side.
- `find_segments` is added for the selectivity probe under
  `filters_and_properties.md`: segments matching a property filter, up
  to `limit + 1`, so the caller can tell "selective" from "broad".
- `get_neighbours` is added for expansion, over the ordering index.
- The ABC splits into `SegmentStoreManager` and `SegmentStore`.
- Errors: `SegmentStorePartitionHandleStaleError` becomes
  `KeyNotLiveError`; `SegmentStorePartitionAlreadyExistsError` becomes
  `KeyExistsError`; `SegmentStoreAttemptsExhaustedError` becomes
  `AttemptsExhaustedError`; `SegmentStorePartitionConfigMismatchError`
  goes with open-or-create.
- Fencing is unchanged: writes `FOR SHARE` the registry row for the
  transaction, the logical delete takes it `FOR UPDATE`, reads carry
  the liveness predicate; on SQLite `BEGIN IMMEDIATE`.
- Segmenter and deriver contracts gain a clause: a segment carries a
  verbatim copy of its event's properties, and a derivative of its
  segment's.

## Schema, after the changes

`segment_store_pt`, the registry row (the fence):

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key |
| `payload_codec_config` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `created_at` | `DateTime(timezone=True)` | not null, `func.now()` |

`segment_store_sg`, the segments:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part |
| `event_uuid` | `Uuid` | not null |
| `index` | `Integer` | not null |
| `offset` | `Integer` | not null |
| `timestamp` | `DateTime(timezone=True)` | not null, UTC |
| `timestamp_timezone_offset` | `Integer` | not null, minutes |
| `context` | `LargeBinary` | not null, codec-encoded |
| `block` | `LargeBinary` | not null, codec-encoded |
| `properties` | `JSON` (`JSONB` on PostgreSQL) | not null |

Indexes: `segment_store_sg__key_event (key, event_uuid, index, offset)`
for lookup by event; `segment_store_sg__key_order (key, timestamp,
event_uuid, index, offset)` for context windows, expansion and `since`
and `before`, which is the one total order the store exposes; a GIN index on
`properties` on PostgreSQL, added by a deployment as its undeclared-key filters
need.

`segment_store_dv_ln`, the derivative links:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part, the derivative uuid |
| `segment_uuid` | `Uuid` | not null; foreign key `(key, segment_uuid)` to `segment_store_sg (key, uuid)` `ON DELETE CASCADE` |

Index: `segment_store_dv_ln__key_segment (key, segment_uuid)`, which the
cascade and `get_derivative_uuids_by_segment_uuids` use.

`segment_store_gc`, the purge queue: `key Uuid` primary key,
`enqueued_at DateTime(timezone=True)` not null `func.now()`, index
`segment_store_gc__enqueued_at`.

No foreign key from the data tables to the registry row, so the logical
delete is O(1); the link table's cascade from segments is kept, and an
engine that does not enforce it leaves link rows the purge removes with
a warning, as today.
