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

```python
class SegmentStore(ABC):
    async def create_partition(self, key: UUID, config: SegmentStorePartitionConfig) -> None
    async def delete_partition(self, key: UUID) -> None
    async def reclaim_partition(self, key: UUID) -> Progress
    async def purge_deleted_partitions(self) -> bool     # library use only

    async def add_segments(self, key: UUID,
                           segments_to_derivative_uuids: Mapping[Segment, Iterable[UUID]]) -> None
    async def get_segment_contexts(self, key: UUID, seed_segment_uuids: Iterable[UUID], *,
                                   max_backward_segments: int, max_forward_segments: int,
                                   since: datetime | None, before: datetime | None,
                                   property_filter: FilterExpr | None) -> dict[UUID, list[Segment]]
    async def get_segment_uuids_by_event_uuids(self, key: UUID,
                                               event_uuids: Iterable[UUID]) -> dict[UUID, list[UUID]]
    async def get_derivative_uuids_by_segment_uuids(self, key: UUID,
                                                    segment_uuids: Iterable[UUID]) -> dict[UUID, list[UUID]]
    async def find_segments(self, key: UUID, property_filter: FilterExpr,
                            limit: int) -> list[UUID]
    async def delete_segments(self, key: UUID, segment_uuids: Iterable[UUID]) -> None
```

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
