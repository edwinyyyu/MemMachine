# Event store

New component. The system of record: a tenant's events as ingested,
plus a per-tenant log of additions and deletions in position order,
which memory subsystems replay. Never derived from anything.

## Constructed with

`SqlAlchemyEventStore(engine: AsyncEngine, settings: EventStoreSettings)`;
settings: `blocks.max_bytes`, `properties.max_keys`,
`properties.max_string_bytes`, `payload_codec` (codec settings for
blocks and context).

## Types

`Event`, `Block`, `Context` from `event_memory/data_types.py`
(`Event.uuid: UUID`). On the read side:

```python
class LogEntry(BaseModel):
    position: int                  # contiguous and commit-ordered within a key
    kind: Literal["added", "deleted"]
    uuid: UUID
    event: Event | None            # the content for "added", unless deleted since
```

## Schema

`event_store_pt`, the registry row (the fence):

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key |
| `codec_config` | `JSON` | not null |
| `next_position` | `BigInteger` | not null, default 1 |
| `created_at` | `DateTime(timezone=True)` | not null, `func.now()` |

`event_store_ev`, the events:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `uuid` | `Uuid` | primary key part |
| `timestamp` | `DateTime(timezone=True)` | not null, UTC |
| `context` | `LargeBinary` | not null, codec-encoded |
| `properties` | `JSON` (`JSONB` on PostgreSQL) | not null |
| `blocks` | `LargeBinary` | not null, codec-encoded |
| `ingested_at` | `DateTime(timezone=True)` | not null, `func.now()` |

Indexes: `event_store_ev__key_timestamp (key, timestamp)` for
`list_events` by time; a GIN index on `properties` on PostgreSQL,
added by a deployment as it needs, since undeclared keys are filtered
here.

`event_store_lg`, the log:

| column | type | constraint |
| --- | --- | --- |
| `key` | `Uuid` | primary key part |
| `position` | `BigInteger` | primary key part |
| `kind` | `String(8)` | check in (`added`, `deleted`) |
| `uuid` | `Uuid` | not null |
| `at` | `DateTime(timezone=True)` | not null, `func.now()` |

No foreign key from `event_store_lg` or `event_store_ev` to the
registry row: the logical delete removes the row in O(1) and the purge
reclaims by key. No foreign key between the log and the events: an
`added` entry outlives its event once the event is deleted, by design.

`event_store_gc`, the purge queue: `key Uuid` primary key, `enqueued_at
DateTime(timezone=True)` not null `func.now()`, index
`event_store_gc__enqueued_at`.

## API

```python
class EventStore(ABC):
    async def create_partition(self, key: UUID, config: EventPartitionConfig) -> None
    async def delete_partition(self, key: UUID) -> None
    async def reclaim_partition(self, key: UUID) -> Progress
    async def purge_deleted_partitions(self) -> bool       # library use only

    async def add_events(self, key: UUID, events: Iterable[Event]) -> AddResult
    async def delete_events(self, key: UUID, uuids: Iterable[UUID]) -> None
    async def get_events(self, key: UUID, uuids: Iterable[UUID]) -> list[Event]
    async def list_events(self, key: UUID, filter: FilterExpr | None,
                          since: datetime | None, before: datetime | None,
                          cursor: Cursor | None, limit: int) -> Page[Event]
    async def read_log(self, key: UUID, after: int, limit: int) -> list[LogEntry]
    async def head(self, key: UUID) -> int                 # last position
```

- `create_partition`: strict; `KeyExistsError` on any row under the key,
  including a purge queue entry.
- `delete_partition`: one transaction: lock the row `FOR UPDATE`, enqueue
  the key, remove the row. O(1), idempotent.
- `reclaim_partition`: delete a bounded number of the key's rows across
  the three data tables; remove the queue entry when none remain;
  `DONE` then.
- `add_events`: one transaction that locks the registry row `FOR
  UPDATE`, so ingests to one tenant serialize at the event store and
  positions are contiguous and commit-ordered within the key; validates
  every event (block sizes, property keys and values, reserved keys,
  raising `InvalidPropertyKeyError` or `InvalidPropertyValueError`);
  inserts each new event and one `added` log entry, taking positions
  from `next_position` and advancing it; events whose uuid exists under
  the key are skipped and returned in `AddResult.skipped`. The lock is
  exclusive rather than shared because positions must be commit-ordered;
  its cost is that a tenant's ingests do not overlap inside this one
  short transaction.
- `delete_events`: the same lock; remove the event rows; append one
  `deleted` log entry per uuid that existed. Idempotent: a uuid with no
  row appends nothing.
- `read_log`: entries with `position > after`, ascending, at most
  `limit`; an `added` entry carries the event unless it has since been
  deleted, in which case `event` is `None` and a subsystem skips it.
- Reads carry `EXISTS (registry row)`; a read on a dead key raises
  `KeyNotLiveError`.

## What positions are for

A position is the log's order within a tenant, contiguous and
commit-ordered. It exists so that a subsystem's progress is one integer
per tenant (the watermark), so that "what is left to process" is a
range the subsystem reads from that integer, so that lag is a
subtraction (`head` minus watermark, the status endpoint), and so that
an event's addition is always replayed before its deletion. Positions
are internal: events are addressed by uuid everywhere else, and a
position is never a request parameter. Commit order is what makes
"read after p" exact, and it is why `add_events` takes the key's row
exclusively rather than a sequence.

## Why a log

A subsystem replays the log from its watermark, so every addition and
deletion a client was acknowledged for reaches every subsystem at least
once without the client retrying: the request path's immediate
processing is a latency optimization and `catch_up` is the guarantee.
The log is the per-tenant data queue; an entry and its event are one
transaction, which a broker could not join without an outbox on top.

## Concurrency scope

`cluster` on PostgreSQL; `host` on a SQLite file; `process` in memory.

## Changes to existing code

Replaces `common/episode_store/` (`EpisodeStorage`, the `episodestore`
table, `CountCachingEpisodeStorage`) and the `Episode` model. Not a
rename: the event's shape is `event_memory/data_types.py`'s `Event`,
blocks and context are codec-encoded, and the log exists. Nothing is
carried over.
