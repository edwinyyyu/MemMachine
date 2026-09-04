# Ingest service

New component. The write path: records additions and deletions in the
event store, whose log is what memory subsystems replay, then hands
them to each subsystem in the configured order for immediate
processing.

## Constructed with

`IngestService(event_store: EventStore,
subsystems: Sequence[MemorySubsystem])`, where `MemorySubsystem` is the
ABC `EpisodicMemoryManager` implements: `name`, `process`, `forget`.

## API

```python
class IngestService:
    async def ingest(self, tenant_id: UUID, events: Sequence[Event]) -> IngestResult
    async def delete_events(self, tenant_id: UUID, uuids: Iterable[UUID]) -> DeleteResult

class IngestResult(BaseModel):
    stored: list[UUID]
    skipped: list[UUID]
    processing: dict[str, Literal["done", "deferred"]]   # per subsystem
```

- `ingest`: `event_store.add_events(key, events)`, which is durable and
  writes the `added` log entries; then, for each subsystem in order,
  `process(tenant_id, stored)`. A subsystem that raises is reported
  `deferred` and has enqueued its own `catch_up`, which replays the
  log; the client's acknowledgment therefore means "recorded and
  processed at least once, eventually", and "processed now" where
  `done` is reported.
- `delete_events`: `event_store.delete_events(key, uuids)`, which
  removes the event rows and writes `deleted` log entries; then
  `forget` on each subsystem. A subsystem that raises is reported
  `deferred` the same way, and its `catch_up` applies the deletion from
  the log. No outcome depends on the client retrying.
- An unknown tenant raises `KeyNotLiveError` from the event store,
  which the router maps by asking the tenant service.

## Changes to existing code

Replaces `MemMachine.add_episodes` and `delete_episodes`
(`main/memmachine.py:735`, `:1186`) and `server/api_v2/service.py`.
Nothing is carried over.
