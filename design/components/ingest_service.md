# Ingest service

New component. The write path: records events in the event store, then
hands them to each memory subsystem in the configured order.

## Constructed with

`IngestService(event_store: EventStore,
subsystems: Sequence[MemorySubsystem])`,
where `MemorySubsystem` is the protocol `EpisodicMemoryManager`
satisfies: `name`, `process`, `forget`.

## API

```python
class IngestService:
    async def ingest(self, tenant_id: UUID, events: Sequence[Event]) -> IngestResult
    async def delete_events(self, tenant_id: UUID, event_uuids: Iterable[UUID]) -> None

class IngestResult(BaseModel):
    stored: list[UUID]
    skipped: list[UUID]
    processing: dict[str, Literal["done", "deferred"]]   # per subsystem
```

- `ingest`: `event_store.add_events(key, events)`; for each subsystem in
  order, `process(tenant_id, stored)`; a subsystem that raises is
  reported `deferred` and has enqueued its own `catch_up`. A tenant the
  event store does not know raises `KeyNotLiveError`, which the router
  maps by asking the tenant service.
- `delete_events`: `event_store.delete_events` first, then `forget` on
  every subsystem, so a `catch_up` that starts after the first step
  cannot read the event again. A `process` that read the event before
  its deletion may still write derived rows after the forget; where the
  event store and the segment store share a database (the default) the
  segment write's transaction carries `EXISTS (event row)` and writes
  nothing for a deleted event, and where they do not, the window is
  closed by repeating the delete, which always re-runs `forget`. A
  crash between the two steps leaves derived rows for a deleted event;
  the caller's retry removes them.

## Open item

Whether processing runs in the request or always through a queue is the
main document's open question; this component is written for the
former.

## Changes to existing code

Replaces `MemMachine.add_episodes` and `delete_episodes`
(`main/memmachine.py:735`, `:1186`) and `server/api_v2/service.py`.
Nothing is carried over.
