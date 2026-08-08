# Refcount System with uuid5 Dedup — Locking Rules

## Context

Derivatives have deterministic UUIDs via `uuid5(namespace, json.dumps([partition_key, text]))`. Multiple concurrent registrations can produce the same derivative UUID. The `active` parameter is removed — `register_segments` does not know which derivatives are new vs existing. ON CONFLICT DO NOTHING handles uuid5 collisions at insert time.

---

## API

```python
async def register_segments(
    self,
    links: Mapping[Segment, Iterable[UUID]],
) -> None:
```

No `active` parameter. The caller provides segments and their derivative UUIDs. The implementation determines new vs existing internally.

---

## Registration (`register_segments`)

### Steps

1. **Compute link counts** per derivative from the `links` mapping.

2. **Insert all derivatives** with ON CONFLICT DO NOTHING, `ref_count = 0`, `state = ACTIVE`.
   - New derivatives: inserted with `ref_count = 0`.
   - Existing derivatives (uuid5 collision): DO NOTHING — their row already exists with its current `ref_count`.
   - Sorted by UUID for consistent insert ordering (deadlock prevention if DB acquires row locks during insert).

3. **Lock all derivatives** (FOR UPDATE, one at a time, sorted by UUID).
   - Covers both newly inserted and pre-existing derivatives.
   - ON CONFLICT DO NOTHING does not acquire a lock on the conflicting row, so this step is necessary to gain exclusive access.

4. **Check PURGING state.** If any locked derivative is PURGING, release all locks (rollback), sleep briefly, retry from step 2. Retries indefinitely — caller manages timeout via async cancellation.

5. **Insert segments.**

6. **Insert links.**

7. **Increment `ref_count`** for each derivative by its link count. Individual UPDATE per derivative.

### Why no `active` parameter?

With uuid5, the caller doesn't track which derivatives already exist in the segment linker. The ON CONFLICT DO NOTHING + lock-all pattern makes this unnecessary:
- New derivatives: inserted with `ref_count = 0`, then incremented.
- Existing derivatives: not re-inserted (DO NOTHING), locked, then incremented.
- PURGING derivatives: detected at lock time, handled by retry.

All three cases are handled uniformly without the caller needing to classify them.

### TOCTOU between insert and lock

Between step 2 (ON CONFLICT DO NOTHING) and step 3 (lock):
- If our insert succeeded: the row is in our transaction, not visible to others. Our lock in step 3 is on our own uncommitted row.
- If our insert conflicted: someone else's committed row exists. Our lock in step 3 acquires FOR UPDATE on their row, seeing its current state and ref_count. We increment on top of whatever they set.

Both cases are correct.

---

## Deletion (`delete_segments_by_episodes`)

### Steps

1. **Find derivative UUIDs** linked to segments being deleted (query via LinkRow JOIN). No locking.

2. **Lock derivatives** (FOR UPDATE, one at a time, sorted by UUID). Skip derivatives in PURGING state.

3. **Count links being removed** per derivative (`SELECT derivative_uuid, COUNT(*) FROM links WHERE segment_uuid IN (...) GROUP BY derivative_uuid`).

4. **Delete segments** (CASCADE deletes links via FK, properties CASCADE via FK).

5. **Decrement `ref_count`** for each derivative by its link count from step 3. Individual UPDATE per derivative.

### No replacement search, no FOR SHARE

Deletion only decrements ref_count. There is no concept of "replacement owner" and no need to lock other segments. This eliminates the concurrent-deletion race that plagues the owner system.

### Concurrent deletion safety

Two deletions of different episode sets that share derivatives:
- Both lock the shared derivative (FOR UPDATE, sorted order — no deadlock).
- One blocks until the other commits.
- Each decrements based on its own links being removed.
- Final ref_count is correct regardless of execution order.

---

## GC: Mark Orphaned Derivatives (`mark_orphaned_derivatives_for_purging`)

### Steps

1. **Find orphan candidates** (`ref_count == 0`, `state == ACTIVE`). FOR UPDATE with SKIP LOCKED. Limit per batch.

2. **Mark as PURGING** (`state = PURGING`) for locked orphans.

### SKIP LOCKED

No ordering needed. SKIP LOCKED never waits — no deadlock risk. Concurrent GC callers pick non-overlapping batches. Derivatives locked by registration or deletion are skipped.

---

## GC: Purge Derivatives (`purge_derivatives`)

### Steps

1. **Caller deletes derivatives from external systems** (vector store).

2. **DELETE derivative rows** where `state == PURGING` and UUID in the provided set.

No locking needed. Registration retries on PURGING. Deletion skips PURGING.

---

## Lock Interaction Matrix

| | Registration (FOR UPDATE) | Deletion (FOR UPDATE) | GC Mark (FOR UPDATE SKIP LOCKED) | GC Purge |
|---|---|---|---|---|
| **Registration** | Blocks (same derivative) — sorted order prevents deadlock | Blocks (same derivative) — sorted order prevents deadlock | GC skips (SKIP LOCKED) | Registration retries on PURGING |
| **Deletion** | Blocks (same derivative) — sorted order prevents deadlock | Blocks (same derivative) — sorted order prevents deadlock | GC skips (SKIP LOCKED) | Deletion skips PURGING |
| **GC Mark** | Skips locked | Skips locked | Skips locked | No conflict — different states |
| **GC Purge** | Registration retries | Deletion skips | No conflict | Safe — DELETE on PURGING only |

---

## Invariants

1. **Ref_count accuracy:** `ref_count` equals the number of link rows for the derivative. Registration increments after inserting links. Deletion decrements after counting links removed. Both hold FOR UPDATE for the duration.

2. **No deadlocks:** All derivative FOR UPDATE locks acquired one at a time in sorted UUID order. No other lock types (no FOR SHARE). SKIP LOCKED in GC never waits.

3. **No PURGING races:** Registration retries on PURGING. Deletion and GC skip PURGING. Only purge_derivatives removes them.

4. **uuid5 collision handling:** ON CONFLICT DO NOTHING + post-insert lock covers concurrent inserts of the same derivative. No `active` parameter needed.

5. **No dangling references:** No `owner_segment_uuid` FK that can become stale. `ref_count` is a self-contained integer — no references to other rows.
