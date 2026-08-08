# Refcount System Locking Rules

## Context

The segment linker uses a refcount-based system to track derivative lifecycle. Each derivative has a `ref_count` integer — when 0, the derivative is considered orphaned and eligible for GC purging. Correctness requires that `ref_count` always equals the number of linked segments.

Three concurrent operations can interact: **registration** (ingestion), **deletion**, and **GC marking/purging**. The locking rules below ensure correctness under high concurrency.

---

## Operations and Their Locks

### 1. Registration (`register_segments`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| All involved derivatives (existing + new after ON CONFLICT DO NOTHING) | FOR UPDATE | Prevents race with GC marking a derivative as PURGING while we're linking to it. Serializes ref_count increments for the same derivative across concurrent registrations. |

**Lock ordering:** One derivative at a time, sorted by UUID (Python-sorted loop). Prevents deadlocks with concurrent registrations and deletions that also lock derivatives.

**Two-phase locking:**
- Phase 1: Lock existing derivatives, insert new derivatives with ON CONFLICT DO NOTHING (initial `ref_count = 0`).
- Phase 2: Re-lock derivatives that went through ON CONFLICT DO NOTHING (because ON CONFLICT DO NOTHING does not acquire a lock on the conflicting row). Check for PURGING state.

**PURGING retry:** If any locked derivative is in PURGING state, release all locks (rollback), sleep briefly, and retry. Retries indefinitely — caller manages timeout via async cancellation.

**Ref_count update:** After inserting links, increment `ref_count` for each derivative by the number of links added. Individual UPDATE per derivative (batched UPDATE was measured to be slower).

---

### 2. Deletion (`delete_segments_by_episodes`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| Derivatives linked to segments being deleted | FOR UPDATE | Serializes ref_count decrements. Prevents concurrent registration from incrementing ref_count on a derivative we're about to decrement, which could interleave incorrectly with GC. |

**Lock ordering:** One derivative at a time, sorted by UUID (same `_lock_derivatives` helper as registration). Two-step: first find derivative UUIDs linked to segments being deleted (query via LinkRow JOIN), then lock them in sorted order.

**No replacement search needed.** No FOR SHARE on segments. This is the key simplification over the owner system — deletion only needs to decrement counts, not find and protect a replacement owner.

**Ref_count update:** Before deleting links, count links being removed per derivative (`GROUP BY derivative_uuid`). After deleting links, decrement `ref_count` for each derivative by the count. Individual UPDATE per derivative.

**Link deletion:** Delete links explicitly before deleting segments, so we can count them for the decrement. Alternatively, count first, then delete segments (CASCADE deletes links).

**PURGING derivatives:** Skip derivatives in PURGING state — they are already being garbage collected. Filter with `DerivativeRow.state == DerivativeState.ACTIVE`.

**Segment deletion:** After ref_count decrements and link deletion, delete segments. Properties CASCADE via FK with `ondelete="CASCADE"`.

---

### 3. GC: Mark Orphaned Derivatives (`mark_orphaned_derivatives_for_purging`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| Orphaned derivatives (`ref_count == 0`, state = ACTIVE) | FOR UPDATE (SKIP LOCKED) | Prevents concurrent registration from incrementing ref_count between our read and our state update. SKIP LOCKED allows concurrent GC callers to pick non-overlapping batches without deadlocking. |

**No ordering needed:** SKIP LOCKED never waits, so no deadlock risk from lock ordering. A GC caller that can't lock a derivative simply skips it for the next batch.

**State transition:** ACTIVE -> PURGING for locked orphans whose `ref_count` is still 0 after lock acquisition.

---

### 4. GC: Purge Derivatives (`purge_derivatives`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| None | None | Deletes only derivatives in PURGING state. Registration retries when it encounters PURGING derivatives, so no conflict. Deletion skips PURGING derivatives. |

**Caller responsibility:** Caller must delete the derivative from external systems (vector store) before calling purge. Purge is the final physical removal from the segment linker.

---

## Lock Interaction Matrix

| | Registration (FOR UPDATE on derivatives) | Deletion (FOR UPDATE on derivatives) | GC Mark (FOR UPDATE SKIP LOCKED) | GC Purge |
|---|---|---|---|---|
| **Registration** | Blocks (same derivative) — sorted order prevents deadlock | Blocks (same derivative) — sorted order prevents deadlock | Blocks — GC skips (SKIP LOCKED) | No conflict — registration retries on PURGING |
| **Deletion** | Blocks (same derivative) — sorted order prevents deadlock | Blocks (same derivative) — sorted order prevents deadlock | Blocks — GC skips (SKIP LOCKED) | No conflict — deletion skips PURGING |
| **GC Mark** | Skips locked derivatives | Skips locked derivatives | Skips locked derivatives | No conflict — different states |
| **GC Purge** | Registration retries | Deletion skips PURGING | No conflict — different states | Safe — DELETE on PURGING only |

---

## Invariants

1. **Ref_count accuracy:** `ref_count` always equals the actual number of rows in `links` for that derivative. Ensured by:
   - Registration increments after inserting links, under FOR UPDATE lock.
   - Deletion decrements after counting links to remove, under FOR UPDATE lock.
   - Both operations hold the derivative lock for the duration, preventing interleaved increments/decrements.

2. **No deadlocks:** All derivative FOR UPDATE locks acquired in sorted UUID order (one at a time). No FOR SHARE locks needed (unlike owner system). SKIP LOCKED in GC never waits.

3. **No PURGING races:** Registration retries indefinitely when it encounters PURGING derivatives. Deletion and GC skip PURGING derivatives. Only purge_derivatives physically removes them.

4. **ON CONFLICT DO NOTHING + Phase 2 re-lock:** Covers the TOCTOU gap where a concurrent registration inserts the same derivative (uuid5 collision). Phase 2 locks the winner's row, checks state, and increments ref_count.

---

## Comparison with Owner System

| Concern | Owner system | Refcount system |
|---------|-------------|-----------------|
| Deletion complexity | Must find replacement owner per derivative, lock replacement FOR SHARE, handle candidate iteration | Decrement ref_count — no replacement search, no FOR SHARE |
| Lock types needed | FOR UPDATE on derivatives + FOR SHARE on replacement segments | FOR UPDATE on derivatives only |
| Dangling reference risk | Owner can point to deleted segment if FOR SHARE is missing | No references to other rows — just an integer |
| Registration complexity | Must rescue orphans (set owner for NULL-owner derivatives) | Increment ref_count for all linked derivatives |
| GC orphan detection | `owner_segment_uuid IS NULL` | `ref_count == 0` |
| Drift risk | Dangling owner from concurrent deletion race (requires FOR SHARE to prevent) | Count drift from bugs (mitigated by periodic recount from links table) |
| Performance | Same derivative FOR UPDATE locks + extra FOR SHARE locks + per-derivative candidate queries during deletion | Same derivative FOR UPDATE locks only |
