# Owner System Locking Rules

## Context

The segment linker uses an owner-based system to track derivative lifecycle. Each derivative has an `owner_segment_uuid` — when NULL, the derivative is considered orphaned and eligible for GC purging. Correctness requires that ownership always accurately reflects whether the derivative has linked segments.

Three concurrent operations can interact: **registration** (ingestion), **deletion**, and **GC marking/purging**. The locking rules below ensure correctness under high concurrency.

---

## Operations and Their Locks

### 1. Registration (`register_segments`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| All involved derivatives (existing + new after ON CONFLICT DO NOTHING) | FOR UPDATE | Prevents race with GC marking a derivative as PURGING while we're linking to it. Prevents concurrent registration from inserting links to the same derivative without coordinating owner assignment. |

**Lock ordering:** One derivative at a time, sorted by UUID (Python-sorted loop). Prevents deadlocks with concurrent registrations and deletions that also lock derivatives.

**Two-phase locking:**
- Phase 1: Lock existing derivatives, insert new derivatives with ON CONFLICT DO NOTHING.
- Phase 2: Re-lock derivatives that went through ON CONFLICT DO NOTHING (because ON CONFLICT DO NOTHING does not acquire a lock on the conflicting row). Check for PURGING state.

**PURGING retry:** If any locked derivative is in PURGING state, release all locks (rollback), sleep briefly, and retry. Retries indefinitely — caller manages timeout via async cancellation.

**Owner assignment:** Set `owner_segment_uuid` for derivatives that have NULL owner (orphan rescue). Derivatives with an existing owner are left untouched.

---

### 2. Deletion (`delete_segments_by_episodes`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| Affected derivatives (those whose `owner_segment_uuid` points to a segment being deleted, state = ACTIVE) | FOR UPDATE | Prevents concurrent registration or GC from modifying these derivatives while we reassign owners. Prevents concurrent deletion from reassigning the same derivatives. |
| Replacement segment (one per affected derivative) | FOR SHARE | Prevents a concurrent deletion from deleting the replacement segment before our transaction commits. Without this, the owner can become a dangling reference (points to a deleted segment), making ownership inaccurate. |

**Lock ordering for derivatives:** One at a time, sorted by UUID (same `_lock_derivatives` helper as registration). Two-step: first find affected derivative UUIDs without locking (query with JOIN on segments being deleted), then lock them in sorted order.

**Lock ordering for replacement segments:** FOR SHARE does not conflict with other FOR SHARE locks, so no deadlock between concurrent reassignments. FOR SHARE only conflicts with exclusive locks (DELETE), which is the desired blocking behavior.

**Replacement search:** For each affected derivative, query linked segments not in the deleted episodes. Try locking each candidate FOR SHARE until one succeeds (row still exists after lock acquired). If no candidate survives, set owner to NULL (derivative is truly orphaned).

**Segment deletion:** After owner reassignment, delete segments. CASCADE deletes links via FK. Properties CASCADE via FK with `ondelete="CASCADE"`.

**PURGING derivatives:** Skip derivatives in PURGING state during reassignment — they are already being garbage collected. Filter with `DerivativeRow.state == DerivativeState.ACTIVE`.

---

### 3. GC: Mark Orphaned Derivatives (`mark_orphaned_derivatives_for_purging`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| Orphaned derivatives (`owner_segment_uuid IS NULL`, state = ACTIVE) | FOR UPDATE (SKIP LOCKED) | Prevents concurrent registration from rescuing the derivative (setting owner) between our read and our state update. SKIP LOCKED allows concurrent GC callers to pick non-overlapping batches without deadlocking. |

**No ordering needed:** SKIP LOCKED never waits, so no deadlock risk from lock ordering. A GC caller that can't lock a derivative simply skips it for the next batch.

**State transition:** ACTIVE -> PURGING for locked orphans.

---

### 4. GC: Purge Derivatives (`purge_derivatives`)

**Locks acquired:**

| Target | Lock type | Reason |
|--------|-----------|--------|
| None | None | Deletes only derivatives in PURGING state. Registration retries when it encounters PURGING derivatives, so no conflict. Deletion skips PURGING derivatives. |

**Caller responsibility:** Caller must delete the derivative from external systems (vector store) before calling purge. Purge is the final physical removal from the segment linker.

---

## Lock Interaction Matrix

| | Registration (FOR UPDATE on derivatives) | Deletion (FOR UPDATE on derivatives) | Deletion (FOR SHARE on replacement segments) | GC Mark (FOR UPDATE SKIP LOCKED) | GC Purge |
|---|---|---|---|---|---|
| **Registration** | Blocks (same derivative) — sorted order prevents deadlock | Blocks (same derivative) — sorted order prevents deadlock | No conflict | Blocks — GC skips (SKIP LOCKED) | No conflict — registration retries on PURGING |
| **Deletion** | Blocks (same derivative) — sorted order prevents deadlock | Blocks (same derivative) — sorted order prevents deadlock | FOR SHARE vs FOR SHARE: no conflict. FOR SHARE vs DELETE: blocks (desired) | Blocks — GC skips (SKIP LOCKED) | No conflict — deletion skips PURGING |
| **GC Mark** | Skips locked derivatives | Skips locked derivatives | No conflict | Skips locked derivatives | No conflict — different states |
| **GC Purge** | Registration retries | Deletion skips PURGING | No conflict | No conflict — different states | Safe — DELETE on PURGING only |

---

## Invariants

1. **Ownership accuracy:** `owner_segment_uuid IS NULL` if and only if the derivative has no linked segments (is truly orphaned). Ensured by:
   - Registration sets owner for NULL-owner derivatives it links to.
   - Deletion reassigns owner (with FOR SHARE protection) or NULLs it when no replacement exists.
   - FOR SHARE on replacement prevents concurrent deletion from invalidating the chosen owner.

2. **No deadlocks:** All derivative FOR UPDATE locks acquired in sorted UUID order (one at a time). FOR SHARE on segments does not conflict with other FOR SHARE. SKIP LOCKED in GC never waits.

3. **No PURGING races:** Registration retries indefinitely when it encounters PURGING derivatives. Deletion and GC skip PURGING derivatives. Only purge_derivatives physically removes them.

4. **ON CONFLICT DO NOTHING + Phase 2 re-lock:** Covers the TOCTOU gap where a concurrent registration inserts the same derivative (uuid5 collision). Phase 2 locks the winner's row, checks state, and rescues orphans if needed.
