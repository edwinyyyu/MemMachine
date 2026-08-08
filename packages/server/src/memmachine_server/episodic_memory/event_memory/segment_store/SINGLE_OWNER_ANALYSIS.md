# Single-Owner Derivative System: Correctness & Optimization Analysis

## Operations that modify derivative rows

1. **Insert** (`register_segments`) — adds links, inserts new derivatives, potentially rescues orphaned derivatives
2. **Delete** (`_delete_segments`) — removes links, reassigns or nullifies owner
3. **GC** (`mark_orphaned_derivatives_for_purging`) — transitions ACTIVE → PURGING
4. **Purge** (`purge_derivatives`) — physically removes PURGING derivatives

## Insert vs Delete — the critical race

Setup: D exists with owner=S1, one link S1→D. Tx A deletes S1. Tx B inserts link S2→D.

### If Insert locks D (current implementation):

**Insert locks first:**
1. Insert: `SELECT ... FOR UPDATE` → locks D (owner=S1)
2. Delete: `SELECT ... FOR UPDATE WHERE owner IN (S1)` → blocks on D
3. Insert: inserts link S2→D, commits, releases lock
4. Delete: unblocks, locks D. Correlated subquery: sees S2→D (committed) → sets owner=S2 ✓

**Delete locks first:**
1. Delete: locks D (owner=S1)
2. Insert: `SELECT ... FOR UPDATE` → blocks on D
3. Delete: correlated subquery finds no replacement → sets owner=NULL, deletes S1→D, commits
4. Insert: unblocks, reads D under lock: owner=NULL → orphaned → rescue sets owner=S2 → inserts link S2→D ✓

Both orderings are correct. The lock serializes the two transactions.

### If Insert does NOT lock D (incorrect optimization):

1. Delete: `SELECT ... FOR UPDATE WHERE owner IN (S1)` → locks D
2. Insert: plain SELECT reads D via MVCC → sees committed owner=S1, non-orphaned → decides no rescue needed
3. Insert: inserts link S2→D (LinkRow is a different table — no lock conflict with DerivativeRow) → commits
4. Delete: Step 2 UPDATE begins (new statement → fresh READ COMMITTED snapshot)

Now it depends on timing:
- If Delete's Step 2 starts **after** Insert commits at (3) → subquery sees S2→D → sets owner=S2 ✓
- If Delete's Step 2 starts **before** Insert commits at (3) → subquery doesn't see S2→D → sets owner=NULL → **BAD: D.owner=NULL with link S2→D existing**

This is a genuine race. We cannot control statement timing across transactions.

## Insert vs GC

Setup: D is orphaned (owner=NULL, state=ACTIVE). Insert rescues D while GC marks it for purging.

### If Insert locks D:

**Insert locks first:**
1. Insert: locks D
2. GC: `SELECT ... FOR UPDATE SKIP LOCKED` → D is locked → **skipped**
3. Insert: rescues D (owner=S2), commits
4. GC next run: D has owner=S2 → not orphaned → not selected ✓

**GC locks first:**
1. GC: locks D, sets state=PURGING, commits
2. Insert: locks D, reads state=PURGING → `DerivativeNotActiveError` ✓

### If Insert does NOT lock D:

1. Insert: plain SELECT reads D: state=ACTIVE, owner=NULL → orphaned → rescue UPDATE: `WHERE uuid=D AND owner IS NULL`
2. GC: `SELECT ... FOR UPDATE SKIP LOCKED` → D is not locked → locks D → sets state=PURGING → commits
3. Insert's rescue UPDATE: PostgreSQL tries to lock D for the update, blocks until GC commits. Then re-evaluates WHERE under READ COMMITTED: owner IS NULL is still true → proceeds → sets owner=S2.
4. Result: D has state=PURGING, owner=S2 → **inconsistent**

The lock is needed to prevent this.

## Insert vs Insert

Two inserts both rescuing the same orphaned D.

The conditional `UPDATE WHERE owner IS NULL` naturally serializes: first UPDATE acquires a row lock, sets owner. Second UPDATE blocks, then re-evaluates: owner is no longer NULL → no-op. **No explicit lock needed for this pair specifically.** But Insert needs the lock for the Delete and GC races above anyway.

## Locking requirements summary

| Pair            | Lock needed in Insert? | Why                                                                      |
|-----------------|------------------------|--------------------------------------------------------------------------|
| Insert vs Delete| **Yes**                | Without lock, Delete can nullify owner between Insert's read and commit  |
| Insert vs GC    | **Yes**                | Without lock, GC can mark PURGING between Insert's validation and rescue |
| Insert vs Insert| No (conditional UPDATE suffices) | But moot since lock is needed for above                        |

**Insert must lock all existing active derivatives.** There is no way around this under READ COMMITTED.

## Optimizations available (within the lock)

1. **Select only needed columns in `_lock_derivatives`**: Only `uuid`, `state`, `partition_key`, `owner_segment_uuid` are needed — not the full ORM object (which includes the `block` JSON blob, `context`, etc.).

2. **Rescue only orphaned derivatives**: After locking, identify the orphaned subset (`owner IS NULL`). Only issue the rescue UPDATE for those. Non-orphaned derivatives are protected by the lock — they cannot become orphaned during the transaction. In the common case (no orphans among active derivatives), the rescue is skipped entirely (zero writes to existing derivative rows).

## Single-owner vs ref_count comparison

### Insert path (30k existing, 70k new, 10k segments, 100k links)

| Step                          | Ref_count                               | Single-owner                                |
|-------------------------------|-----------------------------------------|---------------------------------------------|
| Lock active derivatives       | SELECT FOR UPDATE 30k rows              | Same                                        |
| Validate                      | Check state                             | Same + identify orphaned subset             |
| Insert segments               | 10k rows                                | Same                                        |
| Insert new derivatives        | 70k rows (with ref_count)               | 70k rows (with owner)                       |
| **Update existing derivatives** | **UPDATE 30k rows (increment ref_count)** | **UPDATE ~0 rows (rescue orphaned only)** |
| Insert links                  | 100k rows                               | Same                                        |

Single-owner saves one batched UPDATE of 30k rows. With fast batching (VALUES join), that's ~200–500ms on PG. Real but moderate.

### Delete path (the dominant advantage)

| Step                    | Ref_count                                          | Single-owner                                           |
|-------------------------|----------------------------------------------------|--------------------------------------------------------|
| Find affected derivatives | SELECT DISTINCT from links — **all** linked derivs | `WHERE owner IN (deleted)` — **owner subset** only     |
| Lock                    | Lock **all** linked derivatives                    | Lock **owner subset** only                             |
| Update                  | Decrement ref_count for **all**                    | Reassign owner for **subset**                          |

If a derivative has N links and you delete 1 segment: ref_count locks and updates that derivative. Single-owner locks it only if that segment happens to be its owner (~1/N chance). For high-fanout derivatives, this is **10–100x fewer locks and writes per deletion**.

### GC path

- Ref_count: `WHERE ref_count = 0` on a column written on every insert and delete → high index churn
- Single-owner: `WHERE owner IS NULL` on a column that changes only on deletion and rescue → stable index

### Verdict

Single-owner is meaningfully faster than ref_count — **primarily on the delete path**. The insert advantage (skipping writes to non-orphaned existing derivatives) saves ~200–500ms. The delete advantage (touching only the owner-subset instead of all linked derivatives) can be orders of magnitude better for high-fanout derivatives.

The lock on existing active derivatives is non-negotiable for correctness. What single-owner saves on insert is the *write* to locked rows, not the lock itself.
