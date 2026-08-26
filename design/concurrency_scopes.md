# Concurrency scopes and data-plane contracts

Contributor-facing design notes for `ConcurrencyScope` and the visibility and
authority contracts on `VectorStoreCollection`.

## The problem

The `VectorStore` contract said a logical collection "must be managed by at
most one process at a time". That is true of some implementations and not
others, and it is about to stop being true of more of them: the moment one
backend genuinely supports multi-process management, a blanket sentence is
either false for it or forbids what it supports -- capability the contract
cannot express is capability no consumer may use.

## Declared scope

`ConcurrencyScope` (`common/data_types.py`) is an `IntEnum`,
`PROCESS < MACHINE < CLUSTER`: the widest deployment boundary within which
concurrent instances of a component may safely manage the same resources.
Ordered by breadth, so the effective scope of a composed system is the `min`
of its parts' scopes.

`VectorStore` and `SegmentStore` declare an abstract
`concurrency_scope` property, and the blanket sentences defer to it:
concurrent management of the same collection/partition is safe within the
declared scope; beyond it, at most one instance manages a resource and the
consumer shards names, as before.

Declarations:

| Component | Scope | Why |
|---|---|---|
| `SQLAlchemySegmentStore` | `CLUSTER` on PostgreSQL, `MACHINE` on SQLite | cross-process safe via unique-constraint create and row locks |
| `QdrantVectorStore` | `PROCESS` | bookkeeping guarded only by in-process locks |
| `MilvusVectorStore` | `PROCESS` | bookkeeping guarded only by in-process locks |
| `SQLiteVectorStore` | `PROCESS` | search-engine state lives in process memory |
| `SQLiteVecVectorStore` | `PROCESS` | check-then-write bookkeeping within one process |

A deployment can therefore introspect what it is allowed to do, instead of
relying on out-of-band knowledge, and a component that composes several
reports the `min` of them. `PROCESS` declarations are honest labels for
current implementations, not permanent judgments: a store whose metadata
moves to a cross-process authority widens to whatever that authority
reaches, and the property is the place that change becomes visible.

## Data-plane contracts

Two properties stated on `VectorStoreCollection`. Both are pre-existing
behavior; multi-process deployment is what makes them worth stating, because
single-process consumers could survive assuming their negations.

- Visibility: a returned write is durably accepted but is not guaranteed
  visible to a subsequent query, from this instance or any other. Consumers
  must not build on read-your-writes. (True even single-process today;
  uniform staleness, not a per-client asymmetry.)
- Authority: stored properties and property filters operate on the
  collection's own copy of a record, with no freshness guarantee relative to
  any external source of truth. A vector store can promise internal
  consistency of its own copy, never authority over data governed elsewhere;
  consumers whose records are governed by an external authority (e.g. SQL
  holding access-determining attributes) must re-validate results against it.
  Post-validation at the source of truth can only remove candidates -- an
  attribute may be pre-filtered inside the vector store only if it is
  immutable after ingest or if staleness can only produce false positives.

## Handle lifetime is not scope-conditional

The data-plane contracts include one invariant deliberately stated
without reference to scope: a write through a handle whose collection
has been deleted must never surface in a collection later created under
the same name.

Scope is what makes the uniformity necessary rather than tidy. PROCESS
is the only scope at which the *application* could be the guarantor,
since inside one process it sees every handle it holds and every delete
it issues. Above PROCESS that is impossible -- one process holds a
handle while another deletes and recreates, and the first cannot learn
of it -- so any store declaring MACHINE or CLUSTER has to provide the
invariant outright. Making it conditional would then leave consumers
writing against the lenient reading and breaking when a backend's scope
widens under them, which is what any move to a cross-process metadata
authority would do to Qdrant and Milvus. The discipline it would delegate is also expensive and
silently violable: handle-lifetime tracking across tasks, in every
consumer, to avoid a failure whose symptom is corrupted data rather than
an error.

Known non-conformance: both SQLite-backed vector stores derive their
native resources from (namespace, name) alone and do resurrect such
writes -- immediately in `SQLiteVecVectorStore`, and on the next index
save in `SQLiteVectorStore`, whose stale handle also rewrites the live
index file of the replacement collection. Tracked in
MemMachine/MemMachine#1536 and, for Qdrant, MemMachine/MemMachine#1563.
Incarnation-scoped partition keys are the fix in both cases.

## Rejected alternatives

- Deleting the one-process sentence outright: Milvus and the SQLite stores
  still need it; per-instance declaration is the only shape that tells the
  truth for all implementations at once.
- Boolean "multi-process safe" flag: loses the machine/cluster distinction,
  which is exactly the distinction a shared-SQLite deployment needs.
- Runtime enforcement (detecting out-of-scope concurrent managers): requires
  cross-process membership machinery; the scope is a contract, like the
  identifier constraints, not a guard.
