# Schema creation and DDL

Contributor-facing design notes on where the server runs DDL and why.

## Problem

Horizontal scalability: several server processes should be able to run against
one database. Every SQLAlchemy-backed store creates its own schema during
`startup()` with `metadata.create_all`, which is check-then-create -- SQLAlchemy
reflects, then emits `CREATE TABLE` without `IF NOT EXISTS`. Two processes
booting against a database that does not yet have the tables can both pass the
check; the loser's `CREATE TABLE` fails and takes its boot down with it.

This note records the decision to keep that shape for now, the argument that
makes it survivable, and the conditions under which the argument stops holding.

## What the server does today

Schema creation in `startup()`, via `create_all`:

- `common/episode_store/episode_sqlalchemy_store.py:170`
- `common/session_manager/session_data_manager_sql_impl.py:144`
- `common/vector_store/sqlite_vector_store.py:718`
- `common/vector_store/sqlite_vec_vector_store.py:491`
- `episodic_memory/event_memory/segment_store/sqlalchemy_segment_store.py:793`
- `semantic_memory/cluster_store/cluster_store_sqlalchemy.py:101`
- `semantic_memory/config_store/config_store_sqlalchemy.py:238`
- `semantic_memory/storage/vector_store_semantic_storage.py:196`

Schema creation at runtime, driven by traffic rather than by configuration.
This is worth stating plainly, because it is easy to assume the server only
creates schema when an operator changes something. It does not. The event
backend derives a partition key per session --
`partition_key_for_session(session_id)`, a truncated SHA-256 of the session id
(`long_term_memory/service_locator.py:106` and `:165-183`) -- and uses it as
both the segment store partition key and the vector store collection name. So
the first use of a new session runs DDL:

- On PostgreSQL, two `CREATE TABLE ... PARTITION OF` statements for the segment
  store child tables (`sqlalchemy_segment_store.py:1045-1061`, reached through
  `open_or_create_partition` at `service_locator.py:140`).
- On the SQLite vector stores, `create_all(tables=[records_table])` for the
  collection's records table (`sqlite_vector_store.py:1061`,
  `sqlite_vec_vector_store.py:692`). The sqlite-vec store follows it with
  `CREATE VIRTUAL TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` for the
  vec0 table and property indexes, which are re-emitted on every open.

Runtime DDL is therefore an ordinary, per-session occurrence, not a rare
administrative one, and the crash-and-restart argument below does not cover it.

Of those two, only the segment store's matters for running several processes
against one database. The SQLite stores are single-process by construction --
one file, one writer -- so concurrent creation of the same collection is
outside what they promise, and the DDL they run per session is not a
multi-process hazard. The segment store's partitions are shared, which is why
that path had to be made race-free rather than left to a restart.

The registry is what keeps this list from growing. Qdrant and Milvus also mint
a collection per session, but registering one is an `INSERT` into a table that
already exists, so adding a session adds a row rather than a table. Turning
runtime metadata management into a transactional row write is the point of the
primitive, and it means the only schema those backends need is created once, at
boot, from configuration.

Alembic exists (`semantic_memory/storage/alembic_pg`) but is also driven from
the boot path: `apply_alembic_migrations` runs `command.upgrade(config, "head")`
in-process, called from `SqlAlchemyPgVectorSemanticStorage.startup()`
(`sqlalchemy_pgvector_semantic.py:172-211`). Having Alembic in the repo is
therefore not the same as having schema creation out of the boot path.

## Decision

New SQLAlchemy-backed stores keep `create_all` in `startup()` and match the
existing shape. They do not add per-store race handling of their own.

The reasoning is uniformity, not that the race is imaginary. A one-off guard on
a single new store would leave seven other boot paths with the same behavior
while making the repo harder to reason about as a whole -- the reader could no
longer assume that "SQLAlchemy store, `startup()`, `create_all`" means one
thing. Moving schema creation out of the boot path is a change to the
deployment model, not to one file, and it should be made once, deliberately,
for every store at the same time.

## Why crash-and-restart is acceptable for boot-path DDL

The loser of a boot race raises, the process exits, the supervisor restarts it,
`create_all` reflects, finds the tables present, emits nothing, and boot
proceeds. The database converges on the correct schema and no writes are lost
or misdirected in the meantime, because the failure is entirely before the
store serves traffic.

Three properties are what make that true, and they are properties of the
current code rather than of `create_all` in general:

1. **The schema each `create_all` emits is a single statement, or a set of
   statements with no ordering dependency between them.** This is the load-
   bearing one. `create_all`'s `checkfirst` tests for *table* existence and
   skips the whole `CREATE TABLE` when the table is present, indexes included.
   A partially applied multi-statement schema -- table created, a following
   index not -- is therefore *not* repaired by a restart; the restart skips the
   table and the index stays missing forever. Where a store emits indexes as
   separate `CREATE INDEX IF NOT EXISTS` statements after `create_all`, as the
   sqlite-vec store does, they are re-emitted on every open and the gap closes.
2. **On PostgreSQL, DDL is transactional.** Stores that wrap `create_all` in
   `engine.begin()` get all-or-nothing application, so a failed boot leaves no
   half-built schema behind at all.
3. **The set of tables is fixed by configuration, not by traffic.** Boot-path
   `create_all` covers a static metadata. The race window is a cold database,
   or the first boot after an operator adds a configuration entry that names a
   new table. It is not re-entered per request.

## Where the argument stops holding

Any of these invalidates the reasoning above, and a store that has one needs to
be handled explicitly rather than inheriting this decision:

- **A table whose schema needs more than one statement**, i.e. a `Table` that
  carries `Index()` objects, or DDL emitted after `create_all` that is not
  itself `IF NOT EXISTS`. Then a partial application is permanent, per (1).
- **Table names derived at runtime rather than from configuration.** The
  failure moves from a boot crash, which a supervisor absorbs, to a 500 in a
  process that is already serving traffic, and it recurs for each new name
  rather than once per deployment.
- **Deployments with no supervisor restart**, where a crashed boot is a
  permanent outage rather than a retry.

The collection registry satisfies all three properties: its table
(`sqlalchemy_collection_registry.py:143-148`) is one `Table` with a primary key
and a JSON column and no `Index()` objects, so `create_all` emits exactly one
`CREATE TABLE`; `startup()` wraps it in `engine.begin()`; and registry names
come from the vector store configuration (`qdrant_{name}` per configured Qdrant
instance, built during `DatabaseManager.build_all`), so the table set is fixed
before the server accepts a request.

## The DDL path that is already race-free

Segment store partition creation is the exception worth pointing at. It runs in
steady state, on the first use of every new session, so restarting is not an
available answer -- the failure would be a 500 in a process that is already
serving, recurring per session rather than once per deployment.
`create_partition` and `_open_or_create_partition`
(`sqlalchemy_segment_store.py:805-895`) take an explicit lock on the partitions
table, insert the partition row, and emit
`CREATE TABLE ... PARTITION OF` -- all inside one transaction. PostgreSQL's
transactional DDL means the loser's insert conflicts and the child-table
creation rolls back with it. Runtime DDL that cannot be answered with a restart
has to look like this.

## Deferred: moving schema creation out of the boot path

The standard answer is to make schema creation a distinct deployment step --
`alembic upgrade head` as an init job, an initContainer, or a deploy stage --
and reduce `startup()` to verifying that the expected schema is present and
failing fast when it is not. Exactly one process runs DDL, so there is no race
to handle, and a binary can refuse to serve against a schema it does not
understand instead of silently adapting to it.

This is deferred rather than rejected. It is a change to how the server is
deployed, it touches all eight boot paths plus the in-process Alembic call, and
it needs the operator-facing story (what runs the migration, what happens on
rollback) decided alongside the code. Nothing in the current stack depends on
it: the failure it prevents is a cold-start crash loop that resolves itself,
not data loss or divergence between processes.

## Follow-ups

- Decide the repo-wide model, and apply it to every store in one pass rather
  than store by store.
- The four copies of the engine validator
  (`sqlite_vector_store.py:655`, `sqlite_vec_vector_store.py:447`,
  `sqlalchemy_segment_store.py:749`, `sqlalchemy_collection_registry.py:93`)
  should be deduplicated. They share a gap: the ephemeral-SQLite check tests
  `db is None or db == ":memory:"`, but `sqlite+aiosqlite:///` parses to
  `database == ""` and passes, giving every connection its own private
  temporary database. That one is a plain bug, not a race, and no restart
  converges out of it.
