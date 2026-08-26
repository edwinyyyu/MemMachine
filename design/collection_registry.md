# Vector store collection registry

Contributor-facing design notes for
`memmachine_server.common.vector_store.collection_registry`.

## Problem

Horizontal scalability: multiple server processes should be able to work with
the same collections on one backend. Strict sharding (each collection managed
by exactly one process) has always worked; the blocker to lifting it is
collection *metadata* management. Backends without conditional writes,
transactions, or unique constraints (Qdrant, Milvus) cannot make
create / open / delete atomic, so their ad-hoc catalogs are read-check-write
sequences guarded by per-process locks.

The collection registry is the enabling primitive: a durable catalog mapping
(namespace, name) to an immutable `CollectionRegistryEntry`, with
registration atomic across processes via a real compare-and-set. A vector
store holds a registry dedicated to it and cannot reach any other registry
through it.

## Scope: collections, not a generic registry

The registry is deliberately specific to vector store collections. An
earlier draft generalized it to a key -> config registry with pluggable
types; that was narrowed because it had exactly one consumer kind, and the
realistic reuse axis (other vector store backends) is *within* the
collection domain -- Milvus's needs are the same shape. The compatibility
asymmetry decides the default: widening a specific API later (extracting a
generic ABC when a non-collection consumer actually materializes) is an
additive refactor; narrowing a shipped generic ABC is a breaking change.
Being collection-specific also lets the API speak the domain --
`register(namespace, name, entry)` rather than key/adapter plumbing -- and
turns the key encoding into an implementation detail.

## The entry

`CollectionRegistryEntry` = `{config, native_collection_name,
partition_key}`. One entry format is shared by all vector store backends:
`config` is the ABC-level `VectorStoreCollectionConfig` every backend already
takes, and the two identity fields are the shared mechanism (config-hashed
shared native collections, per-collection partition keys) that Qdrant and
Milvus both use.

- `native_collection_name` is pinned at registration instead of being
  re-derived from `sha256(config)` on every open: any change to config
  serialization -- even an added optional field -- would otherwise re-hash
  every stored config and silently repoint existing collections at new,
  empty native collections.
- `partition_key` is generation-scoped (minted per registration): records
  written through handles held across a deregistration land under the dead
  generation -- invisible, never resurrected by a re-registration, no locks
  and no per-write round trips.

Evolution policy -- covering the whole stored document: the entry envelope
and every model nested in it, `VectorStoreCollectionConfig` included, since
the config is serialized inside the entry's JSON and read back through the
same validation. Extend by adding optional fields with defaults -- no
version bump, no migration, old rows validate under the new model. The rule
has a semantic half: a new field's default must reproduce the behavior that
existed before the field, because stored rows predating it are read through
that default. A change whose default cannot mean the old behavior is not
additive -- it is a breaking change and takes the version bump path.
Backend-specific needs are expressed the same way, not with per-backend
entry types.

Terminology: "generation" is reserved for collection incarnations (the
partition-key uuid); evolution of the entry or config models is a format
"version". Content-hash native naming has the side effect of never sharing
native collections across config-model versions (any model change drifts
the hash) -- conservative quarantine at the cost of gradual fragmentation.

## Portability across vector store providers

Two identity fields locate a collection, and that arity is structural
rather than a Qdrant/Milvus accident. The pair is the *container* -- the
unit a provider creates, deletes, caps, and pins fixed physical
attributes to (dimensionality, metric, index structures) -- and a
*discriminator*, the unit of isolation within a container. Every
surveyed provider decomposes this way, under its own vocabulary:

| Provider   | Container           | Discriminator          |
| ---------- | ------------------- | ---------------------- |
| Qdrant     | collection          | payload key, shard key |
| Milvus     | collection          | partition-key field    |
| Pinecone   | index               | namespace              |
| Weaviate   | collection          | tenant                 |
| S3 Vectors | index within bucket | metadata field         |
| Chroma     | collection          | (none needed)          |

The entry calls the second field `partition_key`, after the two shipped
backends and the house term for the same subdivision in the segment
store. Note that Pinecone's "namespace" is a discriminator, not this
registry's `namespace`, which is a component of the logical key.

The discriminator level is essentially free: it asks nothing of a
provider beyond storing a value per record and filtering it by equality,
which is below the bar for being usable as a vector store at all. Chroma
is the degenerate end -- collections are cheap enough there to give each
logical collection its own, leaving nothing for a discriminator to do.
One container with many discriminators and many containers with none are
the same entry shape, which is the portability claim in one line.

So the only open question is whether one container per logical
collection suffices. Three cases where it would not, none of which argues
for adding a field now:

- A level above the container becomes variable: S3 buckets, Milvus
  databases, Pinecone projects, or simply a second cluster once a
  per-parent cap is reached. Today the parent is fixed by store
  configuration. Lifting that needs a locator, not a second container.
- One logical collection needs several containers -- multiple vectors
  per record with differing dimensions or metrics on a provider that
  indexes one vector per container, or a collection outgrowing a
  per-container capacity cap. The first is excluded a level up:
  `VectorStoreCollectionConfig` declares a single `vector_dimensions`
  and `similarity_metric`. The second is remote at surveyed caps, and
  when a *shared* container fills rather than a single collection it is
  resolved by minting a new container for subsequent collections, which
  needs no entry change since each entry names its own.
- A collection transiently spans containers during an online re-index.
  Handled a level up, by creating a new logical collection and swapping.

The guarantee is not that two fields are provably sufficient forever; it
is that the arity sits on the cheapest axis of change. The registry
stores the entry and never parses it, so a third field is
add-optional-only (`backend_id: str | None = None`, defaulting to the
configured backend) and reproduces prior behavior by construction. What
would be expensive is changing the key shape or the atomicity contract,
and no surveyed provider pressures either.

Both identity fields are opaque strings minted by the store and never
derived by the registry, which keeps per-provider naming budgets out of
this layer -- and those budgets are strict. The current
`{namespace}__{sha256hex}` native name runs to 98 bytes, over Pinecone's
and S3 Vectors' index-name limits; the `{name}#{uuid4hex}` partition key
is 65 bytes and contains "#", over Weaviate's 64-byte tenant limit and
outside its charset. A backend for those providers shortens the digest
and picks a charset-safe generation format without touching the
registry: the ABC requires only that the partition key be unique and
carry a per-registration generation, never a format. That is why the key
format is duplicated between the Qdrant and Milvus stores rather than
shared.

One budget does reach the addressing scheme. The index set stays inside
the native name's config hash deliberately -- collections sharing a
container then have identical index sets by construction, instead of the
container accumulating the union of its tenants' indexes with each
tenant paying memory and build time for another's fields. The cost is
that containers accumulate one per config revision and are never
reclaimed. At Qdrant's ~1000 collections per cluster that is cosmetic
fragmentation; at Pinecone's 20-200 indexes per project it is a hard
operational cap, which would make reclaiming orphaned containers a
prerequisite for such a backend rather than a cleanup nicety.

## Which backends need a registry

Not every backend needs one. The registry does four jobs -- atomic
create, durable storage of the entry, the logical-to-native name
mapping, and the generation that stops a deleted collection's records
from being resurrected -- and a backend that provides all four itself
should use its own and take none. Where those four jobs land is what
decides it:

- **Qdrant and Milvus take one.** Both fail the first job (no
  conditional writes and no unique constraints, so create is a
  read-check-write) and the fourth (native collections are
  name-addressed and shared by config hash). Those two failures are the
  entire reason this primitive exists.
- **`SQLiteVectorStore` cannot be helped by one.** Its authoritative
  index state is a `dict[(namespace, name), VectorSearchEngine]` held in
  process memory and flushed to disk on a threshold, so a second process
  would search a divergent index. The limit is the data plane; repairing
  metadata would not move it, and a registry would only add a
  cross-process authority to a store that cannot be used across
  processes.
- **`SQLiteVecVectorStore` does not need one, but does need the
  compare-and-set it already has the machinery for.** Its vectors live in
  SQLite tables rather than in process memory, and it owns a
  `_CollectionRow` table whose primary key is (namespace, name) -- the
  same kind of arbiter this registry's implementation relies on. Its
  create is nonetheless a check-then-write inside one transaction, which
  is what confines concurrent collection management to a single process.
  Turning that pre-check into an insert that maps `IntegrityError` onto
  the domain error is the change `SQLAlchemyCollectionRegistry.register`
  already makes, done in place rather than delegated. Two caveats: the
  generation gap would remain, since its table names derive from
  (namespace, name), so a handle held across a deletion addresses
  whatever is later recreated under those names; and widening its reach
  past one process needs the usual multi-process SQLite conditions
  besides -- WAL, busy timeouts, a local filesystem.
- **A container-per-collection provider covers all four.** A uniqueness
  constraint on container names for atomic create, container metadata for
  the entry, a direct name lookup for the mapping, and -- the
  load-bearing one -- a container identity that is not its name, so that
  a recreated container is a different object and handles held across the
  deletion fault instead of writing into it. That last check matters
  because the obvious alternative, putting the generation into the name,
  destroys the first: two processes racing a create would mint different
  generations, produce different names, and both succeed, leaving two
  live containers for one logical collection.

Chroma is the worked example of that last case, checked against 1.5.9
rather than assumed. Collections carry a uuid that changes across
delete-and-recreate, and operations route by it, so a handle held across
a recreation raises rather than writing into the new collection -- a
stronger generation than the synthetic one here, because the backend
enforces it instead of merely hiding the records. Creates are genuinely
atomic rather than only rejected: eight concurrent clients racing
`create_collection` against a Chroma server produced exactly one winner
in 25 of 25 rounds and never two collections of one name, and the
`get_or_create` race converged all eight callers on a single id -- the
same contract `get_or_register` provides here. A Chroma backend would
therefore take no registry.

What Chroma does not provide is a classifiable failure. The
already-exists condition surfaces as `ChromaError` code 400 over HTTP and
`InternalError` code 500 through the local binding -- two types, two
codes, neither of them the `UniqueConstraintError` the same module
defines and exports -- leaving the message as the only portable
discriminator. That is a worse version of the error-classification
problem this stack cleaned up for Qdrant and Milvus, and it is a
backend-side concern rather than a reason to reach for a registry.

Adding a registry to a backend that already satisfies all four would be
a regression rather than caution: it introduces a second authority that
can disagree with the first, which is precisely the divergence the
create and delete orderings in this design exist to bound.

## API shape

Docstrings say what each operation does; this section says why the
surface has the shape it has. It is five operations plus a lifecycle
pair, and each shape is a decision rather than a default.

**The key is two parameters, not one.** An earlier draft took a single
opaque `key: str` and let each consumer compose it from its own parts.
Namespacing is enforced by the contract instead: the registry guarantees
that distinct (namespace, name) pairs never collide, so a store cannot
get the encoding wrong and two stores cannot disagree about it. The "/"
join in the SQLAlchemy implementation is consequently an implementation
detail, and the test that ("a__b", "c") and ("a", "b__c") remain
distinct tests the contract rather than the encoding.

**Every parameter is keyword-only.** `namespace` and `name` are both
`str` over the same charset, so a positional swap type-checks cleanly
and then silently addresses the wrong collection. Keyword-only makes the
swap unrepresentable rather than merely discouraged.

**`register` raises and `get` returns `None`.** The asymmetry is
deliberate. Absence on read is an ordinary expected outcome -- it is how
a store learns a collection does not exist, and `open_collection`
returning `None` is that same answer one layer up -- so raising would
put try/except on the common path. A lost registration race is rare,
contended, and has to interrupt a multi-step lifecycle sequence, so it
raises, and the store translates it into its own already-exists error.
A boolean return would invite callers to drop it on the floor.

**`register` raises whether or not the stored entry matches the one
offered.** The registry never compares entries, here or in
`get_or_register`. Equality policy belongs to the store: what counts as
"the same collection" is backend-specific, and two entries can carry
equal configs while naming different native collections. Keeping
comparison out also keeps registry behavior independent of how config
models evolve, which is what lets the evolution policy below be stated
in terms of validation alone.

**`get_or_register` returns `(stored_entry, registered)`.** It returns
the *stored* entry rather than the caller's, which is what makes it the
atomic commit point: both sides of a race leave holding identical
identity and therefore cannot write to different native collections
under one logical name. The boolean is separate because the two outcomes
demand different work from the store -- minting a native collection
versus adopting one that already exists -- and that distinction is not
recoverable from the returned entry.

It exists because the create path genuinely needs "register if absent,
else tell me what is already there" as a single atomic step; this is the
commit point of an otherwise non-atomic sequence, which is the one place
ensure-style semantics earn their keep. It is implemented natively -- a
`SELECT` fast path, then `INSERT .. ON CONFLICT DO NOTHING`, then a
re-read only when that insert lost -- rather than as try/except around
`register`, so ordinary paths cost one statement and no control flow
runs through exceptions.

**`deregister` is idempotent and returns nothing.** It is the tail of a
sequence that begins by destroying data, so a crash partway has to leave
it re-runnable; reporting "already absent" as an error would force every
retry to branch on a distinction with no consequence. It does not hand
back the removed entry, because the caller necessarily read that entry
first in order to find the data to delete, and returning it would invite
a read-modify-write shape the registry deliberately cannot support.

**There is no `update`.** Entries are immutable by contract, and this is
the decision the generation scheme rests on: if `partition_key` could be
rewritten in place, a handle held across the rewrite would silently
begin writing into a different partition, and the guarantee that a
deregistered collection's records are never resurrected would be gone.
Changing a collection's identity is deregister followed by register,
which mints a fresh generation by construction.

**There is no way to enumerate entries.** This is a dated omission
rather than a principle, and it is distinct from the registry-management
surface rejected below -- that one is about listing and deleting whole
registries. Reclaiming orphaned native containers needs enumeration, and
the portability section above shows it stops being optional on providers
with low container caps. A read-only enumeration operation is purely
additive, so it waits for the reclamation design that would consume it.

**`startup` and `shutdown` rather than a constructor that performs
I/O.** Every store in the codebase separates construction from starting,
so wiring code can build the object graph and then start it. `startup`
is idempotent because every process runs it at boot and none is
privileged. `shutdown` is a no-op in the SQLAlchemy implementation
because the engine is shared and externally owned: the registry does not
close what it did not open.

**`concurrency_scope` is declared by the registry** rather than derived
by each store, because the registry is what actually bounds atomicity: a
store reports `min(its own ceiling, its registry's scope)`, so a
cluster-capable backend sitting behind a file-backed SQLite registry
correctly declares itself machine-scoped. One answer per backing store,
not one per consumer. See `concurrency_scopes.md`.

## Storage (SQLAlchemy implementation)

- Table per registry (`collection_registry_<name>`: `key` VARCHAR(255) PK,
  `entry` JSON, JSONB on PostgreSQL). Registry names become SQL identifier
  components, hence the `[a-z0-9_]+`, 32-byte constraint (prefix + name
  stays inside PostgreSQL's 63-byte identifier limit). Storage keys are
  `f"{namespace}/{name}"` -- "/" is outside the identifier charset, so
  distinct pairs can never collide ("__" would be ambiguous).
- The primary key is the concurrency arbiter: `register` is a plain INSERT
  with `IntegrityError` mapped to `CollectionAlreadyRegisteredError` (the
  `SQLAlchemySegmentStore.create_partition` pattern); `get_or_register` is a
  SELECT fast path, then a native `INSERT .. ON CONFLICT DO NOTHING`, then a
  re-SELECT of the winner on conflict. `get_or_register` never compares
  entries -- config equality policy belongs to the store.
- Flat JSON rather than typed columns: every access is get-by-key, pydantic
  validates at the boundary, and typed columns would turn entry evolution
  into DDL migrations in a codebase whose schema management is
  `create_all`-on-startup.
- Dialects: PostgreSQL and SQLite, the two relational providers the server
  offers.

## Format evolution without standing version machinery

There is deliberately no stored format version and no version check. The
evolution policy above is what makes that safe: because every change is
add-optional-only and old rows classify exactly through defaults, a future
breaking change can introduce an entry format version field (default 1)
*at the moment it is first needed* -- the policy guarantees every
pre-existing row is correctly version 1 -- and ship its migration
(migrate-then-deploy with a window; low-downtime migrations are explicitly
not a goal). The accepted residual: an old binary started against migrated
data fails with validation errors at first read rather than being refused
at startup.

## Rejected alternatives

- Distributed locks or leases: operationally heavier, and crash-recovery
  semantics are worse than an insert that is atomic by construction.
- A generic `ConfigRegistry[ConfigT]` (the earlier draft): speculative with
  one consumer kind; see "Scope" above.
- Registry management surface (list/delete registries): registries are
  constructed by wiring code from configuration; the database catalog
  already lists `collection_registry_*` tables, and management operations
  are exactly the power stores must not have.
- Schema-compatibility rules (Confluent-style): the right machinery when
  many independently deployed writers and readers evolve on their own
  schedules; wrong scale for an embedded component with one codebase owning
  both sides.
- A standing registry-of-registries version stamp (an earlier draft:
  declarations table, startup version check, redeclare): under the
  add-optional-only policy the version field is retrofittable with exact
  classification at first need, so the standing machinery insured against a
  scenario the policy already insures, at the cost of real API and doc
  surface. Its one unique protection -- boot-time refusal of old binaries
  against migrated data -- is the residual noted above.
