# Server Config Rewrite Plan

## Problem

Configuration is spread across 3 layers (YAML config, partial/merge system, per-session stored JSON), each with its own model classes. Every new memory type requires touching all 3. The Partial/Full class pairs and recursive merge logic add ~200 lines of boilerplate per memory type.

## Current Architecture

```
YAML config (server-level)
  → EpisodicMemoryConfPartial (partial, with merge logic)
  → MemMachine._with_default_episodic_memory_conf() (merges per-request)
  → EpisodicMemoryConf (full, stored per-session in DB as JSON)
  → service_locator (resolves resource IDs to instances)
  → *Params (constructed objects)
```

## Proposed Architecture

### Two-level config: server defaults + per-tenant overrides

**Server config** (global, from YAML, provides resources and defaults):

```python
class ServerMemoryDefaults(BaseModel):
    event_memory: EventMemoryConf | None = None
    semantic_memory: SemanticMemoryConf | None = None

class Configuration(BaseModel):
    resources: ResourcesConf
    memory_defaults: ServerMemoryDefaults
    server: ServerConf = ServerConf()
```

**Tenant config** (per-session, stored in DB, all fields optional):

```python
class TenantEventMemoryConf(BaseModel):
    embedder: str | None = None
    reranker: str | None = None
    vector_store: str | None = None
    segment_store: str | None = None
    vector_dimensions: int | None = None
    indexed_properties_schema: dict[str, str] | None = None

class TenantMemoryConf(BaseModel):
    event_memory: TenantEventMemoryConf | None = None
    semantic_memory: TenantSemanticMemoryConf | None = None
```

**Resolution** — one level of fallback, no recursive merge:

```python
def resolve(tenant: TenantEventMemoryConf, default: EventMemoryConf) -> EventMemoryConf:
    return EventMemoryConf(
        embedder=tenant.embedder or default.embedder,
        reranker=tenant.reranker or default.reranker,
        vector_store=tenant.vector_store or default.vector_store,
        segment_store=tenant.segment_store or default.segment_store,
        vector_dimensions=tenant.vector_dimensions or default.vector_dimensions,
        indexed_properties_schema=tenant.indexed_properties_schema or default.indexed_properties_schema,
    )
```

### Key design decisions

- `None` means disabled. No separate `enabled` booleans.
- No Partial/Full class pairs. Tenant config has optional fields, server config has required fields.
- No nested merge — resolution is one `or` per field.
- Server config doesn't know about memory types beyond providing defaults.
- Tenant config is self-contained per memory type.
- Resolution produces the same config types that the memory implementations already use.

### Multi-tenant with different memory types

Each tenant declares which memory types are active via `TenantMemoryConf`. `None` disables a memory type for that tenant. No strategy pattern needed — just null checks.

### What gets deleted

- `EpisodicMemoryConfPartial`, `LongTermMemoryConfPartial`, `ShortTermMemoryConfPartial`
- `merge_partial_configs()` and all `merge()` methods
- `MemMachine._with_default_episodic_memory_conf()`
- `MemMachine._initialize_default_episodic_configuration()`
- `Configuration._maybe_disable_*` validators
- `EpisodicMemoryManager` (LRU cache only needed for STM's in-memory state)
- `LongTermMemory`, `DeclarativeMemory`, `ShortTermMemory` (if fully replaced by EventMemory)

### What stays

- `ResourcesConf` / `ResourceManagerImpl` — resource ID → instance resolution
- Memory implementations (`EventMemory`, `SemanticService`)

### Split SessionDataManager

`SessionDataManager` currently mixes three concerns:
1. **Session CRUD** — create, delete, list, get info
2. **Memory config persistence** — episodic config flags, event memory config, per-memory-type methods
3. **Runtime state** — short-term memory summaries (save/get)

Each new memory type adds more methods to this interface. Split into:
- **SessionManager** — session CRUD only (create, delete, list, get info)
- **TenantConfigStore** — per-tenant memory config (replaces `param_data` + `configuration` JSON columns with a structured config per memory type)
- **ShortTermMemoryStore** — STM summary state (or removed if STM is removed)

### Package restructure

Split into four packages to eliminate duplicate data types and support core as a standalone library:

```
memmachine-types  ← memmachine-core   ← memmachine-server
      ↑                                        ↓
memmachine-client ← memmachine-common ←────────┘
```

**memmachine-types** (3.10+, pydantic only)
- Pure data models: Event, Segment, QueryResult, ScoredSegmentContext, PropertyValue, SimilarityMetric, FormatOptions
- Serialization/deserialization: properties JSON encode/decode, field_serializer/field_validator on models
- Formatting: string_from_segment_context, build_query_result_context, timestamp formatting — as methods on data models (e.g. `QueryResult.__str__`, `ScoredSegmentContext.to_string(format_options)`) so client can format results without depending on core
- No memory logic, no ABCs, no heavy dependencies
- Constraint: must avoid 3.11+ features (StrEnum, Self, ExceptionGroup) for client compatibility

**memmachine-core** (3.12+, depends on memmachine-types)
- Memory implementations: EventMemory, SemanticService
- ABCs: VectorStore, SegmentStore, Embedder, Reranker
- Can use 3.12+ features (type aliases, match/case, etc.)

**memmachine-common** (3.10+, depends on memmachine-types)
- HTTP API contracts: AddMemoriesSpec, SearchResult, MemoryType, etc.
- Request/response models only — no data type duplicates
- Today's EventMemorySegment/EventMemoryScoredContext wrappers go away; server serializes core types directly

**memmachine-server** (3.12+, depends on core + common)
- MemMachine orchestration, config, session management
- HTTP routes/service layer
- Episode→Event conversion, lifecycle management

**memmachine-client** (3.10+, depends on common)
- HTTP client, gets types transitively through common → types

This eliminates:
- Duplicate API data types (EventMemorySegment vs Segment, etc.)
- Manual field-by-field mapping in service.py
- Client needing to know about internal vs API type differences

Core consumers (custom servers, notebooks, CLIs) depend on memmachine-core only. API consumers (client apps) depend on memmachine-common only. Nobody pulls in what they don't need.

### Migration

The rewrite happens naturally when LTM/STM are removed. The new config is what's left after deleting the old stuff. Backward compatibility with stored `param_data` JSON requires a one-time migration of existing sessions.
