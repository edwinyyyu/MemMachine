# EpisodicMemory

Existing component, `episodic_memory/event_memory/event_memory.py`,
`EventMemory`, renamed. Processes events into segments and derivative
embeddings and answers searches. A configured object: built by
`EpisodicMemoryManager` per structural configuration, never by the
composition, interchangeable with any other built from the same
configuration.

## Constructed with

```python
EpisodicMemory(
    segment_store: SegmentStore,
    vector_store: VectorStore,      # the view for this embedder's container
    segmenter: Segmenter,
    deriver: Deriver,
    embedder: Embedder,
    metrics_factory: MetricsFactory | None,
)
```

One embedder, one segmenter, one deriver. No reranker: it is a per-call
argument. No store handles: operations take the key.

## API

```python
class EpisodicMemory:
    async def encode(self, key: UUID, events: Iterable[StoredEvent], *,
                      format_options: FormatOptions | None = None) -> None
    async def forget(self, key: UUID, event_uuids: Iterable[UUID]) -> None
    async def query(self, key: UUID, query: str, *,
                    limit: int, expand_context: int,
                    min_score: float | None,
                    reranker: Reranker | None,
                    since: datetime | None, before: datetime | None,
                    source_ids: Iterable[str] | None,
                    filter: FilterExpr | None,
                    format_options: FormatOptions | None = None) -> QueryResult
    async def expand(self, key: UUID, anchor: UUID, *,
                     before: int, after: int,
                     unit: Literal["segments", "events"],
                     source_ids: Iterable[str] | None) -> Expansion
    @staticmethod
    def string_from_segment_context(segment_context: Iterable[Segment], *,
                                    format_options: FormatOptions | None = None) -> str
    @staticmethod
    def string_from_segment_contexts(segment_contexts: Iterable[Iterable[Segment]], *,
                                     format_options: FormatOptions | None = None) -> str
```

- `encode`: for each event, first `forget` its derived rows (so a
  repeat leaves one copy), then segment, derive, embed; write segments
  to the segment store; upsert derivatives to the vector store with the
  declared properties (system fields under reserved keys, plus the
  declared user keys). Segment and derivative uuids are `uuid4`.
- `forget`: look up segments by event uuids and derivatives by segment
  uuids; delete vector records; delete segments.
- `query`: embed the query; split `filter` into the declared part and
  the rest (`filters_and_properties.md`); choose the plan: if the
  undeclared part is selective by `find_segments` up to
  `filter.selective_limit`, score the matching segments' derivatives
  with `get_cosine_similarity`; otherwise `vector_store.query` with the
  declared part, `since`, `before` and `source_ids` as filter predicates
  on reserved keys, over-fetching up to `filter.max_overfetch` and
  dropping seeds the segment store rejects; then `get_segment_contexts`
  with `expand_context` split as today (`event_memory.py:450`); score
  by embedding similarity or, with `reranker`, by
  `reranker.score`; return at most `limit` contexts, fewer when the
  filter admits fewer.

- `expand`: the neighbourhood of an anchor, in the tenant's one total
  order (`timestamp`, then event, index, offset), as claude-memory's
  `memory_expand` walks a conversation around a memory. The anchor is a
  segment uuid (from a search hit) or an event uuid (its first
  segment). `unit` counts what `before` and `after` mean: segments, or
  whole events; `source_ids` restricts the walk. Returns the segments in
  order with the anchor marked, and a cursor at each end so a caller
  walks further by repeating the call from the last segment. Backed by
  `SegmentStore.get_neighbours` over the ordering index; no vector
  search and no embedding, so it is one indexed read.

## Context

```python
class ContextPart(BaseModel):            # registered under a kind
    kind: ClassVar[str]

class Author(ContextPart):               # kind "author"
    name: str

class TimeRanges(ContextPart):           # kind "time_ranges"; from #1436
    time_ranges: list[TimeRange]

Context = Mapping[str, ContextPart]      # at most one part per kind

def get_part[P: ContextPart](context: Context, part: type[P]) -> P | None
def with_part(context: Context, part: ContextPart) -> Context   # merge
```

- Kinds register in a table like store kinds do, by import for
  built-ins and through the `memmachine.context_parts` entry-point
  group for a library user's own; the API validates an incoming
  `context` object against the registered kinds and rejects an unknown
  one, and the codec round-trips a part unchanged.
- Composition is `with_part`: a segmenter that extracts time ranges
  merges a `TimeRanges` part into the segment's context, and whatever
  the event carried stays. There is no order and no nesting, so no step
  depends on which part came first, which was the fault line in #1436's
  `CompositeContext`.
- Reading is `get_part(context, Author)`: the renderer reads `Author`,
  a temporal scorer reads `TimeRanges`, and each ignores the rest. A
  part is never filtered on; what needs filtering is a property or a
  system field.
- Rendering (`string_from_segment_context`) prints the `Author` part's
  `name` when present and nothing otherwise; a source with no good name
  is one with a `source_id` and no `author` part.

## Changes required

- Rename `EventMemory` to `EpisodicMemory`; `EventMemoryParams`
  (`event_memory.py:52`) to constructor parameters.
- `segment_store_partition` and `vector_store_collection` (`:76`, `:80`)
  become the stores; every operation takes `key: UUID`.
- `reranker` leaves the constructor (`:96`) and becomes a `query`
  argument.
- `encode_events` (`:200`) becomes `encode`, idempotent per event by
  forgetting first; `forget_events` (`:680`) becomes `forget`.
- `query` (`:353`): `vector_search_limit` becomes `limit` with
  maximum semantics; `since`, `before`, `source_ids` are added as typed
  parameters; the reserved-key mapping `_to_vector_record_property`
  (`:340`) and the `m.` user prefix go, replaced by
  `filters_and_properties.md`'s reserved namespace; the plan split is
  added.
- `_SEGMENT_UUID_FIELD_NAME` and `_TIMESTAMP_FIELD_NAME` (`:111`, `:112`)
  become reserved keys built by `reserved_property_key`;
  `expected_vector_store_collection_schema` (`:118`) goes, since the
  store's schema is settings.
- Ingest order is unchanged (segments, then vectors).
- `ProducerContext` and `NullContext` (`data_types.py:49`, `:56`) and
  the `Context` discriminated union (`:64`) go; `Event.source_id: str |
  None` is added beside `timestamp` as the filterable identity, and
  `Event.context` becomes the keyed mapping of registered parts above,
  with `Author` and `TimeRanges` as the first kinds. `Segment` and
  `Derivative` carry both. Rendering prints the recorded name; a caller
  that wants current names or ids shown renders from the returned
  `source_id` and context itself. `FormatOptions` stays dates, times,
  locale and timezone.
  `produced_for` and the producer roles of the old episode model are not
  carried over and nothing replaces them.
- `expand` is added, with `get_neighbours` on the segment store.
- Scores are cosine similarity; `SimilarityMetric` goes from the
  embedder, the vector store and the engines, as on the reference
  branch (commit 6ab12098): the embedder exposes `model_id` and
  `dimensions` only, every container and engine is configured for
  cosine, `higher_is_better` and the metric-dependent scoring branches
  in `event_memory.py` go, and the threshold is `min_score` on cosine
  similarity.
- `Embedder.ingest_embed` and `search_embed` take `list[str]`, not
  `list[Any]` (reference branch, commit ae1d616a); the only inputs are
  derivative texts and the query.
