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
    vector_store: VectorStore,          # the view for this embedder's container
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
    async def process(self, key: UUID, events: Iterable[StoredEvent], *,
                      format_options: FormatOptions | None = None) -> None
    async def forget(self, key: UUID, event_uuids: Iterable[UUID]) -> None
    async def query(self, key: UUID, query: str, *,
                    limit: int, expand_context: int,
                    min_score: float | None,
                    reranker: Reranker | None,
                    since: datetime | None, before: datetime | None,
                    producers: Iterable[str] | None,
                    filter: FilterExpr | None,
                    format_options: FormatOptions | None = None) -> QueryResult
    @staticmethod
    def string_from_segment_context(segment_context: Iterable[Segment], *,
                                    format_options: FormatOptions | None = None) -> str
    @staticmethod
    def string_from_segment_contexts(segment_contexts: Iterable[Iterable[Segment]], *,
                                     format_options: FormatOptions | None = None) -> str
```

- `process`: for each event, first `forget` its derived rows (so a
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
  declared part, `since`, `before` and `producers` as filter predicates
  on reserved keys, over-fetching up to `filter.max_overfetch` and
  dropping seeds the segment store rejects; then `get_segment_contexts`
  with `expand_context` split as today (`event_memory.py:450`); score
  by embedding similarity or, with `reranker`, by
  `reranker.score`; return at most `limit` contexts, fewer when the
  filter admits fewer.

## Changes required

- Rename `EventMemory` to `EpisodicMemory`; `EventMemoryParams`
  (`event_memory.py:52`) to constructor parameters.
- `segment_store_partition` and `vector_store_collection` (`:76`, `:80`)
  become the stores; every operation takes `key: UUID`.
- `reranker` leaves the constructor (`:96`) and becomes a `query`
  argument.
- `encode_events` (`:200`) becomes `process`, idempotent per event by
  forgetting first; `forget_events` (`:680`) becomes `forget`.
- `query` (`:353`): `vector_search_limit` becomes `limit` with
  maximum semantics; `since`, `before`, `producers` are added as typed
  parameters; the reserved-key mapping `_to_vector_record_property`
  (`:340`) and the `m.` user prefix go, replaced by
  `filters_and_properties.md`'s reserved namespace; the plan split is
  added.
- `_SEGMENT_UUID_FIELD_NAME` and `_TIMESTAMP_FIELD_NAME` (`:111`, `:112`)
  become reserved keys built by `reserved_property_key`;
  `expected_vector_store_collection_schema` (`:118`) goes, since the
  store's schema is settings.
- Ingest order is unchanged (segments, then vectors).
