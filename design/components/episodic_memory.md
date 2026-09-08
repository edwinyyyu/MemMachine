# EpisodicMemory

Existing component, `episodic_memory/event_memory/event_memory.py`,
`EventMemory`, renamed. Processes events into segments and derivative
embeddings and answers vector searches over them. A configured object
bound to one tenant: built by `EpisodicMemoryManager` per request from
the tenant's handles and configuration, never by the composition, never
cached, interchangeable with any other built the same way; a library
user builds one directly.

## Constructed with

```python
EpisodicMemory(
    partition: SegmentPartition,    # handle: this tenant's segments
    collection: VectorCollection,   # handle: this tenant's records in its embedder's container
    segmenter: Segmenter,
    deriver: Deriver,
    embedder: Embedder,
    format_options: FormatOptions,  # how the deriver renders dates and names into embedded text
    metrics_factory: MetricsFactory | None,
)
```

One tenant, one embedder, one segmenter, one deriver, one way of
formatting the text that is embedded. No reranker: reranking is a stage
the manager runs on top (`episodic_memory_manager.md`). No operation
names a key: the two handles are stateless bindings of the stores to
the tenant's key (`server_redesign.md`, "Vocabulary"), so past
construction nothing can route to another tenant, and a handle to a
deleted tenant raises `KeyNotLiveError` from the store's fence.
`format_options` is a constructor parameter, not a call argument,
because it decides what is written: the deriver formats a segment's
timestamp and author into the text it embeds, and a tenant's
derivatives must be formatted one way. It is a mutable tenant option
(`episodic_memory.format`), applying to events processed after a
change.

## API

```python
class SearchHit(BaseModel):
    score: float                    # cosine similarity of the matched derivative
    seed: int                       # index in `segments` of the matched segment
    segments: list[Segment]         # the context window, in the store's order

class EpisodicMemory:
    async def encode(self, events: Iterable[StoredEvent]) -> None
    async def forget(self, event_uuids: Iterable[UUID]) -> None
    async def query(self, query: str, *,
                    limit: int, min_similarity: float | None,
                    expand_context: int,
                    since: datetime | None, before: datetime | None,
                    session_ids: Iterable[str] | None,
                    source_ids: Iterable[str] | None,
                    block_kinds: Iterable[str] | None,
                    filter: FilterExpr | None) -> list[SearchHit]
    async def expand(self, anchor: UUID, *,
                     before: int, after: int,
                     unit: Literal["segments", "events"],
                     source_ids: Iterable[str] | None,
                     block_kinds: Iterable[str] | None) -> list[Segment]
    @staticmethod
    def render(segments: Iterable[Segment], *,
               format_options: FormatOptions) -> str
```

- `encode`: for each event, first `forget` its derived rows (so a
  repeat leaves one copy), then segment, derive, embed; write segments
  to the segment store; upsert derivatives to the vector store with the
  declared properties (system fields under reserved keys, plus the
  declared user keys). Segment and derivative uuids are `uuid4`. The
  caller's batch is fully written when `encode` returns; the manager
  advances the watermark only then.
- `forget`: look up segments by event uuids and derivatives by segment
  uuids; delete vector records; delete segments.
- `query`: one stage, vector search. Embed the query; split `filter`
  into the declared part and the rest (`filters_and_properties.md`);
  choose the plan. Selective plan: if `find_segments` with the system
  filters and the undeclared part returns at most
  `filter.selective_limit` segments, score their derivatives with
  `get_cosine_similarity`, drop those below `min_similarity`, and keep
  the best `limit`. Broad plan: `collection.query` with the declared
  part and the system filters as predicates on reserved keys, `limit`
  widened up to `filter.max_overfetch` while the segment store rejects
  seeds against the undeclared part, and cut to `limit`. Then
  `get_segment_contexts` for the surviving seeds with `expand_context`
  split as today (`event_memory.py:450`) and the same filters, which
  bound the window rows too. Returns at most `limit` hits in descending
  score, one per matched derivative, each carrying its window and the
  index of the matched segment in it; windows of different hits may
  overlap, and each hit is returned whole. Every count is a maximum:
  a filtered search returns fewer when the filter admits fewer.
- `expand`: the neighbourhood of an anchor in its session's one total
  order (`segment_store.md`), as claude-memory's `memory_expand` walks
  a conversation around a memory. The anchor is a segment uuid (from a
  hit) or an event uuid (its first segment). `unit` says what `before`
  and `after` count, segments or whole events; the walk stays in the
  anchor's session, and `source_ids` and `block_kinds` restrict it.
  Returns the segments in the store's order, the anchor among them; a
  caller walks further by calling again with the first or last segment
  as the anchor and one side zero. Backed by
  `SegmentPartition.get_neighbours` over the ordering index; no vector
  search and no embedding, so it is one indexed read.
- `render`: the reader's text for a run of segments, in their order:
  each segment's timestamp formatted by `format_options`, its context
  parts' contributions, and its block's rendering (`context.md`,
  `blocks.md`). What the API returns as `text`, and what a reranker
  scores.

## Context

Specified in `context.md`: a mapping from part kind to one registered
part, never `None`, composed by `with_part`, read by `get_part`, never
filtered. The deriver reads `Author` to format text, and the temporal
scorer reads `TimeRanges`.

## Blocks

Specified in `blocks.md`: a registered family of kinds, `text` built
in; the segmenter and deriver dispatch on the kind and pass a kind
they do not handle through as one segment with no derivatives; a
segment is one block, so its kind is a system field filtered by
`block_kinds`; rendering calls `block.render`.

## Changes required

- Rename `EventMemory` to `EpisodicMemory`; `EventMemoryParams`
  (`event_memory.py:52`) to constructor parameters.
- `segment_store_partition` and `vector_store_collection` (`:76`, `:80`)
  stay as dependencies and become the stateless handles
  `SegmentPartition` and `VectorCollection`; no operation takes a key.
- `reranker` leaves the constructor (`:96`) and the class: reranking is
  the manager's stage.
- `encode_events` (`:200`) becomes `encode`, idempotent per event by
  forgetting first, taking `StoredEvent`s so a segment can carry the
  event's position; `forget_events` (`:680`) becomes `forget`.
- `query` (`:353`): `vector_search_limit` becomes `limit` with
  maximum semantics, the threshold becomes `min_similarity` on cosine
  similarity; `since`, `before`, `session_ids`, `source_ids` and
  `block_kinds` are added as typed parameters; the reserved-key mapping
  `_to_vector_record_property` (`:340`) and the `m.` user prefix go,
  replaced by `filters_and_properties.md`'s reserved namespace; the plan
  split is added; the result is `list[SearchHit]`.
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
  locale and timezone. `produced_for` and the producer roles of the old
  episode model are not carried over and nothing replaces them.
- `expand` is added, with `get_neighbours` on the segment store;
  `string_from_segment_context` and `string_from_segment_contexts`
  become `render`.
- Scores are cosine similarity; `SimilarityMetric` goes from the
  embedder, the vector store and the engines, as on the reference
  branch (commit 6ab12098): the embedder exposes `model_id` and
  `dimensions` only, every container and engine is configured for
  cosine, `higher_is_better` and the metric-dependent scoring branches
  in `event_memory.py` go.
- `Embedder.ingest_embed` and `search_embed` take `list[str]`, not
  `list[Any]` (reference branch, commit ae1d616a); the only inputs are
  derivative texts and the query.
