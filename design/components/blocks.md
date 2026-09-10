# Blocks

New shape for an existing type. A block is one unit of an event's
content: the leaf the segmenter splits, the deriver embeds, expansion
returns and rendering prints. `Block` in `event_memory/data_types.py`
is a discriminated union closed over `TextBlock` (`block_type:
Literal["text"]`, `data_types.py:36`), so a second kind is an edit to
the core union. This makes block kinds a registered family, like
context parts, and makes the kind a system field of the segment.

## What is wanted from it

1. Content of more than one kind, each processed by its own policy:
   plain text first; structured data, HTML or an image reference
   later. Kinds are processing types, not modalities
   (`data_types.py:26`): plain text, JSON and HTML share a modality and
   are processed differently.
2. A library user adds a kind, with its own processing and rendering,
   without editing anything in the core.
3. The kind is filterable efficiently where a hit is a segment, since a
   segment is exactly one block; never on events, which hold several
   blocks of several kinds.
4. Round-trip through the codec, survive a kind the reader does not
   know, and evolve.
5. Validated at the API with a generated schema.
6. A step's policy for one kind is added or replaced without touching
   its policy for the others: the built-in text policy and a library
   user's policy for their kind compose in one segmenter and one
   deriver, the later registration winning for the kinds it names, so
   there is a base to start from and never a whole segmenter to rewrite.

## Types

```python
class Block(BaseModel, ABC):
    kind: str                                # a Literal on each subclass

    def render(self, options: FormatOptions) -> str | None: ...
        # the reader's text for this block; None renders nothing

class TextBlock(Block):                      # kind = "text"
    kind: Literal["text"] = "text"
    text: str

class UnknownBlock(Block):                   # never registered; see below
    kind: str
    data: dict[str, JsonValue]

Event.blocks: list[Block]                    # one or more
Segment.block: Block                         # exactly one
Derivative.block: Block
```

`kind` is a field, not a class variable as on `ContextPart`, because a
block travels in a list and carries its own discriminator, where a part
is keyed by the mapping it sits in. Kind names are `[a-z0-9_]`, bounded
like a property key, unique.

## Registration

As for context parts: a kind table, filled by import for the built-ins
and through the `memmachine.block_kinds` entry-point group for a
library user's own; the table is the only registration point. The
API's `blocks` schema is the discriminated union over the registered
kinds, built from the table at startup, and the codec decodes with the
same union. A library user adds a kind by writing the class,
registering it, and laying a `BlockSegmenter` and a `BlockDeriver` for
it over the tables ("Processing"), which leaves every other kind's
policy as it was (wants 2 and 6).

## Processing

A step's policy is per kind. `BlockSegmenter` and `BlockDeriver` are
the handler contracts: one block of one kind in, segments or
derivatives out. `Segmenter` and `Deriver` are the two objects a memory
holds (`episodic_memory.md`), each a mapping from kind name to handler.
A table is built from handlers in order, a later handler replacing an
earlier one for every kind it names, so a library user's table is the
base with their handlers laid over it (want 6). Nothing is recursive
and nothing is chained: a step looks its block's kind up once and calls
one handler.

```python
class BlockSegmenter(ABC):
    kinds: ClassVar[frozenset[str]]        # the kinds this handler splits

    async def segment_block(
        self, event: Event, index: int, block: Block, *, format_options: FormatOptions
    ) -> list[Segment]: ...
        # the segments of event.blocks[index], offsets within the block;
        # the kind's join contract holds over what it returns

class BlockDeriver(ABC):
    kinds: ClassVar[frozenset[str]]        # the kinds this handler derives from

    async def derive_block(
        self, segment: Segment, *, format_options: FormatOptions
    ) -> list[Derivative]: ...
        # the derivatives of one segment; a derivative's block may be of
        # another kind (an image's caption is text)

class Segmenter:
    def __init__(self, handlers: Iterable[BlockSegmenter] = ()): ...
        # later handlers replace earlier ones for the kinds they name
    def with_handlers(self, *handlers: BlockSegmenter) -> Segmenter: ...
        # a new table: this one with `handlers` laid over it
    async def segment(
        self, event: Event, *, format_options: FormatOptions
    ) -> list[Segment]: ...
        # for each (index, block): the handler for block.kind, else passthrough

class Deriver:                             # the same shape; no handler: no derivatives
```

A handler sees one block and the event or segment it belongs to, so it
reads context parts with `get_part` and adds its own with `with_part`
(`context.md`, "Propagation"), and it decides the offsets and the join
contract of its kind. `format_options` is the memory's
(`episodic_memory.md`): the segmenter passes it through, the deriver
formats the embedded text with it.

The built-in handlers, all for `text`: `TextBlockSegmenter` (the
recursive character splitter, `max_chunk_length`) and
`PassthroughBlockSegmenter` (one segment per block, unchanged);
`WholeTextBlockDeriver` and `SentenceTextBlockDeriver`. A kind a table
has no handler for has one fixed outcome, and a step never raises on
it: the segmenter emits the block as one segment, unchanged, so the
join contract holds (the passthrough handler is the fallback); the
deriver derives nothing from it. Such a block is stored, reconstructed,
returned by expansion and rendered, and is found by search only through
its event's other blocks. That is the outcome for a library user's kind
until they lay their handlers over the table, and for a built-in kind
under a table with no policy for it.

Base and configuration (proposed; the decision above covers the tables
and the override order, not this). The base tables are the kind
table's defaults: each registered kind names a default segmenter
handler and a default deriver handler, `text` naming
`TextBlockSegmenter` and `WholeTextBlockDeriver`, and the base is those
handlers in registration order, built-ins first. Tenant options
(`episodic_memory_manager.md`, `SegmenterOptions` and `DeriverOptions`)
are per kind: an entry names a registered handler and its options for
that kind, and a kind without an entry keeps its default. Today's
`PassthroughSegmenterConf` / `TextSegmenterConf` choice becomes the
`text` entry of `segmenter`; `WholeTextDeriverConf` /
`SentenceTextDeriverConf` becomes the `text` entry of `deriver`.

## Filtering

The kind is a system field of the segment: `block_kinds` on
episodic-memory search and expansion restricts hits, context windows,
the selectivity probe and neighborhoods to segments of those kinds
(`episodic_memory.md`), evaluated on the vector record under the
reserved key `memmachine_block_kind` and on the segment row's
`block_kind` column. Event listing has no kind filter: an event has
several blocks, and "an event with a text block" is not a question the
server answers (want 3). The kind meets the criterion of "Properties
and filtering" for a system field because the server dispatches
processing and rendering on it.

## Rendering

Rendering a segment prints the timestamp, the context parts'
contributions, then `block.render(options)` (`context.md`,
"Rendering"); `TextBlock` renders its text. A kind decides its own
rendering, and `None` prints nothing.

## Storage and compatibility

Blocks are codec-encoded, as a list on the event row and as one block
on the segment row and the derivative; the segment row also carries
the kind name as a plain column, `block_kind`, since the encoded block
cannot be filtered. Decoding a kind the running server does not
register yields an `UnknownBlock` that keeps the kind name and the
data, round-trips unchanged, renders nothing and is processed by no
step, with one log line; nothing is dropped. A registered kind's model
evolves additively (new optional fields, defaults for old rows); a
change that is not additive is a new kind name (want 4). The API
rejects an unknown kind at ingest with `InvalidBlockError` and bounds
an event's blocks by `blocks.max_bytes` (want 5), so unknown blocks
arise only from what the server itself no longer knows, never from a
caller.

## Changes to existing code

- `block_type` (`data_types.py:36`) becomes `kind`, the discriminator
  every registered family uses; the closed `Block` union (`:40`)
  becomes the union built from the kind table; `encode_block` and
  `decode_block` go through it and produce `UnknownBlock`.
- `TextBlock` gains `render`; `EpisodicMemory.render` calls it instead
  of reading `.text`.
- `Segmenter` and `Deriver` (`segmenter/segmenter.py`,
  `deriver/deriver.py`) stop being the ABCs a whole policy implements
  and become the kind tables above; `BlockSegmenter` and `BlockDeriver`
  are the ABCs. `TextSegmenter` becomes `TextBlockSegmenter`,
  `PassthroughSegmenter` becomes `PassthroughBlockSegmenter` and the
  fallback, `WholeTextDeriver` and `SentenceTextDeriver` become the
  `text` derivers; their `NotImplementedError` on another kind
  (`text_segmenter.py:97`, `text_deriver.py:86`, `:111`) goes, since a
  table never routes a handler another kind. `service_locator.py`
  builds the two tables from the per-kind options.
- `Segment` rows gain `block_kind` (`segment_store.md`); vector records
  gain `memmachine_block_kind` (`vector_store.md`); search and expansion
  gain `block_kinds` (`episodic_memory.md`).
