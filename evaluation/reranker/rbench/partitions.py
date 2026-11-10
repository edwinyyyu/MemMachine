"""Every dataset as EventMemory partitions: one code path for chat histories and documents.

Each query searches one partition of MemMachine's EventMemory (memory.py), chosen by its
dataset's scope:

  history     LongMemEval: the question's own chat history, one event per turn (histories.py)
  doc         the query's own document
  corpus      the whole corpus, one partition every query shares
  scope_docs  the query's candidate documents (SWE-bench: one repository snapshot's files)

A document is an event with no producer (NullContext) and one text block, in corpus order.
Non-chat text has no author or time, so documents are embedded and rendered without dates
(DOCUMENTS); chat histories keep MemMachine's defaults (dated embedding text, short times in
rendered contexts). Partitions are keyed by the question id (history) or a hash of their
documents: EventMemory's partition keys are [a-z0-9_]{1,32}.

Gold, whatever its source, becomes character spans in event texts: a LongMemEval gold turn is
a whole-turn span, a document dataset's span keeps its offsets. Every segment carries its
character offsets within its event's text (found in order: TextSegmenter's segments are
substrings of the text with edge whitespace stripped; the simple chunker's tile it), so a
span maps to the segments overlapping it after metrics.trim cuts it to its non-whitespace
extent, as metrics.gold maps spans to chunks.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from rbench import metrics
from rbench.datasets import info
from rbench.memory import Memory  # imports MemMachine (quietly) before the names below
from rbench.memory import segmenter as memory_segmenter
from rbench.reading import constants
from rbench.schema import corpus, queries
from rbench.text import GEMMA, count

from memmachine_server.episodic_memory.event_memory.data_types import (  # noqa: E402, I001
    Event,
    FormatOptions,
    NullContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import WholeTextDeriver  # noqa: E402
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory  # noqa: E402

NAMESPACE = uuid.UUID("6f1d8a52-3c1e-4c4f-9a43-2b6f1f0e9d11")  # event uuids: uuid5 of their source
TICK = datetime.timedelta(microseconds=1)  # orders events; never rendered
EPOCH = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)  # documents' timestamps (never rendered)
DOCUMENTS = FormatOptions(date_style=None, time_style=None)  # no date or time anywhere


@dataclass(frozen=True)
class Formats:
    """What EventMemory is given for a dataset: embed for encode_events (the deriver's embedding
    text; None is MemMachine's default), rerank for query (the contexts its reranker scores),
    read for string_from_query_result (what a reader gets; None is its default)."""

    embed: FormatOptions | None
    rerank: FormatOptions
    read: FormatOptions | None


CHAT = Formats(None, FormatOptions(time_style="short"), None)
DOCS = Formats(DOCUMENTS, DOCUMENTS, DOCUMENTS)


def formats(dataset: str) -> Formats:
    return CHAT if info(dataset).partitioned else DOCS


@dataclass(frozen=True)
class Part:
    """One partition's events in order, what each is (a turn or a document), and each event's
    text (spans are offsets into it)."""

    key: str
    events: tuple[Event, ...]
    labels: tuple[str, ...]  # "session s, turn t" or the document id
    texts: tuple[str, ...]

    @property
    def number(self) -> dict[str, int]:
        """Event number by label."""
        return {label: i for i, label in enumerate(self.labels)}


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:20]


def scope_of(dataset: str, scope: str | None = None) -> str:
    return scope or info(dataset).scope


def key(dataset: str, q: dict, scope: str | None = None) -> str:
    """The partition a query searches."""
    scope = scope_of(dataset, scope)
    if scope == "history":
        return q["qid"]
    if scope == "corpus":
        return "corpus"
    if scope == "doc":
        return "d" + _hash(q["doc_id"])
    if scope == "scope_docs":
        return "s" + _hash("\n".join(sorted(q["scope_docs"])))
    raise ValueError(f"{dataset}: unknown scope {scope!r}")


def documents(dataset: str, q: dict, scope: str | None = None) -> list[str]:
    """The documents a query's partition holds, in corpus order."""
    scope = scope_of(dataset, scope)
    docs = corpus(dataset)
    if scope == "corpus":
        return list(docs.doc_ids)
    names = [q["doc_id"]] if scope == "doc" else q["scope_docs"]
    return sorted(names, key=docs.index.__getitem__)


@lru_cache(maxsize=8)
def _document_part(dataset: str, k: str, names: tuple[str, ...]) -> Part:
    docs = corpus(dataset)
    texts = tuple(docs.texts[docs.index[n]] for n in names)
    events = tuple(Event(uuid=uuid.uuid5(NAMESPACE, f"{dataset}/{n}"), timestamp=EPOCH + i * TICK,
                         context=NullContext(), blocks=[TextBlock(text=t)])
                   for i, (n, t) in enumerate(zip(names, texts, strict=True)))
    return Part(k, events, names, texts)


def part(dataset: str, q: dict, scope: str | None = None) -> Part:
    """A query's partition: its events, labels and texts."""
    k = key(dataset, q, scope)
    if scope_of(dataset, scope) == "history":
        from rbench import histories

        return histories.part(dataset, q["qid"])
    return _document_part(dataset, k, tuple(documents(dataset, q, scope)))


def gold_spans(dataset: str, q: dict, p: Part) -> list[list[tuple[int, int, int]]]:
    """A query's gold as alternatives (any one answers it) of spans (all needed), each span
    (event number, start, end) in its event's text."""
    if info(dataset).partitioned:  # LongMemEval: its gold turns, whole, all needed
        number = p.number
        spans = [number[f"session {s}, turn {t}"] for s, t in q["gold_turns"][0]]
        return [[(e, 0, len(p.texts[e])) for e in spans]]
    number = p.number
    return [[(number[doc_id], s, e) for doc_id, s, e in alternative] for alternative in q["gold"]]


@dataclass(frozen=True)
class Segments:
    """One partition's segments in EventMemory's order (ids are positions)."""

    uuid: tuple[str, ...]  # the segment store's uuids (query results refer to them)
    event: np.ndarray  # the event's number in the partition (Part order): one event per turn or document
    offset: np.ndarray  # the segment's position within its event
    start: np.ndarray  # character offsets within the event's text
    end: np.ndarray
    tokens: np.ndarray  # embeddinggemma tokens of the segment text
    inner: np.ndarray  # reader-cost pieces (reading.Reader)
    run: np.ndarray

    def __len__(self) -> int:
        return len(self.uuid)

    @property
    def chars(self) -> np.ndarray:
        """Each segment's length in characters."""
        return self.end - self.start

    def text(self, p: Part, i: int) -> str:
        """Segment i's text: its span of its event's text (segments are substrings of it, found
        by offsets), so the table keeps no copy."""
        return p.texts[self.event[i]][self.start[i] : self.end[i]]


COLUMNS = ("event", "offset", "start", "end", "tokens", "inner", "run")


def _costs(segments: list[Segment], read: FormatOptions | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Token counts per segment: its text, its JSON-escaped text (inner), and a run starting
    at it (header, quotes and newline), with the header as EventMemory renders it for a reader
    (string_from_segment_context in the dataset's reading format)."""
    quotes, newline, _ = constants()
    texts = [s.block.text for s in segments]
    headers = [EventMemory.string_from_segment_context([s.model_copy(update={"block": TextBlock(text="")})],
                                                       format_options=read)[:-2] for s in segments]
    return (count(texts, GEMMA), count([json.dumps(t, ensure_ascii=False)[1:-1] for t in texts], GEMMA),
            count(headers, GEMMA) + quotes + newline)


def _ordered(segments: Iterable[Segment]) -> list[Segment]:
    return sorted(segments, key=lambda s: (s.timestamp, s.event_uuid, s.index, s.offset))


def offsets(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Where each of an event's segments lies in its text, in order. Segments are the text's
    pieces in order, each a substring (TextSegmenter strips edge whitespace between them)."""
    out, cursor = [], 0
    for piece in pieces:
        at = text.find(piece, cursor)
        if at < 0:
            raise ValueError(f"segment not found in its event's text after offset {cursor}: {piece[:60]!r}")
        out.append((at, at + len(piece)))
        cursor = at + len(piece)
    return out


def _table(dataset: str, p: Part, segments: list[Segment]) -> Segments:
    number = {e.uuid: i for i, e in enumerate(p.events)}
    event = [number[s.event_uuid] for s in segments]
    bounds: list[tuple[int, int]] = [(0, 0)] * len(segments)
    for e in sorted(set(event)):
        rows = [i for i, x in enumerate(event) if x == e]
        for i, b in zip(rows, offsets(p.texts[e], [segments[i].block.text for i in rows]), strict=True):
            bounds[i] = b
    tokens, inner, run = _costs(segments, formats(dataset).read)
    ints = lambda v: np.asarray(v, dtype=np.int64)  # noqa: E731
    return Segments(tuple(str(s.uuid) for s in segments), ints(event), ints([s.offset for s in segments]),
                    ints([b[0] for b in bounds]), ints([b[1] for b in bounds]), tokens, inner, run)


def _save_table(path, stamp: str, seg: Segments) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, stamp=np.str_(stamp), uuid=np.asarray(seg.uuid, dtype="S"),
                        **{c: getattr(seg, c) for c in COLUMNS})


async def table(m: Memory, dataset: str, p: Part) -> Segments:
    """A stored partition's segments, read once per encoding and cached beside the stores,
    compressed. No texts (Segments.text reads them from the Part; segments.sqlite holds them
    too), and uuids as ASCII bytes: numpy's str arrays are fixed-width UTF-32. A table written
    before (with a UTF-32 copy of the texts, about 2 KB per segment) is rewritten in this form
    when read."""
    path = m.root / "tables" / f"{p.key}.npz"
    stamp = m.done(p.key).read_text()
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if str(data["stamp"]) == stamp and all(c in data for c in ("uuid", *COLUMNS)):
            seg = Segments(tuple(data["uuid"].astype(str).tolist()), *(data[c] for c in COLUMNS))
            if "texts" in data:  # the earlier form: drop the copy of the texts
                data.close()
                _save_table(path, stamp, seg)
            return seg
    segment_store, _, _ = await m.stores()
    partition = await segment_store.open_partition(p.key)
    by_event = await partition.get_segment_uuids_by_event_uuids([e.uuid for e in p.events])
    ids = [u for e in p.events for u in by_event.get(e.uuid, [])]
    found: dict = {}
    for start in range(0, len(ids), 1000):
        found.update({k: v[0] for k, v in (await partition.get_segment_contexts(ids[start : start + 1000])).items()})
    seg = _table(dataset, p, _ordered(found.values()))
    _save_table(path, stamp, seg)
    return seg


def groups(dataset: str, picked: list[int], scope: str | None = None) -> dict[str, list[int]]:
    """Sampled query indices by the partition they search (first-seen order), so each
    partition's stores are opened once."""
    rows = queries(dataset)
    out: dict[str, list[int]] = {}
    for i in picked:
        out.setdefault(key(dataset, rows[i], scope), []).append(i)
    return out


def ingest(embedder: str, dataset: str, chunking: str, picked: list[int], scope: str | None = None) -> None:
    """Encode each sampled query's partition (embedding it), skipping those already in."""
    rows = queries(dataset)

    from rbench.progress import Progress

    async def go() -> None:
        keyed = groups(dataset, picked, scope)
        progress, encoded = Progress(f"{dataset} {chunking} ingest", len(keyed), "partitions"), 0
        for k, members in keyed.items():
            p = part(dataset, rows[members[0]], scope)
            async with Memory(embedder, dataset, chunking) as m:
                encoded += await m.ingest(k, list(p.events), formats(dataset).embed)
            progress.step(note=f"{encoded} embedded now, the rest already were")

    asyncio.run(go())


@lru_cache(maxsize=4)
def offline(dataset: str, k: str, chunking: str, scope: str | None = None) -> tuple[Segments, tuple[Segment, ...], tuple[str, ...]]:
    """(segments, MemMachine Segment objects, embedding texts) of partition k without a store:
    its events through the chunking's segmenter and MemMachine's WholeTextDeriver in process.
    Texts and order are the stored ones; only the uuids differ (the segmenter draws them)."""
    rows = queries(dataset)
    q = next(r for r in rows if key(dataset, r, scope) == k)
    p = part(dataset, q, scope)
    embed = formats(dataset).embed

    async def go() -> tuple[list[Segment], list[str]]:
        segmenter, deriver = memory_segmenter(chunking), WholeTextDeriver()
        segments = [s for e in p.events for s in await segmenter.segment(e, format_options=embed)]
        texts = [d.block.text for s in segments for d in await deriver.derive(s, format_options=embed)]
        return segments, texts

    segments, texts = asyncio.run(go())
    order = sorted(range(len(segments)), key=lambda i: (segments[i].timestamp, segments[i].event_uuid,
                                                         segments[i].index, segments[i].offset))
    segments, texts = [segments[i] for i in order], [texts[i] for i in order]
    return _table(dataset, p, segments), tuple(segments), tuple(texts)


def embedding_tokens(dataset: str, k: str, chunking: str, scope: str | None = None, embedder: str | None = None) -> int:
    """Tokens a partition's embedding costs: its embedding texts (sent as they are, as
    EventMemory's OpenAIEmbedder sends them) plus the embedder's special tokens per text,
    counted with the embedder's tokenizer (embeddinggemma's by default)."""
    from rbench.embedding import EMBEDDERS

    _, _, texts = offline(dataset, k, chunking, scope)
    spec = EMBEDDERS.get(embedder) if embedder else None
    tokenizer, special = (spec.tokenizer, spec.special_tokens) if spec else (GEMMA, 2)
    return int(count(list(texts), tokenizer).sum()) + special * len(texts)


def gold(spans: list[list[tuple[int, int, int]]], p: Part, seg: Segments) -> metrics.Gold:
    """Gold in segment terms (metrics.Gold): each span trimmed to its non-whitespace extent,
    then the segments of its event overlapping it, with the characters of it each holds."""
    by_event: dict[int, np.ndarray] = {}
    alternatives, pieces = [], {}
    for alternative in spans:
        out = []
        for e, s, t in alternative:
            s, t = metrics.trim(p.texts[e], s, t)
            rows = by_event.setdefault(e, np.flatnonzero(seg.event == e))
            hit = rows[(seg.start[rows] < t) & (seg.end[rows] > s)]
            span = []
            for c in hit:
                a, b = max(s, int(seg.start[c])), min(t, int(seg.end[c]))
                span.append((int(c), b - a))
                pieces.setdefault(int(c), []).append((a, b))
            out.append(tuple(span))
        alternatives.append(tuple(out))
    cost = [sum(int(seg.tokens[c]) for c in {c for span in alt for c, _ in span}) for alt in alternatives]
    best = alternatives[int(np.argmin(cost))]
    return metrics.Gold(tuple(alternatives), {c: metrics.merged_length(v) for c, v in pieces.items()},
                        tuple(sorted({c for span in best for c, _ in span})))
