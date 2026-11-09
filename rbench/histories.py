"""Chat-history datasets (LongMemEval) as EventMemory partitions, one per question (memory.py).

Each turn is an EventMemory event: producer User or Assistant, one text block, a uuid derived
from the question, session and turn. EventMemory orders a partition by (timestamp, event,
index, offset). LongMemEval dates sessions to the minute, gives turns no time, and its files
are not in date order (40 of 94 sampled M histories are not; most have sessions starting in
the same minute). So sessions are sorted by date, ties kept in file order, and each turn's
timestamp is its session's date plus one microsecond per earlier turn in the history: every
turn is ordered, and nothing rendered or embedded changes (renders show the minute, the
embedding text the date).

Everything else is MemMachine's own code: segmenting (TextSegmenter), the text embedded
(WholeTextDeriver), storing, searching, expanding and rendering. This module adds:
  ingest   encode a sample's histories, each embedded once (the stores are kept)
  table    a stored history's segments in EventMemory's order, read through the segment
           store's interface, with the token counts the reader-cost model needs (reading.py)
  offline  the same segments without a store (MemMachine's segmenter and deriver run in
           process), for estimates, profiles and the embedder comparison
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache

import numpy as np

from rbench.datasets.longmemeval import history, session_datetime
from rbench.memory import Memory  # imports MemMachine (quietly) before the names below
from rbench.reading import constants
from rbench.schema import queries
from rbench.text import GEMMA, count

from memmachine_server.episodic_memory.event_memory.data_types import Event, ProducerContext, Segment, TextBlock  # noqa: E402, I001
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import WholeTextDeriver  # noqa: E402
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory  # noqa: E402
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter  # noqa: E402

PRODUCERS = {"user": "User", "assistant": "Assistant"}
TICK = timedelta(microseconds=1)  # per turn: orders turns, never rendered
NAMESPACE = uuid.UUID("6f1d8a52-3c1e-4c4f-9a43-2b6f1f0e9d11")  # event uuids: uuid5 of dataset/qid/session/turn


def chronological(sessions: list[dict]) -> list[int]:
    """Session indices in date order, ties in file order."""
    return sorted(range(len(sessions)), key=lambda s: (session_datetime(sessions[s]["date"]), s))


@lru_cache(maxsize=4)
def events(dataset: str, qid: str) -> tuple[tuple[Event, ...], dict[uuid.UUID, tuple[int, int, int]]]:
    """A history's turns as events in chronological order, and event uuid -> (session index
    in the file, turn index, chronological event number)."""
    sessions = history(dataset, qid)["sessions"]
    out, where = [], {}
    for s in chronological(sessions):
        start = session_datetime(sessions[s]["date"])
        for t, turn in enumerate(sessions[s]["turns"]):
            e = Event(uuid=uuid.uuid5(NAMESPACE, f"{dataset}/{qid}/{s}/{t}"), timestamp=start + len(out) * TICK,
                      context=ProducerContext(producer=PRODUCERS[turn["role"]]), blocks=[TextBlock(text=turn["content"])])
            where[e.uuid] = (s, t, len(out))
            out.append(e)
    assert len(out) * TICK < timedelta(seconds=1), "increments must stay below the rendered resolution"
    return tuple(out), where


def ingest(embedder: str, dataset: str, size: int, qids: list[str]) -> None:
    """Encode each history into its partition (embedding it), skipping those already in."""

    async def go() -> None:
        done = 0
        for qid in qids:
            async with Memory(embedder, dataset, size) as m:
                done += await m.ingest(qid, list(events(dataset, qid)[0]))
            if done and done % 25 == 0:
                print(f"{dataset} seg{size}: {done} histories encoded", flush=True)

    asyncio.run(go())


@dataclass(frozen=True)
class Segments:
    """One history's segments in EventMemory's order (ids are positions)."""

    uuid: tuple[str, ...]  # the segment store's uuids (query results refer to them)
    session: np.ndarray  # the session's index in the history file (gold refers to it)
    turn: np.ndarray
    event: np.ndarray  # the turn's chronological number: one event per turn, one block each
    offset: np.ndarray
    tokens: np.ndarray  # embeddinggemma tokens of the segment text
    inner: np.ndarray  # reader-cost pieces (reading.Reader)
    run: np.ndarray
    texts: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.texts)


def _costs(segments: list[Segment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Token counts per segment: its text, its JSON-escaped text (inner), and a run starting
    at it (header, quotes and newline), with the header as EventMemory renders it for a
    reader (string_from_segment_context with its default format)."""
    quotes, newline, _ = constants()
    texts = [s.block.text for s in segments]
    headers = [EventMemory.string_from_segment_context([s.model_copy(update={"block": TextBlock(text="")})])[:-2]
               for s in segments]
    return (count(texts, GEMMA), count([json.dumps(t, ensure_ascii=False)[1:-1] for t in texts], GEMMA),
            count(headers, GEMMA) + quotes + newline)


def _ordered(segments: list[Segment]) -> list[Segment]:
    return sorted(segments, key=lambda s: (s.timestamp, s.event_uuid, s.index, s.offset))


def _segments_table(dataset: str, qid: str, segments: list[Segment]) -> Segments:
    _, where = events(dataset, qid)
    rows = [(*where[s.event_uuid], s.offset) for s in segments]
    tokens, inner, run = _costs(segments)
    cols = [np.asarray(c, dtype=np.int64) for c in zip(*rows, strict=True)] if rows else [np.zeros(0, np.int64)] * 4
    return Segments(tuple(str(s.uuid) for s in segments), *cols, tokens, inner, run, tuple(s.block.text for s in segments))


async def table(m: Memory, dataset: str, qid: str) -> Segments:
    """A stored history's segments, read once per encoding and cached beside the stores."""
    path = m.root / "tables" / f"{qid}.npz"
    stamp = m.done(qid).read_text()
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if str(data["stamp"]) == stamp:
            return Segments(tuple(data["uuid"].tolist()), data["session"], data["turn"], data["event"], data["offset"],
                            data["tokens"], data["inner"], data["run"], tuple(data["texts"].tolist()))
    segment_store, _, _ = await m.stores()
    partition = await segment_store.open_partition(qid)
    evs = events(dataset, qid)[0]
    by_event = await partition.get_segment_uuids_by_event_uuids([e.uuid for e in evs])
    ids = [u for e in evs for u in by_event.get(e.uuid, [])]
    found: dict = {}
    for start in range(0, len(ids), 1000):
        found.update({k: v[0] for k, v in (await partition.get_segment_contexts(ids[start : start + 1000])).items()})
    seg = _segments_table(dataset, qid, _ordered(list(found.values())))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, stamp=np.str_(stamp), uuid=np.asarray(seg.uuid), session=seg.session, turn=seg.turn, event=seg.event,
             offset=seg.offset, tokens=seg.tokens, inner=seg.inner, run=seg.run, texts=np.asarray(seg.texts))
    return seg


@lru_cache(maxsize=4)
def offline(dataset: str, qid: str, size: int) -> tuple[Segments, tuple[Segment, ...], tuple[str, ...]]:
    """(segments, MemMachine Segment objects, embedding texts) without a store: the history's
    events through MemMachine's TextSegmenter and WholeTextDeriver in process. The texts and
    order are the stored ones; only the uuids differ (the segmenter draws them)."""

    async def go() -> tuple[list[Segment], list[str]]:
        segmenter, deriver = TextSegmenter(max_chunk_length=size), WholeTextDeriver()
        segments = [s for e in events(dataset, qid)[0] for s in await segmenter.segment(e)]
        texts = [d.block.text for s in segments for d in await deriver.derive(s)]
        return segments, texts

    segments, texts = asyncio.run(go())
    order = sorted(range(len(segments)), key=lambda i: (segments[i].timestamp, segments[i].event_uuid,
                                                         segments[i].index, segments[i].offset))
    segments, texts = [segments[i] for i in order], [texts[i] for i in order]
    return _segments_table(dataset, qid, segments), tuple(segments), tuple(texts)


def embedding_tokens(dataset: str, qid: str, size: int) -> int:
    """Tokens a history's embedding costs: its embedding texts plus special tokens (the
    no-prompt setup EventMemory's OpenAIEmbedder sends)."""
    _, _, texts = offline(dataset, qid, size)
    return int(count(list(texts), GEMMA).sum()) + 2 * len(texts)


def gold_segments(dataset: str, qid: str, seg: Segments) -> tuple[list[list[int]], list[list[int]]]:
    """Segment ids of each gold turn, and of each evidence session."""
    q = next(r for r in queries(dataset) if r["qid"] == qid)
    turns = [[int(i) for i in np.flatnonzero((seg.session == s) & (seg.turn == t))] for s, t in q["gold_turns"][0]]
    sessions = [[int(i) for i in np.flatnonzero(seg.session == s)] for s in q["gold_sessions"]]
    return turns, sessions
