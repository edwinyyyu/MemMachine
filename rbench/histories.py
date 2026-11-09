"""Partitioned datasets (one chat history per query, e.g. LongMemEval): segments, vectors,
first-stage pools, expansion windows and reranker units, one history at a time.

Everything follows EventMemory (eventmemory.py):
  order     EventMemory orders a partition's segments by (timestamp, event, index, offset).
            LongMemEval dates sessions to the minute and gives turns no time, and its files
            are not in date order (40 of 94 sampled M histories are not; most have sessions
            starting in the same minute). So sessions are sorted by date, ties kept in file
            order, and each turn's timestamp is its session's date plus one microsecond per
            turn before it in the history. The increments order every turn without being
            rendered (embedding text shows the date, rendered contexts the minute). Segment
            ids are positions in this order, which is EventMemory's segment order.
  segments  each turn is an event (producer User/Assistant); its stripped content is split by
            the production TextSegmenter at `size` characters.
  vectors   the embedder's document prompt + derivative_text (date, producer, JSON text);
            queries embed the raw question. One float16 file per history is kept.
  pools     exact search over the history's segments (no index), the head kept to
            retrieval.HEAD_TOKENS tokens of seed text; every later depth and budget lies inside it.
  windows   expand_context e: e // 3 segments before the seed and e - e // 3 after, in the
            history's order (they cross turns and sessions, as a partition's do).
  units     what the reranker scores: string_from_segment_context of the seed alone or of
            its window, with the query path's default format (date and short time).

Cached token counts, vectors and pools carry the history's segment fingerprint (texts,
producers, timestamps). Token counts are recomputed when it changes; stale vectors and
pools are refused, so a change to segmentation or ordering can never pair cached numbers
with other segments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

from rbench import embedding as emb
from rbench import eventmemory as em
from rbench import retrieval
from rbench.datasets.longmemeval import history, session_datetime
from rbench.paths import CHUNKS, POOLS, VECTORS
from rbench.reading import constants
from rbench.schema import queries
from rbench.text import GEMMA, count, tokenizer

PRODUCERS = {"user": "User", "assistant": "Assistant"}
TICK = timedelta(microseconds=1)  # per turn: orders turns, never rendered


@dataclass(frozen=True)
class Segments:
    """One history's segments in chronological order (ids are positions)."""

    session: np.ndarray  # the session's index in the history file (gold refers to it)
    turn: np.ndarray
    event: np.ndarray  # the turn's chronological number in the history: one event per turn, one block each
    offset: np.ndarray
    tokens: np.ndarray  # embeddinggemma tokens of the segment text
    inner: np.ndarray  # reader-cost pieces (reading.Reader)
    run: np.ndarray
    texts: tuple[str, ...]
    timestamps: tuple
    fingerprint: str

    def __len__(self) -> int:
        return len(self.texts)

    def segment(self, i: int, producers: tuple[str, ...]) -> em.Segment:
        return em.Segment(i, int(self.event[i]), 0, int(self.offset[i]), self.timestamps[i], producers[i], self.texts[i])


def _chunk_path(dataset: str, size: int, qid: str) -> Path:
    return CHUNKS / dataset / f"seg{size}" / f"{qid}.npz"


def chronological(sessions: list[dict]) -> list[int]:
    """Session indices in date order, ties in file order."""
    return sorted(range(len(sessions)), key=lambda s: (session_datetime(sessions[s]["date"]), s))


def _fingerprint(texts: list[str], producers: list[str], stamps: list) -> str:
    """Segments plus every setting the cached numbers depend on: the embedding and rendering
    formats and locale (header and embedding text), the reader cost's constants."""
    h = hashlib.sha1(repr((em.EMBED_FORMAT, em.RENDER_FORMAT, em.LOCALE, GEMMA)).encode())
    for text, producer, stamp in zip(texts, producers, stamps, strict=True):
        h.update(f"{producer}\x1f{stamp.isoformat()}\x1f{text}\x1e".encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=4)
def build_segments(dataset: str, qid: str, size: int) -> tuple[Segments, tuple[str, ...]]:
    """(segments, producer per segment) for one history; token counts cached on disk."""
    sessions = history(dataset, qid)["sessions"]
    rows, producers, texts, stamps = [], [], [], []
    event = 0
    for s in chronological(sessions):
        start = session_datetime(sessions[s]["date"])
        for t, turn in enumerate(sessions[s]["turns"]):
            stamp = start + event * TICK
            for o, piece in enumerate(em.segment_text(turn["content"], size)):
                rows.append((s, t, event, o))
                producers.append(PRODUCERS[turn["role"]])
                texts.append(piece)
                stamps.append(stamp)
            event += 1
    assert event * TICK < timedelta(seconds=1), "increments must stay below the rendered resolution"
    fingerprint = _fingerprint(texts, producers, stamps)
    path = _chunk_path(dataset, size, qid)
    data = np.load(path) if path.exists() else None
    if data is not None and "fingerprint" in data and str(data["fingerprint"]) == fingerprint:
        tokens, inner, run = data["tokens"], data["inner"], data["run"]
    else:  # new, or built from other segments: recount (free)
        quotes, newline, _ = constants()
        tokens = count(texts, GEMMA)
        inner = count([json.dumps(x, ensure_ascii=False)[1:-1] for x in texts], GEMMA)
        headers = [em._header(em.Segment(i, 0, 0, 0, stamps[i], producers[i], ""), em.RENDER_FORMAT)
                   for i in range(len(texts))]
        run = count(headers, GEMMA) + quotes + newline
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, tokens=tokens, inner=inner, run=run, fingerprint=np.str_(fingerprint))
    cols = [np.asarray(c, dtype=np.int64) for c in zip(*rows, strict=True)]
    return Segments(*cols, tokens, inner, run, tuple(texts), tuple(stamps), fingerprint), tuple(producers)


def gold_segments(dataset: str, qid: str, seg: Segments) -> tuple[list[list[int]], list[list[int]]]:
    """Segment ids of each gold turn, and of each evidence session."""
    q = next(r for r in queries(dataset) if r["qid"] == qid)
    turns = [[int(i) for i in np.flatnonzero((seg.session == s) & (seg.turn == t))] for s, t in q["gold_turns"][0]]
    sessions = [[int(i) for i in np.flatnonzero(seg.session == s)] for s in q["gold_sessions"]]
    return turns, sessions


# ---------------------------------------------------------------------------- vectors


def vector_path(embedder: str, dataset: str, size: int, qid: str) -> Path:
    return VECTORS / emb.EMBEDDERS[embedder].documents / dataset / f"seg{size}" / f"{qid}.npz"


def _stale(path: Path) -> RuntimeError:
    return RuntimeError(f"{path} was built from other segments (segmentation or ordering changed); "
                        f"delete it (and the pools built from it) and run prepare again")


def derivative_texts(seg: Segments, producers: tuple[str, ...]) -> list[str]:
    """What EventMemory embeds for each segment (WholeTextDeriver)."""
    return [em.derivative_text(seg.segment(i, producers)) for i in range(len(seg))]


SHARD = 1024  # segments per saved shard while a history is being embedded


def _vector_fingerprint(seg: Segments, embedder: str) -> str:
    spec = emb.EMBEDDERS[embedder]
    return hashlib.sha1(f"{seg.fingerprint}|{spec.documents}|{spec.doc_prompt}".encode()).hexdigest()[:16]


def embed_history(embedder: str, dataset: str, qid: str, size: int, load: bool = True) -> np.ndarray | None:
    """A history's segment vectors, embedded once and kept; load=False only makes sure they exist.

    Embedding is saved in shards of SHARD segments as it goes, under a directory named for the
    fingerprint, so an interrupted history resumes from its last shard and never mixes shards
    of other segments."""
    import shutil

    seg, producers = build_segments(dataset, qid, size)
    path = vector_path(embedder, dataset, size, qid)
    fingerprint = _vector_fingerprint(seg, embedder)
    if path.exists():
        data = np.load(path)
        if "fingerprint" not in data or str(data["fingerprint"]) != fingerprint:
            raise _stale(path)
        return data["vectors"] if load else None
    spec = emb.EMBEDDERS[embedder]
    texts = derivative_texts(seg, producers)
    parts = path.with_name(f"{qid}.{fingerprint}.parts")
    parts.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(range(0, len(texts), SHARD)):
        shard = parts / f"{i:05d}.npy"
        if not shard.exists():
            np.save(shard, emb.embed(spec, texts[start : start + SHARD], spec.doc_prompt).astype(np.float16))
    vectors = np.concatenate([np.load(parts / f"{i:05d}.npy") for i in range(-(-len(texts) // SHARD))])
    assert len(vectors) == len(texts)
    np.savez(path, vectors=vectors, fingerprint=np.str_(fingerprint))
    shutil.rmtree(parts)
    return vectors


def embedding_tokens(embedder: str, dataset: str, qid: str, size: int) -> int:
    """Tokens a history's embedding costs: its derivative texts, each with the document prompt."""
    seg, producers = build_segments(dataset, qid, size)
    return int(count(derivative_texts(seg, producers), GEMMA).sum()) + prompt_overhead(embedder) * len(seg)


@cache
def prompt_overhead(embedder: str) -> int:
    """Tokens each embedded text adds beyond its own: the document prompt and special tokens."""
    t = tokenizer(GEMMA)
    return len(t.encode(emb.EMBEDDERS[embedder].doc_prompt + "x", add_special_tokens=True).ids) - len(
        t.encode("x", add_special_tokens=False).ids)


@cache
def query_vectors(embedder: str, dataset: str, variant: str = "question") -> np.ndarray:
    return emb.query_vectors(embedder, dataset, variant)


# ------------------------------------------------------------------------------ pools


def pool_path(embedder: str, dataset: str, size: int, sample_name: str, variant: str = "question") -> Path:
    return POOLS / embedder / dataset / "history" / f"seg{size}__{variant}__{sample_name}.npz"


def build_pools(embedder: str, dataset: str, size: int, sample_name: str, variant: str = "question") -> Path:
    """Exact search per history for a sample's queries; heads saved together (CSR over all
    the dataset's queries, empty for queries outside the sample), with each history's
    segment fingerprint."""
    from rbench.samples import sample

    path = pool_path(embedder, dataset, size, sample_name, variant)
    if path.exists():
        return path
    rows = queries(dataset)
    qids = {rows[i]["qid"] for i in sample(dataset, sample_name)}
    qv = query_vectors(embedder, dataset, variant).astype(np.float32)
    ids, scores, offsets, fingerprints = [], [], [0], []
    for qi, q in enumerate(rows):
        if q["qid"] not in qids:
            offsets.append(offsets[-1])
            fingerprints.append("")
            continue
        seg, _ = build_segments(dataset, q["qid"], size)
        sims = embed_history(embedder, dataset, q["qid"], size).astype(np.float32) @ qv[qi]
        done = sum(1 for f in fingerprints if f) + 1
        if done % 25 == 0:
            print(f"{dataset} seg{size}: {done}/{len(qids)} histories embedded and searched", flush=True)
        local = retrieval.head(sims, seg.tokens)
        ids.append(local.astype(np.int32))
        scores.append(sims[local].astype(np.float32))
        offsets.append(offsets[-1] + len(local))
        fingerprints.append(seg.fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, qids=np.asarray([q["qid"] for q in rows]), offsets=np.asarray(offsets, np.int64),
             ids=np.concatenate(ids) if ids else np.zeros(0, np.int32),
             scores=np.concatenate(scores) if scores else np.zeros(0, np.float32), fingerprints=np.asarray(fingerprints))
    return path


@dataclass(frozen=True)
class HistoryPools:
    """Pools whose heads are handed out only for the segments they were built from."""

    pools: retrieval.Pools
    fingerprints: np.ndarray
    path: Path

    def head(self, qi: int, seg: Segments) -> tuple[np.ndarray, np.ndarray]:
        if str(self.fingerprints[qi]) != seg.fingerprint:
            raise _stale(self.path)
        return self.pools.head(qi)


def load_pools(embedder: str, dataset: str, size: int, sample_name: str, variant: str = "question") -> HistoryPools:
    path = pool_path(embedder, dataset, size, sample_name, variant)
    data = np.load(path)
    if data["qids"].tolist() != [q["qid"] for q in queries(dataset)] or "fingerprints" not in data:
        raise RuntimeError(f"{path} was built for other queries or by an older version; delete it")
    return HistoryPools(retrieval.Pools(data["qids"], data["offsets"], data["ids"], data["scores"]), data["fingerprints"], path)


# ------------------------------------------------------------------ windows and units


def window(seed: int, n: int, e: int) -> list[int]:
    before, after = e // 3, e - e // 3
    return list(range(max(0, seed - before), min(n, seed + after + 1)))


def unit_text(seg: Segments, producers: tuple[str, ...], ids: list[int]) -> str:
    return em.string_from_segment_context([seg.segment(i, producers) for i in ids], em.RENDER_FORMAT)
