"""EventMemory as MemMachine runs it, over this benchmark's backends.

MemMachine is this repository's own packages/server and packages/common (path dependencies). Its
own code does everything retrieval depends on: segmenting events (TextSegmenter), deriving the
text to embed (WholeTextDeriver), storing segments (SQLAlchemySegmentStore on SQLite), context
expansion, reranking, and rendering. The benchmark supplies only what MemMachine leaves to a
deployment, as implementations of its interfaces:

  ContentVectorStore a VectorStore (vectorstore.py) whose vectors are the permanent copy
                     (EmbeddingCache), searched exhaustively: exact search keeps index misses
                     out of the results. Stores written before it keep MemMachine's
                     SQLiteVectorStore with ExactSearchEngine (float16), below.
  QueryCachingEmbedder  MemMachine's OpenAIEmbedder, against the local llama-server or DeepInfra
                     (embeddinggemma, or Qwen3-Embedding-8B with its query instruction), with
                     concurrency limited (adaptive for DeepInfra: limits.py), long retries,
                     document vectors cached by text until their partition is stored
                     (EmbeddingCache), and question embeddings cached, so replaying
                     configurations does not re-embed them.
  FixedCutSegmenter  a Segmenter that cuts every N characters: the simple chunker, an option
                     beside MemMachine's TextSegmenter (chunking names: segmenter()).
  CachedReranker     Qwen3-Reranker scores from the score cache (rerankers.py). EventMemory calls
                     it as it calls any reranker: score(question, rendered contexts). Replay mode
                     treats a missing score as an error, so replays never spend; record mode
                     collects the misses (scored 0) as the work a paid run scores.

One partition per history or document set (partitions.py). Stores live under
data/memory/<document store>/<dataset>/<chunking>/ (store_root: the embedder's document store, shared by
embedders that embed documents alike; seg<N> or cut<N>) and are kept; question vectors are cached in
data/memory/questions/ (questions_path).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Container, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

# MemMachine's package imports an optional agent that logs a traceback when spacy is missing.
logging.getLogger("memmachine_server.retrieval_agent.agents.rarag_query_agent").setLevel(logging.CRITICAL)

import openai  # noqa: E402
from memmachine_server.common.data_types import SimilarityMetric  # noqa: E402
from memmachine_server.common.embedder import Embedder  # noqa: E402
from memmachine_server.common.embedder.openai_embedder import OpenAIEmbedder, OpenAIEmbedderParams  # noqa: E402
from memmachine_server.common.reranker import Reranker  # noqa: E402
from memmachine_server.common.sql import enable_sqlite_explicit_begin, enable_sqlite_foreign_keys  # noqa: E402
from memmachine_server.common.vector_store import VectorStoreSchema  # noqa: E402
from memmachine_server.common.vector_store.sqlite_vector_store import SQLiteVectorStore, SQLiteVectorStoreParams  # noqa: E402
from memmachine_server.common.vector_store.vector_search_engine.vector_search_engine import (  # noqa: E402
    SearchMatch,
    SearchResult,
    VectorSearchEngine,
)
from memmachine_server.episodic_memory.event_memory.data_types import (  # noqa: E402
    Event,
    FormatOptions,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import WholeTextDeriver  # noqa: E402
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory, EventMemoryParams  # noqa: E402
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (  # noqa: E402
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import Segmenter  # noqa: E402
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter  # noqa: E402
from sqlalchemy import event  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from rbench import credits, limits  # noqa: E402
from rbench.dbs import connect, sqlite_pragmas  # noqa: E402
from rbench import embedding as emb  # noqa: E402
from rbench.paths import DATA, EMBEDDINGS, SECRETS  # noqa: E402
from rbench.rerankers import RERANKERS, ScoreCache, prompt_key, to_logit  # noqa: E402
from rbench.vectorstore import ContentVectorStore, ensure_content, vector_hash  # noqa: E402

MEMORY = DATA / "memory"
RERANK_FORMAT = FormatOptions(time_style="short")  # what EventMemory.query renders for its reranker
RENDER = "em"  # score-cache render tag: units are EventMemory-rendered contexts
# Segments per encode_events call: a workaround for a MemMachine limitation, left unchanged
# upstream (the user's decision, 2026-11-09: "work around it"). MemMachine's SQLiteVectorStore
# binds one SQL variable per vector in an upsert, and SQLite allows 32,766 per statement: one
# call with more fails ("too many SQL variables"; a corpus partition, or a LongMemEval-M history
# at 250 characters), and the failed upsert's pending operations then keep the store from
# reopening. Batches stay under the limit (with save_threshold=1, one index save per batch);
# tests/test_memory.py checks a partition encoded in batches stores the same segments. Redoing a
# half-written partition goes through MemMachine's SQLiteVectorStore.delete_partition, which
# drops the partition's records table and unlinks its index rather than deleting rows by an IN
# list, so it avoids the limit by construction.
ENCODE_SEGMENTS = 20_000
# The same limitation bounds a query: EventMemory fetches its vector_search_limit seeds in one
# statement, so no query asks for more than this (depth.py ranks a partition this deep at most).
QUERY_SEEDS = 30_000


class FixedCutSegmenter(Segmenter):
    """The simple chunker: cuts each text block every `length` characters (code points),
    whatever the text, the last piece shorter. Like TextSegmenter it segments one event at a
    time (one LongMemEval turn), so no segment spans turns; unlike it, nothing is stripped, so
    the pieces join back to the turn exactly."""

    def __init__(self, length: int) -> None:
        self._length = length

    async def segment(self, event: Event, *, format_options: FormatOptions | None = None) -> list[Segment]:
        segments: list[Segment] = []
        for index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    pieces = [text[i : i + self._length] for i in range(0, len(text), self._length)]
                    segments.extend(Segment(uuid=uuid.uuid4(), event_uuid=event.uuid, index=index, offset=offset,
                                            timestamp=event.timestamp, block=TextBlock(text=piece), context=event.context,
                                            properties=event.properties) for offset, piece in enumerate(pieces))
                case _:
                    raise NotImplementedError(f"Unsupported block type: {type(block).__name__}")
        return segments


SEGMENTERS = {"seg": TextSegmenter, "cut": FixedCutSegmenter}


def segmenter(chunking: str) -> Segmenter:
    """The Segmenter a history chunking names: seg<C> is MemMachine's TextSegmenter (recursive
    splitting, EventMemory's default) at a limit of C characters; cut<C> the simple chunker,
    cutting every C characters."""
    kind, length = chunking[:3], chunking[3:]
    if kind not in SEGMENTERS or not length.isdigit():
        raise ValueError(f"unknown history chunking {chunking!r}: expected seg<characters> or cut<characters>")
    return SEGMENTERS[kind](int(length))


class ExactSearchEngine(VectorSearchEngine):
    """Exhaustive cosine search. Vectors are kept as float16 (as saved), so a reloaded index
    ranks exactly as the one that was built; ties keep insertion order (a stable sort), so a
    shallower search is always a prefix of a deeper one."""

    def __init__(self, num_dimensions: int, similarity_metric: SimilarityMetric) -> None:
        if similarity_metric != SimilarityMetric.COSINE:
            raise ValueError(f"ExactSearchEngine supports cosine only, not {similarity_metric}")
        self._keys = np.zeros(0, np.int64)
        self._vectors = np.zeros((0, num_dimensions), np.float16)
        self._float: np.ndarray | None = None  # the vectors as float32, made once per change (searches reuse it)
        self._dirty = False  # changed since the last load or save (the store re-saves every engine it holds)

    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        matrix = np.asarray(list(vectors.values()), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self._keys = np.concatenate([self._keys, np.fromiter(vectors.keys(), np.int64, len(vectors))])
        self._vectors = np.concatenate([self._vectors, matrix.astype(np.float16)])
        self._float = None
        self._dirty = True

    async def search(self, vectors: Iterable[Sequence[float]], *, limit: int,
                     allowed_keys: Container[int] | None = None) -> list[SearchResult]:
        if self._float is None:
            self._float = self._vectors.astype(np.float32)
        matrix = self._float
        out = []
        for query in vectors:
            q = np.asarray(query, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            scores = matrix @ q
            order = np.argsort(-scores, kind="stable")
            matches = []
            for i in order:
                key = int(self._keys[i])
                if allowed_keys is None or key in allowed_keys:
                    matches.append(SearchMatch(score=float(scores[i]), key=key))
                    if len(matches) == limit:
                        break
            out.append(SearchResult(matches=matches))
        return out

    async def remove(self, keys: Iterable[int]) -> None:
        drop = np.isin(self._keys, np.fromiter(keys, np.int64))
        if drop.any():
            self._keys, self._vectors = self._keys[~drop], self._vectors[~drop]
            self._float = None
            self._dirty = True

    async def save(self, path: str) -> None:
        """Written only when changed (MemMachine's store saves every engine it holds at shutdown,
        read-only sessions too), and atomically: a temporary file, flushed and fsynced, then
        moved into place, so an interrupted save never leaves a truncated index."""
        import os

        if not self._dirty and os.path.exists(path):
            return
        temporary = f"{path}.tmp"
        with open(temporary, "wb") as file:
            np.savez(file, keys=self._keys, vectors=self._vectors)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        self._dirty = False

    async def load(self, path: str) -> None:
        with open(path, "rb") as file:
            data = np.load(file)
            self._keys, self._vectors = data["keys"], data["vectors"]
        self._float = None
        self._dirty = False


_LOCKS: dict[str, list] = {}  # store root -> [open lock file, holders in this process]


def lock_store(root: Path) -> None:
    """One process per store (MemMachine's SQLite vector store assumes it: its startup replays
    every partition's pending operations and its shutdown saves every engine it holds, so a
    second process can overwrite a partition another is writing). An exclusive lock on
    <root>/.lock, held until the last Memory on it in this process closes; another process
    fails fast."""
    import fcntl

    key = str(root.resolve())
    if key in _LOCKS:
        _LOCKS[key][1] += 1
        return
    file = open(root / ".lock", "a")  # noqa: SIM115 - held open until unlock_store
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        file.close()
        raise SystemExit(f"another rbench process is using {root}; wait for it to finish or stop it") from None
    _LOCKS[key] = [file, 1]


def unlock_store(root: Path) -> None:
    key = str(root.resolve())
    if key in _LOCKS:
        _LOCKS[key][1] -= 1
        if _LOCKS[key][1] == 0:
            _LOCKS.pop(key)[0].close()  # closing releases the lock


def store_root(embedder: str, dataset: str, chunking: str) -> Path:
    """Where a chunking's EventMemory stores live: under the embedder's document store, so
    embedders that embed documents the same way (the same model with or without a query
    instruction) share one encoding instead of paying for it twice."""
    documents = emb.EMBEDDERS[embedder].documents if embedder in emb.EMBEDDERS else embedder
    return MEMORY / documents / dataset / chunking


def questions_path(embedder: str) -> Path:
    """The question-embedding cache of an embedder, keyed by everything that changes a query's
    vector (the model, its URL, the query prompt, the dimensions), so a changed instruction or
    size never reuses old vectors."""
    spec = emb.EMBEDDERS.get(embedder)
    key = json.dumps([spec.model, spec.url, spec.query_prompt, spec.dimensions, spec.request_dimensions]) if spec else embedder
    return MEMORY / "questions" / f"{embedder}__{hashlib.sha1(key.encode()).hexdigest()[:12]}.jsonl"


def _read_questions(path: Path) -> dict[str, list[float]]:
    """The cached question vectors; a line cut short by a crash is skipped (asked again)."""
    out: dict[str, list[float]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[item["q"]] = item["v"]
    return out


_QUESTIONS: dict[str, dict[str, list[float]]] = {}  # question cache path -> its vectors, shared in a process
EMBEDDED_TOKENS: dict[str, int] = {}  # document tokens sent per embedder in this process (the run summary prices them)


class EmbeddingCache:
    """Document vectors by content, like the score cache: sha1 of the exact text EventMemory
    sends (the deriver's output) -> its vector, in data/embeddings/<embedder>__<setup>.sqlite,
    where the setup hashes everything that changes a vector (the model, server, dimensions asked
    and kept, prompts, MemMachine's max input length, the stored precision).

    Kept permanently (the user, 2026-11-10: the vectors must survive a failure of EventMemory's
    storage layers and be reusable without its id mappings), so re-encoding the same chunking
    and derivation from nothing costs no embedding even if both SQLite stores and every index
    file are lost; duplicate texts anywhere are paid once. Each request's vectors are written as
    it returns, so a failure mid-partition keeps what was paid for. Stored as float32: the
    engine casts each vector to float32 and normalizes it before rounding to float16, so a
    float32 copy rebuilds the index bit for bit (a float16 copy would round twice and could
    reorder near ties). It is also the content store's vectors (vectorstore.py), stored once:
    about 4.7 KB per distinct text at 1,024 dimensions and 17 KB at 4,096, 32 GB and 116 GB for
    LongMemEval-M at seg500 (costs.disk_bytes)."""

    @staticmethod
    def path_for(spec: emb.Embedder) -> Path:
        """The cache file of an embedder setup (what to back up to keep the paid vectors)."""
        setup = json.dumps([spec.model, spec.url, spec.route, spec.request_dimensions, spec.dimensions, spec.query_prompt,
                            spec.doc_prompt, spec.max_input_chars, "float32"])
        return EMBEDDINGS / f"{spec.name}__{hashlib.sha1(setup.encode()).hexdigest()[:12]}.sqlite"

    def __init__(self, spec: emb.Embedder) -> None:
        path = self.path_for(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = connect(path)
        ensure_content(self.db)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def get(self, keys: Iterable[str]) -> dict[str, list[float]]:
        keys, out = list(keys), {}
        for start in range(0, len(keys), 900):  # SQLite's bound-parameter limit
            part = keys[start : start + 900]
            rows = self.db.execute(f"SELECT key, v FROM vectors WHERE key IN ({','.join('?' * len(part))})", part)
            out.update({k: np.frombuffer(v, dtype=np.float32).tolist() for k, v in rows})
        return out

    def put(self, rows: dict[str, Sequence[float]]) -> None:
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO vectors (key, v, vhash) VALUES (?, ?, ?)",
                                [(k, b, vector_hash(b)) for k, b in
                                 ((k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in rows.items())])

    def drop(self, keys: Iterable[str]) -> None:
        keys = list(keys)
        with self.db:
            for start in range(0, len(keys), 900):
                part = keys[start : start + 900]
                self.db.execute(f"DELETE FROM vectors WHERE key IN ({','.join('?' * len(part))})", part)

    def close(self) -> None:
        self.db.close()


MAX_REQUEST_CHARS = OpenAIEmbedder.max_total_input_length_per_request  # 75,000: MemMachine splits a longer call
RETRYABLE = (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError)


class QueryCachingEmbedder(Embedder):
    """MemMachine's OpenAIEmbedder with document vectors cached by text (EmbeddingCache) and
    question vectors cached on disk by text (replays query the same questions many times).
    Requests go through the embedder's concurrency limit (limits.py: adaptive for a hosted
    model, fixed for the shared local server) and are retried here, one attempt per MemMachine
    call, for up to an hour (limits.PATIENCE). Documents are sent as they are (MemMachine's
    OpenAIEmbedder adds no prompt); a query prompt (Qwen3-Embedding's instruction) is put
    before each question, as a deployment would format its queries. Vectors come back at the
    embedder's size: asked for directly, or truncated and re-normalized when the API is asked
    for more (request_dimensions)."""

    def __init__(self, embedder: str) -> None:
        super().__init__()
        spec = emb.EMBEDDERS[embedder]
        if spec.doc_prompt:
            raise ValueError(f"{embedder} has a document prompt; EventMemory's OpenAIEmbedder sends documents as they are")
        key = (SECRETS / spec.key_file).read_text().strip() if spec.key_file else "none"
        base = spec.url + spec.route.removesuffix("/embeddings")
        self._spec = spec
        self._client = openai.AsyncOpenAI(base_url=base, api_key=key, max_retries=0)
        self._inner = OpenAIEmbedder(OpenAIEmbedderParams(client=self._client, model=spec.model or "embeddinggemma",
                                                          dimensions=spec.request_dimensions or spec.dimensions,
                                                          max_input_length=spec.max_input_chars))
        hosted = spec.key_file is not None
        self._gate = limits.Gate(limits.limit_for(f"embed:{spec.name}", spec.in_flight, adaptive=hosted))
        self._cache = EmbeddingCache(spec)
        self._path = questions_path(embedder)
        if str(self._path) not in _QUESTIONS:  # parsed once per process (about 10 MB of JSON lines)
            _QUESTIONS[str(self._path)] = _read_questions(self._path)
        self._questions: dict[str, list[float]] = _QUESTIONS[str(self._path)]

    async def _attempt(self, batch: list, call, stop: list) -> list[list[float]] | None:
        """One batch through the limit, retried while the failure is retryable (an hour at most);
        None, unsent, once another batch of the call has failed (`stop`)."""
        failing, attempt = None, 0
        while True:
            async with self._gate:
                if stop or limits.CANCEL.is_set():
                    return None  # another batch failed, or Ctrl-C: send nothing more
                try:
                    vectors = await call(batch, 1)
                    self._gate.state.success()
                    await self._gate.wake()
                    break
                except Exception as err:  # MemMachine's ExternalServiceAPIError, from the API's answer
                    broke = credits.from_embedder(err)
                    if broke is not None:
                        raise broke from err
                    cause = err.__cause__ if err.__cause__ is not None else err
                    if not isinstance(cause, RETRYABLE):
                        raise
                    if isinstance(cause, openai.RateLimitError):
                        self._gate.state.throttled()
                    last = type(cause).__name__
            failing = failing or time.monotonic()
            if time.monotonic() - failing > limits.PATIENCE:
                raise limits.GaveUp("embedding requests", last)
            attempt += 1
            await limits.sleep(limits.backoff(attempt))
        if vectors and len(vectors[0]) > self._spec.dimensions:  # asked for more: keep the Matryoshka prefix
            vectors = emb.fit_dimensions(self._spec, np.asarray(vectors, dtype=np.float32)).tolist()
        return vectors

    def _requests(self, texts: list) -> list[tuple[int, int]]:
        """Index ranges of one request each: at most request_inputs inputs and MAX_REQUEST_CHARS
        characters, so MemMachine's OpenAIEmbedder never splits a call into several requests
        (each would hold one slot of the limit while counting as more)."""
        out, start, chars = [], 0, 0
        for i, text in enumerate(texts):
            n = len(str(text))
            if i > start and (i - start == self._spec.request_inputs or chars + n > MAX_REQUEST_CHARS):
                out.append((start, i))
                start, chars = i, 0
            chars += n
        if start < len(texts):
            out.append((start, len(texts)))
        return out

    async def _batches(self, texts: list, call, keys: list[str] | None = None) -> list[list[float]]:
        """Embed texts in requests (_requests); with keys, each request's vectors are cached as
        it returns. If a request fails, or its vectors cannot be cached (disk full), no more are
        sent; those in flight finish and are cached (they are paid for: vectors that could not be
        written are kept and written once more before the failure is raised)."""
        stop: list[BaseException] = []
        unsaved: dict[str, Sequence[float]] = {}

        async def one(start: int, end: int) -> list[list[float]] | None:
            try:
                vectors = await self._attempt(texts[start:end], call, stop)
            except Exception as err:
                stop.append(err)
                raise
            if vectors is not None and keys is not None:
                got = dict(zip(keys[start:end], vectors, strict=True))
                try:
                    self._cache.put(got)
                except Exception as err:  # disk full, a lock: stop sending, keep what was paid for
                    unsaved.update(got)
                    stop.append(err)
                    raise
            return vectors

        parts = await asyncio.gather(*(one(a, b) for a, b in self._requests(texts)), return_exceptions=True)
        if unsaved:
            self._cache.put(unsaved)  # once more; raises (with the disk-full message: __main__) if it cannot
        failed = [x for x in parts if isinstance(x, BaseException)]
        if failed:
            raise failed[0]
        if any(part is None for part in parts):  # Ctrl-C: what returned is cached
            raise limits.Interrupted()
        return [v for part in parts for v in part]

    async def _ingest_embed(self, inputs: list, max_attempts: int = 1) -> list[list[float]]:
        from rbench.text import count

        texts = [str(x) for x in inputs]
        keys = [EmbeddingCache.key(t) for t in texts]
        have = self._cache.get(set(keys))
        first = {}  # each text not cached yet, once: its position
        for i, k in enumerate(keys):
            if k not in have and k not in first:
                first[k] = i
        todo = [texts[i] for i in first.values()]
        if todo:
            EMBEDDED_TOKENS[self._spec.name] = (EMBEDDED_TOKENS.get(self._spec.name, 0)
                                                + int(count(todo, self._spec.tokenizer).sum())
                                                + self._spec.special_tokens * len(todo))
            have.update(zip(first, await self._batches([inputs[i] for i in first.values()], self._inner._ingest_embed,
                                                       list(first)), strict=True))
        return [have[k] for k in keys]

    async def _search_embed(self, queries: list, max_attempts: int = 1) -> list[list[float]]:
        missing = [q for q in dict.fromkeys(queries) if q not in self._questions]
        if missing:
            prompted = [self._spec.query_prompt + q for q in missing]
            vectors = await self._batches(prompted, self._inner._search_embed)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as file:  # appended: a crash loses at most the line being written
                for q, v in zip(missing, vectors, strict=True):
                    self._questions[q] = [float(x) for x in v]
                    file.write(json.dumps({"q": q, "v": self._questions[q]}) + "\n")
        return [self._questions[q] for q in queries]

    async def aclose(self) -> None:
        await self._client.close()
        self._cache.close()

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return self._spec.dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class CachedReranker(Reranker):
    """A Qwen3 reranker's scores, from the score cache. mode "replay": a missing score is an
    error. mode "record": misses are collected in `work` (scored 0) for a paid run to score.
    Either way, every pair asked for is kept in `asked`, in order."""

    def __init__(self, model: str, mode: str = "replay") -> None:
        self.model, self.mode = model, mode
        self.cache = ScoreCache(model)
        self.prompt = prompt_key(RERANKERS[model].template)
        self.work: dict[str, list[tuple[str, str]]] = {}  # question -> [(cache key, rendered context)] not scored
        self.asked: dict[str, list[tuple[str, str]]] = {}  # question -> every (cache key, rendered context)

    def keys(self, query: str, candidates: list[str]) -> list[str]:
        return [ScoreCache.key(self.prompt, RENDER, query, c) for c in candidates]

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        keys = self.keys(query, candidates)
        known = {k for k, _ in self.asked.setdefault(query, [])}
        for k, c in zip(keys, candidates, strict=True):  # in order, each key once (identical contexts render alike)
            if k not in known:
                known.add(k)
                self.asked[query].append((k, c))
        got = self.cache.get(keys)
        missing = [(k, c) for k, c in zip(keys, candidates, strict=True) if k not in got]
        if missing and self.mode == "replay":
            raise KeyError(f"{self.model}: {len(missing)} of {len(keys)} contexts for {query[:60]!r} are not scored")
        if missing:
            seen = {k for k, _ in self.work.setdefault(query, [])}
            for k, c in missing:  # each key once: a duplicate would be billed twice
                if k not in seen:
                    seen.add(k)
                    self.work[query].append((k, c))
        raw = np.asarray([got[k][0] if k in got else np.nan for k in keys], dtype=np.float64)
        return np.nan_to_num(to_logit(self.model, raw), nan=0.0).tolist()


class Memory:
    """One chunking's EventMemory stores for an embedder and dataset, and EventMemory
    instances over them (with or without a reranker; they share the stores).

    Use one per partition (`async with Memory(...)`): the SQLite vector store keeps every opened
    partition's index in memory until it shuts down, so closing after each partition keeps
    memory flat. A partition's events are encoded in one call (batches past ENCODE_SEGMENTS), so its index is saved once."""

    def __init__(self, embedder: str, dataset: str, chunking: str, root: Path | None = None,
                 backend: Embedder | None = None, width: int | None = None, vector_store: str | None = None) -> None:
        """chunking: seg<C> or cut<C> (segmenter()). root and backend replace the stores'
        directory and the embedder (tests). width: the dimensions searches use (the leading ones
        of the stored vectors, re-normalized; default the embedder's search_dimensions, else all).
        vector_store: "content" (vectorstore.ContentVectorStore) or "sqlite" (MemMachine's
        SQLiteVectorStore with ExactSearchEngine); default "sqlite" for a root written by it
        before (vectors.sqlite and no records.sqlite), else "content"."""
        self.embedder, self.dataset, self.chunking = embedder, dataset, chunking
        self.segmenter = segmenter(chunking)
        self.root = root or store_root(embedder, dataset, chunking)
        self.root.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        spec = emb.EMBEDDERS.get(embedder)
        self.dimensions = spec.dimensions if spec else 768
        self.width = width or (spec.search_dimensions if spec else None)
        legacy = (self.root / "vectors.sqlite").exists() and not (self.root / "records.sqlite").exists()
        self.vector_store = vector_store or ("sqlite" if legacy else "content")
        if self.vector_store == "sqlite" and self.width not in (None, self.dimensions):
            raise ValueError("the legacy SQLite vector store searches at the stored width only")
        self._embedder: Embedder | None = None
        self._stores = None

    @property
    def content_path(self) -> Path:
        """Where the content store keeps the vectors: the embedder's permanent copy (EmbeddingCache),
        or the root's own file for a test embedder."""
        if self._backend is None and self.embedder in emb.EMBEDDERS:
            return EmbeddingCache.path_for(emb.EMBEDDERS[self.embedder])
        return self.root / "content.sqlite"

    async def stores(self):
        if self._stores is None:
            lock_store(self.root)  # one process per store, readers too (they replay and save)
            schema = VectorStoreSchema(vector_dimensions=self.dimensions, similarity_metric=SimilarityMetric.COSINE,
                                       indexed_properties=EventMemory.expected_vector_store_indexed_properties())
            engines = []
            for name in ("segments", "vectors") if self.vector_store == "sqlite" else ("segments",):
                engine = create_async_engine(f"sqlite+aiosqlite:///{self.root / name}.sqlite")
                event.listen(engine.sync_engine, "connect", lambda connection, _record: sqlite_pragmas(connection))
                enable_sqlite_foreign_keys(engine)
                enable_sqlite_explicit_begin(engine)
                engines.append(engine)
            segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engines[0]))
            if self.vector_store == "sqlite":
                vector_store = SQLiteVectorStore(SQLiteVectorStoreParams(
                    sqlalchemy_engine=engines[1], vector_store_name="rbench", store_schema=schema,
                    vector_search_engine_factory=ExactSearchEngine, index_directory=str(self.root / "index"),
                    save_threshold=1))  # one upsert per encode_events call (ingest), so its index is saved once per batch
            else:
                vector_store = ContentVectorStore(self.root, schema, self.content_path, width=self.width)
            await segment_store.startup()
            await vector_store.startup()
            self._embedder = self._backend or QueryCachingEmbedder(self.embedder)
            self._stores = (segment_store, vector_store, engines)
        return self._stores

    async def event_memory(self, reranker: Reranker | None = None) -> EventMemory:
        segment_store, vector_store, _ = await self.stores()
        return EventMemory(EventMemoryParams(segment_store=segment_store, vector_store=vector_store,
                                             segmenter=self.segmenter, deriver=WholeTextDeriver(),
                                             embedder=self._embedder, reranker=reranker))

    def done(self, key: str) -> Path:
        return self.root / "done" / key

    async def ingest(self, key: str, events: list[Event], format_options: FormatOptions | None = None) -> bool:
        """Encode a partition's events once; False if it was already in. A partition left
        half-written by an interrupted run is deleted and encoded again. format_options: what
        encode_events gives the deriver (partitions.formats: None for chat histories, MemMachine's
        default; no dates for documents)."""
        if self.done(key).exists():
            return False
        memory = await self.event_memory()
        await memory.delete_partition(key)
        segment_store, _, _ = await self.stores()
        while await segment_store.purge_deleted_partitions():  # reclaim what the deletion deferred (bounded per call)
            pass
        partition = await memory.open_or_create_partition(key)
        for batch in await self._batches(events, format_options):
            await partition.encode_events(batch, format_options=format_options)
        self.done(key).parent.mkdir(parents=True, exist_ok=True)
        self.done(key).write_text(str(uuid.uuid4()))  # an encoding's stamp: its segment uuids are new each time
        return True

    async def _batches(self, events: list[Event], format_options: FormatOptions | None) -> list[list[Event]]:
        """Events in order, in groups of at most ENCODE_SEGMENTS segments (counted with the same
        segmenter; an event is never split). One group, so one index save, for most partitions."""
        out: list[list[Event]] = [[]]
        n = 0
        for e in events:
            k = len(await self.segmenter.segment(e, format_options=format_options))
            if out[-1] and n + k > ENCODE_SEGMENTS:
                out.append([])
                n = 0
            out[-1].append(e)
            n += k
        return out

    async def __aenter__(self) -> Memory:
        await self.stores()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._stores is not None:
            segment_store, vector_store, engines = self._stores
            await vector_store.shutdown()
            await segment_store.shutdown()
            for engine in engines:
                await engine.dispose()
            if isinstance(self._embedder, QueryCachingEmbedder):
                await self._embedder.aclose()
            self._stores = None
            unlock_store(self.root)
