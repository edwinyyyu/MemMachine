"""EventMemory as MemMachine runs it, over this benchmark's backends.

MemMachine is a dependency pinned to its feat/horizontal-scaling branch (pyproject.toml). Its
own code does everything retrieval depends on: segmenting events (TextSegmenter), deriving the
text to embed (WholeTextDeriver), storing segments (SQLAlchemySegmentStore on SQLite) and
vectors (SQLiteVectorStore), vector search, context expansion, reranking, and rendering. The
benchmark supplies only what MemMachine leaves to a deployment, as implementations of its
interfaces:

  ExactSearchEngine  a VectorSearchEngine that searches exhaustively. MemMachine's engines
                     (hnswlib, usearch) are approximate; exact search keeps index misses out of
                     the results. Its saved index is the history's vectors (float16), kept.
  QueryCachingEmbedder  MemMachine's OpenAIEmbedder (the no-prompt path), against the local
                     llama-server or DeepInfra, with concurrency bounded and question
                     embeddings cached, so replaying configurations does not re-embed them.
  CachedReranker     Qwen3-Reranker scores from the score cache (rerankers.py). EventMemory calls
                     it as it calls any reranker: score(question, rendered contexts). Replay mode
                     treats a missing score as an error, so replays never spend; record mode
                     collects the misses (scored 0) as the work a paid run scores.

One partition per history, keyed by the question id. Stores live under
data/memory/<embedder>/<dataset>/seg<N>/ and are kept.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from memmachine_server.episodic_memory.event_memory.data_types import Event, FormatOptions  # noqa: E402
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import WholeTextDeriver  # noqa: E402
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory, EventMemoryParams  # noqa: E402
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (  # noqa: E402
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from rbench import embedding as emb  # noqa: E402
from rbench.paths import DATA, SECRETS  # noqa: E402
from rbench.rerankers import ScoreCache, prompt_key, to_logit  # noqa: E402

MEMORY = DATA / "memory"
RERANK_FORMAT = FormatOptions(time_style="short")  # what EventMemory.query renders for its reranker
RENDER = "em"  # score-cache render tag: units are EventMemory-rendered contexts


class ExactSearchEngine(VectorSearchEngine):
    """Exhaustive cosine search. Vectors are kept as float16 (as saved), so a reloaded index
    ranks exactly as the one that was built; ties keep insertion order (a stable sort), so a
    shallower search is always a prefix of a deeper one."""

    def __init__(self, num_dimensions: int, similarity_metric: SimilarityMetric) -> None:
        if similarity_metric != SimilarityMetric.COSINE:
            raise ValueError(f"ExactSearchEngine supports cosine only, not {similarity_metric}")
        self._keys = np.zeros(0, np.int64)
        self._vectors = np.zeros((0, num_dimensions), np.float16)

    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        matrix = np.asarray(list(vectors.values()), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self._keys = np.concatenate([self._keys, np.fromiter(vectors.keys(), np.int64, len(vectors))])
        self._vectors = np.concatenate([self._vectors, matrix.astype(np.float16)])

    async def search(self, vectors: Iterable[Sequence[float]], *, limit: int,
                     allowed_keys: Container[int] | None = None) -> list[SearchResult]:
        matrix = self._vectors.astype(np.float32)
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
        self._keys, self._vectors = self._keys[~drop], self._vectors[~drop]

    async def save(self, path: str) -> None:
        with open(path, "wb") as file:
            np.savez(file, keys=self._keys, vectors=self._vectors)

    async def load(self, path: str) -> None:
        with open(path, "rb") as file:
            data = np.load(file)
            self._keys, self._vectors = data["keys"], data["vectors"]


class QueryCachingEmbedder(Embedder):
    """MemMachine's OpenAIEmbedder with bounded concurrency, and question embeddings cached on
    disk by text (replays query the same questions many times)."""

    def __init__(self, embedder: str) -> None:
        super().__init__()
        spec = emb.EMBEDDERS[embedder]
        if spec.query_prompt or spec.doc_prompt:
            raise ValueError(f"{embedder} uses prompts; EventMemory's OpenAIEmbedder sends none (use a -raw embedder)")
        key = (SECRETS / spec.key_file).read_text().strip() if spec.key_file else "none"
        base = spec.url + spec.route.removesuffix("/embeddings")
        self._spec = spec
        self._client = openai.AsyncOpenAI(base_url=base, api_key=key, max_retries=0)
        self._inner = OpenAIEmbedder(OpenAIEmbedderParams(client=self._client, model=spec.model or "embeddinggemma",
                                                          dimensions=768, max_input_length=spec.max_input_chars))
        self._gate = asyncio.Semaphore(spec.in_flight)
        self._path = MEMORY / embedder / "questions.json"
        self._questions: dict[str, list[float]] = json.loads(self._path.read_text()) if self._path.exists() else {}

    async def _batches(self, texts: list, call) -> list[list[float]]:
        size = self._spec.request_inputs

        async def one(batch: list) -> list[list[float]]:
            async with self._gate:
                return await call(batch, 10)

        parts = await asyncio.gather(*(one(texts[i : i + size]) for i in range(0, len(texts), size)))
        return [v for part in parts for v in part]

    async def _ingest_embed(self, inputs: list, max_attempts: int = 1) -> list[list[float]]:
        return await self._batches(inputs, self._inner._ingest_embed)

    async def _search_embed(self, queries: list, max_attempts: int = 1) -> list[list[float]]:
        missing = [q for q in dict.fromkeys(queries) if q not in self._questions]
        if missing:
            for q, v in zip(missing, await self._batches(missing, self._inner._search_embed), strict=True):
                self._questions[q] = [float(x) for x in v]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._questions))
        return [self._questions[q] for q in queries]

    async def aclose(self) -> None:
        await self._client.close()

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimensions(self) -> int:
        return 768

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class CachedReranker(Reranker):
    """A Qwen3 reranker's scores, from the score cache. mode "replay": a missing score is an
    error. mode "record": misses are collected in `work` (scored 0) for a paid run to score."""

    def __init__(self, model: str, instruction: str, mode: str = "replay") -> None:
        self.model, self.instruction, self.mode = model, instruction, mode
        self.cache = ScoreCache(model)
        self.prompt = prompt_key(instruction)
        self.work: dict[str, list[tuple[str, str]]] = {}  # question -> [(cache key, rendered context)]

    def keys(self, query: str, candidates: list[str]) -> list[str]:
        return [ScoreCache.key(self.prompt, RENDER, query, c) for c in candidates]

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        keys = self.keys(query, candidates)
        got = self.cache.get(keys)
        missing = [(k, c) for k, c in zip(keys, candidates, strict=True) if k not in got]
        if missing and self.mode == "replay":
            raise KeyError(f"{self.model}: {len(missing)} of {len(keys)} contexts for {query[:60]!r} are not scored")
        if missing:
            seen = {k for k, _ in self.work.setdefault(query, [])}
            self.work[query] += [(k, c) for k, c in missing if k not in seen]
        raw = np.asarray([got[k][0] if k in got else np.nan for k in keys], dtype=np.float64)
        return np.nan_to_num(to_logit(self.model, raw), nan=0.0).tolist()


class Memory:
    """One segment length's EventMemory stores for an embedder and dataset, and EventMemory
    instances over them (with or without a reranker; they share the stores).

    Use one per history (`async with Memory(...)`): the SQLite vector store keeps every opened
    partition's index in memory until it shuts down, so closing after each history keeps
    memory flat. A history's events are encoded in one call, so its index is saved once."""

    def __init__(self, embedder: str, dataset: str, size: int, root: Path | None = None,
                 backend: Embedder | None = None) -> None:
        """root and backend replace the stores' directory and the embedder (tests)."""
        self.embedder, self.dataset, self.size = embedder, dataset, size
        self.root = root or MEMORY / embedder / dataset / f"seg{size}"
        self.root.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self._embedder: Embedder | None = None
        self._stores = None

    async def stores(self):
        if self._stores is None:
            engines = []
            for name in ("segments", "vectors"):
                engine = create_async_engine(f"sqlite+aiosqlite:///{self.root / name}.sqlite")
                enable_sqlite_foreign_keys(engine)
                enable_sqlite_explicit_begin(engine)
                engines.append(engine)
            segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engines[0]))
            vector_store = SQLiteVectorStore(SQLiteVectorStoreParams(
                sqlalchemy_engine=engines[1], vector_store_name="rbench",
                store_schema=VectorStoreSchema(vector_dimensions=768, similarity_metric=SimilarityMetric.COSINE,
                                               indexed_properties=EventMemory.expected_vector_store_indexed_properties()),
                vector_search_engine_factory=ExactSearchEngine, index_directory=str(self.root / "index"),
                save_threshold=1))  # one upsert per history (ingest), so its index is saved once
            await segment_store.startup()
            await vector_store.startup()
            self._embedder = self._backend or QueryCachingEmbedder(self.embedder)
            self._stores = (segment_store, vector_store, engines)
        return self._stores

    async def event_memory(self, reranker: Reranker | None = None) -> EventMemory:
        segment_store, vector_store, _ = await self.stores()
        return EventMemory(EventMemoryParams(segment_store=segment_store, vector_store=vector_store,
                                             segmenter=TextSegmenter(max_chunk_length=self.size), deriver=WholeTextDeriver(),
                                             embedder=self._embedder, reranker=reranker))

    def done(self, key: str) -> Path:
        return self.root / "done" / key

    async def ingest(self, key: str, events: list[Event]) -> bool:
        """Encode a history's events into its partition once; False if it was already in. A
        partition left half-written by an interrupted run is deleted and encoded again."""
        if self.done(key).exists():
            return False
        memory = await self.event_memory()
        await memory.delete_partition(key)
        partition = await memory.open_or_create_partition(key)
        await partition.encode_events(events)
        self.done(key).parent.mkdir(parents=True, exist_ok=True)
        self.done(key).write_text(str(uuid.uuid4()))  # an encoding's stamp: its segment uuids are new each time
        return True

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
