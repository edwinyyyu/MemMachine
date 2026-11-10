"""The benchmark's own vector store, kept minimal (the user, 2026-11-10: "our store doesn't even
have to use the sqlite vector store, if it would be better to make our own"; "keep it simple if
we're writing our own, since more logic is more chance of mistakes").

It implements what EventMemory and rbench call: the store's schema, startup, shutdown,
open_or_create_partition and delete_partition, and a partition's upsert and query. The other
methods the interface requires (create_partition, open_partition, purge_deleted_partitions, a
partition's delete: called only by EventMemory.forget_events) raise NotImplementedError, as do a
query's property filter and score threshold (EventMemory passes a filter only when its caller
does, and no threshold; rbench passes neither).

What it adds to storing records:
  (a) Vectors are stored once, in the permanent copy (memory.EmbeddingCache's table), found by
      the sha1 of their float32 bytes (vector_hash). A record is (partition, uuid, vector hash)
      in root/records.sqlite. A vector the copy lacks is added under key "v:" + its hash (on the
      ingest path every vector is already there, cached as the embedder returned it), committed
      before the records that name it: SQLite in WAL mode does not make a transaction over two
      files atomic, and in this order a crash leaves at most a vector no record names.
  (b) Searches at a width w no larger than the stored one: the leading w dimensions of each
      vector, re-normalized, when a partition's vectors are loaded.

A query is an exact cosine search over the partition's vectors, loaded once per handle (again
after an upsert through it) as float32 unit vectors. Ties keep insertion order (a stable sort),
so a shallower search is a prefix of a deeper one. delete_partition removes the partition's
records; its vectors stay (they are paid for). A handle is not used after its partition is
deleted (Memory.ingest opens a new one).

memory.Memory uses this store for new store roots and MemMachine's SQLiteVectorStore (with
memory.ExactSearchEngine, float16) for roots written before it.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from uuid import UUID

import numpy as np
from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.vector_store import VectorStore, VectorStorePartition
from memmachine_server.common.vector_store.data_types import QueryMatch, QueryResult, Record, VectorStoreSchema
from memmachine_server.common.vector_store.utils import validate_dimensions, validate_limit, validate_query_vector

from rbench.dbs import connect


def vector_hash(data: bytes) -> str:
    """What the store finds a vector by: sha1 of its float32 bytes (the store gets vectors, not
    texts; float32 -> Python float -> float32 round-trips exactly)."""
    return hashlib.sha1(data).hexdigest()


def ensure_content(db) -> None:
    """The content-addressed vector table (the permanent copy, and the vector store's vectors):
    key (sha1 of the exact derived text, or "v:" + the vector's hash for a vector stored
    without its text), the float32 vector, and its hash, indexed. Adds and fills the hash
    column in a copy written before it existed."""
    db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, v BLOB, vhash TEXT)")
    if "vhash" not in {row[1] for row in db.execute("PRAGMA table_info(vectors)")}:
        with db:
            db.execute("ALTER TABLE vectors ADD COLUMN vhash TEXT")
    missing = db.execute("SELECT key, v FROM vectors WHERE vhash IS NULL").fetchall()
    if missing:
        with db:
            db.executemany("UPDATE vectors SET vhash = ? WHERE key = ?", [(vector_hash(v), k) for k, v in missing])
    db.execute("CREATE INDEX IF NOT EXISTS vectors_vhash ON vectors (vhash)")


class ContentVectorStore(VectorStore):
    """Records in root/records.sqlite; vectors in content_path (an EmbeddingCache file, or any
    path for a test embedder); searches at `width` dimensions (default: all)."""

    def __init__(self, root: Path, schema: VectorStoreSchema, content_path: Path, width: int | None = None) -> None:
        if schema.similarity_metric != SimilarityMetric.COSINE:
            raise ValueError(f"ContentVectorStore supports cosine only, not {schema.similarity_metric}")
        self.width = width or schema.vector_dimensions
        if not 0 < self.width <= schema.vector_dimensions:
            raise ValueError(f"width {self.width} is not within the stored {schema.vector_dimensions} dimensions")
        self._root, self._schema, self._content = Path(root), schema, Path(content_path)
        self.db: sqlite3.Connection | None = None

    @property
    def store_schema(self) -> VectorStoreSchema:
        return self._schema

    async def startup(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._content.parent.mkdir(parents=True, exist_ok=True)
        content = connect(self._content)  # create or migrate the content table before attaching it
        try:
            ensure_content(content)
        finally:
            content.close()
        self.db = connect(self._root / "records.sqlite")
        self.db.execute("ATTACH DATABASE ? AS content", (str(self._content),))
        self.db.execute("PRAGMA content.synchronous = NORMAL")  # set per file: an attached one starts at FULL
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS records (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                            "partition TEXT NOT NULL, uuid TEXT NOT NULL, vhash TEXT NOT NULL, UNIQUE (partition, uuid))")
            self.db.execute("CREATE INDEX IF NOT EXISTS records_partition ON records (partition, seq)")

    async def shutdown(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    async def open_or_create_partition(self, partition_key: str) -> ContentVectorStorePartition:
        return ContentVectorStorePartition(self, partition_key)

    async def delete_partition(self, partition_key: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM records WHERE partition = ?", (partition_key,))

    async def create_partition(self, partition_key: str) -> None:
        raise NotImplementedError("not called by EventMemory or rbench")

    async def open_partition(self, partition_key: str) -> ContentVectorStorePartition | None:
        raise NotImplementedError("not called by EventMemory or rbench")

    async def purge_deleted_partitions(self) -> bool:
        raise NotImplementedError("not called by rbench (delete_partition removes the records at once)")


class ContentVectorStorePartition(VectorStorePartition):
    def __init__(self, store: ContentVectorStore, key: str) -> None:
        self._store, self._key = store, key
        self._loaded: tuple[list[UUID], np.ndarray] | None = None  # record uuids and unit vectors, in insertion order

    async def upsert(self, *, records: Iterable[Record]) -> None:
        db, dims = self._store.db, self._store.store_schema.vector_dimensions
        rows, vectors = [], {}
        for record in records:
            validate_dimensions(record.vector, dims)
            data = np.asarray(record.vector, dtype=np.float32).tobytes()
            h = vector_hash(data)
            vectors[h] = data
            rows.append((self._key, str(record.uuid), h))
        with db:  # vectors the copy lacks first (see the module docstring)
            db.executemany("INSERT INTO content.vectors (key, v, vhash) SELECT ?, ?, ? "
                           "WHERE NOT EXISTS (SELECT 1 FROM content.vectors WHERE vhash = ?)",
                           [(f"v:{h}", data, h, h) for h, data in vectors.items()])
        with db:
            db.executemany("INSERT OR REPLACE INTO records (partition, uuid, vhash) VALUES (?, ?, ?)", rows)
        self._loaded = None

    def _vectors(self) -> tuple[list[UUID], np.ndarray]:
        if self._loaded is None:
            store = self._store
            rows = store.db.execute("SELECT r.uuid, (SELECT v FROM content.vectors WHERE vhash = r.vhash LIMIT 1) "
                                    "FROM records r WHERE r.partition = ? ORDER BY r.seq", (self._key,)).fetchall()
            if any(v is None for _, v in rows):
                raise RuntimeError(f"vectors of partition {self._key!r} are missing from {store._content}: delete "
                                   f"{store._root / 'done' / self._key} and rerun to encode it again (vectors still in "
                                   "the copy cost nothing)")
            matrix = np.frombuffer(b"".join(v for _, v in rows), dtype=np.float32)
            matrix = matrix.reshape(len(rows), store.store_schema.vector_dimensions)[:, : store.width]
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._loaded = ([UUID(u) for u, _ in rows], np.divide(matrix, norms, out=np.zeros_like(matrix),
                                                                  where=norms > 0))
        return self._loaded

    async def query(self, *, query_vectors: Iterable[Sequence[float]], limit: int, score_threshold: float | None = None,
                    property_filter=None) -> list[QueryResult]:
        if property_filter is not None or score_threshold is not None:
            raise NotImplementedError("rbench's vector store takes no property filter or score threshold")
        validate_limit(limit)
        store = self._store
        queries = [list(q) for q in query_vectors]
        for q in queries:
            validate_query_vector(q, store.store_schema.vector_dimensions)
        uuids, matrix = self._vectors()
        out = []
        for query in queries:
            q = np.asarray(query, dtype=np.float32)[: store.width]
            scores = matrix @ (q / (np.linalg.norm(q) or 1.0))
            order = np.argsort(-scores, kind="stable")[:limit]
            out.append(QueryResult(matches=[QueryMatch(score=float(scores[i]), record_uuid=uuids[i]) for i in order]))
        return out

    async def delete(self, *, record_uuids: Iterable[UUID]) -> None:
        raise NotImplementedError("called only by EventMemory.forget_events, which rbench never calls")
