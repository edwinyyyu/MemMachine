"""Disk-primary turbovec (TurboQuant) implementation of VectorSearchEngine.

Same quantized brute-force search as
:class:`~.turbovec_engine.TurboVecVectorSearchEngine`, but backed by
``turbovec.DiskIndex``: the quantized codes live in a memory-mapped ``.tvdm``
file in the SIMD-blocked layout the scoring kernel consumes, so searches run
directly over the mapped bytes. Resident memory stays bounded by the OS page
cache plus a small in-RAM delta of vectors added since the last save;
``save()`` compacts the delta and any tombstoned removals into a fresh file.

Use with :class:`~..sqlite_vector_store.SQLiteVectorStore` and an
``index_directory`` — without one, ``save()`` is never called and the delta
grows unbounded, degenerating to the in-RAM engine's footprint.
"""

import asyncio
from collections.abc import Iterable
from typing import override

import numpy as np
from turbovec import DiskIndex

from memmachine_server.common.data_types import SimilarityMetric

from .turbovec_engine import TurboVecVectorSearchEngine


class TurboVecDiskVectorSearchEngine(TurboVecVectorSearchEngine):
    """Vector search engine backed by a memory-mapped turbovec DiskIndex.

    ``target_partition_size``, when set, enables SPFresh-style partitioning:
    at each save the codes are clustered into partitions of roughly that many
    vectors and searches probe only the nearest partitions — queries touch a
    fraction of the file instead of all of it, at the cost of approximate
    routing. ``None`` (default) keeps the index flat and the quantized scan
    exact.

    Three opt-in recall levers on top of routed search:

    * ``store_vectors=True`` keeps the full-precision vectors in the file
      (and enables ``get_vectors``); searches then exact-rescore the top
      quantized candidates by f32 inner product, lifting the quantization
      ceiling for a handful of mapped page reads per query. ``rescore_k``
      overrides the rescore depth (default ``4 * limit``; ``0`` disables).
    * ``probe_epsilon`` switches a partitioned index to distance-bounded
      adaptive probing: each query scans every partition whose centroid is
      within ``(1 + probe_epsilon)`` of its nearest, up to the ``nprobe``
      cap, so boundary queries probe more partitions than confident ones.
    * ``replica_epsilon`` enables SPANN-style boundary multi-assignment at
      save time: vectors near partition boundaries are also stored in the
      adjacent partitions (RNG-rule pruned, at most 8 copies), making them
      findable at small probe counts. Costs the replication factor in file
      size.
    """

    def __init__(
        self,
        *,
        num_dimensions: int,
        similarity_metric: SimilarityMetric,
        bit_width: int = 4,
        target_partition_size: int | None = None,
        store_vectors: bool = False,
        replica_epsilon: float | None = None,
        nprobe: int | None = None,
        probe_epsilon: float | None = None,
        rescore_k: int | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(
            num_dimensions=num_dimensions,
            similarity_metric=similarity_metric,
            bit_width=bit_width,
        )
        self._target_partition_size = target_partition_size
        self._replica_epsilon = replica_epsilon
        self._nprobe = nprobe
        self._probe_epsilon = probe_epsilon
        self._rescore_k = rescore_k
        self._index = DiskIndex(
            dim=num_dimensions,
            bit_width=bit_width,
            target_partition_size=target_partition_size,
            store_vectors=store_vectors,
            replica_epsilon=replica_epsilon,
        )

    @override
    def _index_search(
        self, queries: np.ndarray, limit: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._index.search(
            queries,
            limit,
            nprobe=self._nprobe,
            probe_epsilon=self._probe_epsilon,
            rescore_k=self._rescore_k,
        )

    @override
    async def get_vectors(self, keys: Iterable[int]) -> dict[int, list[float]]:
        if not self._index.store_vectors:
            raise NotImplementedError(
                "this index stores only TurboQuant-compressed vectors; "
                "construct the engine with store_vectors=True to retrieve "
                "the originals"
            )
        keys = list(keys)
        if not keys:
            return {}
        async with self._lock.read_lock():
            return await asyncio.to_thread(self._sync_get_vectors, keys)

    def _sync_get_vectors(self, keys: list[int]) -> dict[int, list[float]]:
        present = [key for key in keys if key in self._index]
        if not present:
            return {}
        vectors = self._index.get_vectors(np.array(present, dtype=np.uint64))
        return {
            key: vector.tolist() for key, vector in zip(present, vectors, strict=True)
        }

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            index = await asyncio.to_thread(DiskIndex.open, path)
            # The engine's configuration wins over what the file recorded,
            # but only when explicitly configured — None means "follow the
            # file" so reopening a partitioned/replicated index keeps its
            # settings. store_vectors always follows the file: it is fixed
            # at build time and cannot be retrofitted onto existing rows.
            if self._target_partition_size is not None:
                index.target_partition_size = self._target_partition_size
            if self._replica_epsilon is not None:
                index.replica_epsilon = self._replica_epsilon
            self._index = index
