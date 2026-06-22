"""Incrementally-updatable turbovec (SPFresh) VectorSearchEngine.

Same quantized search, routing, and recall levers as
:class:`~.turbovec_disk_engine.TurboVecDiskVectorSearchEngine`, but backed by
``turbovec.FreshIndex``: the index is a *directory* with one append-only
segment file per partition, a write-ahead log, and an atomically-replaced
manifest. ``save()`` appends buffered vectors to the partitions they belong
to and runs local split/merge/reassign maintenance — it never rewrites the
whole index, untouched partitions keep their page-cache contents, and
mutations since the last save survive a crash via the write-ahead log.

Prefer this engine over the ``.tvdm`` disk engine when the corpus sees
ongoing inserts/removes; the single-file engine remains the simpler artifact
for build-once corpora (and the two formats convert losslessly).
"""

import asyncio
from collections.abc import Iterable
from typing import override

import numpy as np
from turbovec import FreshIndex

from memmachine_server.common.data_types import SimilarityMetric

from .turbovec_engine import TurboVecVectorSearchEngine


class TurboVecFreshVectorSearchEngine(TurboVecVectorSearchEngine):
    """Vector search engine backed by an incrementally-updatable FreshIndex.

    Constructor knobs are identical to the disk engine:
    ``target_partition_size`` (SPFresh partitioning), ``store_vectors``
    (exact rescoring + ``get_vectors``), ``replica_epsilon`` (boundary
    multi-assignment), and the per-search ``nprobe`` / ``probe_epsilon`` /
    ``rescore_k``.
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
        self._index = FreshIndex(
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
            key: vector.tolist()
            for key, vector in zip(present, vectors, strict=True)
        }

    @override
    async def save(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._index.save, path)

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            index = await asyncio.to_thread(FreshIndex.open, path)
            # Explicit engine configuration wins over what the directory
            # recorded; None means "follow the index". store_vectors always
            # follows the index (fixed at build time).
            if self._target_partition_size is not None:
                index.target_partition_size = self._target_partition_size
            if self._replica_epsilon is not None:
                index.replica_epsilon = self._replica_epsilon
            self._index = index
