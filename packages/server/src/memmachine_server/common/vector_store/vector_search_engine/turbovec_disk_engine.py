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
from typing import override

from turbovec import DiskIndex

from memmachine_server.common.data_types import SimilarityMetric

from .turbovec_engine import TurboVecVectorSearchEngine


class TurboVecDiskVectorSearchEngine(TurboVecVectorSearchEngine):
    """Vector search engine backed by a memory-mapped turbovec DiskIndex."""

    def __init__(
        self,
        *,
        num_dimensions: int,
        similarity_metric: SimilarityMetric,
        bit_width: int = 4,
    ) -> None:
        """Initialize."""
        super().__init__(
            num_dimensions=num_dimensions,
            similarity_metric=similarity_metric,
            bit_width=bit_width,
        )
        self._index = DiskIndex(dim=num_dimensions, bit_width=bit_width)

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            self._index = await asyncio.to_thread(DiskIndex.open, path)
