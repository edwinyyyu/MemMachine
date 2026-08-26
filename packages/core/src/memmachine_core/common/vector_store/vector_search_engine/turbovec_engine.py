"""turbovec (TurboQuant) implementation of VectorSearchEngine."""

import asyncio
import math
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import ClassVar, override

import numpy as np
from turbovec import IdMapIndex

from memmachine_core.common.rw_locks import AsyncRWLock

from .vector_search_engine import SearchMatch, SearchResult, VectorSearchEngine


class TurboVecVectorSearchEngine(VectorSearchEngine):
    """
    Vector search engine backed by turbovec.

    turbovec indexes a dimensionality that is a multiple of 8, so a vector of
    any other width is zero-padded up to one. Padding is exact -- a zero
    coordinate adds nothing to an inner product and nothing to an L2 norm --
    so the padded index answers as the unpadded one would, and any embedding
    width the other engines accept works here too.

    Vectors are L2-normalized on the way in, so the inner-product index
    returns cosine similarities.
    """

    _VALID_BIT_WIDTHS: ClassVar[frozenset[int]] = frozenset({2, 3, 4})
    _DEFAULT_BIT_WIDTH: ClassVar[int] = 4

    _OVERFETCH_BASE: ClassVar[int] = 4

    def __init__(
        self,
        *,
        num_dimensions: int,
        bit_width: int = _DEFAULT_BIT_WIDTH,
    ) -> None:
        """Initialize."""
        if bit_width not in self._VALID_BIT_WIDTHS:
            raise ValueError(
                f"turbovec bit_width must be one of "
                f"{sorted(self._VALID_BIT_WIDTHS)}, got {bit_width}"
            )

        self._num_dimensions = num_dimensions
        self._padded_dimensions = math.ceil(num_dimensions / 8) * 8
        self._index = IdMapIndex(dim=self._padded_dimensions, bit_width=bit_width)
        self._lock = AsyncRWLock()

    @override
    async def add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        if not vectors:
            return
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_add, vectors)

    def _sync_add(self, vectors: Mapping[int, Sequence[float]]) -> None:
        keys = np.array(list(vectors.keys()), dtype=np.uint64)
        array = self._prepare_vectors(list(vectors.values()))
        self._index.add_with_ids(array, keys)

    @override
    async def search(
        self,
        vectors: Iterable[Sequence[float]],
        *,
        limit: int,
        allowed_keys: Container[int] | None = None,
    ) -> list[SearchResult]:
        vectors = list(vectors)
        if not vectors or limit <= 0:
            return [SearchResult(matches=[]) for _ in vectors]
        async with self._lock.read_lock():
            return await asyncio.to_thread(
                self._sync_search, vectors, limit, allowed_keys
            )

    def _sync_search(
        self,
        vectors: Sequence[Sequence[float]],
        limit: int,
        allowed_keys: Container[int] | None,
    ) -> list[SearchResult]:
        query = self._prepare_vectors(vectors)

        if allowed_keys is None:
            inner_products, ids = self._index.search(query, limit)
            return [
                SearchResult(
                    matches=self._to_matches(inner_products[i], ids[i], limit, None)
                )
                for i in range(query.shape[0])
            ]

        return [
            self._search_one_filtered(query[i : i + 1], limit, allowed_keys)
            for i in range(query.shape[0])
        ]

    def _search_one_filtered(
        self,
        query: np.ndarray,
        limit: int,
        allowed_keys: Container[int],
    ) -> SearchResult:
        fetch_limit = limit * self._OVERFETCH_BASE
        while True:
            inner_products, ids = self._index.search(query, fetch_limit)
            matches = self._to_matches(inner_products[0], ids[0], limit, allowed_keys)
            exhausted = ids.shape[1] < fetch_limit
            if len(matches) >= limit or exhausted:
                return SearchResult(matches=matches)
            fetch_limit *= self._OVERFETCH_BASE

    def _to_matches(
        self,
        inner_products: Iterable[float],
        keys: Iterable[int],
        limit: int,
        allowed_keys: Container[int] | None,
    ) -> list[SearchMatch]:
        matches: list[SearchMatch] = []
        for inner_product, key in zip(inner_products, keys, strict=True):
            int_key = int(key)
            if allowed_keys is not None and int_key not in allowed_keys:
                continue
            matches.append(
                SearchMatch(
                    key=int_key,
                    cosine_similarity=self._to_cosine_similarity(inner_product),
                )
            )
            if len(matches) >= limit:
                break
        return matches

    @staticmethod
    def _to_cosine_similarity(inner_product: float) -> float:
        return min(1.0, max(-1.0, float(inner_product)))

    def _prepare_vectors(self, vectors: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.zeros((len(vectors), self._padded_dimensions), dtype=np.float32)
        try:
            array[:, : self._num_dimensions] = vectors
        except ValueError as error:
            raise ValueError(
                f"vectors must have {self._num_dimensions} dimensions"
            ) from error
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return array / norms

    @override
    async def get_vectors(self, keys: Iterable[int]) -> dict[int, list[float]]:
        raise NotImplementedError(
            "turbovec stores only TurboQuant-compressed vectors; "
            "the original vectors cannot be retrieved"
        )

    @override
    async def remove(self, keys: Iterable[int]) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_remove, keys)

    def _sync_remove(self, keys: Iterable[int]) -> None:
        for key in keys:
            self._index.remove(key)

    @override
    async def save(self, path: str) -> None:
        async with self._lock.write_lock():
            await asyncio.to_thread(self._sync_save, path)

    def _sync_save(self, path: str) -> None:
        self._index.sync(path)

    @override
    async def load(self, path: str) -> None:
        async with self._lock.write_lock():
            self._index = await asyncio.to_thread(self._sync_load, path)

    def _sync_load(self, path: str) -> IdMapIndex:
        return IdMapIndex.load(path)
