"""Embedder that returns random vectors instantly, for benchmarking."""

from typing import Any, override

import numpy as np

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.embedder.embedder import Embedder


class RandomEmbedder(Embedder):
    """Returns random unit vectors. Useful for isolating non-embedding bottlenecks."""

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    @override
    async def ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        return self._random_unit_vectors(len(inputs))

    @override
    async def search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        return self._random_unit_vectors(len(queries))

    @property
    @override
    def model_id(self) -> str:
        return "random"

    @property
    @override
    def dimensions(self) -> int:
        return self._dimensions

    @property
    @override
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE

    def _random_unit_vectors(self, n: int) -> list[list[float]]:
        vecs = np.random.standard_normal((n, self._dimensions))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).tolist()
