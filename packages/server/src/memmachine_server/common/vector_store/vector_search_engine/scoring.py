"""Shared numpy scoring helpers for vector search engines."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .vector_search_engine import SearchMatch


def cosine_similarities(
    query_vector: Sequence[float], matrix: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Cosine similarity of `query_vector` against each row of `matrix`."""
    query = np.asarray(query_vector, dtype=np.float32)
    denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    denominators[denominators == 0.0] = np.inf
    return (matrix @ query) / denominators


def top_k_matches(
    query_vector: Sequence[float],
    keys: Sequence[int],
    matrix: npt.NDArray[np.float32],
    limit: int,
) -> list[SearchMatch]:
    """The best `limit` rows of `matrix` as matches keyed by `keys`."""
    similarities = cosine_similarities(query_vector, matrix)
    k = min(limit, similarities.shape[0])
    if k <= 0:
        return []
    top = np.argpartition(-similarities, k - 1)[:k]
    top = top[np.argsort(-similarities[top])]
    return [
        SearchMatch(key=keys[index], cosine_similarity=float(similarities[index]))
        for index in top
    ]
