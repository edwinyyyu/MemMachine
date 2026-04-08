"""Abstract base class for a vector search engine."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class SearchResult:
    """Result of a nearest-neighbor search.

    Attributes:
        keys: Engine keys for the matched vectors.
        scores: Pure metric scores (e.g. cosine similarity,
            L2 distance).  Ordered from best to worst.
    """

    keys: list[int]
    scores: list[float]


class KeyFilter(ABC):
    """Predicate that decides which keys are allowed in search results.

    Engines call ``key in filter`` per candidate during search.
    Implementations may back this with a set, a SQL query, a bitmap, etc.
    """

    @abstractmethod
    def __contains__(self, key: object) -> bool:
        """Return whether the key is allowed."""
        ...


class VectorSearchEngine(ABC):
    """A vector search engine that indexes vectors by integer key.

    Provides nearest-neighbor search (exact or approximate) over vectors
    identified by caller-provided integer keys (typically SQLite rowids).

    Scores are pure metric values:
    - Cosine: cosine similarity (1.0 = identical, higher is better)
    - Dot: inner product (higher is better)
    - Euclidean: L2 distance (0.0 = identical, lower is better)

    Results are returned ordered from best to worst.

    All operations are synchronous and NOT thread-safe.
    The caller is responsible for serializing writes and wrapping
    calls in ``asyncio.to_thread`` for async contexts.

    Subclasses must implement :meth:`_raw_search` (unfiltered search).
    The default :meth:`search` handles ``key_filter`` via overfetch +
    post-filter.  Engines with native filtering (e.g. FAISS callback)
    should override :meth:`search` directly.
    """

    _OVERFETCH_FACTORS: ClassVar[list[int]] = [20, 100]

    @abstractmethod
    def add(self, keys: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        """Add or replace vectors.

        Args:
            keys: Integer keys, one per vector.
            vectors: Vectors to add, each of length ndim.

        If a key already exists, its vector is replaced.
        """

    @abstractmethod
    def _raw_search(self, vector: Sequence[float], k: int) -> SearchResult:
        """Unfiltered nearest-neighbor search.  Subclasses must implement."""

    def search(
        self,
        vector: Sequence[float],
        k: int,
        *,
        key_filter: KeyFilter | None = None,
    ) -> SearchResult:
        """Find the k nearest neighbors of ``vector``.

        If ``key_filter`` is provided, only keys present in the filter
        are returned.  The default implementation uses overfetch +
        post-filter.  Override for native filtering support.

        Returns a :class:`SearchResult` with keys and scores
        ordered from best to worst.
        """
        if key_filter is None:
            return self._raw_search(vector, k)

        engine_size = len(self)
        if engine_size == 0:
            return SearchResult(keys=[], scores=[])

        for factor in self._OVERFETCH_FACTORS:
            overfetch_k = min(k * factor, engine_size)
            result = self._raw_search(vector, overfetch_k)
            filtered = _apply_key_filter(result, key_filter, k)
            if len(filtered.keys) >= k or overfetch_k >= engine_size:
                return filtered

        # Full scan fallback
        result = self._raw_search(vector, engine_size)
        return _apply_key_filter(result, key_filter, k)

    @abstractmethod
    def remove(self, keys: Iterable[int]) -> None:
        """Remove vectors by key. Missing keys are ignored."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of vectors in the engine."""

    @abstractmethod
    def __contains__(self, key: int) -> bool:
        """Whether a key exists in the engine."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""


def _apply_key_filter(
    result: SearchResult, key_filter: KeyFilter, k: int
) -> SearchResult:
    """Post-filter a SearchResult, keeping only keys in the filter."""
    filtered_keys: list[int] = []
    filtered_scores: list[float] = []
    for key, score in zip(result.keys, result.scores, strict=True):
        if key in key_filter:
            filtered_keys.append(key)
            filtered_scores.append(score)
            if len(filtered_keys) >= k:
                break
    return SearchResult(keys=filtered_keys, scores=filtered_scores)
