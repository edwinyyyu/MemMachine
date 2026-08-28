"""Identity reranker implementation."""

from typing import override

from .reranker import Reranker


class IdentityReranker(Reranker):
    """Reranker that returns candidates in their original order."""

    @override
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        _ = query
        return list(map(float, reversed(range(len(candidates)))))
