"""
Factory for Reranker instances.
"""

from typing import Any

from memmachine.common.data_types import ConfigValue, Nested
from memmachine.common.factory import Factory

from .reranker import Reranker


class RerankerFactory(Factory):
    """
    Factory for Reranker instances.
    """

    @staticmethod
    def create(
        provider: str,
        config: dict[str, ConfigValue],
        dependencies: dict[str, Nested[str]],
        injections: dict[str, Any],
    ) -> Reranker:
        match provider:
            case "bm25":
                from .bm25_reranker import BM25Reranker

                return BM25Reranker(config)
            case "cross-encoder":
                try:
                    from .cross_encoder_reranker import CrossEncoderReranker
                except ImportError as e:
                    raise ValueError(
                        "sentence-transformers is required "
                        "for CrossEncoderReranker. "
                        "Please install it with "
                        "`pip install sentence-transformers`, "
                        "or by including GPU dependencies with "
                        "`pip install memmachine[gpu]`."
                    ) from e

                return CrossEncoderReranker(config)
            case "embedder":
                from .embedder_reranker import EmbedderReranker

                return EmbedderReranker(
                    dict(config) | Factory.inject_dependencies(dependencies, injections)
                )
            case "identity":
                from .identity_reranker import IdentityReranker

                return IdentityReranker()
            case "rrf-hybrid":
                from .rrf_hybrid_reranker import RRFHybridReranker

                return RRFHybridReranker(
                    dict(config) | Factory.inject_dependencies(dependencies, injections)
                )
            case _:
                raise ValueError(f"Unknown Reranker provider: {provider}")
