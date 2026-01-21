"""FlagEmbedding-based embedder implementation."""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, InstanceOf
from FlagEmbedding import BGEM3FlagModel

from memmachine.common.data_types import ExternalServiceAPIError, SimilarityMetric

from .embedder import Embedder

logger = logging.getLogger(__name__)


class FlagEmbedderParams(BaseModel):
    """Parameters for FlagEmbedder."""

    flag_model: InstanceOf[BGEM3FlagModel] = Field(
        ...,
        description="The flag model to use for generating embeddings.",
    )


class FlagEmbedder(Embedder):
    """Embedder powered by a FlagEmbedding model."""

    def __init__(self, params: FlagEmbedderParams) -> None:
        """Initialize the FlagEmbedder."""
        super().__init__()

        self._flag_model = params.flag_model

        self._dimensions = len(self._flag_model.encode("")["dense_vecs"])

    async def ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed input documents using the FlagEmbedding model."""
        return await self._embed(inputs, max_attempts)

    async def search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed search queries using the FlagEmbedding model."""
        return await self._embed(queries, max_attempts)

    async def _embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Generate embeddings with retry logic."""
        if not inputs:
            return []
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        embed_call_uuid = uuid4()

        start_time = time.monotonic()

        try:
            logger.debug(
                "[call uuid: %s] "
                "Attempting to create embeddings using BAAI/bge-m3 FlagEmbedding model",
                embed_call_uuid,
            )
            response = (
                await asyncio.to_thread(
                    self._flag_model.encode,
                    inputs,
                )
            )["dense_vecs"]
        except Exception as e:
            # Exception may not be retried.
            error_message = (
                f"[call uuid: {embed_call_uuid}] "
                "Giving up creating embeddings "
                f"due to assumed non-retryable {type(e).__name__}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message) from e

        end_time = time.monotonic()
        logger.debug(
            "[call uuid: %s] Embeddings created in %.3f seconds",
            embed_call_uuid,
            end_time - start_time,
        )

        return response.astype(float).tolist()

    @property
    def model_id(self) -> str:
        """Return the underlying model identifier."""
        return "BAAI/bge-m3"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        """Return the similarity metric used."""
        return SimilarityMetric.COSINE
