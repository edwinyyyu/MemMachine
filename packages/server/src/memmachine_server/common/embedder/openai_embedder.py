"""OpenAI-based embedder implementation."""

import asyncio
import base64
import logging
import re
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import numpy as np
import openai
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common import fast_json
from memmachine_server.common.data_types import (
    ExternalServiceAPIError,
    SimilarityMetric,
)
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker
from memmachine_server.common.raw_http import RawHTTPPool
from memmachine_server.common.utils import (
    chunk_text_balanced,
    cluster_texts,
    unflatten_like,
)

from .embedder import Embedder

logger = logging.getLogger(__name__)


class OpenAIEmbedderParams(BaseModel):
    """Parameters for OpenAIEmbedder."""

    client: InstanceOf[openai.AsyncOpenAI] = Field(
        ...,
        description="AsyncOpenAI client to use for making API calls.",
    )
    model: str = Field(
        ...,
        description=(
            "Name of the OpenAI embedding model to use (e.g. 'text-embedding-3-small')."
        ),
    )
    dimensions: int = Field(
        ...,
        description=(
            "Dimensionality of the embedding vectors "
            "produced by the OpenAI embedding model."
        ),
        gt=0,
    )
    max_input_length: int | None = Field(
        default=None,
        description="Maximum input length for the model (in Unicode code points).",
        gt=0,
    )
    max_retry_interval_seconds: int = Field(
        default=120,
        description="Maximal retry interval in seconds when retrying API calls.",
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        default=None,
        description="An instance of MetricsFactory for collecting usage metrics.",
    )
    batch_size: int | None = Field(
        None,
        description="Batch size for embedding requests.",
    )


class OpenAIEmbedder(Embedder):
    """Embedder that uses OpenAI embedding models."""

    # https://platform.openai.com/docs/api-reference/embeddings/create#embeddings_create-input
    max_num_inputs_per_request = 2048
    max_total_input_length_per_request = (
        75000  # Assume at most 4 tokens per Unicode code point.
    )

    # Tokens that cause 500 errors on text-embedding-3-small.
    _SPECIAL_TOKEN_PATTERN = re.compile(
        r"<\|endoftext\|>"
        r"|<\|im_start\|>"
        r"|<\|im_end\|>"
        r"|<\|fim_prefix\|>"
        r"|<\|fim_middle\|>"
        r"|<\|fim_suffix\|>"
        r"|<\|endofprompt\|>"
    )

    def __init__(self, params: OpenAIEmbedderParams) -> None:
        """Initialize the OpenAI embedder with configuration parameters."""
        super().__init__(batch_size=params.batch_size)

        self._client = params.client

        # https://platform.openai.com/docs/guides/embeddings#embedding-models
        self._model = params.model

        self._dimensions = params.dimensions
        self._use_dimensions_parameter = True

        self._max_retry_interval_seconds = params.max_retry_interval_seconds

        self._max_input_length = params.max_input_length

        # Direct fast path for the embeddings call: RawHTTPPool for plain
        # http, httpx for https. None = not yet probed; False = unavailable
        # (always use the canonical SDK path).
        self._fast_http: RawHTTPPool | httpx.AsyncClient | bool | None = None

        metrics_factory = params.metrics_factory

        self._tracker = OperationTracker(metrics_factory, prefix="embedder_openai")

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True

            self._prompt_tokens_usage_counter = metrics_factory.get_counter(
                "embedder_openai_usage_prompt_tokens",
                "Number of tokens used by prompts to OpenAI embedder",
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "embedder_openai_usage_total_tokens",
                "Number of tokens used by requests to OpenAI embedder",
            )

    async def _ingest_embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed the provided inputs with retries."""
        async with self._tracker("ingest_embed"):
            return await self._embed(inputs, max_attempts)

    async def _search_embed(
        self,
        queries: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Embed search queries with retries."""
        async with self._tracker("search_embed"):
            return await self._embed(queries, max_attempts)

    async def _embed(
        self,
        inputs: list[Any],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        """Shared retrying embed logic."""
        if not inputs:
            return []
        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

        inputs = [
            OpenAIEmbedder._SPECIAL_TOKEN_PATTERN.sub("", input_text) or "."
            for input_text in inputs
        ]

        effective_max = (
            self._max_input_length or self.max_total_input_length_per_request
        )
        inputs_chunks = [
            chunk_text_balanced(input_text, effective_max) for input_text in inputs
        ]

        chunks = [chunk for input_chunks in inputs_chunks for chunk in input_chunks]
        chunk_clusters = cluster_texts(
            chunks,
            self.max_num_inputs_per_request,
            self.max_total_input_length_per_request,
        )

        embed_call_uuid = uuid4()

        logger.debug(
            "[call uuid: %s] "
            "Attempting to create embeddings using %s OpenAI model: "
            "%d total chunks in %d clusters with max attempts %d",
            embed_call_uuid,
            self._model,
            len(chunks),
            len(chunk_clusters),
            max_attempts,
        )
        clusters_chunk_embeddings_awaitables = [
            self._embed_chunk_cluster(
                embed_call_uuid=embed_call_uuid,
                cluster_number=cluster_number,
                chunk_cluster=chunk_cluster,
                max_attempts=max_attempts,
            )
            for cluster_number, chunk_cluster in enumerate(chunk_clusters)
        ]
        clusters_chunk_embeddings = await asyncio.gather(
            *clusters_chunk_embeddings_awaitables
        )

        chunk_embeddings = [
            chunk_embedding
            for cluster_chunk_embeddings in clusters_chunk_embeddings
            for chunk_embedding in cluster_chunk_embeddings
        ]
        inputs_chunk_embeddings = unflatten_like(
            chunk_embeddings,
            inputs_chunks,
        )

        # Average chunk embeddings to get input embeddings.
        return [
            np.mean(chunk_embeddings, axis=0).astype(float).tolist()
            for chunk_embeddings in inputs_chunk_embeddings
        ]

    def _fast_http_client(self) -> RawHTTPPool | httpx.AsyncClient | None:
        """Build (once) an httpx client for the direct embeddings fast path.

        Returns None when the SDK client's attributes are not the expected
        shape, so callers fall back to the canonical SDK path.
        """
        if self._fast_http is False:
            return None
        if isinstance(self._fast_http, (RawHTTPPool, httpx.AsyncClient)):
            return self._fast_http
        api_key = getattr(self._client, "api_key", None)
        base_url = getattr(self._client, "base_url", None)
        if not isinstance(api_key, str) or base_url is None:
            self._fast_http = False
            return None
        headers = {"Authorization": f"Bearer {api_key}"}
        organization = getattr(self._client, "organization", None)
        if isinstance(organization, str):
            headers["OpenAI-Organization"] = organization
        project = getattr(self._client, "project", None)
        if isinstance(project, str):
            headers["OpenAI-Project"] = project
        # The canonical SDK allows up to 1000 concurrent connections; a small
        # pool here would throttle a high-latency remote embedder relative to
        # the canonical path.
        base = str(base_url)
        if base.startswith("http://"):
            self._fast_http = RawHTTPPool(base, headers=headers, max_connections=256)
        else:
            self._fast_http = httpx.AsyncClient(
                base_url=base,
                headers=headers,
                timeout=60,
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
            )
        return self._fast_http

    async def _fast_embed(
        self,
        embed_call_uuid: UUID,
        chunk_cluster: list[str],
    ) -> list[list[float]] | None:
        """Embed one cluster over direct httpx, base64 wire format.

        Returns None on ANY request failure so the caller runs the canonical
        SDK path, which owns retries, the dimensions-parameter fallback, and
        canonical error types. Usage metrics and the dimensionality check are
        replicated here so a fast-path success is indistinguishable from a
        canonical one.
        """
        http = self._fast_http_client()
        if http is None or not self._use_dimensions_parameter:
            return None
        body: dict[str, object] = {
            "input": chunk_cluster,
            "model": self._model,
            "dimensions": self._dimensions,
            "encoding_format": "base64",
        }
        try:
            if isinstance(http, RawHTTPPool):
                content = await http.post("/embeddings", fast_json.dumps(body))
            else:
                response = await http.post(
                    "embeddings",
                    content=fast_json.dumps(body),
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code != 200:
                    return None
                content = response.content
        except Exception:  # any transport failure -> canonical SDK path
            return None
        try:
            payload = fast_json.loads(content)
            data = sorted(
                cast(list[dict[str, Any]], payload["data"]), key=lambda d: d["index"]
            )
            embeddings = [
                np.frombuffer(
                    base64.b64decode(datum["embedding"]), dtype="float32"
                ).tolist()
                if isinstance(datum["embedding"], str)
                else cast(list[float], datum["embedding"])
                for datum in data
            ]
        except Exception:  # unexpected response shape -> canonical SDK path
            return None
        if self._should_collect_metrics:
            usage = cast(dict[str, int], payload.get("usage") or {})
            self._prompt_tokens_usage_counter.increment(
                value=usage.get("prompt_tokens", 0),
            )
            self._total_tokens_usage_counter.increment(
                value=usage.get("total_tokens", 0),
            )
        if len(embeddings[0]) != self._dimensions:
            error_message = (
                f"[call uuid: {embed_call_uuid}] "
                f"Received embedding dimensionality {len(embeddings[0])} "
                f"does not match expected dimensionality {self._dimensions}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message)
        return embeddings

    async def _embed_chunk_cluster(
        self,
        embed_call_uuid: UUID,
        cluster_number: int,
        chunk_cluster: list[str],
        max_attempts: int = 1,
    ) -> list[list[float]]:
        fast_embeddings = await self._fast_embed(embed_call_uuid, chunk_cluster)
        if fast_embeddings is not None:
            return fast_embeddings

        sleep_seconds = 1
        for attempt in range(1, max_attempts + 1):
            logger.debug(
                "[call uuid: %s] "
                "Attempting to create embeddings for cluster number %d: "
                "on attempt %d with max attempts %d",
                embed_call_uuid,
                cluster_number,
                attempt,
                max_attempts,
            )

            try:
                # Internal try-except is required
                # for models that do not support dimensions parameter

                # Avoid concurrency issues by tracking whether dimensions parameter is used for this request only.
                dimensions_parameter_used = self._use_dimensions_parameter
                try:
                    response = (
                        await self._client.embeddings.create(
                            input=chunk_cluster,
                            model=self._model,
                            dimensions=self._dimensions,
                        )
                        if dimensions_parameter_used
                        else await self._client.embeddings.create(
                            input=chunk_cluster,
                            model=self._model,
                        )
                    )
                except openai.BadRequestError as err:
                    if "dimension" in str(err).lower() and dimensions_parameter_used:
                        response = await self._client.embeddings.create(
                            input=chunk_cluster,
                            model=self._model,
                        )
                        self._use_dimensions_parameter = False
                        break
                    raise
                break
            except (
                openai.BadRequestError,
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as err:
                # Exception may be retried.
                if attempt >= max_attempts:
                    error_message = (
                        f"[call uuid: {embed_call_uuid}] "
                        "Giving up creating embeddings "
                        f"for cluster number {cluster_number} "
                        f"after failed attempt {attempt} "
                        f"due to retryable {type(err).__name__}: "
                        f"max attempts {max_attempts} reached"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from err

                logger.info(
                    "[call uuid: %s] "
                    "Retrying creating embeddings for cluster number %d "
                    "in %d seconds "
                    "after failed attempt %d due to retryable %s...",
                    embed_call_uuid,
                    cluster_number,
                    sleep_seconds,
                    attempt,
                    type(err).__name__,
                )
                await asyncio.sleep(
                    min(sleep_seconds, self._max_retry_interval_seconds),
                )
                sleep_seconds *= 2
                continue
            except (openai.APIError, openai.OpenAIError) as err:
                error_message = (
                    f"[call uuid: {embed_call_uuid}] "
                    "Giving up creating embeddings "
                    f"for cluster number {cluster_number} "
                    f"after failed attempt {attempt} "
                    f"due to non-retryable {type(err).__name__}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from err

        if self._should_collect_metrics:
            self._prompt_tokens_usage_counter.increment(
                value=response.usage.prompt_tokens,
            )
            self._total_tokens_usage_counter.increment(
                value=response.usage.total_tokens,
            )

        if len(response.data[0].embedding) != self._dimensions:
            error_message = (
                f"[call uuid: {embed_call_uuid}] "
                f"Received embedding dimensionality {len(response.data[0].embedding)} "
                f"does not match expected dimensionality {self._dimensions}"
            )
            logger.exception(error_message)
            raise ExternalServiceAPIError(error_message)

        return [datum.embedding for datum in response.data]

    @property
    def model_id(self) -> str:
        """Return the embedding model identifier."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        """Return the similarity metric used by this embedder."""
        # https://platform.openai.com/docs/guides/embeddings
        return SimilarityMetric.COSINE
