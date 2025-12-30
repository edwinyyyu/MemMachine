"""Declarative memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections import deque
from collections.abc import Iterable
from typing import cast
from uuid import uuid4

from nltk import sent_tokenize
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder.embedder import Embedder
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_store import VectorStore

from .data_types import (
    ContentType,
    Derivative,
    Episode,
    FilterablePropertyValue,
    demangle_filterable_property_key,
    is_mangled_filterable_property_key,
    mangle_filterable_property_key,
)
from .declarative_episode_store import DeclarativeEpisodeStore

logger = logging.getLogger(__name__)


class DeclarativeMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_id (str):
            Session identifier.
        episode_store (DeclarativeEpisodeStore):
            DeclarativeEpisodeStore instance for episode storage and retrieval.
        vector_store (VectorStore):
            VectorStore instance for vector storage and retrieval.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.

    """

    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    episode_store: InstanceOf[DeclarativeEpisodeStore] = Field(
        ...,
        description="DeclarativeEpisodeStore instance for episode storage and retrieval",
    )
    vector_store: InstanceOf[VectorStore] = Field(
        ...,
        description="VectorStore instance for vector storage and retrieval",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )


class DeclarativeMemory:
    """Declarative memory system."""

    def __init__(self, params: DeclarativeMemoryParams) -> None:
        """
        Initialize a DeclarativeMemory with the provided parameters.

        Args:
            params (DeclarativeMemoryParams):
                Parameters for the DeclarativeMemory.

        """
        session_id = params.session_id

        self._episode_store = params.episode_store
        self._vector_store = params.vector_store
        self._embedder = params.embedder
        self._reranker = params.reranker

        self._score_single_episodes_threshold = 10
        self._episode_context_content_length_quota_factor = 20
        self._episode_context_content_length_max_quota = 400

    async def add_episodes(
        self,
        episodes: Iterable[Episode],
    ) -> None:
        """
        Add episodes.

        Episodes are sorted by timestamp.
        Episodes with the same timestamp are sorted by UID.

        Args:
            episodes (Iterable[Episode]): The episodes to add.

        """
        episodes = sorted(
            episodes,
            key=lambda episode: (episode.timestamp, episode.uid),
        )
        derive_derivatives_tasks = [
            self._derive_derivatives(episode) for episode in episodes
        ]

        episodes_derivatives = await asyncio.gather(*derive_derivatives_tasks)

        derivatives = [
            derivative
            for episode_derivatives in episodes_derivatives
            for derivative in episode_derivatives
        ]

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.content for derivative in derivatives],
        )

        await self._vector_store.add(
            [Entry()]
        )

    async def _derive_derivatives(
        self,
        episode: Episode,
    ) -> list[Derivative]:
        """
        Derive derivatives from an episode.

        Args:
            episode (Episode):
                The episode from which to derive derivatives.

        Returns:
            list[Derivative]: A list of derived derivatives.

        """
        match episode.content_type:
            case ContentType.MESSAGE:
                sentences = []
                for line in episode.content.strip().splitlines():
                    sentences.extend(sent_tokenize(line.strip()))

                message_timestamp = episode.timestamp.strftime(
                    "%A, %B %d, %Y at %I:%M %p",
                )
                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.MESSAGE,
                        content=f"[{message_timestamp}] {episode.source}: {sentence}",
                        filterable_properties=episode.filterable_properties,
                    )
                    for sentence in sentences
                ]
            case ContentType.TEXT:
                text_content = episode.content
                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.TEXT,
                        content=text_content,
                        filterable_properties=episode.filterable_properties,
                    ),
                ]
            case _:
                logger.warning(
                    "Unsupported content type for derivative derivation: %s",
                    episode.content_type,
                )
                return []

    async def search(
        self,
        query: str,
        *,
        max_num_episodes: int = 20,
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        """
        Search declarative memory for episodes relevant to the query.

        Args:
            query (str):
                The search query.
            max_num_episodes (int):
                The maximum number of episodes to return
                (default: 20).
            property_filter (FilterExpr | None):
                Filterable property keys and values
                to use for filtering episodes
                (default: None).

        Returns:
            list[Episode]:
                A list of episodes relevant to the query, ordered chronologically.

        """
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Search vector store for matches.
        (
            matched_derivative_entries,
            _,
        ) = await self._vector_store.search(
            collection=self._derivative_collection,
            embedding_name=(
                DeclarativeMemory._embedding_name(
                    self._embedder.model_id,
                    self._embedder.dimensions,
                )
            ),
            query_embedding=query_embedding,
            similarity_metric=self._embedder.similarity_metric,
            limit=100,
            property_filter=mangled_property_filter,
        )

        # Get origin episodes of matched derivatives.
        derivative_origin_episode_uids = dict.fromkeys(
            matched_derivative_entry.uid for matched_derivative_entry in matched_derivative_entries
        )

        derivative_origin_episodes = await self._episode_store.get_episodes(
            episode_uids=derivative_origin_episode_uids,
        )

        # Use origin episodes as nuclei for contextualization.
        contextualize_episode_tasks = [
            self._contextualize_episode(
                nuclear_episode,
                episode_context_content_length_quota=min(
                    self._episode_context_content_length_max_quota,
                    self._episode_context_content_length_quota_factor
                    * max_num_episodes,
                ),
                mangled_property_filter=mangled_property_filter,
            )
            for nuclear_episode in nuclear_episodes
        ]

        episode_contexts = await asyncio.gather(*contextualize_episode_tasks)

        if max_num_episodes <= self._score_single_episodes_threshold:
            nuclear_episodes = [
                episode
                for episode_context in episode_contexts
                for episode in episode_context
            ]
            episode_contexts = [
                [nuclear_episode] for nuclear_episode in nuclear_episodes
            ]

        # Rerank episode contexts.
        episode_context_scores = await self._score_episode_contexts(
            query,
            episode_contexts,
        )

        reranked_anchored_episode_contexts = [
            (nuclear_episode, episode_context)
            for _, nuclear_episode, episode_context in sorted(
                zip(
                    episode_context_scores,
                    nuclear_episodes,
                    episode_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        # Unify episode contexts.
        unified_episode_context = DeclarativeMemory._unify_anchored_episode_contexts(
            reranked_anchored_episode_contexts,
            max_num_episodes=max_num_episodes,
        )
        return unified_episode_context

    async def _contextualize_episode(
        self,
        nuclear_episode: Episode,
        episode_context_content_length_quota: int,
        max_backward_episodes: int = 4,
        max_forward_episodes: int = 8,
        mangled_property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        episode_context = deque([nuclear_episode])
        episode_context_content_length = len(nuclear_episode.content)

        if episode_context_content_length >= episode_context_content_length_quota:
            return list(episode_context)

        previous_episodes = (
            await self._episode_store.search(
                collection=self._episode_collection,
                by_properties=("timestamp", "uid"),
                starting_at=(
                    nuclear_episode.timestamp,
                    str(nuclear_episode.uid),
                ),
                order_ascending=(False, False),
                include_equal_start=False,
                limit=max_backward_episodes,
                property_filter=mangled_property_filter,
            )
        )

        next_episodes = await self._episode_store.search(
            collection=self._episode_collection,
            by_properties=("timestamp", "uid"),
            starting_at=(
                nuclear_episode.timestamp,
                str(nuclear_episode.uid),
            ),
            order_ascending=(True, True),
            include_equal_start=False,
            limit=max_forward_episodes,
            property_filter=mangled_property_filter,
        )

        previous_episode_index = 0
        next_episode_index = 0
        while previous_episode_index < len(
            previous_episodes
        ) or next_episode_index < len(
            next_episodes,
        ):
            for _ in range(2):
                if next_episode_index < len(next_episodes):
                    next_episode = next_episodes[next_episode_index]
                    next_episode_index += 1

                    episode_context.append(next_episode)
                    episode_context_content_length += len(next_episode.content)
                    if (
                        episode_context_content_length
                        >= episode_context_content_length_quota
                    ):
                        return list(episode_context)

            if previous_episode_index < len(previous_episodes):
                previous_episode = previous_episodes[previous_episode_index]
                previous_episode_index += 1

                episode_context.appendleft(previous_episode)
                episode_context_content_length += len(previous_episode.content)
                if (
                    episode_context_content_length
                    >= episode_context_content_length_quota
                ):
                    return list(episode_context)

        return list(episode_context)

    async def _score_episode_contexts(
        self,
        query: str,
        episode_contexts: Iterable[Iterable[Episode]],
    ) -> list[float]:
        """Score episode contexts based on their relevance to the query."""
        context_strings = []
        for episode_context in episode_contexts:
            context_string = DeclarativeMemory.string_from_episode_context(
                episode_context
            )
            context_strings.append(context_string)

        episode_context_scores = await self._reranker.score(query, context_strings)

        return episode_context_scores

    @staticmethod
    def string_from_episode_context(episode_context: Iterable[Episode]) -> str:
        """Format episode context as a string."""
        context_string = ""

        for episode in episode_context:
            match episode.content_type:
                case ContentType.MESSAGE:
                    context_date = DeclarativeMemory._format_date(
                        episode.timestamp.date(),
                    )
                    context_time = DeclarativeMemory._format_time(
                        episode.timestamp.time(),
                    )
                    context_string += f"[{context_date} at {context_time}] {episode.source}: {json.dumps(episode.content)}\n"
                case ContentType.TEXT:
                    context_string += json.dumps(episode.content) + "\n"

        return context_string

    @staticmethod
    def _format_date(date: datetime.date) -> str:
        """Format the date as a string."""
        return date.strftime("%A, %B %d, %Y")

    @staticmethod
    def _format_time(time: datetime.time) -> str:
        """Format the time as a string."""
        return time.strftime("%I:%M %p")

    async def get_episodes(self, uids: Iterable[str]) -> list[Episode]:
        """Get episodes by their UIDs."""
        episodes = await self._episode_store.get_episodes(
            collection=self._episode_collection,
            episode_uids=uids,
        )

        return episodes

    async def get_matching_episodes(
        self,
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        """Filter episodes by their properties."""
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        matching_episodes = await self._episode_store.search(
            collection=self._episode_collection,
            property_filter=mangled_property_filter,
        )

        return matching_episodes

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        """Delete episodes by their UIDs."""
        uids = list(uids)

        search_derived_derivative_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._derivative_collection,
                this_collection=self._episode_collection,
                this_node_uid=episode_uid,
                find_sources=True,
                find_targets=False,
            )
            for episode_uid in uids
        ]

        derived_derivative_nodes = [
            derivative_node
            for derivative_nodes in await asyncio.gather(
                *search_derived_derivative_nodes_tasks,
            )
            for derivative_node in derivative_nodes
        ]

        delete_nodes_tasks = [
            self._episode_store.delete_episodes(
                collection=self._episode_collection,
                episode_uids=uids,
            ),
            self._vector_graph_store.delete_nodes(
                collection=self._derivative_collection,
                node_uids=[
                    derivative_node.uid for derivative_node in derived_derivative_nodes
                ],
            ),
        ]

        await asyncio.gather(*delete_nodes_tasks)

    @staticmethod
    def _unify_anchored_episode_contexts(
        anchored_episode_contexts: Iterable[tuple[Episode, Iterable[Episode]]],
        max_num_episodes: int,
    ) -> list[Episode]:
        """Unify anchored episode contexts into a single list within the limit."""
        episode_set: set[Episode] = set()

        for nuclear_episode, context in anchored_episode_contexts:
            context = list(context)

            if len(episode_set) >= max_num_episodes:
                break
            if (len(episode_set) + len(context)) <= max_num_episodes:
                # It is impossible that the context exceeds the limit.
                episode_set.update(context)
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize episodes near the nuclear episode.

                # Sort chronological episodes by weighted index-proximity to the nuclear episode.
                nuclear_index = context.index(nuclear_episode)

                nuclear_context = sorted(
                    context,
                    key=lambda episode: DeclarativeMemory._weighted_index_proximity(
                        episode=episode,
                        context=context,
                        nuclear_index=nuclear_index,
                    ),
                )

                # Add episodes to unified context until limit is reached,
                # or until the context is exhausted.
                for episode in nuclear_context:
                    if len(episode_set) >= max_num_episodes:
                        break
                    episode_set.add(episode)

        unified_episode_context = sorted(
            episode_set,
            key=lambda episode: (
                episode.timestamp,
                episode.uid,
            ),
        )

        return unified_episode_context

    @staticmethod
    def _weighted_index_proximity(
        episode: Episode,
        context: list[Episode],
        nuclear_index: int,
    ) -> float:
        proximity = context.index(episode) - nuclear_index
        if proximity >= 0:
            # Forward recall is better than backward recall.
            return (proximity - 0.5) / 2
        return -proximity

    @staticmethod
    def _episode_from_episode_node(episode_node: Node) -> Episode:
        return Episode(
            uid=cast("str", episode_node.properties["uid"]),
            timestamp=cast("datetime.datetime", episode_node.properties["timestamp"]),
            source=cast("str", episode_node.properties["source"]),
            content_type=ContentType(episode_node.properties["content_type"]),
            content=episode_node.properties["content"],
            filterable_properties={
                demangle_filterable_property_key(key): cast(
                    "FilterablePropertyValue",
                    value,
                )
                for key, value in episode_node.properties.items()
                if is_mangled_filterable_property_key(key)
            },
            user_metadata=json.loads(
                cast("str", episode_node.properties["user_metadata"]),
            ),
        )

    @staticmethod
    def _embedding_name(model_id: str, dimensions: int) -> str:
        """
        Generate a standardized property name for embeddings based on the model ID and embedding dimensions.

        Args:
            model_id (str): The identifier of the embedding model.
            dimensions (int): The dimensionality of the embedding.

        Returns:
            str: A standardized property name for the embedding.

        """
        return f"embedding_{model_id}_{dimensions}d"

    @staticmethod
    def _mangle_property_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        if property_filter is None:
            return None

        return DeclarativeMemory._mangle_filter_expr(property_filter)

    @staticmethod
    def _mangle_filter_expr(expr: FilterExpr | None) -> FilterExpr | None:
        if expr is None:
            return None

        if isinstance(expr, FilterComparison):
            return FilterComparison(
                field=mangle_filterable_property_key(expr.field),
                op=expr.op,
                value=expr.value,
            )
        if isinstance(expr, FilterAnd):
            return FilterAnd(
                left=DeclarativeMemory._mangle_filter_expr(expr.left),
                right=DeclarativeMemory._mangle_filter_expr(expr.right),
            )
        if isinstance(expr, FilterOr):
            return FilterOr(
                left=DeclarativeMemory._mangle_filter_expr(expr.left),
                right=DeclarativeMemory._mangle_filter_expr(expr.right),
            )
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
