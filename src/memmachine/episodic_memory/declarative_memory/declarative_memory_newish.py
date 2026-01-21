"""Declarative memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
import math
import re
from collections.abc import Iterable
from typing import cast
from uuid import uuid4

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
from memmachine.common.reranker import Reranker
from memmachine.common.language_model import LanguageModel
from memmachine.common.utils import compute_similarity, rank_by_mmr
# from memmachine.common.semantic import get_semantic_units
from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore

from .data_types import (
    ContentType,
    Derivative,
    Episode,
    FilterablePropertyValue,
    demangle_filterable_property_key,
    is_mangled_filterable_property_key,
    mangle_filterable_property_key,
)

logger = logging.getLogger(__name__)


class DeclarativeMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_id (str):
            Session identifier.
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
        language_model (LanguageModel):
            LanguageModel instance for language generation.

    """

    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    vector_graph_store: InstanceOf[VectorGraphStore] = Field(
        ...,
        description="VectorGraphStore instance for storing and retrieving memories",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    language_model: InstanceOf[LanguageModel] | None = Field(
        None,
        description="LanguageModel instance for language generation",
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

        self._vector_graph_store = params.vector_graph_store
        self._embedder = params.embedder
        self._reranker = params.reranker
        self._language_model = params.language_model

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,  # Default separators are ["\n\n", "\n", " ", ""]
            separators=[
                "\n\n",
                "],\n",
                "},\n",
                "),\n",
                "]\n",
                "}\n",
                ")\n",
                ",\n",
                "\uff1f\n",  # Fullwidth question mark
                "?\n",
                "\uff01\n",  # Fullwidth exclamation mark
                "!\n",
                "\u3002\n",  # Ideographic full stop
                ".\n",
                "\uff1f",  # Fullwidth question mark
                "? ",
                "\uff01",  # Fullwidth exclamation mark
                "! ",
                "\u3002",  # Ideographic full stop
                ". ",
                "; ",
                ": ",
                "—",
                "--",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                ", ",
                "\u200b",  # Zero-width space
                " ",
                "",
            ],
            keep_separator="end",
        )

        self._episode_collection = f"Episode_{session_id}"
        self._derivative_collection = f"Derivative_{session_id}"

        self._derived_from_relation = f"DERIVED_FROM_{session_id}"

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

        episodes = await asyncio.to_thread(self._chunk_episodes, episodes)

        derive_derivatives_tasks = [
            self._derive_derivatives(episode) for episode in episodes
        ]

        episodes_derivatives = await asyncio.gather(*derive_derivatives_tasks)

        derivatives = [
            derivative
            for episode_derivatives in episodes_derivatives
            for derivative in episode_derivatives
        ]

        embeddings = await self._embedder.ingest_embed(
            [f"{episode.source}: {episode.content}" for episode in episodes] + [derivative.content for derivative in derivatives],
            max_attempts=5,
        )

        episode_embeddings = embeddings[:len(episodes)]
        derivative_embeddings = embeddings[len(episodes):]

        # derivative_embeddings = [
        #     datetime_rotary_decay(embedding, derivative.timestamp)
        #     for derivative, embedding in zip(
        #         derivatives,
        #         derivative_embeddings,
        #         strict=True,
        #     )
        # ]

        # derivative_embeddings = await self._pattern_separation(derivative_embeddings)

        # derivatives_indexes = [
        #     self._ou_recruiter.recruit_indices(derivative.timestamp)
        #     for derivative in derivatives
        # ]

        # derivative_embeddings_np = [
        #     np.zeros_like(derivative_embedding)
        #     for derivative_embedding in derivative_embeddings
        # ]

        # for derivative_embedding, derivative_embedding_np, derivative_indexes in zip(
        #     derivative_embeddings,
        #     derivative_embeddings_np,
        #     derivatives_indexes,
        #     strict=True,
        # ):
        #     derivative_embedding_np[derivative_indexes] = np.array(
        #         derivative_embedding,
        #     )[derivative_indexes]

        episode_nodes = [
            Node(
                uid=episode.uid,
                properties={
                    "uid": str(episode.uid),
                    "timestamp": episode.timestamp,
                    "source": episode.source,
                    "content_type": episode.content_type.value,
                    "content": episode.content,
                    "user_metadata": json.dumps(episode.user_metadata),
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in episode.filterable_properties.items()
                },
                embeddings={
                    DeclarativeMemory._embedding_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    ): (embedding, self._embedder.similarity_metric),
                },
            )
            for episode, embedding in zip(
                episodes,
                episode_embeddings,
                strict=True,
            )
        ]

        derivative_nodes = [
            Node(
                uid=derivative.uid,
                properties={
                    "uid": derivative.uid,
                    "timestamp": derivative.timestamp,
                    "source": derivative.source,
                    "content_type": derivative.content_type.value,
                    "content": derivative.content,
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in derivative.filterable_properties.items()
                },
                embeddings={
                    DeclarativeMemory._embedding_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    ): (embedding, self._embedder.similarity_metric),
                },
            )
            for derivative, embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        derivative_episode_edges = [
            Edge(
                uid=str(uuid4()),
                source_uid=derivative.uid,
                target_uid=episode.uid,
            )
            for episode, episode_derivatives in zip(
                episodes,
                episodes_derivatives,
                strict=True,
            )
            for derivative in episode_derivatives
        ]

        add_nodes_tasks = [
            self._vector_graph_store.add_nodes(
                collection=self._episode_collection,
                nodes=episode_nodes,
            ),
            self._vector_graph_store.add_nodes(
                collection=self._derivative_collection,
                nodes=derivative_nodes,
            ),
        ]
        await asyncio.gather(*add_nodes_tasks)

        await self._vector_graph_store.add_edges(
            relation=self._derived_from_relation,
            source_collection=self._derivative_collection,
            target_collection=self._episode_collection,
            edges=derivative_episode_edges,
        )

    def _chunk_episodes(
        self,
        episodes: Iterable[Episode],
    ) -> list[Episode]:
        return [
            Episode(
                uid=f"{episode.uid}_chunk_{i}",
                timestamp=episode.timestamp,
                source=episode.source,
                content_type=ContentType.MESSAGE,
                content=chunk,
                filterable_properties=episode.filterable_properties,
                user_metadata=episode.user_metadata,
            )
            for episode in episodes
            for i, chunk in enumerate(self._text_splitter.split_text(episode.content))
        ]

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
                partitions = {
                    partition
                    for line in episode.content.strip().splitlines()
                    for partition in sent_tokenize(line.strip())
                }

                sentences = {
                    sentence
                    for partition in partitions
                    for sentence in re.findall(
                        r".*?(?:[?!\.\uff1f\uff01\u3002]*[?!\uff1f\uff01\u3002][?!\.\uff1f\uff01\u3002]*)+|.+$",
                        partition,
                    )
                    if sentence
                }

                # get_semantic_units_tasks = [
                #     asyncio.to_thread(get_semantic_units, sentence)
                #     for sentence in sentences
                # ]

                # semantic_units = [
                #     semantic_unit
                #     for semantic_units in await asyncio.gather(*get_semantic_units_tasks)
                #     for semantic_unit in semantic_units
                # ]

                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.MESSAGE,
                        content=f"{episode.source}: {sentence}",
                        filterable_properties=episode.filterable_properties,
                    )
                    for sentence in sentences
                ]
            case ContentType.TEXT:
                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.TEXT,
                        content=episode.content,
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

        # Search graph store for vector matches.
        (
            matched_derivative_nodes,
            _,
        ) = await self._vector_graph_store.search_similar_nodes(
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

        matched_derivative_embeddings = [
            derivative_node.embeddings[
                DeclarativeMemory._embedding_name(
                    self._embedder.model_id,
                    self._embedder.dimensions,
                )
            ][0]
            for derivative_node in matched_derivative_nodes
        ]

        # pairwise_candidate_similarities = [
        #     compute_similarity(
        #         matched_derivative_embedding,
        #         matched_derivative_embeddings,
        #         self._embedder.similarity_metric,
        #     )
        #     for matched_derivative_embedding in matched_derivative_embeddings
        # ]

        # Get source episodes of matched derivatives.
        search_derivatives_source_episode_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._episode_collection,
                this_collection=self._derivative_collection,
                this_node_uid=matched_derivative_node.uid,
                find_sources=False,
                find_targets=True,
                node_property_filter=mangled_property_filter,
            )
            for matched_derivative_node in matched_derivative_nodes
        ]

        # Use a dict instead of a set to preserve order.
        source_episode_nodes = dict.fromkeys(
            episode_node
            for episode_nodes in await asyncio.gather(
                *search_derivatives_source_episode_nodes_tasks,
            )
            for episode_node in episode_nodes
        )

        # Use source episodes as nuclei for contextualization.
        nuclear_episodes = [
            DeclarativeMemory._episode_from_episode_node(source_episode_node)
            for source_episode_node in source_episode_nodes
        ]

        nuclear_episode_embeddings = [
            source_episode_node.embeddings[
                DeclarativeMemory._embedding_name(
                    self._embedder.model_id,
                    self._embedder.dimensions,
                )
            ][0]
            for source_episode_node in source_episode_nodes
        ]

        contextualize_episode_tasks = [
            self._contextualize_episode(
                nuclear_episode,
                nuclear_episode_embedding=nuclear_episode_embedding,
                mangled_property_filter=mangled_property_filter,
            )
            for nuclear_episode, nuclear_episode_embedding in zip(nuclear_episodes, nuclear_episode_embeddings)
        ]

        episode_contexts = await asyncio.gather(*contextualize_episode_tasks)

        # Rerank episode contexts.
        episode_context_scores = await self._score_episode_contexts(
            query,
            episode_contexts,
        )

        # ranked_indexes = rank_by_mmr(
        #     candidate_relevances=episode_context_scores,
        #     pairwise_candidate_similarities=pairwise_candidate_similarities,
        #     lambda_param=0.5,
        # )

        # reranked_anchored_episode_contexts = [
        #     (nuclear_episodes[index], episode_contexts[index])
        #     for index in ranked_indexes
        # ]

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

        # reranked_anchored_episode_contexts = [
        #     (episode_context_score, nuclear_episode, episode_context)
        #     for episode_context_score, nuclear_episode, episode_context in sorted(
        #         zip(
        #             episode_context_scores,
        #             nuclear_episodes,
        #             episode_contexts,
        #             strict=True,
        #         ),
        #         key=lambda triple: triple[0],
        #         reverse=True,
        #     )
        # ]

        # return reranked_anchored_episode_contexts

    async def _contextualize_episode(
        self,
        nuclear_episode: Episode,
        nuclear_episode_embedding: list[float],
        max_backward_episodes: int = 1,
        max_forward_episodes: int = 1,
        mangled_property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        previous_episode_nodes = (
            await self._vector_graph_store.search_directional_nodes(
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

        next_episode_nodes = await self._vector_graph_store.search_directional_nodes(
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

        episodes = (
            [
                DeclarativeMemory._episode_from_episode_node(episode_node)
                for episode_node in reversed(previous_episode_nodes)
            ]
            + [nuclear_episode]
            + [
                DeclarativeMemory._episode_from_episode_node(episode_node)
                for episode_node in next_episode_nodes
            ]
        )

        combined_episode_embeddings = DeclarativeMemory._combine_consecutive_episode_embeddings(
            episodes,
            [
                episode_node.embeddings[
                    DeclarativeMemory._embedding_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    )
                ][0]
                for episode_node in previous_episode_nodes
            ]
            + [nuclear_episode_embedding]
            + [
                episode_node.embeddings[
                    DeclarativeMemory._embedding_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    )
                ][0]
                for episode_node in next_episode_nodes
            ],
        )

        distances = DeclarativeMemory._calculate_cosine_distances(
            combined_episode_embeddings,
        )

        start, end = DeclarativeMemory._get_fh_span(
            index=len(previous_episode_nodes),
            distances=distances,
        )

        context = episodes[start:end + 1]
        return context

    @staticmethod
    def _calculate_cosine_distances(episode_embeddings: list[list[float]]):
        distances = []
        for i in range(len(episode_embeddings) - 1):
            embedding_current = episode_embeddings[i]
            embedding_next = episode_embeddings[i + 1]

            # Calculate cosine similarity
            similarity = compute_similarity(embedding_current, [embedding_next])[0]

            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

        return distances

    @staticmethod
    def _combine_consecutive_episode_embeddings(
        episodes: list[Episode],
        episode_embeddings: list[list[float]],
        max_distance: int = 1,
    ) -> list[list[float]]:
        combined_episode_embeddings = []
        for i in range(len(episodes)):
            combined_episode_embedding = np.zeros_like(episode_embeddings[0])

            # Add episodes before the current one, based on the buffer size.
            for j in range(i - max_distance, i):
                # Check if the index j is not negative
                # (to avoid index out of range like on the first one)
                if j >= 0:
                    combined_episode_embedding += math.sqrt(len(episodes[j].content)) * np.array(episode_embeddings[j])

            # Add the current episode
            combined_episode_embedding += math.sqrt(len(episodes[i].content)) * np.array(episode_embeddings[i])

            # Add episodes after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + max_distance):
                # Check if the index j is within the range of the episodes list
                if j < len(episodes):
                    combined_episode_embedding += math.sqrt(len(episodes[j].content)) * np.array(episode_embeddings[j])

            combined_episode_embeddings.append(combined_episode_embedding.astype(float).tolist())

        return combined_episode_embeddings

    @staticmethod
    def _get_fh_span(index, distances, k=0.55):
        d_low = np.percentile(distances, 5)
        d_high = np.percentile(distances, 95)
        denom = max(d_high - d_low, 0.05)
        norm_dist = np.clip((distances - d_low) / denom, 0, 1)

        n_edges = len(norm_dist)

        # Setup Priority Expansion
        backward_ptr = index - 1
        forward_ptr = index

        curr_int = 0.0
        size = 1 # Start with the seed episode

        # Track the limits of our span
        span_start = index
        span_end = index

        while True:
            # Get distances to the next potential neighbors
            d_back = norm_dist[backward_ptr] if backward_ptr >= 0 else float('inf')
            d_fwd = norm_dist[forward_ptr] if forward_ptr < n_edges else float('inf')

            # If both directions are blocked by the end of the window
            if d_back == float('inf') and d_fwd == float('inf'):
                break

            # 3. Choose the "easier" (smaller distance) direction
            if d_back < d_fwd:
                candidate_d = d_back
                direction = 'back'
            else:
                candidate_d = d_fwd
                direction = 'fwd'

            # 4. Apply F-H Merging Criterion
            # Threshold = max_internal_edge + k/size
            threshold = curr_int + (k / size)

            if candidate_d <= threshold:
                # Successful Merge
                size += 1
                curr_int = max(curr_int, candidate_d)

                if direction == 'back':
                    span_start = backward_ptr
                    backward_ptr -= 1
                else:
                    span_end = forward_ptr + 1
                    forward_ptr += 1
            else:
                # The easier side failed the threshold, meaning the
                # harder side definitely will too. Stop expanding.
                break

        return span_start, span_end

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

        last_episode: Episode | None = None
        accumulated_content = ""
        for episode in episode_context:
            if (
                last_episode is not None
                and episode.timestamp == last_episode.timestamp
                and episode.source == last_episode.source
                and episode.content_type == last_episode.content_type
            ):
                accumulated_content += episode.content
                continue

            match episode.content_type:
                case ContentType.MESSAGE:
                    context_date = DeclarativeMemory._format_date(
                        episode.timestamp.date(),
                    )
                    context_time = DeclarativeMemory._format_time(
                        episode.timestamp.time(),
                    )
                    context_string += f"{json.dumps(accumulated_content)}\n[{context_date} at {context_time}] {episode.source}: "
                    accumulated_content = episode.content

                case ContentType.TEXT:
                    context_string += f"{json.dumps(episode.content)}\n---\n"
                    accumulated_content = episode.content
                case _:
                    logger.warning(
                        "Unsupported content type for episode context formatting: %s",
                        episode.content_type,
                    )

            last_episode = episode

        if accumulated_content:
            context_string += json.dumps(accumulated_content) + "\n"

        return context_string.strip()

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
        episode_nodes = await self._vector_graph_store.get_nodes(
            collection=self._episode_collection,
            node_uids=uids,
        )

        episodes = [
            DeclarativeMemory._episode_from_episode_node(episode_node)
            for episode_node in episode_nodes
        ]

        return episodes

    async def get_matching_episodes(
        self,
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        """Filter episodes by their properties."""
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        matching_episode_nodes = await self._vector_graph_store.search_matching_nodes(
            collection=self._episode_collection,
            property_filter=mangled_property_filter,
        )

        matching_episodes = [
            DeclarativeMemory._episode_from_episode_node(matching_episode_node)
            for matching_episode_node in matching_episode_nodes
        ]

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
            self._vector_graph_store.delete_nodes(
                collection=self._episode_collection,
                node_uids=uids,
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
