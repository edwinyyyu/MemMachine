"""
Declarative memory system for storing and retrieving
episodic and semantic memory.
"""

import asyncio
import functools
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime
from string import Template
from typing import Any, Self, cast
from uuid import uuid4

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.embedder.embedder import Embedder
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore
from nltk import sent_tokenize
from pydantic import BaseModel, Field, InstanceOf

from .data_types import (
    ContentType,
    Episode,
    Chunk,
    Derivative,
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
        max_chunk_length (int):
            Maximum length of a chunk in characters
            (default: 1000).
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
    """

    max_chunk_length: int = Field(
        1000,
        description="Maximum length of a chunk in characters.",
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


class DeclarativeMemory:
    """
    Declarative memory system.
    """

    def __init__(self, params: DeclarativeMemoryParams):
        """
        Initialize a DeclarativeMemory with the provided parameters.

        Args:
            params (DeclarativeMemoryParams):
                Parameters for the DeclarativeMemory.
        """

        self._max_chunk_length = params.max_chunk_length
        self._vector_graph_store = params.vector_graph_store
        self._embedder = params.embedder
        self._reranker = params.reranker

        self._episode_collection = "Episode"
        self._chunk_collection = "Chunk"
        self._derivative_collection = "Derivative"

        self._contains_relation = "CONTAINS"
        self._derived_from_relation = "DERIVED_FROM"

        self._message_metadata_template = Template("[$timestamp] $source: $content")
        self._temporal_context: deque[Chunk] | None = None


    async def add_episodes(
        self,
        episodes: Iterable[Episode],
    ):
        """
        Add episodes.

        Args:
            episodes (Iterable[Episode]): The episodes to add.
        """

        # TODO @edwinyyyu: lock this
        # The memory is server-side so it can be stateful.
        # Remember previous episodes for context.
        # This is like coming back to work on a Monday.
        # if self._temporal_context is None:
        # task = self._vector_graph_store.get_last_episodes

        episodes = list(episodes)
        episode_nodes = [
            Node(
                uuid=episode.uuid,
                data_properties={
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
            )
            for episode in episodes
        ]

        chunk_episode_tasks = [
            DeclarativeMemory._chunk_episode(episode) for episode in episodes
        ]
        episodes_chunks = await asyncio.gather(*chunk_episode_tasks)

        chunks = [
            chunk for episode_chunks in episodes_chunks for chunk in episode_chunks
        ]

        chunk_nodes = [
            Node(
                uuid=chunk.uuid,
                data_properties={
                    "sequence_number": chunk.sequence_number,
                    "timestamp": chunk.timestamp,
                    "source": chunk.source,
                    "content_type": chunk.content_type.value,
                    "content": chunk.content,
                    "user_metadata": json.dumps(chunk.user_metadata),
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in chunk.filterable_properties.items()
                },
            )
            for chunk in chunks
        ]

        episode_chunk_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=episode.uuid,
                target_uuid=chunk.uuid,
            )
            for episode, episode_chunks in zip(episodes, episodes_chunks)
            for chunk in episode_chunks
        ]

        derive_derivatives_tasks = []
        for chunk in chunks:
            derive_derivatives_tasks.append(
                self._derive_derivatives(
                    chunk,
                    list(self._temporal_context),
                )
            )
            self._temporal_context.append(chunk)

        chunks_derivatives = await asyncio.gather(*derive_derivatives_tasks)

        derivatives = [
            derivative
            for chunk_derivatives in chunks_derivatives
            for derivative in chunk_derivatives
        ]

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.content for derivative in derivatives],
        )

        derivative_nodes = [
            Node(
                uuid=derivative.uuid,
                data_properties={
                    "content_type": derivative.content_type.value,
                    "content": derivative.content,
                },
                embedding_properties={
                    DeclarativeMemory._embedding_property_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    ): embedding,
                },
            )
            for derivative, embedding in zip(derivatives, derivative_embeddings)
        ]

        derivative_chunk_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=derivative.uuid,
                target_uuid=chunk.uuid,
            )
            for chunk, chunk_derivatives in zip(chunks, chunks_derivatives)
            for derivative in chunk_derivatives
        ]

        await self._vector_graph_store.add_nodes(
            colleciton=self._episode_collection,
            nodes=episode_nodes,
        )
        await self._vector_graph_store.add_nodes(
            collection=self._chunk_collection,
            nodes=chunk_nodes,
        )
        await self._vector_graph_store.add_edges(
            relation=self._contains_relation,
            source_collection=self._episode_collection,
            target_collection=self._chunk_collection,
            edges=episode_chunk_edges,
        )
        await self._vector_graph_store.add_nodes(
            collection=self._derivative_collection,
            nodes=derivative_nodes,
        )
        await self._vector_graph_store.add_edges(
            relation=self._derived_from_relation,
            source_collection=self._derivative_collection,
            target_collection=self._chunk_collection,
            edges=derivative_chunk_edges,
        )

    def _chunk_episode(self, episode: Episode) -> list[Chunk]:
        """
        Partition episode into chunks.

        Args:
            episode (Episode): The episode whose content to partition.

        Returns:
            list[Chunk]: A list of chunks.
        """
        match episode.content_type:
            case ContentType.MESSAGE | ContentType.TEXT:
                sentences = []

                for line in episode.content.strip().splitlines():
                    for sentence in sent_tokenize(line.strip()):
                        if len(sentence) > self._max_sentence_length:
                            sentence_splits = self._split_sentence(sentence)
                            sentences.extend(sentence_splits)
                        else:
                            sentences.append(sentence)

                return [
                    Chunk(
                        uuid=uuid4(),
                        episode_uuid=episode.uuid,
                        sequence_number=index,
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=episode.content_type,
                        content=sentence,
                        filterable_properties=episode.filterable_properties,
                        user_metadata=episode.user_metadata,
                    )
                    for index, sentence in enumerate(sentences)
                ]
            case _:
                return [
                    Chunk(
                        uuid=uuid4(),
                        episode_uuid=episode.uuid,
                        sequence_number=0,
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=episode.content_type,
                        content=episode.content,
                        filterable_properties=episode.filterable_properties,
                        user_metadata=episode.user_metadata,
                    )
                ]

    def _split_sentence(self, sentence: str) -> list[str]:
        """
        Split a long sentence into smaller chunks.

        Args:
            sentence (str): The sentence to split.

        Returns:
            list[str]: A list of sentence chunks.
        """
        return []  # TODO @edwinyyyu: Implement sentence splitting logic here.

    async def _derive_derivatives(
        self,
        chunk: Chunk,
        temporal_context: Iterable[Chunk],
    ) -> list[Derivative]:
        """
        Derive derivatives from a chunk from the temporal context.

        Args:
            chunk (Chunk):
                The chunk from which to derive derivatives.
            temporal_context (Iterable[Chunk]):
                The temporal context containing previous chunks.

        Returns:
            list[Derivative]: A list of derived derivatives.
        """
        match chunk.content_type:
            case ContentType.MESSAGE:
                message_content = self._message_metadata_template.safe_substitute(
                    {
                        "timestamp": chunk.timestamp,
                        "source": chunk.source,
                        "content": chunk.content,
                    },
                    **{
                        key: value
                        for key, value in {
                            **chunk.filterable_properties,
                            **(
                                chunk.user_metadata
                                if isinstance(chunk.user_metadata, dict)
                                else {}
                            ),
                        }.items()
                    },
                )

                return [
                    Derivative(
                        uuid=uuid4(),
                        content_type=ContentType.MESSAGE,
                        content=message_content,
                    ),
                ]
            case ContentType.TEXT:
                text_content = chunk.content
                return [
                    Derivative(
                        uuid=uuid4(),
                        content_type=ContentType.TEXT,
                        content=text_content,
                    )
                ]
            case _:
                return []

    async def search(
        self,
        query: str,
        num_chunks_limit: int = 20,
        property_filter: Mapping[str, FilterablePropertyValue] | None = None,
    ) -> list[Chunk]:
        """
        Search declarative memory for chunks relevant to the query.

        Args:
            query (str):
                The search query.
            num_chunks_limit (int):
                The maximum number of chunks to return
                (default: 20).
            property_filter (Mapping[str, FilterablePropertyValue] | None):
                Filterable property keys and values
                to use for filtering episodes
                (default: None).

        Returns:
            list[Chunk]:
                A list of chunks relevant to the query, sorted by time.
        """

        query_embedding = await self._embedder.search_embed(
            [query],
        )

        # Search graph store for vector matches.
        matched_derivative_nodes = await self._vector_graph_store.search_similar_nodes(
            collection=self._derivative_collection,
            embedding_property_name=(
                DeclarativeMemory._embedding_property_name(
                    self._embedder.model_id,
                    self._embedder.dimensions,
                )
            ),
            query_embedding=query_embedding,
            similarity_metric=self._embedder.similarity_metric,
            limit=100,
            required_properties={
                mangle_filterable_property_key(key): value
                for key, value in property_filter.items()
            },
            include_missing_properties=True,
        )

        # Get source chunks of matched derivatives.
        search_derivatives_source_chunk_tasks = [
            self._vector_graph_store.search_related_nodes(
                node_uuid=matched_derivative_node.uuid,
                allowed_relations={"DERIVED_FROM"},
                find_sources=False,
                find_targets=True,
                required_labels={"EpisodeCluster"},
                required_properties={
                    mangle_filterable_property_key(key): value
                    for key, value in property_filter.items()
                },
                include_missing_properties=True,
            )
            for matched_derivative_node in matched_derivative_nodes
        ]

        derivatives_source_episode_cluster_nodes = await asyncio.gather(
            *search_derivatives_source_episode_cluster_nodes_tasks
        )

        # Flatten into a single list of episode cluster nodes.
        matched_episode_cluster_nodes = [
            episode_cluster_node
            for derivative_source_episode_cluster_nodes in (
                derivatives_source_episode_cluster_nodes
            )
            for episode_cluster_node in derivative_source_episode_cluster_nodes
        ]

        # Get source episodes of matched episode clusters.
        search_episode_clusters_source_episode_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                node_uuid=matched_episode_cluster_node.uuid,
                allowed_relations={"CONTAINS"},
                find_sources=False,
                find_targets=True,
                required_labels={"Episode"},
                required_properties={
                    mangle_filterable_property_key(key): value
                    for key, value in property_filter.items()
                },
            )
            for matched_episode_cluster_node in matched_episode_cluster_nodes
        ]

        episode_clusters_source_episode_nodes = await asyncio.gather(
            *search_episode_clusters_source_episode_nodes_tasks
        )

        # Flatten into a single list of episode nodes.
        # Use source episode nodes as nuclei for context expansion.
        nuclear_episode_nodes = [
            source_episode_node
            for episode_cluster_source_episode_nodes in (
                episode_clusters_source_episode_nodes
            )
            for source_episode_node in episode_cluster_source_episode_nodes
        ]

        # Get contexts for nuclear episode nodes.
        expand_episode_node_contexts_tasks = [
            self._expand_episode_node_context(
                nuclear_episode_node,
                property_filter=property_filter,
                retrieval_depth_limit=4,
            )
            for nuclear_episode_node in nuclear_episode_nodes
        ]

        episode_node_contexts = await asyncio.gather(
            *expand_episode_node_contexts_tasks
        )

        # Rerank contexts.
        episode_node_context_scores = await self._score_episode_node_contexts(
            query, episode_node_contexts
        )

        reranked_anchored_episode_node_contexts = [
            (nuclear_episode_node, episode_node_context)
            for _, nuclear_episode_node, episode_node_context in sorted(
                zip(
                    episode_node_context_scores,
                    nuclear_episode_nodes,
                    episode_node_contexts,
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]

        # Unify contexts.
        unified_episode_node_context = (
            DeclarativeMemory._unify_anchored_episode_node_contexts(
                reranked_anchored_episode_node_contexts,
                num_episodes_limit=num_episodes_limit,
            )
        )

        # Return episodes sorted by timestamp.
        episodes = DeclarativeMemory._episodes_from_episode_nodes(
            list(unified_episode_node_context)
        )

        return sorted(
            episodes,
            key=lambda episode: episode.timestamp,
        )

    async def _expand_episode_node_context(
        self,
        nucleus_episode_node: Node,
        retrieval_depth_limit: int = 1,
        property_filter: dict[str, FilterablePropertyValue] = {},
    ) -> set[Node]:
        """
        Expand the context of a nucleus episode node
        by retrieving related episode nodes
        up to a specified depth limit.
        """
        retrieved_context = {nucleus_episode_node}
        frontier = [nucleus_episode_node]

        for _ in range(1, retrieval_depth_limit + 1):
            get_new_frontier_tasks = [
                self._vector_graph_store.search_related_nodes(
                    node_uuid=frontier_node.uuid,
                    find_sources=True,
                    find_targets=True,
                    limit=10,
                    required_labels={"Episode"},
                    required_properties={
                        mangle_filterable_property_key(key): value
                        for key, value in property_filter.items()
                    },
                )
                for frontier_node in frontier
            ]

            node_neighborhoods = await asyncio.gather(*get_new_frontier_tasks)
            frontier = [
                neighbor_node
                for node_neighborhood in node_neighborhoods
                for neighbor_node in node_neighborhood
                if neighbor_node not in retrieved_context
            ]

            if not frontier:
                break

            retrieved_context.update(frontier)

        return retrieved_context

    async def _score_episode_node_contexts(
        self, query: str, episode_node_contexts: list[set[Node]]
    ) -> list[float]:
        """
        Score episode node contexts
        based on their relevance to the query.
        """
        contexts_episodes = [
            DeclarativeMemory._episodes_from_episode_nodes(list(episode_node_context))
            for episode_node_context in episode_node_contexts
        ]

        def get_formatted_episode_content(episode: Episode) -> str:
            # Format episode content for reranker using metadata.
            return self._episode_metadata_template.safe_substitute(
                {
                    "episode_type": episode.episode_type,
                    "content_type": episode.content_type.value,
                    "content": episode.content,
                    "timestamp": episode.timestamp,
                    "filterable_properties": (episode.filterable_properties),
                    "user_metadata": episode.user_metadata,
                },
                **{
                    key: value
                    for key, value in {
                        **episode.filterable_properties,
                        **(
                            episode.user_metadata
                            if isinstance(episode.user_metadata, dict)
                            else {}
                        ),
                    }.items()
                },
            )

        contexts_content = [
            "\n".join(
                [
                    get_formatted_episode_content(context_episode)
                    for context_episode in sorted(
                        context_episodes,
                        key=lambda episode: episode.timestamp,
                    )
                    if context_episode.content_type == ContentType.STRING
                ]
            )
            for context_episodes in contexts_episodes
        ]

        episode_node_context_scores = await self._reranker.score(
            query, contexts_content
        )

        return episode_node_context_scores

    @staticmethod
    def _unify_anchored_episode_node_contexts(
        anchored_episode_node_contexts: list[tuple[Node, set[Node]]],
        num_episodes_limit: int,
    ) -> set[Node]:
        """
        Unify episode node contexts
        anchored on their nuclear episode nodes
        into a single set of episode nodes,
        respecting the episode limit.
        """
        unified_episode_node_context: set[Node] = set()

        for nucleus, context in anchored_episode_node_contexts:
            if (len(unified_episode_node_context) + len(context)) <= num_episodes_limit:
                # It is impossible that the context exceeds the limit.
                unified_episode_node_context.update(context)
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize episodes near the nucleus.

                # Sort context episodes by timestamp.
                chronological_context = sorted(
                    context,
                    key=lambda node: cast(
                        datetime,
                        node.properties.get("timestamp", datetime.min),
                    ),
                )

                # Sort chronological episodes by index-proximity to nucleus.
                nucleus_index = chronological_context.index(nucleus)
                nuclear_context = sorted(
                    chronological_context,
                    key=lambda node: abs(
                        chronological_context.index(node) - nucleus_index
                    ),
                )

                # Add episodes to unified context until limit is reached,
                # or until the context is exhausted.
                for episode_node in nuclear_context:
                    if len(unified_episode_node_context) >= num_episodes_limit:
                        return unified_episode_node_context
                    unified_episode_node_context.add(episode_node)

        return unified_episode_node_context

    async def forget_all(self):
        """
        Forget all episodes and data derived from them.
        """
        await self._vector_graph_store.clear_data()

    async def forget_filtered_episodes(
        self,
        property_filter: dict[str, FilterablePropertyValue] = {},
    ):
        """
        Forget all episodes matching the given filterable properties
        and data derived from them.
        """
        matching_episode_nodes = await self._vector_graph_store.search_matching_nodes(
            required_labels={"Episode"},
            required_properties={
                mangle_filterable_property_key(key): value
                for key, value in property_filter.items()
            },
        )

        search_related_episode_cluster_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                node_uuid=episode_node.uuid,
                allowed_relations={"CONTAINS"},
                required_labels={"EpisodeCluster"},
                find_sources=True,
                find_targets=False,
            )
            for episode_node in matching_episode_nodes
        ]

        episode_nodes_related_episode_cluster_nodes = await asyncio.gather(
            *search_related_episode_cluster_nodes_tasks
        )

        # Flatten into a single list of episode cluster nodes.
        matching_episode_cluster_nodes = [
            episode_cluster_node
            for episode_node_related_episode_cluster_nodes in (
                episode_nodes_related_episode_cluster_nodes
            )
            for episode_cluster_node in (episode_node_related_episode_cluster_nodes)
        ]

        search_related_derivative_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                node_uuid=episode_cluster_node.uuid,
                allowed_relations={"DERIVED_FROM"},
                required_labels={"Derivative"},
                find_sources=True,
                find_targets=False,
            )
            for episode_cluster_node in matching_episode_cluster_nodes
        ]

        episode_cluster_nodes_related_derivative_nodes = await asyncio.gather(
            *search_related_derivative_nodes_tasks
        )

        # Flatten into a single list of derivative nodes.
        matching_derivative_nodes = [
            derivative_node
            for episode_cluster_node_related_derivative_nodes in (
                episode_cluster_nodes_related_derivative_nodes
            )
            for derivative_node in (episode_cluster_node_related_derivative_nodes)
        ]

        episode_uuids = [node.uuid for node in matching_episode_nodes]
        episode_cluster_uuids = [node.uuid for node in matching_episode_cluster_nodes]
        derivative_uuids = [node.uuid for node in matching_derivative_nodes]

        node_uuids_to_delete = episode_uuids + episode_cluster_uuids + derivative_uuids
        await self._vector_graph_store.delete_nodes(node_uuids_to_delete)

    @staticmethod
    def _episodes_from_episode_nodes(
        episode_nodes: list[Node],
    ) -> list[Episode]:
        """
        Convert a list of episode Nodes to a list of Episodes.

        Args:
            episode_nodes (list[Node]):
                A list of Nodes representing episodes.

        Returns:
            list[Episode]:
                A list of Episodes constructed from the episode Nodes.
        """
        return [
            Episode(
                uuid=node.uuid,
                episode_type=cast(str, node.properties["episode_type"]),
                content_type=ContentType(node.properties["content_type"]),
                content=node.properties["content"],
                timestamp=cast(
                    datetime,
                    node.properties.get("timestamp", datetime.min),
                ),
                filterable_properties={
                    demangle_filterable_property_key(key): cast(
                        FilterablePropertyValue, value
                    )
                    for key, value in node.properties.items()
                    if is_mangled_filterable_property_key(key)
                },
                user_metadata=json.loads(cast(str, node.properties["user_metadata"])),
            )
            for node in episode_nodes
        ]

    @staticmethod
    def _embedding_property_name(model_id: str, dimensions: int) -> str:
        """
        Generate a standardized property name for embeddings
        based on the model ID and embedding dimensions.

        Args:
            model_id (str): The identifier of the embedding model.
            dimensions (int): The dimensionality of the embedding.

        Returns:
            str: A standardized property name for the embedding.
        """
        return f"embedding_{model_id}_{dimensions}d"
