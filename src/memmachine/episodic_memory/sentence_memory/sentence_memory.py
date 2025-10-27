"""
Declarative memory system for storing and retrieving
episodic and semantic memory.
"""

import asyncio
import functools
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from string import Template
from typing import Any, Self, cast
from uuid import uuid4

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.embedder.embedder import Embedder
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore
from pydantic import BaseModel, Field

from .data_types import (
    ContentType,
    Episode,
    Chunk,
    Sentence,
    FilterablePropertyValue,
    demangle_filterable_property_key,
    is_mangled_filterable_property_key,
    mangle_filterable_property_key,
)

logger = logging.getLogger(__name__)


class SentenceMemoryParams(BaseModel):
    """
    Parameters for SentenceMemory.

    Attributes:
        max_sentence_length (int):
            Maximum length of a sentence in characters.
            Sentences longer than this will be split
            (default: 1000).
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        reranker (Reranker):
            Reranker instance for reranking search results.
        metadata_templates (dict[str, str]):
            Template strings supporting $-substitutions
            for different episode types
            (default: {"default": "[$timestamp] $content"}).
    """
    max_sentence_length: int = Field(
        1000,
        description=(
            "Maximum length of a sentence in characters. "
            "Sentences longer than this will be split"
        ),
    )
    vector_graph_store: VectorGraphStore = Field(
        ...,
        description="VectorGraphStore instance for storing and retrieving memories",
    )
    reranker: Reranker = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    metadata_templates: dict[str, str] = Field(
        default_factory=lambda: {"default": "[$timestamp] $content"},
        description=(
            "Template strings supporting $-substitutions for different episode types"
        ),
    )


class SentenceMemory:
    """
    Sentence memory system.
    """

    def __init__(self, params: SentenceMemoryParams):
        """
        Initialize a SentenceMemory with the provided parameters.

        Args:
            params (SentenceMemoryParams):
                Parameters for the SentenceMemory.
        """

        self._vector_graph_store = params.vector_graph_store
        self._reranker = params.reranker
        self._episode_metadata_templates = {
            episode_type: Template(template_string)
            for episode_type, template_string in params.metadata_templates.items()
        }

    async def add_episode(
        self,
        episode: Episode,
    ):
        """
        Add an episode.

        Args:
            episode (Episode): The episode to add.
        """
        episode_node = Node(
            uuid=episode.uuid,
            labels={"Episode"},
            properties={
                "episode_type": episode.episode_type,
                "content_type": episode.content_type.value,
                "content": episode.content,
                "timestamp": episode.timestamp,
                "user_metadata": json.dumps(episode.user_metadata),
            }
            | {
                mangle_filterable_property_key(key): value
                for key, value in episode.filterable_properties.items()
            },
        )

        await self._vector_graph_store.add_nodes([episode_node])

        episode_type_derivation_workflows = self._derivation_workflows.get(
            episode.episode_type
        ) or self._derivation_workflows.get("default", [])

        # Create nodes and edges for episode clusters and derivatives.
        derivation_workflow_tasks = [
            derivation_workflow.execute(episode)
            for derivation_workflow in episode_type_derivation_workflows
        ]

        derivation_workflows_nodes, derivation_workflows_edges = zip(
            *(await asyncio.gather(*derivation_workflow_tasks))
        )

        derivation_nodes = [
            node
            for workflow_nodes in derivation_workflows_nodes
            for node in workflow_nodes
        ]
        derivation_edges = [
            edge
            for workflow_edges in derivation_workflows_edges
            for edge in workflow_edges
        ]

        related_episodes = [
            postulated_related_episode
            for postulated_related_episodes in await asyncio.gather(
                *[
                    related_episode_postulator.postulate(episode)
                    for related_episode_postulator in (
                        self._related_episode_postulators
                    )
                ]
            )
            for postulated_related_episode in postulated_related_episodes
        ]

        # Create postulated edges between episodes.
        related_episode_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=episode.uuid,
                target_uuid=related_episode.uuid,
                relation="RELATED_TO",
            )
            for related_episode in related_episodes
        ]

        await self._vector_graph_store.add_nodes(derivation_nodes)
        await self._vector_graph_store.add_edges(
            derivation_edges + related_episode_edges
        )

    async def search(
        self,
        query: str,
        num_episodes_limit: int = 20,
        property_filter: dict[str, FilterablePropertyValue] = {},
    ) -> list[Episode]:
        """
        Search declarative memory for episodes relevant to the query.

        Args:
            query (str):
                The search query.
            num_episodes_limit (int, optional):
                The maximum number
                of episodes to return (default: 20).
            property_filter (
                dict[str, FilterablePropertyValue], optional
            ):
                Filterable property keys and values to use
                for filtering episodes.
                If not provided, no filtering is applied.

        Returns:
            list[Episode]:
                A list of episodes relevant to the query,
                sorted by timestamp.
        """

        # Derive derivatives from query.
        derivatives = await self._query_derivative_deriver.derive(
            EpisodeCluster(
                uuid=uuid4(),
                episodes=[
                    Episode(
                        uuid=uuid4(),
                        episode_type="query",
                        content_type=ContentType.STRING,
                        content=query,
                        timestamp=datetime.now(),
                    ),
                ],
            )
        )

        # Embed derivatives.
        try:
            derivative_embeddings = await self._embedder.search_embed(
                [derivative.content for derivative in derivatives],
                max_attempts=3,
            )
        except (ExternalServiceAPIError, ValueError, RuntimeError):
            logger.error("Failed to create embeddings for query derivatives")
            return []

        # Search graph store for vector matches.
        search_similar_nodes_tasks = [
            self._vector_graph_store.search_similar_nodes(
                query_embedding=derivative_embedding,
                embedding_property_name=(
                    DeclarativeMemory._embedding_property_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    )
                ),
                similarity_metric=self._embedder.similarity_metric,
                required_labels={"Derivative"},
                required_properties={
                    mangle_filterable_property_key(key): value
                    for key, value in property_filter.items()
                },
                include_missing_properties=True,
            )
            for derivative_embedding in derivative_embeddings
        ]

        matched_derivative_nodes = [
            similar_node
            for similar_nodes in await asyncio.gather(*search_similar_nodes_tasks)
            for similar_node in similar_nodes
        ]

        # Get source episode clusters of matched derivatives.
        search_derivatives_source_episode_cluster_nodes_tasks = [
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
