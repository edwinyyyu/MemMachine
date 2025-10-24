"""
A related episode postulator implementation
that postulates related similar episodes.

This is suitable for use cases
where recent episodes are likely to be relevant to the current episode.
"""

import asyncio
import json
from datetime import datetime
from typing import cast

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder import Embedder
from memmachine.common.vector_graph_store import VectorGraphStore
from ..declarative_memory import DeclarativeMemory

from ..data_types import (
    ContentType,
    Derivative,
    Episode,
    FilterablePropertyValue,
    demangle_filterable_property_key,
    is_mangled_filterable_property_key,
    mangle_filterable_property_key,
)
from .related_episode_postulator import RelatedEpisodePostulator


class SimilarRelatedEpisodePostulatorParams(BaseModel):
    """
    Parameters for SimilarRelatedEpisodePostulator.

    Attributes:
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance to use for searching episodes.
        embedder (Embedder):
            Embedder instance for derivative embeddings.
        search_limit (int):
            The maximum number of related similar episodes
            to postulate per derivative (default: 2).
        filterable_property_keys (set[str]):
            A set of property keys
            to use for filtering episodes (default: set()).
    """

    vector_graph_store: InstanceOf[VectorGraphStore] = Field(
        ..., description="VectorGraphStore instance to use for searching episodes"
    )
    embedder: InstanceOf[Embedder] = Field(
        ..., description="Embedder instance for derivative embeddings"
    )
    search_limit: int = Field(
        2,
        description="The maximum number of related similar episodes to postulate per derivative",
        gt=0,
    )
    filterable_property_keys: set[str] = Field(
        default_factory=set,
        description="A set of property keys to use for filtering episodes",
    )


class SimilarRelatedEpisodePostulator(RelatedEpisodePostulator):
    """
    RelatedEpisodePostulator implementation
    that postulates related similar episodes.
    """

    def __init__(self, params: SimilarRelatedEpisodePostulatorParams):
        """
        Initialize a SimilarRelatedEpisodePostulator
        with the provided parameters.

        Args:
            params (SimilarRelatedEpisodePostulatorParams):
                Parameters for the SimilarRelatedEpisodePostulator.
        """
        self._vector_graph_store = params.vector_graph_store
        self._embedder = params.embedder
        self._search_limit = params.search_limit
        self._filterable_property_keys = params.filterable_property_keys

    async def postulate(self, episode: Episode) -> list[Episode]:
        # Get episode clusters containing the episode.
        episode_cluster_nodes = await self._vector_graph_store.search_related_nodes(
            node_uuid=episode.uuid,
            allowed_relations={"CONTAINS"},
            find_sources=True,
            find_targets=False,
            required_labels={"EpisodeCluster"},
        )

        # Get derivatives derived from the episode clusters.
        search_derivative_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                node_uuid=episode_cluster_node.uuid,
                allowed_relations={"DERIVED_FROM"},
                find_sources=True,
                find_targets=False,
                required_labels={"Derivative"},
                required_properties={
                    "derivative_type": "entity"
                },
            )
            for episode_cluster_node in episode_cluster_nodes
        ]

        related_derivative_nodes = [
            derivative_node
            for derivative_nodes in await asyncio.gather(*search_derivative_nodes_tasks)
            for derivative_node in derivative_nodes
        ]

        related_derivatives = [
            Derivative(
                uuid=derivative_node.uuid,
                derivative_type=cast(
                    str,
                    derivative_node.properties["derivative_type"],
                ),
                content_type=ContentType(
                    derivative_node.properties["content_type"]
                ),
                content=derivative_node.properties["content"],
                filterable_properties={
                    demangle_filterable_property_key(key): cast(
                        FilterablePropertyValue, value
                    )
                    for key, value in derivative_node.properties.items()
                    if is_mangled_filterable_property_key(key)
                },
                user_metadata=json.loads(
                    cast(
                        str,
                        derivative_node.properties["user_metadata"],
                    )
                ),
            )
            for derivative_node in related_derivative_nodes
        ]

        # Search graph store for vector matches.
        # TODO @edwinyyyu: Figure out better way to share embedding property name.
        embedding_property_name = DeclarativeMemory._embedding_property_name(
            self._embedder.model_id, self._embedder.dimensions
        )

        search_similar_nodes_tasks = [
            self._vector_graph_store.search_similar_nodes(
                query_embedding=derivative_node.properties.get(embedding_property_name),
                embedding_property_name=embedding_property_name,
                similarity_metric=self._embedder.similarity_metric,
                limit=2 * self._search_limit,
                required_labels={"Derivative"},
                required_properties={
                    mangle_filterable_property_key(key): derivative.filterable_properties[
                        key
                    ]
                    for key in self._filterable_property_keys
                    if key in derivative.filterable_properties
                } | {
                    "derivative_type": "entity"
                },
                include_missing_properties=True,
            )
            for derivative_node, derivative in zip(related_derivative_nodes, related_derivatives)
        ]

        similar_derivative_nodes = [
            similar_node
            for similar_nodes in await asyncio.gather(*search_similar_nodes_tasks)
            for similar_node in similar_nodes
        ]

        # Get source episode clusters of matched derivatives.
        search_derivatives_source_episode_cluster_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                node_uuid=similar_derivative_node.uuid,
                allowed_relations={"DERIVED_FROM"},
                find_sources=False,
                find_targets=True,
                required_labels={"EpisodeCluster"},
            )
            for similar_derivative_node in similar_derivative_nodes
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
            )
            for matched_episode_cluster_node in matched_episode_cluster_nodes
        ]

        episode_clusters_source_episode_nodes = await asyncio.gather(
            *search_episode_clusters_source_episode_nodes_tasks
        )

        # Flatten into a single list of episode nodes.
        # Use source episode nodes as nuclei for context expansion.
        similar_episode_nodes = set(
            source_episode_node
            for episode_cluster_source_episode_nodes in (
                episode_clusters_source_episode_nodes
            )
            for source_episode_node in episode_cluster_source_episode_nodes
        )

        similar_episodes = [
            Episode(
                uuid=similar_episode_node.uuid,
                episode_type=cast(
                    str,
                    similar_episode_node.properties["episode_type"],
                ),
                content_type=ContentType(
                    similar_episode_node.properties["content_type"]
                ),
                content=similar_episode_node.properties["content"],
                timestamp=cast(
                    datetime,
                    similar_episode_node.properties.get("timestamp", datetime.min),
                ),
                filterable_properties={
                    demangle_filterable_property_key(key): cast(
                        FilterablePropertyValue, value
                    )
                    for key, value in similar_episode_node.properties.items()
                    if is_mangled_filterable_property_key(key)
                },
                user_metadata=json.loads(
                    cast(
                        str,
                        similar_episode_node.properties["user_metadata"],
                    )
                ),
            )
            for similar_episode_node in similar_episode_nodes if similar_episode_node.uuid != episode.uuid
        ]

        return similar_episodes
