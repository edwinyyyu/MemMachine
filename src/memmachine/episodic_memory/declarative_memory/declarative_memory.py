"""
Declarative memory system for storing and retrieving
episodic and semantic memory.
"""

import asyncio
import functools
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Awaitable
from datetime import datetime
from string import Template
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, Field

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.embedder.embedder import Embedder
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore

from .data_types import (
    ContentType,
    Derivative,
    Episode,
    EpisodeCluster,
    FilterablePropertyValue,
    demangle_filterable_property_key,
    is_mangled_filterable_property_key,
    mangle_filterable_property_key,
)
from .derivative_deriver import DerivativeDeriver
from .derivative_mutator import DerivativeMutator
from .related_episode_postulator import RelatedEpisodePostulator

logger = logging.getLogger(__name__)


class DeclarativeMemoryConfig(BaseModel):
    """
    Configuration for the declarative memory system.

    Attributes:
        episode_metadata_template (str, optional):
            Template string supporting $-substitutions
            (default: "[$timestamp] $content").
    """

    episode_metadata_template: str = Field(
        "[$timestamp] $content",
        description=(
            "Template string supporting $-substitutions "
            "to format episodes with episode metadata for the reranker "
            "(default: '[$timestamp] $content')."
        ),
    )


class IngestionWorkflow:
    def __init__(
        self,
        cluster_related_episode_postulator: RelatedEpisodePostulator,
        derivative_deriver: DerivativeDeriver,
        derivative_mutator: DerivativeMutator,
        embedder: Embedder,
    ):
        if not isinstance(cluster_related_episode_postulator, RelatedEpisodePostulator):
            raise TypeError(
                "cluster_related_episode_postulator must be a RelatedEpisodePostulator"
            )
        if not isinstance(derivative_deriver, DerivativeDeriver):
            raise TypeError("derivative_deriver must be a DerivativeDeriver")
        if not isinstance(derivative_mutator, DerivativeMutator):
            raise TypeError("derivative_mutator must be a DerivativeMutator")
        if not isinstance(embedder, Embedder):
            raise TypeError("embedder must be an Embedder")

        self.cluster_related_episode_postulator = cluster_related_episode_postulator
        self.derivative_deriver = derivative_deriver
        self.derivative_mutator = derivative_mutator


class QueryWorkflow:
    def __init__(
        self,
        derivative_deriver: DerivativeDeriver,
        derivative_mutator: DerivativeMutator,
        embedder: Embedder,
    ):
        if not isinstance(derivative_deriver, DerivativeDeriver):
            raise TypeError("derivative_deriver must be a DerivativeDeriver")
        if not isinstance(derivative_mutator, DerivativeMutator):
            raise TypeError("derivative_mutator must be a DerivativeMutator")
        if not isinstance(embedder, Embedder):
            raise TypeError("embedder must be an Embedder")

        self.derivative_deriver = derivative_deriver
        self.derivative_mutator = derivative_mutator
        self.embedder = embedder


class DeclarativeMemory:
    """
    Memory system for episodic and semantic memory.
    """

    def __init__(
        self,
        config: DeclarativeMemoryConfig,
        reranker: Reranker,
        vector_graph_store: VectorGraphStore,
        ingestion_workflows_map: Mapping[str, Iterable[IngestionWorkflow]],
        query_workflows: Iterable[QueryWorkflow],
        adjacent_related_episode_postulators: Iterable[RelatedEpisodePostulator],
    ):
        """
        Initialize a DeclarativeMemory.

        Args:
            config (DeclarativeMemoryConfig):
                Configuration for the declarative memory system.
            reranker (Reranker):
                Reranker for reranking search results.
            vector_graph_store (VectorGraphStore):
                Vector graph store for storing and retrieving memories.
            ingestion_workflows_map (Mapping[str, Iterable[IngestionWorkflow]]):
                Mapping from episode types to iterables of ingestion workflows.
            query_workflows (Iterable[QueryWorkflow]):
                Iterable of query workflows.
            adjacentn_related_episode_postulators (Iterable[RelatedEpisodePostulator]):
                Iterable of related episode postulators
                for postulating adjacent episodes.
        """

        self._episode_metadata_template = Template(config.episode_metadata_template)

        self._reranker: Reranker = reranker
        self._vector_graph_store: VectorGraphStore = vector_graph_store
        self._ingestion_workflows_map = {
            episode_type: self._build_ingestion_workflows(ingestion_workflows)
            for episode_type, ingestion_workflows in ingestion_workflows_map.items()
        }
        self._query_workflows = self._build_query_workflow(query_workflows)
        self._adjacent_related_episode_postulators: list[RelatedEpisodePostulator] = (
            adjacent_related_episode_postulators
        )

    @staticmethod
    async def _build_workflow(
        executable: Callable,
        subworkflows: Iterable[Callable],
        callback: Callable | None = None,
    ) -> Callable:
        """
        Execute the workflow with the provided arguments.

        Args:
            executable (Callable):
                The main executable function of the workflow.
            subworkflows (Iterable[Callable]):
                The subworkflows to execute on the result of the main executable.

        Returns:
            Callable:
                The assembled workflow function.
        """

        async def execute_workflow(*args, **kwargs):
            execution_result = await executable(*args, **kwargs)
            subworkflow_results = await asyncio.gather(
                *[subworkflow(execution_result) for subworkflow in subworkflows]
            )

            if callback is None:
                return execution_result, subworkflow_results

            return await callback(execution_result, subworkflow_results)

        return execute_workflow

    def _build_ingestion_workflows(
        self,
        ingestion_workflows: Iterable[IngestionWorkflow],
    ):
        workflow_trees = {}
        for ingestion_workflow in ingestion_workflows:
            workflow_trees.setdefault(
                id(ingestion_workflow.cluster_related_episode_postulator),
                (ingestion_workflow.cluster_related_episode_postulator, {}),
            )[1].setdefault(
                id(ingestion_workflow.derivative_deriver),
                (ingestion_workflow.derivative_deriver, {}),
            )[1].setdefault(
                id(ingestion_workflow.derivative_mutator),
                (ingestion_workflow.derivative_mutator, {}),
            )[1].setdefault(
                id(ingestion_workflow.embedder),
                ingestion_workflow.embedder
            )

        return [
            DeclarativeMemory._build_workflow(
                executable=functools.partial(
                    DeclarativeMemory._assemble_episode_cluster,
                    cluster_related_episode_postulator,
                ),
                subworkflows=[
                    DeclarativeMemory._build_workflow(
                        executable=functools.partial(
                            DeclarativeMemory._derive_derivatives,
                            derivative_deriver,
                        ),
                        subworkflows=[
                            DeclarativeMemory._build_workflow(
                                executable=functools.partial(
                                    DeclarativeMemory._mutate_derivatives,
                                    derivative_mutator,
                                ),
                                subworkflows=[
                                    DeclarativeMemory._build_workflow(
                                        executable=functools.partial(
                                            DeclarativeMemory._embed_derivatives,
                                            functools.partial(
                                                embedder.ingest_embed,
                                                max_attempts=3,
                                            )
                                        ),
                                        subworkflows=[],
                                        callback=self._process_ingestion_embedding,
                                    )
                                    for embedder in derivative_mutator_workflow_tree
                                ],
                                callback=self._process_ingestion_derivative_mutation,
                            )
                            for (
                                derivative_mutator,
                                derivative_mutator_workflow_tree,
                            ) in derivative_deriver_workflow_tree
                        ],
                        callback=self._process_ingestion_derivative_derivation,
                    )
                    for (
                        derivative_deriver,
                        derivative_deriver_workflow_tree,
                    ) in cluster_related_episode_postulator_workflow_tree.values()
                ],
                callback=self._process_ingestion_episode_cluster_assembly,
            )
            for (
                cluster_related_episode_postulator,
                cluster_related_episode_postulator_workflow_tree,
            ) in workflow_trees.values()
        ]

    def _build_query_workflow(self, query_workflows: Iterable[QueryWorkflow]):
        workflow_trees = {}
        for query_workflow in query_workflows:
            workflow_trees.setdefault(
                id(query_workflow.derivative_deriver),
                (query_workflow.derivative_deriver, {}),
            )[1].setdefault(
                id(query_workflow.derivative_mutator),
                query_workflow.derivative_mutator,
            )

        return [
            DeclarativeMemory._build_workflow(
                executable=functools.partial(
                    DeclarativeMemory._derive_derivatives,
                    derivative_deriver,
                ),
                subworkflows=[
                    DeclarativeMemory._build_workflow(
                        executable=functools.partial(
                            DeclarativeMemory._mutate_derivatives,
                            derivative_mutator,
                        ),
                        subworkflows=[
                            DeclarativeMemory._build_workflow(
                                executable=functools.partial(
                                    DeclarativeMemory._embed_derivatives,
                                    functools.partial(
                                        embedder.query_embed,
                                        max_attempts=3,
                                    )
                                ),
                                subworkflows=[],
                                callback=self._process_query_embedding,
                            )
                            for embedder in derivative_mutator_workflow_tree
                        ],
                        callback=self._process_query_derivative_mutation,
                    )
                    for (
                        derivative_mutator,
                        derivative_mutator_workflow_tree,
                    ) in derivative_deriver_workflow_tree
                ],
                callback=self._process_query_derivative_derivation,
            )
            for (
                derivative_deriver,
                derivative_deriver_workflow_tree,
            ) in workflow_trees.values()
        ]

    @staticmethod
    async def _assemble_episode_cluster(
        related_episode_postulator: RelatedEpisodePostulator,
        episode: Episode,
    ) -> EpisodeCluster:
        """
        Assemble an episode cluster given an episode.
        """
        related_episodes = await related_episode_postulator.postulate(episode)
        cluster_episodes = sorted(
            [episode] + related_episodes,
            key=lambda episode: episode.timestamp,
        )
        episode_cluster = EpisodeCluster(
            uuid=uuid4(),
            episodes=cluster_episodes,
            timestamp=cluster_episodes[-1].timestamp,
            filterable_properties=dict(
                set.intersection(
                    *(
                        set(cluster_episode.filterable_properties.items())
                        for cluster_episode in cluster_episodes
                    )
                )
            ),
            user_metadata=episode.user_metadata,
        )

        return episode_cluster

    @staticmethod
    async def _derive_derivatives(
        derivative_deriver: DerivativeDeriver,
        episode_cluster: EpisodeCluster,
    ) -> tuple[list[Derivative], EpisodeCluster]:
        """
        Derive derivatives from an episode cluster.
        """
        derivatives = await derivative_deriver.derive(episode_cluster)
        return derivatives, episode_cluster

    @staticmethod
    async def _mutate_derivatives(
        derivative_mutator: DerivativeMutator,
        derivative_derivation_result: tuple[list[Derivative], EpisodeCluster],
    ) -> list[Derivative]:
        """
        Mutate derived derivatives.
        """
        derivatives, episode_cluster = derivative_derivation_result

        mutate_derivative_tasks = [
            derivative_mutator.mutate(derivative, episode_cluster)
            for derivative in derivatives
        ]
        derivatives_mutated_derivatives = await asyncio.gather(*mutate_derivative_tasks)

        # Flatten into a single list of mutated derivatives.
        mutated_derivatives = [
            derivative
            for derivative_mutated_derivatives in (derivatives_mutated_derivatives)
            for derivative in derivative_mutated_derivatives
        ]
        return mutated_derivatives

    @staticmethod
    async def _create_embedded_derivative_nodes(
        async_embed_func: Callable[[list[str]], Awaitable[list[list[float]]]],
        mutated_derivatives: list[Derivative]
    ) -> tuple[list[Derivative], list[list[float]]]:
        """
        Embed mutated derivatives.
        """
        try:
            derivative_embeddings = await async_embed_func(
                [derivative.content for derivative in mutated_derivatives]
            )
        except (ExternalServiceAPIError, ValueError, RuntimeError):
            logger.error("Failed to create embeddings for mutated derivatives")
            return [], []

        return mutated_derivatives, derivative_embeddings

    async def _process_ingestion_embedding(
        self,
        derivative_embedding_result: tuple[list[Derivative], list[list[float]]],
        none: list[None],
    ) -> list[Node]:
        """
        Process the result of derivative embedding
        by creating nodes for the embedded derivatives.
        """
        embedded_derivatives, derivative_embeddings = derivative_embedding_result

        embedded_derivative_nodes = [
            Node(
                uuid=derivative.uuid,
                labels={"Derivative"},
                properties={
                    "content": derivative.content,
                    DeclarativeMemory._embedding_property_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    ): derivative_embedding,
                    "timestamp": derivative.timestamp,
                    "user_metadata": json.dumps(derivative.user_metadata),
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in derivative.filterable_properties.items()
                },
            )
            for derivative, derivative_embedding in zip(
                embedded_derivatives, derivative_embeddings
            )
        ]
        return embedded_derivative_nodes

    async def _process_ingestion_derivative_mutation(
        self,
        mutated_derivatives: list[Derivative],
        embedded_derivative_nodes: list[list[Node]],
    ) -> list[Node]:
        """
        Process the result of derivative mutation
        by embedding and creating nodes for the mutated derivatives.
        """
        mutated_derivative_nodes = [
            Node(
                uuid=derivative.uuid,
                labels={"Derivative"},
                properties={
                    "content": derivative.content,
                    DeclarativeMemory._embedding_property_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    ): derivative_embedding,
                    "timestamp": derivative.timestamp,
                    "user_metadata": json.dumps(derivative.user_metadata),
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in derivative.filterable_properties.items()
                },
            )
            for derivative, derivative_embedding in zip(
                mutated_derivatives, mutated_derivative_embeddings
            )
        ]
        return mutated_derivative_nodes

    async def _process_ingestion_derivative_derivation(
        self,
        derived_derivatives: list[Derivative],
        mutation_workflows_derivative_nodes: list[list[Node]],
    ) -> list[Node]:
        """
        Process the result of derivative derivation
        by flattening the list of mutated derivative nodes.
        Do nothing with the unprocessed derived derivatives.
        """
        derivative_nodes = [
            derivative_node
            for mutation_workflow_derivative_nodes in (
                mutation_workflows_derivative_nodes
            )
            for derivative_node in mutation_workflow_derivative_nodes
        ]
        return derivative_nodes

    async def _process_ingestion_episode_cluster_assembly(
        self,
        episode_cluster: EpisodeCluster,
        derivation_workflows_derivative_nodes: list[list[Node]],
    ) -> tuple[list[Node], list[Edge]]:
        """
        Process the result of episode cluster assembly
        by creating nodes and edges
        for the episode cluster and its derivatives.
        """

        # Create episode cluster nodes.
        episode_cluster_node = Node(
            uuid=episode_cluster.uuid,
            labels={"EpisodeCluster"},
            properties=dict(
                {
                    "timestamp": episode_cluster.timestamp,
                    "user_metadata": json.dumps(episode_cluster.user_metadata),
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in (episode_cluster.filterable_properties.items())
                }
            ),
        )

        # Create edges from episode cluster nodes
        # to source episode nodes.
        episode_cluster_source_episodes_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=episode_cluster.uuid,
                target_uuid=episode.uuid,
                relation="CONTAINS",
            )
            for episode in episode_cluster.episodes
        ]

        # Flatten into a single list of derivative nodes.
        derivative_nodes = [
            derivative_node
            for derivation_workflow_derivative_nodes in (
                derivation_workflows_derivative_nodes
            )
            for derivative_node in derivation_workflow_derivative_nodes
        ]

        # Create edges from derivative nodes to episode cluster node.
        derivatives_source_episode_cluster_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=derivative_node.uuid,
                target_uuid=episode_cluster_node.uuid,
                relation="DERIVED_FROM",
            )
            for derivative_node in derivative_nodes
        ]

        nodes = [episode_cluster_node] + derivative_nodes
        edges = (
            episode_cluster_source_episodes_edges
            + derivatives_source_episode_cluster_edges
        )
        return nodes, edges

    async def add_episode(
        self,
        episode: Episode,
    ):
        """
        Add an episode to declarative memory.

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

        episode_type_ingestion_workflows = self._ingestion_workflows_map.get(
            episode.episode_type
        )
        if episode_type_ingestion_workflows is None:
            episode_type_ingestion_workflows = self._ingestion_workflows_map.get(
                "_default", []
            )

        # Create nodes and edges for episode clusters and derivatives.
        ingestion_workflow_tasks = [
            execute_ingestion_workflow(episode)
            for execute_ingestion_workflow in episode_type_ingestion_workflows
        ]

        ingestion_workflows_nodes, ingestion_workflows_edges = zip(
            *(await asyncio.gather(*ingestion_workflow_tasks))
        )

        derivation_nodes = [
            node
            for workflow_nodes in ingestion_workflows_nodes
            for node in workflow_nodes
        ]
        derivation_edges = [
            edge
            for workflow_edges in ingestion_workflows_edges
            for edge in workflow_edges
        ]

        adjacent_episodes = [
            postulated_related_episode
            for postulated_related_episodes in await asyncio.gather(
                *[
                    related_episode_postulator.postulate(episode)
                    for related_episode_postulator in (
                        self._adjacent_related_episode_postulators
                    )
                ]
            )
            for postulated_related_episode in postulated_related_episodes
        ]

        # Create edges between episode and postulated connected episodes.
        adjacent_episode_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=episode.uuid,
                target_uuid=adjacent_episode.uuid,
                relation="RELATED_TO",
            )
            for adjacent_episode in adjacent_episodes
        ]

        await self._vector_graph_store.add_nodes(derivation_nodes)
        await self._vector_graph_store.add_edges(
            derivation_edges + adjacent_episode_edges
        )

    async def _process_query_derivative_mutation(
        self,
        mutated_derivatives: list[Derivative],
        none: list[None],
    ) -> list[list[float]]:
        """
        Process the result of derivative mutation
        by embedding the mutated derivatives.
        """
        try:
            mutated_derivative_embeddings = await self._embedder.search_embed(
                [derivative.content for derivative in mutated_derivatives],
                max_attempts=3,
            )
        except (ExternalServiceAPIError, ValueError, RuntimeError):
            logger.error("Failed to create embeddings for mutated derivatives")
            return []

        return mutated_derivative_embeddings

    async def _process_query_derivative_derivation(
        self,
        derived_derivatives: list[Derivative],
        mutation_workflows_derivative_embeddings: list[list[list[float]]],
    ) -> tuple[list[Derivative], list[list[float]]]:
        """
        Process the result of derivative derivation
        by flattening the list of mutated derivative embeddings.
        Do nothing with the unprocessed derived derivatives.
        """
        derivative_embeddings = [
            derivative_embedding
            for mutation_workflow_derivative_embeddings in (
                mutation_workflows_derivative_embeddings
            )
            for derivative_embedding in mutation_workflow_derivative_embeddings
        ]
        return derivative_embeddings

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

        # Derive and embed derivatives from query.
        query_dummy_cluster = EpisodeCluster(
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

        query_workflow_tasks = [
            execute_query_workflow(query_dummy_cluster)
            for execute_query_workflow in self._query_workflows
        ]

        workflows_derivative_embeddings = await asyncio.gather(*query_workflow_tasks)
        derivative_embeddings = [
            derivative_embedding
            for workflow_derivative_embeddings in (workflows_derivative_embeddings)
            for derivative_embedding in workflow_derivative_embeddings
        ]

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
