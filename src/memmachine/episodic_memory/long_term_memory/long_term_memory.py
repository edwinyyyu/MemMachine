from collections.abc import Mapping

from pydantic import BaseModel, Field, model_validator

from memmachine.common.data_types import ResourceDefinition
from memmachine.common.factory import Factory
from memmachine.common.reranker import Reranker
from memmachine.common.resource_manager import ResourceManager
from memmachine.common.utils import get_nested_values
from memmachine.common.vector_graph_store import VectorGraphStore

from ..data_types import ContentType, Episode
from ..declarative_memory import ContentType as DeclarativeMemoryContentType
from ..declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryConfig,
    IngestionWorkflow,
    QueryWorkflow,
)
from ..declarative_memory import Episode as DeclarativeMemoryEpisode
from ..declarative_memory.derivative_deriver import (
    DerivativeDeriverFactory,
)
from ..declarative_memory.derivative_mutator import (
    DerivativeMutatorFactory,
)
from ..declarative_memory.related_episode_postulator import (
    RelatedEpisodePostulatorFactory,
)

content_type_to_declarative_memory_content_type_map = {
    ContentType.STRING: DeclarativeMemoryContentType.STRING,
}

declarative_memory_content_type_to_content_type_map = {
    DeclarativeMemoryContentType.STRING: ContentType.STRING,
}

workflow_resource_type_factory_map: dict[str, Factory] = {
    "derivative_deriver": DerivativeDeriverFactory,
    "derivative_mutator": DerivativeMutatorFactory,
    "related_episode_postulator": RelatedEpisodePostulatorFactory,
}


class IngestionWorkflowDefinition(BaseModel):
    cluster_related_episode_postulator: str
    derivative_deriver: str
    derivative_mutator: str
    embedder: str


class QueryWorkflowDefinition(BaseModel):
    derivative_deriver: str
    derivative_mutator: str
    embedder: str


class LongTermMemoryConfig(BaseModel):
    metadata_prefix: str = Field(
        "[$timestamp] $producer_id: ",
        description=(
            "Template prefix supporting $-substitutions "
            "to format episodes with episode metadata for the reranker "
            "(default: '[$timestamp] $producer_id: ')."
        ),
    )
    # TODO: Convert from config file format (type first) to this format (homogeneous) in previous layer.
    workflow_resources: dict[str, ResourceDefinition]
    ingestion_workflows_map: dict[str, list[IngestionWorkflowDefinition]]
    query_workflows: list[QueryWorkflowDefinition]
    adjacent_related_episode_postulators: list[str]

    @model_validator(mode="after")
    def check_workflow_resources_used(
        self,
    ):
        """
        Check that all resource ids in workflow resource definitions re used
        and that all workflow resource ids are defined in workflow resource definitions.
        """
        workflow_resource_ids = set(self.workflow_resources.keys())
        used_resource_ids = set()

        # Check ingestion workflows.
        for ingestion_workflows in self.ingestion_workflows_map.values():
            for ingestion_workflow in ingestion_workflows:
                related_episode_postulator_id = (
                    ingestion_workflow.cluster_related_episode_postulator
                )
                derivative_deriver_id = ingestion_workflow.derivative_deriver
                derivative_mutator_id = ingestion_workflow.derivative_mutator

                if related_episode_postulator_id not in workflow_resource_ids:
                    raise ValueError(
                        f"Related episode postulator id '{related_episode_postulator_id}' "
                        "in ingestion workflow not found in resource definitions"
                    )
                if derivative_deriver_id not in workflow_resource_ids:
                    raise ValueError(
                        f"Derivative deriver id '{derivative_deriver_id}' "
                        "in ingestion workflow not found in resource definitions"
                    )
                if derivative_mutator_id not in workflow_resource_ids:
                    raise ValueError(
                        f"Derivative mutator id '{derivative_mutator_id}' "
                        "in ingestion workflow not found in resource definitions"
                    )

                used_resource_ids.add(
                    ingestion_workflow.cluster_related_episode_postulator
                )
                used_resource_ids.add(ingestion_workflow.derivative_deriver)
                used_resource_ids.add(ingestion_workflow.derivative_mutator)

        # Check query workflows.
        for query_workflow in self.query_workflows:
            derivative_deriver_id = query_workflow.derivative_deriver
            derivative_mutator_id = query_workflow.derivative_mutator

            if derivative_deriver_id not in workflow_resource_ids:
                raise ValueError(
                    f"Derivative deriver id '{derivative_deriver_id}' "
                    "in query workflow not found in resource definitions"
                )
            if derivative_mutator_id not in workflow_resource_ids:
                raise ValueError(
                    f"Derivative mutator id '{derivative_mutator_id}' "
                    "in query workflow not found in resource definitions"
                )

            used_resource_ids.add(query_workflow.derivative_deriver)
            used_resource_ids.add(query_workflow.derivative_mutator)

        # Check adjacent related episode postulators.
        for related_episode_postulator_id in self.adjacent_related_episode_postulators:
            if related_episode_postulator_id not in workflow_resource_ids:
                raise ValueError(
                    f"Related episode postulator id '{related_episode_postulator_id}' "
                    "not found in resource definitions"
                )
            used_resource_ids.add(related_episode_postulator_id)

        unused_resource_ids = workflow_resource_ids - used_resource_ids
        if len(unused_resource_ids) > 0:
            raise ValueError(
                f"Workflow resources contain unused resource ids: {unused_resource_ids}"
            )

        return self


class LongTermMemory:
    def __init__(
        self,
        config: LongTermMemoryConfig,
        reranker: Reranker,
        vector_graph_store: VectorGraphStore,
        resource_manager: ResourceManager,
    ):
        episode_metadata_template = f"{config.metadata_prefix}$content"

        # Only execute the following couple of blocks if workflow overriden. TODO @edwinyyyu: REMOVE THIS MESSAGE AFTER IMPL
        # TODO @edwinyyyu: check that all resource ids are unique since it's typed in config but untyped here, else mangle ids based on type

        # LongTermMemory is responsible for providing dependencies
        # to its internal DeclarativeMemory.
        # Create all derivative derivers, mutators, related episode postulators here based on workflow spec and workflow resources.
        workflow_resources = {}

        for (
            workflow_resource_id,
            workflow_resource_definition,
        ) in config.workflow_resources.items():
            workflow_resource_factory = workflow_resource_type_factory_map.get(
                workflow_resource_definition.type
            )
            if workflow_resource_factory is None:
                raise ValueError(
                    f"Unknown workflow resource type: {workflow_resource_definition.type}"
                )

            dependency_ids = get_nested_values(
                workflow_resource_definition.dependencies
            )
            injections = resource_manager.resolve_resources(dependency_ids)

            workflow_resource = workflow_resource_factory.create(
                workflow_resource_definition.provider,
                workflow_resource_definition.config,
                workflow_resource_definition.dependencies,
                injections,
            )

            workflow_resources[workflow_resource_id] = workflow_resource

        ingestion_workflows_map = {
            episode_type: [
                IngestionWorkflow(
                    related_episode_postulator=workflow_resources[
                        ingestion_workflow.related_episode_postulator
                    ],
                    derivative_deriver=workflow_resources[
                        ingestion_workflow.derivative_deriver
                    ],
                    derivative_mutator=workflow_resources[
                        ingestion_workflow.derivative_mutator
                    ],
                    embedder=resource_manager.get_resource(ingestion_workflow.embedder),
                )
                for ingestion_workflow in ingestion_workflows
            ]
            for episode_type, ingestion_workflows in config.ingestion_workflows_map.items()
        }

        query_workflows = [
            QueryWorkflow(
                derivative_deriver=workflow_resources[
                    query_workflow.derivative_deriver
                ],
                derivative_mutator=workflow_resources[
                    query_workflow.derivative_mutator
                ],
                embedder=resource_manager.get_resource(query_workflow.embedder),
            )
            for query_workflow in config.query_workflows
        ]

        adjacent_related_episode_postulators = [
            workflow_resources[related_episode_postulator_id]
            for related_episode_postulator_id in config.adjacent_related_episode_postulators
        ]

        declarative_memory_config = DeclarativeMemoryConfig(
            episode_metadata_template=episode_metadata_template,
        )

        self._declarative_memory = DeclarativeMemory(
            declarative_memory_config,
            reranker=reranker,
            vector_graph_store=vector_graph_store,
            ingestion_workflows_map=ingestion_workflows_map,
            query_workflows=query_workflows,
            adjacent_related_episode_postulators=adjacent_related_episode_postulators,
        )

    async def add_episode(self, episode: Episode):
        declarative_memory_episode = DeclarativeMemoryEpisode(
            uuid=episode.uuid,
            episode_type=episode.episode_type,
            content_type=content_type_to_declarative_memory_content_type_map[
                episode.content_type
            ],
            content=episode.content,
            timestamp=episode.timestamp,
            filterable_properties={
                key: value
                for key, value in {
                    "group_id": episode.group_id,
                    "session_id": episode.session_id,
                    "producer_id": episode.producer_id,
                    "produced_for_id": episode.produced_for_id,
                }.items()
                if value is not None
            },
            user_metadata=episode.user_metadata,
        )
        await self._declarative_memory.add_episode(declarative_memory_episode)

    async def search(
        self,
        query: str,
        num_episodes_limit: int,
        id_filter: Mapping[str, str] | None = None,
    ):
        declarative_memory_episodes = await self._declarative_memory.search(
            query,
            num_episodes_limit=num_episodes_limit,
            property_filter=id_filter,
        )
        return [
            Episode(
                uuid=declarative_memory_episode.uuid,
                episode_type=declarative_memory_episode.episode_type,
                content_type=(
                    declarative_memory_content_type_to_content_type_map[
                        declarative_memory_episode.content_type
                    ]
                ),
                content=declarative_memory_episode.content,
                timestamp=declarative_memory_episode.timestamp,
                group_id=(
                    declarative_memory_episode.filterable_properties.get("group_id", "")
                ),
                session_id=(
                    declarative_memory_episode.filterable_properties.get(
                        "session_id", ""
                    )
                ),
                producer_id=(
                    declarative_memory_episode.filterable_properties.get(
                        "producer_id", ""
                    )
                ),
                produced_for_id=(
                    declarative_memory_episode.filterable_properties.get(
                        "produced_for_id", ""
                    )
                ),
                user_metadata=declarative_memory_episode.user_metadata,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def clear(self):
        self._declarative_memory.forget_all()

    async def forget_session(self, group_id: str, session_id: str):
        await self._declarative_memory.forget_filtered_episodes(
            property_filter={
                "group_id": group_id,
                "session_id": session_id,
            }
        )
