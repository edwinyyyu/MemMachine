from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator, field_validator

from memmachine.common.data_types import Nested
from memmachine.common.utils import get_nested_values
from memmachine.common.factory import Factory
from memmachine.common.embedder import Embedder
from memmachine.common.reranker import Reranker
from memmachine.common.resource_manager import ResourceManager
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.common.data_types import ResourceDefinition

from ..data_types import ContentType, Episode
from ..declarative_memory import ContentType as DeclarativeMemoryContentType
from ..declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryConfig,
    IngestionWorkflow,
    DerivationWorkflow,
    MutationWorkflow,
    Derivation,
)
from ..declarative_memory import Episode as DeclarativeMemoryEpisode
from ..declarative_memory.derivative_deriver import (
    DerivativeDeriver,
    DerivativeDeriverFactory,
)
from ..declarative_memory.derivative_mutator import (
    DerivativeMutator,
    DerivativeMutatorFactory,
)
from ..declarative_memory.related_episode_postulator import (
    RelatedEpisodePostulator,
    RelatedEpisodePostulatorFactory,
)

content_type_to_declarative_memory_content_type_map = {
    ContentType.STRING: DeclarativeMemoryContentType.STRING,
}

declarative_memory_content_type_to_content_type_map = {
    DeclarativeMemoryContentType.STRING: ContentType.STRING,
}

declarative_memory_resource_type_factory_map: dict[str, Factory] = {
    "derivative_deriver": DerivativeDeriverFactory,
    "derivative_mutator": DerivativeMutatorFactory,
    "related_episode_postulator": RelatedEpisodePostulatorFactory,
}


class MutationWorkflowSpec(BaseModel):
    derivative_mutator: str


class DerivationWorkflowSpec(BaseModel):
    derivative_deriver: str = Field()
    mutation_workflows: list[MutationWorkflowSpec]

    @field_validator("mutation_workflows", mode="after")
    @classmethod
    def check_mutation_workflows(cls, v):
        if len(v) == 0:
            raise ValueError("At least one mutation workflow must be specified")
        return v


class IngestionWorkflowSpec(BaseModel):
    related_episode_postulator: str
    derivation_workflows: list[DerivationWorkflowSpec]

    @field_validator("derivation_workflows", mode="after")
    @classmethod
    def check_derivation_workflows(cls, v):
        if len(v) == 0:
            raise ValueError("At least one derivation workflow must be specified")
        return v


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
    declarative_memory_workflows_resource_definitions: dict[str, ResourceDefinition]
    declarative_memory_ingestion_workflows_specs: dict[str, list[IngestionWorkflowSpec]]

    @model_validator(mode="after")
    def check_workflows_resource_definitions_and_specs(
        self,
    ):
        resource_definition_ids = set(
            self.declarative_memory_workflows_resource_definitions.keys()
        )

        # Check that all resource ids in ingestion workflows are defined in resource definitions
        # and that all resource ids in resource definitions are used in ingestion workflows.
        used_resource_ids = set()
        for (
            ingestion_workflow_spec
        ) in self.declarative_memory_ingestion_workflows_specs.values():
            related_episode_postulator_id = (
                ingestion_workflow_spec.related_episode_postulator
            )
            if related_episode_postulator_id not in resource_definition_ids:
                raise ValueError(
                    f"Related episode postulator id '{related_episode_postulator_id}' "
                    "in ingestion workflow not found in resource definitions"
                )
            used_resource_ids.add(related_episode_postulator_id)

            for (
                derivation_workflow_spec
            ) in ingestion_workflow_spec.derivation_workflows:
                derivative_deriver_id = derivation_workflow_spec.derivative_deriver
                if derivative_deriver_id not in resource_definition_ids:
                    raise ValueError(
                        f"Derivative deriver id '{derivative_deriver_id}' "
                        "in ingestion workflow not found in resource definitions"
                    )
                used_resource_ids.add(derivative_deriver_id)

                for (
                    mutation_workflow_spec
                ) in derivation_workflow_spec.mutation_workflows:
                    derivative_mutator_id = mutation_workflow_spec.derivative_mutator
                    if derivative_mutator_id not in resource_definition_ids:
                        raise ValueError(
                            f"Derivative mutator id '{derivative_mutator_id}' "
                            "in ingestion workflow not found in resource definitions"
                        )
                    used_resource_ids.add(derivative_mutator_id)

        unused_resource_ids = resource_definition_ids - used_resource_ids
        if len(unused_resource_ids) > 0:
            raise ValueError(
                f"Resource definitions contain unused resource ids: {unused_resource_ids}"
            )

        return self


class LongTermMemory:
    def __init__(
        self,
        config: LongTermMemoryConfig,
        embedder: Embedder,  # TODO: make this part of workflow spec
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
        declarative_memory_resources = {}

        for (
            declarative_memory_resource_id,
            declarative_memory_resource_definition,
        ) in config.resource_definitions.items():
            declarative_memory_resource_factory = (
                declarative_memory_resource_type_factory_map.get(
                    declarative_memory_resource_definition.type
                )
            )
            if declarative_memory_resource_factory is None:
                raise ValueError(
                    f"Unknown declarative memory resource type: {declarative_memory_resource_definition.type}"
                )

            dependency_ids = get_nested_values(
                declarative_memory_resource_definition.dependencies
            )
            injections = resource_manager.resolve_resources(dependency_ids)

            declarative_memory_resource = declarative_memory_resource_factory.create(
                declarative_memory_resource_definition.variant,
                declarative_memory_resource_definition.config,
                declarative_memory_resource_definition.dependencies,
                injections,
            )

            declarative_memory_resources[declarative_memory_resource_id] = (
                declarative_memory_resource
            )

        ingestion_workflows = {
            episode_type: [
                LongTermMemory._build_ingestion_workflow(
                    ingestion_workflow_spec, declarative_memory_resources
                )
                for ingestion_workflow_spec in ingestion_workflow_specs
            ]
            for episode_type, ingestion_workflow_specs in config.declarative_memory_ingestion_workflows_specs.items()
        }

        declarative_memory_config = DeclarativeMemoryConfig(
            episode_metadata_template=episode_metadata_template,
        )

        self._declarative_memory = DeclarativeMemory(
            declarative_memory_config,
            embedder=embedder,
            reranker=reranker,
            vector_graph_store=vector_graph_store,
            ingestion_workflows=ingestion_workflows,
        )

    @staticmethod
    def _build_ingestion_workflow(
        ingestion_workflow: IngestionWorkflowSpec,
        resources: Mapping[str, Any],
    ) -> IngestionWorkflow:
        return IngestionWorkflow(
            related_episode_postulator=resources[
                ingestion_workflow.related_episode_postulator
            ],
            derivation_workflows=[
                DerivationWorkflow(
                    derivative_deriver=resources[
                        derivation_workflow.derivative_deriver
                    ],
                    mutation_workflows=[
                        MutationWorkflow(
                            derivative_mutator=resources[
                                mutation_workflow.derivative_mutator
                            ],
                        )
                        for mutation_workflow in derivation_workflow.mutation_workflows
                    ],
                )
                for derivation_workflow in ingestion_workflow.derivation_workflows
            ],
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
