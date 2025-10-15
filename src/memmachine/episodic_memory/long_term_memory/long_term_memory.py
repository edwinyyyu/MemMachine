from collections.abc import Mapping

from pydantic import BaseModel, Field

from memmachine.common.utils import get_nested_values
from memmachine.common.factory import Factory
from memmachine.common.embedder import Embedder
from memmachine.common.reranker import Reranker
from memmachine.common.resource_manager import ResourceManager
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.common.data_types import ResourceDefinition

from ..data_types import ContentType, Episode
from ..declarative_memory import ContentType as DeclarativeMemoryContentType
from ..declarative_memory import DeclarativeMemory, DeclarativeMemoryConfig
from ..declarative_memory import Episode as DeclarativeMemoryEpisode
from ..declarative_memory.derivative_deriver import DerivativeDeriver, DerivativeDeriverFactory
from ..declarative_memory.derivative_mutator import DerivativeMutator, DerivativeMutatorFactory
from ..declarative_memory.related_episode_postulator import RelatedEpisodePostulator, RelatedEpisodePostulatorFactory

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

class LongTermMemoryConfig(BaseModel):
    metadata_prefix: str = Field(
        "[$timestamp] $producer_id: ",
        description=(
            "Template prefix supporting $-substitutions "
            "to format episodes with episode metadata for the reranker "
            "(default: '[$timestamp] $producer_id: ')."
        )
    )

    workflow_specification: None = None

    # TODO: Convert from config file format (type first) to this format (homogeneous) in previous layer.
    resource_definitions: dict[str, ResourceDefinition]

class LongTermMemory:
    def __init__(
        self,
        config: LongTermMemoryConfig,
        embedder: Embedder, # TODO: make this part of workflow spec
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

        for declarative_memory_resource_id, declarative_memory_resource_definition in config.resource_definitions.items():
            declarative_memory_resource_factory = declarative_memory_resource_type_factory_map.get(declarative_memory_resource_definition.type)
            if declarative_memory_resource_factory is None:
                raise ValueError(f"Unknown declarative memory resource type: {declarative_memory_resource_definition.type}")

            dependency_ids = get_nested_values(declarative_memory_resource_definition.dependencies)
            injections = resource_manager.resolve_resources(dependency_ids)

            declarative_memory_resource = declarative_memory_resource_factory.create(
                declarative_memory_resource_definition.variant,
                declarative_memory_resource_definition.config,
                declarative_memory_resource_definition.dependencies,
                injections,
            )

            declarative_memory_resources[declarative_memory_resource_id] = declarative_memory_resource

        ingestion_workflows = {
            episode_type:
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
            query_workflow=query_workflow,
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
