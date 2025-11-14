from collections.abc import Iterable, Mapping
from typing import cast
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf

from ..data_types import ContentType, Episode, ResourceMgrProto
from memmachine.common.data_types import FilterablePropertyValue
from memmachine.common.embedder import Embedder
from memmachine.common.reranker import Reranker
from memmachine.common.vector_graph_store import VectorGraphStore

from ..data_types import ContentType, Episode, EpisodeType
from ..declarative_memory import DeclarativeMemory, DeclarativeMemoryParams
from ..declarative_memory.data_types import (
    ContentType as DeclarativeMemoryContentType,
    Episode as DeclarativeMemoryEpisode,
)
from ...common.configuration.episodic_config import LongTermMemoryConf



class LongTermMemoryParams(BaseModel):
    """
    Parameters for LongTermMemory.

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


class LongTermMemory:
    def __init__(self,
                 resource_mgr: ResourceMgrProto,
                 conf: LongTermMemoryConf):
        # Note: Things look a bit weird during refactor...
        # Internal session_id is used for external group_id. This is intentional.
        embedder = resource_mgr.get_embedder(conf.embedder)
        reranker = resource_mgr.get_reranker(conf.reranker)
        graph_store = resource_mgr.get_graph_store(conf.vector_graph_store)
        self._conf = conf
        self._declarative_memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=conf.session_id,
                vector_graph_store=graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

    async def add_episodes(self, episodes: Iterable[Episode]):
        declarative_memory_episodes = [
            DeclarativeMemoryEpisode(
                uuid=episode.uuid,
                timestamp=episode.timestamp,
                source=episode.producer_id,
                content_type=LongTermMemory._declarative_memory_content_type_from_episode(
                    episode
                ),
                content=episode.content,
                filterable_properties={
                    key: value
                    for key, value in {
                        "sequence_num": episode.sequence_num,
                        "session_key": episode.session_key,
                        "episode_type": episode.episode_type.value,
                        "content_type": episode.content_type.value,
                        "producer_id": episode.producer_id,
                        "producer_role": episode.producer_role,
                        "produced_for_id": episode.produced_for_id,
                    }.items()
                    if value is not None
                },
                user_metadata=episode.user_metadata,
            )
            for episode in episodes
        ]
        await self._declarative_memory.add_episodes(declarative_memory_episodes)

    async def search(
        self,
        query: str,
        num_episodes_limit: int,
        property_filter: Mapping[str, FilterablePropertyValue] | None = None,
    ):
        declarative_memory_episodes = await self._declarative_memory.search(
            query,
            max_num_episodes=num_episodes_limit,
            property_filter=property_filter,
        )
        return [
            LongTermMemory._episode_from_declarative_memory_episode(
                declarative_memory_episode
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def get_episodes(self, uuids: Iterable[UUID]) -> list[Episode]:
        declarative_memory_episodes = await self._declarative_memory.get_episodes(uuids)
        return [
            LongTermMemory._episode_from_declarative_memory_episode(
                declarative_memory_episode
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def get_matching_episodes(
        self,
        property_filter: Mapping[str, FilterablePropertyValue] | None = None,
    ) -> list[Episode]:
        declarative_memory_episodes = (
            await self._declarative_memory.get_matching_episodes(
                property_filter=property_filter,
            )
        )
        return [
            LongTermMemory._episode_from_declarative_memory_episode(
                declarative_memory_episode
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def delete_episodes(self, uuids: Iterable[UUID]):
        await self._declarative_memory.delete_episodes(uuids)

    async def delete_matching_episodes(
        self,
        property_filter: Mapping[str, FilterablePropertyValue] | None = None,
    ):
        self._declarative_memory.delete_episodes(
            episode.uuid
            for episode in await self._declarative_memory.get_matching_episodes(
                property_filter=property_filter
            )
        )

    async def close(self):
        # Do nothing.
        pass

    @staticmethod
    def _declarative_memory_content_type_from_episode(
        episode: Episode,
    ) -> DeclarativeMemoryContentType:
        match episode.episode_type:
            case EpisodeType.MESSAGE:
                match episode.content_type:
                    case ContentType.STRING:
                        return DeclarativeMemoryContentType.MESSAGE
                    case _:
                        return DeclarativeMemoryContentType.TEXT
            case _:
                match episode.content_type:
                    case ContentType.STRING:
                        return DeclarativeMemoryContentType.TEXT
                    case _:
                        return DeclarativeMemoryContentType.TEXT

    @staticmethod
    def _episode_from_declarative_memory_episode(
        declarative_memory_episode: DeclarativeMemoryEpisode,
    ) -> Episode:
        return Episode(
            uuid=declarative_memory_episode.uuid,
            sequence_num=cast(
                int,
                declarative_memory_episode.filterable_properties.get("sequence_num", 0),
            ),
            session_key=cast(
                str,
                declarative_memory_episode.filterable_properties.get("session_key", ""),
            ),
            episode_type=EpisodeType(
                cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "episode_type", ""
                    ),
                )
            ),
            content_type=ContentType(
                cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "content_type", ""
                    ),
                )
            ),
            content=declarative_memory_episode.content,
            timestamp=declarative_memory_episode.timestamp,
            producer_id=cast(
                str,
                declarative_memory_episode.filterable_properties.get("producer_id", ""),
            ),
            producer_role=cast(
                str,
                declarative_memory_episode.filterable_properties.get(
                    "producer_role", ""
                ),
            ),
            produced_for_id=cast(
                str | None,
                declarative_memory_episode.filterable_properties.get("produced_for_id"),
            ),
            user_metadata=declarative_memory_episode.user_metadata,
        )
