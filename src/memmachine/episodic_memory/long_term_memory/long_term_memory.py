"""Long-term declarative memory coordination."""

from collections.abc import Iterable
from typing import cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf, JsonValue

from memmachine.common.data_types import PropertyValue
from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import ContentType, Episode, EpisodeType
from memmachine.common.filter.filter_parser import (
    FilterExpr,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine.common.reranker import Reranker
from memmachine.common.vector_store import Collection
from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
    Segment,
    SegmentStore,
)
from memmachine.episodic_memory.declarative_memory.data_types import (
    ContentType as DeclarativeMemoryContentType,
)
from memmachine.episodic_memory.declarative_memory.data_types import (
    Episode as DeclarativeMemoryEpisode,
)


class LongTermMemoryParams(BaseModel):
    """
    Parameters for LongTermMemory.

    Attributes:
        session_key (str):
            Session key.
        collection (Collection):
            Collection instance in a vector store.
        segment_store (SegmentStore):
            Segment store instance for managing segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.

    """

    session_key: str = Field(
        ...,
        description="Session key",
    )
    collection: InstanceOf[Collection] = Field(
        ...,
        description="Collection instance in a vector store",
    )
    segment_store: InstanceOf[SegmentStore] = Field(
        ...,
        description="Segment store instance for managing segments",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    message_sentence_chunking: bool = Field(
        False,
        description="Whether to chunk message episodes into sentences for embedding",
    )


class LongTermMemory:
    """High-level facade around the declarative memory store."""

    _FILTERABLE_METADATA_NONE_FLAG = "_filterable_metadata_none"

    def __init__(self, params: LongTermMemoryParams) -> None:
        """Wire up the declarative memory backing store."""
        self._declarative_memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_key=params.session_key,
                collection=params.collection,
                segment_store=params.segment_store,
                embedder=params.embedder,
                reranker=params.reranker,
                message_sentence_chunking=params.message_sentence_chunking,
            ),
        )

    async def add_episodes(self, episodes: Iterable[Episode]) -> None:
        declarative_memory_episodes = [
            DeclarativeMemoryEpisode(
                uuid=UUID(episode.uid) if episode.uid else uuid4(),
                timestamp=episode.created_at,
                context=episode.producer_id,
                content_type=LongTermMemory._declarative_memory_content_type_from_episode(
                    episode,
                ),
                content=episode.content,
                attributes=cast(
                    dict[str, PropertyValue],
                    {
                        key: value
                        for key, value in {
                            "created_at": episode.created_at,
                            "session_key": episode.session_key,
                            "producer_id": episode.producer_id,
                            "producer_role": episode.producer_role,
                            "produced_for_id": episode.produced_for_id,
                            "sequence_num": episode.sequence_num,
                            "episode_type": episode.episode_type.value,
                            "content_type": episode.content_type.value,
                        }.items()
                        if value is not None
                    }
                    | (
                        {
                            LongTermMemory._mangle_filterable_metadata_key(key): value
                            for key, value in (
                                episode.filterable_metadata or {}
                            ).items()
                        }
                        if episode.filterable_metadata is not None
                        else {LongTermMemory._FILTERABLE_METADATA_NONE_FLAG: True}
                    ),
                ),
                payload=cast("dict[str, JsonValue] | None", episode.metadata),
            )
            for episode in episodes
        ]
        await self._declarative_memory.encode_episodes(declarative_memory_episodes)

    async def search(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        scored_episodes = await self.search_scored(
            query,
            num_episodes_limit=num_episodes_limit,
            expand_context=expand_context,
            score_threshold=score_threshold,
            property_filter=property_filter,
        )
        return [episode for _, episode in scored_episodes]

    async def search_scored(
        self,
        query: str,
        *,
        num_episodes_limit: int,
        expand_context: int = 0,
        score_threshold: float = -float("inf"),
        property_filter: FilterExpr | None = None,
    ) -> list[tuple[float, Episode]]:
        scored_segments = await self._declarative_memory.search_scored(
            query,
            max_num_segments=num_episodes_limit,
            expand_context=expand_context,
            property_filter=LongTermMemory._sanitize_property_filter(property_filter),
        )
        return [
            (
                score,
                LongTermMemory._episode_from_segment(segment),
            )
            for score, segment in scored_segments
            if score >= score_threshold
        ]

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        await self._declarative_memory.forget_episodes(UUID(uid) for uid in uids)

    async def forget_episodes(self, episode_uuids: Iterable[UUID]) -> None:
        await self._declarative_memory.forget_episodes(episode_uuids)

    async def delete_matching_episodes(
        self,
        property_filter: FilterExpr | None = None,
    ) -> None:
        if property_filter is None:
            await self._declarative_memory.forget_all_episodes()
        else:
            raise NotImplementedError(
                "delete_matching_episodes with a property filter is not yet supported"
            )

    async def close(self) -> None:
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
    def _episode_from_segment(
        segment: Segment,
    ) -> Episode:
        attributes = segment.attributes or {}
        return Episode(
            uid=str(segment.episode_uuid),
            sequence_num=cast(
                "int",
                attributes.get("sequence_num", 0),
            ),
            session_key=cast(
                "str",
                attributes.get("session_key", ""),
            ),
            episode_type=EpisodeType(
                cast(
                    "str",
                    attributes.get(
                        "episode_type",
                        "",
                    ),
                ),
            ),
            content_type=ContentType(
                cast(
                    "str",
                    attributes.get(
                        "content_type",
                        "",
                    ),
                ),
            ),
            content=segment.content,
            created_at=segment.timestamp,
            producer_id=cast(
                "str",
                attributes.get("producer_id", ""),
            ),
            producer_role=cast(
                "str",
                attributes.get(
                    "producer_role",
                    "",
                ),
            ),
            produced_for_id=cast(
                "str | None",
                attributes.get("produced_for_id"),
            ),
            filterable_metadata={
                LongTermMemory._demangle_filterable_metadata_key(key): value
                for key, value in attributes.items()
                if LongTermMemory._is_mangled_filterable_metadata_key(key)
            }
            if LongTermMemory._FILTERABLE_METADATA_NONE_FLAG not in attributes
            else None,
            metadata=None,
        )

    _MANGLE_FILTERABLE_METADATA_KEY_PREFIX = "metadata."

    @staticmethod
    def _mangle_filterable_metadata_key(key: str) -> str:
        return LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX + key

    @staticmethod
    def _demangle_filterable_metadata_key(mangled_key: str) -> str:
        return mangled_key.removeprefix(
            LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX
        )

    @staticmethod
    def _is_mangled_filterable_metadata_key(candidate_key: str) -> bool:
        return candidate_key.startswith(
            LongTermMemory._MANGLE_FILTERABLE_METADATA_KEY_PREFIX
        )

    @staticmethod
    def _sanitize_property_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        if property_filter is None:
            return None

        return LongTermMemory._sanitize_filter_expr(property_filter)

    @staticmethod
    def _sanitize_field(field: str) -> str:
        internal_name, _ = normalize_filter_field(field)
        return internal_name

    @staticmethod
    def _sanitize_filter_expr(expr: FilterExpr) -> FilterExpr:
        return map_filter_fields(expr, LongTermMemory._sanitize_field)
