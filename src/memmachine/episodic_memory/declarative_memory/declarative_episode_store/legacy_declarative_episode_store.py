import json
from collections.abc import Iterable

from memmachine.common.vector_graph_store import VectorGraphStore, Node, Edge

from .declarative_episode_store import DeclarativeEpisodeStore

from ..data_types import Episode as DeclarativeEpisode, mangle_filterable_property_key, demangle_filterable_property_key, is_mangled_filterable_property_key

class LegacyDeclarativeEpisodeStore(DeclarativeEpisodeStore):
    def __init__(self, vector_graph_store: VectorGraphStore):
        self._episode_store = vector_graph_store

        self._episode_collection = f"Episode_{session_id}"
        self._derivative_collection = f"Derivative_{session_id}"

        self._derived_from_relation = f"DERIVED_FROM_{session_id}"


    async def add_episodes(session_id: str, episodes: Iterable[DeclarativeEpisode]) -> None:
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
            )
            for episode in episodes
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

    async def get_episodes(episode_uids: Iterable[str]) -> list[DeclarativeEpisode]:
        pass

        await asyncio.gather(*[
            self._episode_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._episode_collection,
                this_collection=self._derivative_collection,
                this_node_uid=matched_derivative_entry.uid,
                find_sources=False,
                find_targets=True,
                node_property_filter=mangled_property_filter,
            )
            for matched_derivative_entry in matched_derivative_entries
        ])


    def contextualize():
        previous_episodes = (
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

        next_episodes = await self._vector_graph_store.search_directional_nodes(
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

    def delete_derivatives():
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
