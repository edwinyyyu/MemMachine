"""Declarative memory system for storing and retrieving episodic memory."""

import asyncio
import datetime
import json
import logging
from collections.abc import Iterable
from typing import cast
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder import Embedder
from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.common.language_model import LanguageModel
from memmachine.common.reranker import Reranker
from memmachine.common.utils import extract_sentences
from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore

from .data_types import (
    ContentType,
    Derivative,
    Episode,
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
        session_id (str):
            Session identifier.
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        language_model (LanguageModel | None):
            LanguageModel instance. Required when derivative_dedup_threshold > 0.
        reranker (Reranker):
            Reranker instance for reranking search results.
        derivative_dedup_threshold (float):
            Cosine similarity threshold for derivative deduplication.
            Set to 0.0 to disable (default).

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
    language_model: InstanceOf[LanguageModel] | None = Field(
        None,
        description="LanguageModel instance. Required when derivative_dedup_threshold > 0.",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    derivative_dedup_threshold: float = Field(
        0.0,
        description="Cosine similarity threshold for derivative deduplication. Set to 0.0 to disable.",
        ge=0.0,
        le=1.0,
    )
    message_sentence_chunking: bool = Field(
        False,
        description="Whether to chunk message episodes into sentences for embedding",
    )



class DeclarativeMemory:
    """Declarative memory system."""

    def __init__(self, params: DeclarativeMemoryParams) -> None:
        """
        Initialize a DeclarativeMemory with the provided parameters.

        Args:
            params (DeclarativeMemoryParams):
                Parameters for the DeclarativeMemory.

        """
        session_id = params.session_id

        self._vector_graph_store = params.vector_graph_store
        self._embedder = params.embedder
        self._language_model = params.language_model
        self._reranker = params.reranker

        self._derivative_dedup_threshold = params.derivative_dedup_threshold
        self._message_sentence_chunking = params.message_sentence_chunking

        self._episode_collection = f"Episode_{session_id}"
        self._derivative_collection = f"Derivative_{session_id}"

        self._derived_from_relation = f"DERIVED_FROM_{session_id}"

    async def _verify_duplicate(self, text_a: str, text_b: str) -> bool:
        """Use the language model to verify if two texts are semantically equivalent."""
        if self._language_model is None:
            return True

        system_prompt = (
            "You are a semantic equivalence judge. Given two texts, determine if they "
            "convey the same meaning. Respond with only 'true' or 'false'."
        )
        user_prompt = f"Text A: {text_a}\n\nText B: {text_b}"

        response_text, _ = await self._language_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if not response_text:
            return False

        return "true" in response_text.lower()

    async def _deduplicate_derivatives(
        self,
        episodes: list[Episode],
        episodes_derivatives: list[list[Derivative]],
        derivatives: list[Derivative],
        derivative_embeddings: list[list[float]],
    ) -> tuple[list[Node], list[Edge]]:
        """Deduplicate derivatives globally across batch and DB."""
        embedding_name = DeclarativeMemory._embedding_name(
            self._embedder.model_id,
            self._embedder.dimensions,
        )

        # Build mapping: derivative index -> source episode
        derivative_to_episode: list[Episode] = []
        for episode, episode_derivs in zip(
            episodes, episodes_derivatives, strict=True
        ):
            derivative_to_episode.extend(episode for _ in episode_derivs)

        # Compute similarities (batch + DB in parallel)
        batch_sim_matrix = DeclarativeMemory._compute_batch_similarity_matrix(
            derivative_embeddings
        )
        db_results = await self._lookup_db_similarities(
            derivative_embeddings, embedding_name
        )

        # Find best global candidate per derivative, LLM verify, resolve groups
        batch_candidates, db_candidates = self._find_dedup_candidates(
            derivatives, batch_sim_matrix, db_results
        )
        verify_results = await self._verify_dedup_candidates(
            derivatives, batch_candidates, db_candidates
        )
        groups, group_db_match = DeclarativeMemory._apply_dedup_results(
            batch_candidates, db_candidates, verify_results, len(derivatives)
        )

        return self._build_dedup_output(
            groups,
            group_db_match,
            derivatives,
            derivative_embeddings,
            derivative_to_episode,
            embedding_name,
        )

    @staticmethod
    def _compute_batch_similarity_matrix(
        derivative_embeddings: list[list[float]],
    ) -> np.ndarray:
        """Compute pairwise cosine similarity matrix for batch embeddings."""
        embeddings_matrix = np.array(derivative_embeddings)
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings_matrix / norms
        return normalized @ normalized.T

    async def _lookup_db_similarities(
        self,
        derivative_embeddings: list[list[float]],
        embedding_name: str,
    ) -> list[tuple[Node, float] | None]:
        """DB lookup for each derivative. Returns (node, cosine_sim) or None."""
        threshold = self._derivative_dedup_threshold

        db_lookup_results = await asyncio.gather(
            *(
                self._vector_graph_store.search_similar_nodes(
                    collection=self._derivative_collection,
                    embedding_name=embedding_name,
                    query_embedding=emb,
                    similarity_metric=self._embedder.similarity_metric,
                    limit=1,
                )
                for emb in derivative_embeddings
            )
        )

        results: list[tuple[Node, float] | None] = []
        for emb, db_nodes in zip(
            derivative_embeddings, db_lookup_results, strict=True
        ):
            if not db_nodes:
                results.append(None)
                continue
            db_node = db_nodes[0]
            if embedding_name not in db_node.embeddings:
                results.append(None)
                continue
            db_embedding = db_node.embeddings[embedding_name][0]
            query_emb = np.array(emb)
            db_emb = np.array(db_embedding)
            query_norm = np.linalg.norm(query_emb)
            db_norm = np.linalg.norm(db_emb)
            if query_norm == 0 or db_norm == 0:
                results.append(None)
                continue
            cos_sim = float(np.dot(query_emb, db_emb) / (query_norm * db_norm))
            if cos_sim >= threshold:
                results.append((db_node, cos_sim))
            else:
                results.append(None)

        return results

    def _find_dedup_candidates(
        self,
        derivatives: list[Derivative],
        batch_sim_matrix: np.ndarray,
        db_results: list[tuple[Node, float] | None],
    ) -> tuple[dict[int, int], dict[int, Node]]:
        """Find best global dedup candidate per derivative (batch or DB)."""
        threshold = self._derivative_dedup_threshold
        batch_candidates: dict[int, int] = {}
        db_candidates: dict[int, Node] = {}

        for i in range(len(derivatives)):
            best_batch_j = -1
            best_batch_sim = -1.0
            for j in range(i):
                sim = float(batch_sim_matrix[i, j])
                if sim > best_batch_sim:
                    best_batch_sim = sim
                    best_batch_j = j

            db_result = db_results[i]
            db_sim = db_result[1] if db_result is not None else -1.0

            if best_batch_sim >= threshold and best_batch_sim >= db_sim:
                batch_candidates[i] = best_batch_j
            elif db_result is not None and db_sim >= threshold:
                db_candidates[i] = db_result[0]

        return batch_candidates, db_candidates

    async def _verify_dedup_candidates(
        self,
        derivatives: list[Derivative],
        batch_candidates: dict[int, int],
        db_candidates: dict[int, Node],
    ) -> dict[int, bool]:
        """LLM-verify all dedup candidates. Returns derivative_idx -> is_duplicate."""
        all_indices = sorted(set(batch_candidates) | set(db_candidates))
        if not all_indices:
            return {}

        verify_results_list = await asyncio.gather(
            *(
                self._verify_duplicate(
                    derivatives[i].content,
                    derivatives[batch_candidates[i]].content
                    if i in batch_candidates
                    else cast("str", db_candidates[i].properties.get("content", "")),
                )
                for i in all_indices
            )
        )
        return dict(zip(all_indices, verify_results_list, strict=True))

    @staticmethod
    def _apply_dedup_results(
        batch_candidates: dict[int, int],
        db_candidates: dict[int, Node],
        verify_results: dict[int, bool],
        num_derivatives: int,
    ) -> tuple[dict[int, list[int]], dict[int, str]]:
        """Apply verified results into groups + DB match map."""
        representative: dict[int, int] = {i: i for i in range(num_derivatives)}
        group_db_match: dict[int, str] = {}

        # Process in derivative order so union-find roots are resolved
        for i in sorted(set(batch_candidates) | set(db_candidates)):
            if not verify_results.get(i, False):
                continue
            if i in batch_candidates:
                root = batch_candidates[i]
                while representative[root] != root:
                    root = representative[root]
                representative[i] = root
            else:
                root = i
                while representative[root] != root:
                    root = representative[root]
                group_db_match[root] = db_candidates[i].uid

        groups: dict[int, list[int]] = {}
        for idx in range(num_derivatives):
            groups.setdefault(representative[idx], []).append(idx)

        return groups, group_db_match

    def _build_dedup_output(
        self,
        groups: dict[int, list[int]],
        db_match: dict[int, str],
        derivatives: list[Derivative],
        derivative_embeddings: list[list[float]],
        derivative_to_episode: list[Episode],
        embedding_name: str,
    ) -> tuple[list[Node], list[Edge]]:
        """Build output nodes and edges from dedup groups."""
        nodes_to_create: list[Node] = []
        edges_to_create: list[Edge] = []

        for rep_idx, group_indices in groups.items():
            source_episode_uids = [
                derivative_to_episode[idx].uid for idx in group_indices
            ]

            if rep_idx in db_match:
                existing_uid = db_match[rep_idx]
                edges_to_create.extend(
                    Edge(
                        uid=str(uuid4()),
                        source_uid=existing_uid,
                        target_uid=ep_uid,
                    )
                    for ep_uid in source_episode_uids
                )
            else:
                rep_derivative = derivatives[rep_idx]
                rep_embedding = derivative_embeddings[rep_idx]
                nodes_to_create.append(
                    Node(
                        uid=rep_derivative.uid,
                        properties={
                            "uid": rep_derivative.uid,
                            "timestamp": rep_derivative.timestamp,
                            "source": rep_derivative.source,
                            "content_type": rep_derivative.content_type.value,
                            "content": rep_derivative.content,
                        }
                        | {
                            mangle_filterable_property_key(key): value
                            for key, value in rep_derivative.filterable_properties.items()
                        },
                        embeddings={
                            embedding_name: (
                                rep_embedding,
                                self._embedder.similarity_metric,
                            ),
                        },
                    )
                )
                edges_to_create.extend(
                    Edge(
                        uid=str(uuid4()),
                        source_uid=rep_derivative.uid,
                        target_uid=ep_uid,
                    )
                    for ep_uid in source_episode_uids
                )

        return nodes_to_create, edges_to_create

    async def add_episodes(
        self,
        episodes: Iterable[Episode],
    ) -> None:
        """
        Add episodes.

        Episodes are sorted by timestamp.
        Episodes with the same timestamp are sorted by UID.

        Args:
            episodes (Iterable[Episode]): The episodes to add.

        """
        episodes = sorted(
            episodes,
            key=lambda episode: (episode.timestamp, episode.uid),
        )
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

        derive_derivatives_tasks = [
            self._derive_derivatives(episode) for episode in episodes
        ]

        episodes_derivatives = await asyncio.gather(*derive_derivatives_tasks)

        derivatives = [
            derivative
            for episode_derivatives in episodes_derivatives
            for derivative in episode_derivatives
        ]

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.content for derivative in derivatives],
        )

        if self._derivative_dedup_threshold > 0.0 and derivatives:
            derivative_nodes, derivative_episode_edges = (
                await self._deduplicate_derivatives(
                    episodes,
                    episodes_derivatives,
                    derivatives,
                    derivative_embeddings,
                )
            )
        else:
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

    async def _derive_derivatives(
        self,
        episode: Episode,
    ) -> list[Derivative]:
        """
        Derive derivatives from an episode.

        Args:
            episode (Episode):
                The episode from which to derive derivatives.

        Returns:
            list[Derivative]: A list of derived derivatives.

        """
        match episode.content_type:
            case ContentType.MESSAGE:
                if not self._message_sentence_chunking:
                    return [
                        Derivative(
                            uid=str(uuid4()),
                            timestamp=episode.timestamp,
                            source=episode.source,
                            content_type=ContentType.MESSAGE,
                            content=f"{episode.source}: {episode.content}",
                            filterable_properties=episode.filterable_properties,
                        ),
                    ]

                sentences = extract_sentences(episode.content)

                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.MESSAGE,
                        content=f"{episode.source}: {sentence}",
                        filterable_properties=episode.filterable_properties,
                    )
                    for sentence in sentences
                ]
            case ContentType.TEXT:
                text_content = episode.content
                return [
                    Derivative(
                        uid=str(uuid4()),
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=ContentType.TEXT,
                        content=text_content,
                        filterable_properties=episode.filterable_properties,
                    ),
                ]
            case _:
                logger.warning(
                    "Unsupported content type for derivative derivation: %s",
                    episode.content_type,
                )
                return []

    async def search(
        self,
        query: str,
        *,
        max_num_episodes: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        """
        Search declarative memory for episodes relevant to the query.

        Args:
            query (str):
                The search query.
            max_num_episodes (int):
                The maximum number of episodes to return
                (default: 20).
            expand_context (int):
                The number of additional episodes to include
                around each matched episode for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Filterable property keys and values
                to use for filtering episodes
                (default: None).

        Returns:
            list[Episode]:
                A list of episodes relevant to the query, ordered chronologically.

        """
        scored_episodes = await self.search_scored(
            query,
            max_num_episodes=max_num_episodes,
            expand_context=expand_context,
            property_filter=property_filter,
        )
        return [episode for _, episode in scored_episodes]

    async def search_scored(
        self,
        query: str,
        *,
        max_num_episodes: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
    ) -> list[tuple[float, Episode]]:
        """
        Search declarative memory for episodes relevant to the query, returning scored episodes.

        Args:
            query (str):
                The search query.
            max_num_episodes (int):
                The maximum number of episodes to return
                (default: 20).
            expand_context (int):
                The number of additional episodes to include
                around each matched episode for additional context
                (default: 0).
            property_filter (FilterExpr | None):
                Filterable property keys and values
                to use for filtering episodes
                (default: None).

        Returns:
            list[tuple[float, Episode]]:
                A list of scored episodes relevant to the query, ordered chronologically.

        """
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Search graph store for vector matches.
        matched_derivative_nodes = await self._vector_graph_store.search_similar_nodes(
            collection=self._derivative_collection,
            embedding_name=(
                DeclarativeMemory._embedding_name(
                    self._embedder.model_id,
                    self._embedder.dimensions,
                )
            ),
            query_embedding=query_embedding,
            similarity_metric=self._embedder.similarity_metric,
            limit=max(5 * max_num_episodes, 200),
            property_filter=mangled_property_filter,
        )

        # Get source episodes of matched derivatives.
        search_derivatives_source_episode_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._episode_collection,
                this_collection=self._derivative_collection,
                this_node_uid=matched_derivative_node.uid,
                find_sources=False,
                find_targets=True,
                node_property_filter=mangled_property_filter,
            )
            for matched_derivative_node in matched_derivative_nodes
        ]

        # Use a dict instead of a set to preserve order.
        source_episode_nodes = dict.fromkeys(
            episode_node
            for episode_nodes in await asyncio.gather(
                *search_derivatives_source_episode_nodes_tasks,
            )
            for episode_node in episode_nodes
        )

        # Use source episodes as nuclei for contextualization.
        nuclear_episodes = [
            DeclarativeMemory._episode_from_episode_node(source_episode_node)
            for source_episode_node in source_episode_nodes
        ]

        expand_context = min(max(0, expand_context), max_num_episodes - 1)
        max_backward_episodes = expand_context // 3
        max_forward_episodes = expand_context - max_backward_episodes

        contextualize_episode_tasks = [
            self._contextualize_episode(
                nuclear_episode,
                max_backward_episodes=max_backward_episodes,
                max_forward_episodes=max_forward_episodes,
                mangled_property_filter=mangled_property_filter,
            )
            for nuclear_episode in nuclear_episodes
        ]

        episode_contexts = await asyncio.gather(*contextualize_episode_tasks)

        # Rerank episode contexts.
        episode_context_scores = await self._score_episode_contexts(
            query,
            episode_contexts,
        )

        reranked_scored_anchored_episode_contexts = [
            (episode_context_score, nuclear_episode, episode_context)
            for episode_context_score, nuclear_episode, episode_context in sorted(
                zip(
                    episode_context_scores,
                    nuclear_episodes,
                    episode_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        # Unify episode contexts.
        unified_scored_episode_context = (
            DeclarativeMemory._unify_scored_anchored_episode_contexts(
                reranked_scored_anchored_episode_contexts,
                max_num_episodes=max_num_episodes,
            )
        )
        return unified_scored_episode_context

    async def _contextualize_episode(
        self,
        nuclear_episode: Episode,
        max_backward_episodes: int = 0,
        max_forward_episodes: int = 0,
        mangled_property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        previous_episode_nodes = []
        next_episode_nodes = []

        if max_backward_episodes > 0:
            previous_episode_nodes = (
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

        if max_forward_episodes > 0:
            next_episode_nodes = (
                await self._vector_graph_store.search_directional_nodes(
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
            )

        context = (
            [
                DeclarativeMemory._episode_from_episode_node(episode_node)
                for episode_node in reversed(previous_episode_nodes)
            ]
            + [nuclear_episode]
            + [
                DeclarativeMemory._episode_from_episode_node(episode_node)
                for episode_node in next_episode_nodes
            ]
        )

        return context

    async def _score_episode_contexts(
        self,
        query: str,
        episode_contexts: Iterable[Iterable[Episode]],
    ) -> list[float]:
        """Score episode contexts based on their relevance to the query."""
        context_strings = []
        for episode_context in episode_contexts:
            context_string = DeclarativeMemory.string_from_episode_context(
                episode_context
            )
            context_strings.append(context_string)

        episode_context_scores = await self._reranker.score(query, context_strings)

        return episode_context_scores

    @staticmethod
    def string_from_episode_context(episode_context: Iterable[Episode]) -> str:
        """Format episode context as a string."""
        context_string = ""

        for episode in episode_context:
            context_date = DeclarativeMemory._format_date(
                episode.timestamp.date(),
            )
            context_time = DeclarativeMemory._format_time(
                episode.timestamp.time(),
            )
            context_string += f"[{context_date} at {context_time}] {episode.source}: {json.dumps(episode.content)}\n"

        return context_string

    @staticmethod
    def _format_date(date: datetime.date) -> str:
        """Format the date as a string."""
        return date.strftime("%A, %B %d, %Y")

    @staticmethod
    def _format_time(time: datetime.time) -> str:
        """Format the time as a string."""
        return time.strftime("%I:%M %p")

    async def get_episodes(self, uids: Iterable[str]) -> list[Episode]:
        """Get episodes by their UIDs."""
        episode_nodes = await self._vector_graph_store.get_nodes(
            collection=self._episode_collection,
            node_uids=uids,
        )

        episodes = [
            DeclarativeMemory._episode_from_episode_node(episode_node)
            for episode_node in episode_nodes
        ]

        return episodes

    async def get_matching_episodes(
        self,
        property_filter: FilterExpr | None = None,
    ) -> list[Episode]:
        """Filter episodes by their properties."""
        mangled_property_filter = DeclarativeMemory._mangle_property_filter(
            property_filter,
        )

        matching_episode_nodes = await self._vector_graph_store.search_matching_nodes(
            collection=self._episode_collection,
            property_filter=mangled_property_filter,
        )

        matching_episodes = [
            DeclarativeMemory._episode_from_episode_node(matching_episode_node)
            for matching_episode_node in matching_episode_nodes
        ]

        return matching_episodes

    async def delete_episodes(self, uids: Iterable[str]) -> None:
        """Delete episodes by their UIDs."""
        uids = list(uids)
        uids_set = set(uids)

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

        derived_derivative_nodes = list(
            dict.fromkeys(
                derivative_node
                for derivative_nodes in await asyncio.gather(
                    *search_derived_derivative_nodes_tasks,
                )
                for derivative_node in derivative_nodes
            )
        )

        # Check which derivatives are safe to delete (not shared with episodes
        # outside the deletion set).
        search_derivative_targets_tasks = [
            self._vector_graph_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._episode_collection,
                this_collection=self._derivative_collection,
                this_node_uid=derivative_node.uid,
                find_sources=False,
                find_targets=True,
            )
            for derivative_node in derived_derivative_nodes
        ]

        derivative_targets = await asyncio.gather(*search_derivative_targets_tasks)

        deletable_derivative_uids = [
            derivative_node.uid
            for derivative_node, target_episode_nodes in zip(
                derived_derivative_nodes, derivative_targets, strict=True
            )
            if all(
                cast("str", ep_node.properties["uid"]) in uids_set
                for ep_node in target_episode_nodes
            )
        ]

        delete_nodes_tasks = [
            self._vector_graph_store.delete_nodes(
                collection=self._episode_collection,
                node_uids=uids,
            ),
            self._vector_graph_store.delete_nodes(
                collection=self._derivative_collection,
                node_uids=deletable_derivative_uids,
            ),
        ]

        await asyncio.gather(*delete_nodes_tasks)

    @staticmethod
    def _unify_scored_anchored_episode_contexts(
        scored_anchored_episode_contexts: Iterable[
            tuple[float, Episode, Iterable[Episode]]
        ],
        max_num_episodes: int,
    ) -> list[tuple[float, Episode]]:
        """Unify anchored episode contexts into a single list within the limit."""
        episode_scores: dict[Episode, float] = {}

        for score, nuclear_episode, context in scored_anchored_episode_contexts:
            context = list(context)

            if len(episode_scores) >= max_num_episodes:
                break
            if (len(episode_scores) + len(context)) <= max_num_episodes:
                # It is impossible that the context exceeds the limit.
                episode_scores.update(
                    {
                        episode: score
                        for episode in context
                        if episode not in episode_scores
                    }
                )
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize episodes near the nuclear episode.

                # Sort chronological episodes by weighted index-proximity to the nuclear episode.
                nuclear_index = context.index(nuclear_episode)

                nuclear_context = sorted(
                    context,
                    key=lambda episode: DeclarativeMemory._weighted_index_proximity(
                        episode=episode,
                        context=context,
                        nuclear_index=nuclear_index,
                    ),
                )

                # Add episodes to unified context until limit is reached,
                # or until the context is exhausted.
                for episode in nuclear_context:
                    if len(episode_scores) >= max_num_episodes:
                        break
                    episode_scores.setdefault(episode, score)

        unified_episode_context = sorted(
            [(score, episode) for episode, score in episode_scores.items()],
            key=lambda scored_episode: (
                scored_episode[1].timestamp,
                scored_episode[1].uid,
            ),
        )

        return unified_episode_context

    @staticmethod
    def _weighted_index_proximity(
        episode: Episode,
        context: list[Episode],
        nuclear_index: int,
    ) -> float:
        proximity = context.index(episode) - nuclear_index
        if proximity >= 0:
            # Forward recall is better than backward recall.
            return (proximity - 0.5) / 2
        return -proximity

    @staticmethod
    def _episode_from_episode_node(episode_node: Node) -> Episode:
        return Episode(
            uid=cast("str", episode_node.properties["uid"]),
            timestamp=cast("datetime.datetime", episode_node.properties["timestamp"]),
            source=cast("str", episode_node.properties["source"]),
            content_type=ContentType(episode_node.properties["content_type"]),
            content=episode_node.properties["content"],
            filterable_properties={
                demangle_filterable_property_key(key): cast(
                    "FilterablePropertyValue",
                    value,
                )
                for key, value in episode_node.properties.items()
                if is_mangled_filterable_property_key(key)
            },
            user_metadata=json.loads(
                cast("str", episode_node.properties["user_metadata"]),
            ),
        )

    @staticmethod
    def _embedding_name(model_id: str, dimensions: int) -> str:
        """
        Generate a standardized property name for embeddings based on the model ID and embedding dimensions.

        Args:
            model_id (str): The identifier of the embedding model.
            dimensions (int): The dimensionality of the embedding.

        Returns:
            str: A standardized property name for the embedding.

        """
        return f"embedding_{model_id}_{dimensions}d"

    @staticmethod
    def _mangle_property_filter(
        property_filter: FilterExpr | None,
    ) -> FilterExpr | None:
        if property_filter is None:
            return None

        return DeclarativeMemory._mangle_filter_expr(property_filter)

    @staticmethod
    def _mangle_filter_expr(expr: FilterExpr | None) -> FilterExpr | None:
        if expr is None:
            return None

        if isinstance(expr, FilterComparison):
            return FilterComparison(
                field=mangle_filterable_property_key(expr.field),
                op=expr.op,
                value=expr.value,
            )
        if isinstance(expr, FilterAnd):
            return FilterAnd(
                left=DeclarativeMemory._mangle_filter_expr(expr.left),
                right=DeclarativeMemory._mangle_filter_expr(expr.right),
            )
        if isinstance(expr, FilterOr):
            return FilterOr(
                left=DeclarativeMemory._mangle_filter_expr(expr.left),
                right=DeclarativeMemory._mangle_filter_expr(expr.right),
            )
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")
