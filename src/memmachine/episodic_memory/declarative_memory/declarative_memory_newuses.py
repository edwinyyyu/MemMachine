"""Declarative memory system for storing and retrieving episodic memory."""

import asyncio
import math
import datetime
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import cast
from uuid import uuid4

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
    hyde_language_model: InstanceOf[LanguageModel] = Field(
        ...,
        description="LanguageModel instance for generating hypothetical answers",
    )
    language_model: InstanceOf[LanguageModel] = Field(
        ...,
        description="LanguageModel instance for relevance filtering",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
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
        self._hyde_language_model = params.hyde_language_model
        self._language_model = params.language_model
        self._reranker = params.reranker

        self._message_sentence_chunking = params.message_sentence_chunking

        self._episode_collection = f"Episode_{session_id}"
        self._derivative_collection = f"Derivative_{session_id}"

        self._derived_from_relation = f"DERIVED_FROM_{session_id}"

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

        # hyde_query = await self._generate_hyde_query(query)
        # print(f"Original query: {query}\nHyDE query: {hyde_query}")

        query_embedding = (
            await self._embedder.search_embed(
                # [hyde_query],
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
            limit=min(5 * max_num_episodes, 200),
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

        if expand_context > 0:
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
        else:
            episode_contexts = [[ep] for ep in nuclear_episodes]

        # Deduplicate and sort all episodes chronologically.
        all_episodes = sorted(
            {ep for episode_context in episode_contexts for ep in episode_context},
            key=lambda ep: (ep.timestamp, ep.uid),
        )

        num_episodes = len(all_episodes)
        BATCH_SIZE = 20

        if num_episodes <= BATCH_SIZE:
            # Small pool: single LLM call.
            useless_indexes = await self._identify_useless_episodes(
                query, all_episodes
            )
            useless_uids = {str(all_episodes[i].uid) for i in useless_indexes}
        else:
            batches = DeclarativeMemory._build_circular_batches(
                all_episodes, BATCH_SIZE
            )

            # Evaluate ALL batches in parallel.
            batch_results = await asyncio.gather(
                *[
                    self._identify_useless_episodes(query, batch)
                    for _, batch in batches
                ]
            )

            # Consensus: count useless votes per episode.
            useless_votes: dict[str, int] = defaultdict(int)
            for (indexes, _), useless_batch_indexes in zip(
                batches, batch_results
            ):
                for batch_idx in useless_batch_indexes:
                    original_idx = indexes[batch_idx]
                    useless_votes[str(all_episodes[original_idx].uid)] += 1

            # Only remove if useless in BOTH batches containing it.
            useless_uids = {
                uid for uid, votes in useless_votes.items() if votes > 1
            }

        # Filter to useful episodes.
        useful = [ep for ep in all_episodes if str(ep.uid) not in useless_uids]

        # Rerank by relevance and take top max_num_episodes.
        scores = await self._score_episode_contexts(
            query, [[ep] for ep in useful]
        )
        scored = sorted(
            zip(scores, useful), key=lambda pair: pair[0], reverse=True
        )
        top = scored[:max_num_episodes]

        # Return sorted by timestamp.
        return sorted(top, key=lambda pair: (pair[1].timestamp, pair[1].uid))

    async def _generate_hyde_query(self, query: str) -> str:
        system_prompt = (
            "You are rewriting user inputs into retrieval cues for a semantic search system. The goal is to produce output that maximally overlaps with the actual content stored in documents, notes, chat logs, or records — because that is what the retrieval system will be searching against. The stored content may include source prefixes (e.g., \"User:\", \"Assistant:\", \"[Name]:\") as part of the indexed text.\n"
            "\n"
            "Ask yourself: \"What was actually written down or said at the time this happened?\" Your output should be as close to that original recorded content as possible. Your output can be a statement, a question, a phrase, or a keyword — whatever is most likely to resemble the stored content. Ensure the cue retains enough semantic context for the embedding to be meaningful — if stripping or converting words would make the cue ambiguous or too short to distinguish from unrelated content, keep the key terms that anchor the meaning.\n"
            "\n"
            "## Rules\n"
            "\n"
            "1. **Always rewrite question inputs. If the input is a question, the output must not be that same question.**\n"
            "  - **Most inputs are questions about facts or events.** Convert these into the declarative statement that would appear in content describing that fact or event. This is the default behavior. Strip the question structure entirely and rephrase as a statement — even if the input includes a source prefix.\n"
            "    - \"Where did I go on vacation?\" → \"I went on vacation to\"\n"
            "    - \"How many times have I gone on a hike recently?\" → \"I went on a hike\"\n"
            "    - \"What car do I drive?\" → \"I drive a car\"\n"
            "    - \"User: Who taught me to play guitar?\" → \"User: [someone] taught me to play guitar\"\n"
            "    - \"User: How long did Alice stay in London?\" → \"Alice stayed in London for [time]\"\n"
            "  - **If the input is a request for recommendations, suggestions, or advice,** the useful stored content is the user's related preferences, history, or interests — not the request itself. Rewrite the cue to target that underlying context.\n"
            "    - \"User: recommend me a book\" → \"User: I like reading [genre]\"\n"
            "    - \"User: what should I eat tonight?\" → \"User: I like to eat [cuisine/dish]\"\n"
            "  - **If the input is a pure knowledge question directed at the assistant** (e.g., asking for an explanation, definition, or general information), the source label is irrelevant — drop it and output only the topical content.\n"
            "    - \"User: Can you explain concurrency?\" → \"concurrency\"\n"
            "    - \"User: What is the capital of France?\" → \"capital of France\"\n"
            "  - **If the input asks about something the assistant previously said or explained,** the stored content is from the Assistant. Replace the source label with \"Assistant:\" and extract the topic, preserving all specific details mentioned.\n"
            "    - \"User: What was that Italian restaurant you suggested near downtown?\" → \"Assistant: Italian restaurant near downtown\"\n"
            "    - \"User: Can you remind me which Python framework you recommended for web scraping?\" → \"Assistant: Python framework for web scraping\"\n"
            "    - \"User: What did you say about the side effects of melatonin?\" → \"Assistant: side effects of melatonin\"\n"
            "  - **Rarely, the input asks about something that was itself previously said or written** — meaning the stored content is literally a past utterance or message. Only in this case, extract that utterance directly. Look for explicit signals like \"asked,\" \"said,\" \"wrote,\" or \"searched for\" combined with a referenced phrase or question.\n"
            "    - \"How many times have I asked what the weather is like?\" → \"What's the weather like?\"\n"
            "  - **If the input contains multiple retrieval targets** (e.g., comparisons, conjunctions, or multi-part questions), include all targets in the cue. Do not reduce to a single target — the stored content relevant to each part may be in different places, and retrieving them individually may lose the relationship between them.\n"
            "    - \"User: Do I prefer coffee or tea?\" → \"User: I prefer coffee or tea\" (keep both so the comparison can be resolved)\n"
            "    - \"User: What did I eat for breakfast and where did I go after?\" → \"User: I ate breakfast and I went to\"\n"
            "\n"
            "2. **Source labels and named references.** If the input has an explicit source prefix (e.g., \"User:\", \"Alice:\") and the question is about the speaker themselves, keep the source label on the cue. When the question is about a different named person mentioned *inside* the question, drop the source label entirely — the name in the content itself provides the semantic anchor. When the question asks about what the assistant said, replace the source label with \"Assistant:\". Do not infer or add source labels beyond these cases. If there is no explicit source in the input, do not add any label.\n"
            "    - \"Alice: Where did I go on vacation?\" → \"Alice: I went on vacation to\" (Alice asking about herself — keep label)\n"
            "    - \"User: Who taught me to play guitar?\" → \"User: [someone] taught me to play guitar\" (User asking about themselves — keep label)\n"
            "    - \"User: How long did Alice stay in London?\" → \"Alice stayed in London for [time]\" (about a third party — drop source label)\n"
            "    - \"Bob: How long did Alice stay in London?\" → \"Alice stayed in London for [time]\" (about a third party — drop source label)\n"
            "    - \"User: How many times have I asked what the weather is like?\" → \"User: What's the weather like?\"\n"
            "    - \"How many times have I asked what the weather is like?\" → \"What's the weather like?\" (no explicit source)\n"
            "    - \"Python concurrency patterns\" → \"Python concurrency patterns\" (no source evident)\n"
            "\n"
            "3. **Strip all meta-framing.** Remove quantifiers (\"how many times\"), frequency language (\"how often\"), analytical framing (\"frequency of,\" \"number of,\" \"tell me about,\" \"can you tell me\"), ordering or sorting instructions (\"in order,\" \"starting from the earliest,\" \"ranked by\"), and conversational wrappers (\"do you know,\" \"I was wondering,\" \"can you remind me\"). What remains should be the underlying content or event.\n"
            "\n"
            "4. **Convert temporal and relative time expressions into context-appropriate phrasing or omit them.** Raw temporal references like \"last night,\" \"yesterday,\" \"recently\" do not appear in content written at the time of the event. Infer what natural phrasing the original content would use (e.g., a question about eating \"last night\" maps to content about eating \"for dinner\"). If no natural replacement exists, omit the temporal reference. Never insert specific dates.\n"
            "\n"
            "5. **Preserve all specific details present in the input. Do not invent new ones.** Keep every concrete entity, name, or term the user provided. Never fabricate specific entities and never replace specific terms the user already gave you with vaguer ones. If a detail is missing from the input, leave it out of the cue.\n"
            "\n"
            "6. **Preserve keywords and technical terms as-is.** If the input is already a short phrase or technical query, return it unchanged or nearly unchanged.\n"
            "\n"
            "7. **Do not over-correct bad input.** If the input is vague or poorly formed, produce the closest natural phrasing you can without guessing missing context.\n"
            "\n"
            "8. **Return only the cue.** No commentary or explanations."
        )
        user_prompt = (
            "## Input\n"
            "\n"
            f"{query}"
        )

        hyde_query, _ = await self._hyde_language_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return hyde_query.strip()

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
    def _build_circular_batches(
        episodes: list[Episode],
        batch_size: int,
    ) -> list[tuple[list[int], list[Episode]]]:
        """Build overlapping circular batches for parallel LLM evaluation.

        Each episode appears in exactly 2 batches. All batch sizes are within
        1 of each other. This is achieved by dividing episodes into B balanced
        segments, then forming each batch as the union of two adjacent segments.

        Args:
            episodes: Episodes sorted chronologically.
            batch_size: Target number of episodes per batch.

        Returns:
            List of (indexes, batch) tuples where indexes maps batch-local
            positions back to episodes positions.

        """
        n = len(episodes)
        num_batches = max(2, math.ceil(2 * n / batch_size))
        num_segments = num_batches  # each batch = 2 adjacent segments

        # Balanced partition into num_segments segments.
        # Segment k spans episodes[seg_start[k] : seg_start[k+1]].
        seg_start = [(k * n) // num_segments for k in range(num_segments + 1)]

        # Each batch k = segment k ∪ segment (k+1) % num_segments.
        batches: list[tuple[list[int], list[Episode]]] = []
        for k in range(num_batches):
            next_k = (k + 1) % num_segments
            indexes = list(range(seg_start[k], seg_start[k + 1]))
            indexes += list(range(seg_start[next_k], seg_start[next_k + 1]))
            # Sort for chronological order (only matters for wrap-around batch).
            indexes.sort()
            batch = [episodes[i] for i in indexes]
            batches.append((indexes, batch))

        return batches

    async def _identify_useless_episodes(
        self,
        query: str,
        episodes: list[Episode],
    ) -> set[int]:
        """Identify episodes that are not useful for answering the query."""

        class UselessEpisodes(BaseModel):
            useless_episode_indexes: list[int] = Field(
                description="Indexes of episodes that are not useful for answering the query",
            )

        indexed_context = DeclarativeMemory.indexed_string_from_episode_context(episodes)

        system_prompt = (
            "You are a relevance judge. Given a query and a list of indexed episodes, "
            "identify which episodes are NOT useful for answering the query.\n\n"
            "An episode is useless unless it contains information that would be "
            "directly cited or referenced when answering the query, or removing "
            "it would make a cited episode ambiguous or uninterpretable.\n\n"
            "Return the indexes of useless episodes. "
            "If all episodes are useful, return an empty list."
        )
        user_prompt = f"Query: {query}\n\nEpisodes:\n{indexed_context}"

        result = await self._language_model.generate_parsed_response(
            output_format=UselessEpisodes,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if result is None:
            return set()

        return {i for i in result.useless_episode_indexes if 0 <= i < len(episodes)}

    async def _identify_useless_episodes_binary(
        self,
        query: str,
        episodes: list[Episode],
    ) -> set[int]:
        """Identify episodes that are not useful for answering the query.

        Uses a binary 0/1 output format: one value per episode where
        0 = useful, 1 = useless.
        """
        indexed_context = DeclarativeMemory.indexed_string_from_episode_context(episodes)
        num_episodes = len(episodes)

        system_prompt = (
            "You are a relevance judge. Given a query and a list of indexed episodes, "
            "identify which episodes are NOT useful for answering the query.\n\n"
            "An episode is useless unless it contains information that would be "
            "directly cited or referenced when answering the query, or removing "
            "it would make a cited episode ambiguous or uninterpretable.\n\n"
            "For each episode, output 0 if it is useful or 1 if it is useless. "
            f"Output exactly {num_episodes} values (one per episode, in order), "
            "separated by spaces, on a single line. No other text."
        )
        user_prompt = f"Query: {query}\n\nEpisodes:\n{indexed_context}"

        response, _ = await self._language_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        tokens = response.strip().split()
        if len(tokens) != num_episodes:
            logger.warning(
                "Binary useless-episodes response had %d tokens, expected %d. "
                "Falling back to empty set.",
                len(tokens),
                num_episodes,
            )
            return set()

        return {i for i, token in enumerate(tokens) if token == "1"}

    async def _identify_useless_episodes_checkbox(
        self,
        query: str,
        episodes: list[Episode],
    ) -> set[int]:
        """Identify episodes that are not useful for answering the query.

        Uses a checkbox output format: the LLM receives unchecked checkboxes
        and checks the ones corresponding to useless episodes.
        """
        num_episodes = len(episodes)
        checkbox_lines = []
        for episode in episodes:
            context_date = DeclarativeMemory._format_date(episode.timestamp.date())
            context_time = DeclarativeMemory._format_time(episode.timestamp.time())
            content = json.dumps(episode.content)
            checkbox_lines.append(
                f"- [ ] [{context_date} at {context_time}] {episode.source}: {content}"
            )
        checkboxes = "\n".join(checkbox_lines)

        system_prompt = (
            "You are a relevance judge. Given a query and a checklist of episodes, "
            "identify which episodes are NOT useful for answering the query by "
            "checking their checkboxes.\n\n"
            "An episode is useless unless it contains information that would be "
            "directly cited or referenced when answering the query, or removing "
            "it would make a cited episode ambiguous or uninterpretable.\n\n"
            "Return the full checklist with useless episodes checked (replace "
            "\"[ ]\" with \"[x]\"). Keep useful episodes unchecked. "
            f"Output exactly {num_episodes} checkbox lines. No other text."
        )
        user_prompt = f"Query: {query}\n\nEpisodes:\n{checkboxes}"

        response, _ = await self._language_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        lines = [
            line for line in response.strip().split("\n")
            if line.strip().startswith("- [")
        ]
        if len(lines) != num_episodes:
            logger.warning(
                "Checkbox useless-episodes response had %d checkbox lines, "
                "expected %d. Falling back to empty set.",
                len(lines),
                num_episodes,
            )
            return set()

        useless: set[int] = set()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
                useless.add(i)
            elif not stripped.startswith("- [ ]"):
                logger.warning(
                    "Unexpected checkbox format at line %d: %r. "
                    "Falling back to empty set.",
                    i,
                    stripped[:40],
                )
                return set()

        return useless

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
    def indexed_string_from_episode_context(episode_context: Iterable[Episode]) -> str:
        """Format episode context as an indexed string."""
        context_string = ""

        for index, episode in enumerate(episode_context):
            context_date = DeclarativeMemory._format_date(
                episode.timestamp.date(),
            )
            context_time = DeclarativeMemory._format_time(
                episode.timestamp.time(),
            )
            context_string += f"[{index}] [{context_date} at {context_time}] {episode.source}: {json.dumps(episode.content)}\n"

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

        derived_derivative_nodes = [
            derivative_node
            for derivative_nodes in await asyncio.gather(
                *search_derived_derivative_nodes_tasks,
            )
            for derivative_node in derivative_nodes
        ]

        delete_nodes_tasks = [
            self._vector_graph_store.delete_nodes(
                collection=self._episode_collection,
                node_uids=uids,
            ),
            self._vector_graph_store.delete_nodes(
                collection=self._derivative_collection,
                node_uids=[
                    derivative_node.uid for derivative_node in derived_derivative_nodes
                ],
            ),
        ]

        await asyncio.gather(*delete_nodes_tasks)

    @staticmethod
    def _unify_scored_anchored_episode_contexts(
        scored_anchored_episode_contexts: Iterable[
            tuple[float, Episode, Iterable[Episode]]
        ],
        max_num_episodes: int,
        excluded_uids: set[str] | None = None,
    ) -> list[tuple[float, Episode]]:
        """Unify anchored episode contexts into a single list within the limit."""
        episode_scores: dict[Episode, float] = {}

        for score, nuclear_episode, context in scored_anchored_episode_contexts:
            context = [
                ep for ep in context
                if excluded_uids is None or str(ep.uid) not in excluded_uids
            ]

            if not context:
                continue

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
                try:
                    nuclear_index = context.index(nuclear_episode)
                except ValueError:
                    nuclear_index = 0

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
