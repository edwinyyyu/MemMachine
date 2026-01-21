"""Common utility functions."""

import asyncio
import functools
import math
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Any, ParamSpec, TypeVar, Literal

import cvxpy as cp
import numpy as np
from kneed import KneeLocator

from .data_types import SimilarityMetric

T = TypeVar("T")
P = ParamSpec("P")


async def async_with[T](
    async_context_manager: AbstractAsyncContextManager,
    awaitable: Awaitable[T],
) -> T:
    """
    Use an async context manager while awaiting a coroutine.

    Args:
        async_context_manager (AbstractAsyncContextManager):
            The async context manager to use.
        awaitable (Awaitable):
            The awaitable to execute within the context.

    Returns:
        Any:
            The result of the awaitable.

    """
    async with async_context_manager:
        return await awaitable


def async_locked[**P, T](func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    Ensure that a coroutine function is executed with a shared lock.

    The lock is shared across all invocations of the decorated coroutine function.
    """
    lock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        async with lock:
            return await func(*args, **kwargs)

    return wrapper


def compute_similarity(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    similarity_metric: SimilarityMetric | None = None,
) -> list[float]:
    """
    Compute similarity scores between a query embedding and candidate embeddings.

    Args:
        query_embedding (list[float]): The embedding of the query.
        candidate_embeddings (list[list[float]]): A list of candidate embeddings to compare against.
        similarity_metric (SimilarityMetric | None): The similarity metric to use (default: None).

    Returns:
        list[float]: A list of similarity scores for each candidate embedding.

    """
    if not candidate_embeddings:
        return []

    query_embedding_np = np.array(query_embedding)
    candidate_embeddings_np = np.array(candidate_embeddings)

    match similarity_metric:
        case SimilarityMetric.COSINE:
            magnitude_products = np.linalg.norm(
                candidate_embeddings_np,
                axis=-1,
            ) * np.linalg.norm(query_embedding_np)
            magnitude_products[magnitude_products == 0] = float("inf")

            scores = (
                1
                + np.dot(candidate_embeddings_np, query_embedding_np)
                / magnitude_products
            ) / 2
        case SimilarityMetric.DOT:
            scores = np.dot(candidate_embeddings_np, query_embedding_np)
        case SimilarityMetric.EUCLIDEAN:
            scores = -np.linalg.norm(
                candidate_embeddings_np - query_embedding_np,
                axis=-1,
            )
        case SimilarityMetric.MANHATTAN:
            scores = -np.sum(
                np.abs(candidate_embeddings_np - query_embedding_np),
                axis=-1,
            )
        case _:
            # Default to cosine similarity.
            magnitude_products = np.linalg.norm(
                candidate_embeddings_np,
                axis=-1,
            ) * np.linalg.norm(query_embedding_np)
            magnitude_products[magnitude_products == 0] = float("inf")

            scores = (
                np.dot(candidate_embeddings_np, query_embedding_np) / magnitude_products
            )

    return scores.astype(float).tolist()


def rank_by_mmr(
    candidate_relevances: list[float],
    pairwise_candidate_similarities: list[list[float]],
    lambda_param: float = 0.5,
) -> list[int]:
    """
    Rank candidates using Maximal Marginal Relevance (MMR).

    Args:
        candidate_relevances (list[float]): A list of relevance scores for each candidate.
        pairwise_candidate_similarities (list[list[float]]): A matrix of pairwise similarities between candidates.
        lambda_param (float): The trade-off parameter between relevance and diversity (default: 0.5).

    """
    selected_indexes: list[int] = []
    remaining_indexes = set(range(len(candidate_relevances)))

    while remaining_indexes:
        if not selected_indexes:
            # Select the candidate with the highest relevance to the query.
            best_index = max(
                remaining_indexes,
                key=lambda index: candidate_relevances[index],
            )

        else:
            # Select the candidate that maximizes the MMR score.
            best_index = max(
                remaining_indexes,
                key=lambda index: (
                    lambda_param * candidate_relevances[index]
                    - (1 - lambda_param)
                    * max(
                        pairwise_candidate_similarities[index][selected_index]
                        for selected_index in selected_indexes
                    )
                ),
            )

        selected_indexes.append(best_index)
        remaining_indexes.remove(best_index)

    return selected_indexes


def unflatten_like[T](
    flat_list: list[T],
    template_list: list[list[Any]],
) -> list[list[T]]:
    """
    Unflatten a flat list into a nested list structure based on a template.

    Args:
        flat_list (list): The flat list to unflatten.
        template_list (list): The template nested list structure.

    Returns:
        list: The unflattened nested list.

    """
    unflattened_list = []
    current_index = 0

    for template in template_list:
        unflattened_list.append(
            flat_list[current_index : current_index + len(template)]
        )
        current_index += len(template)

    return unflattened_list


def max_logit_gap(
    scores: Iterable[float],
) -> int:
    """
    Return the number of scores before the maximum logit gap between consecutive sorted scores between 0 and 1.
    """
    scores_np = np.array(scores)
    logits = np.log(scores_np / (1 - scores_np + 1e-16) + 1e-16)
    logit_gaps = np.abs(np.diff(logits))
    return int(np.argmax(logit_gaps)) + 1


def second_max_logit_gap(
    scores: Iterable[float],
) -> int:
    """
    Return the number of scores before the second maximum logit gap between consecutive sorted scores between 0 and 1.
    """
    scores_np = np.array(scores)
    logits = np.log(scores_np / (1 - scores_np + 1e-16) + 1e-16)
    logit_gaps = np.abs(np.diff(logits))
    first_max_logit_gap = int(np.argmax(logit_gaps)) + 1
    return (
        int(np.argmax(logit_gaps[first_max_logit_gap:])) + first_max_logit_gap + 1
        if len(logit_gaps) > first_max_logit_gap
        else first_max_logit_gap
    )


def std_dev_cutoff(
    scores: Iterable[float],
) -> int:
    """
    Return the number of scores above one standard deviation above the mean.
    """
    scores_np = np.array(scores)
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1)
    return int(np.sum(scores_np > mean + std_dev))


def kneedle_cutoff(
    scores: Iterable[float],
    concavity: Literal["convex", "concave"] = "convex",
    direction: Literal["decreasing", "increasing"] = "decreasing",
) -> int:
    """
    Return the number of scores above the kneedle cutoff.

    Args:
        scores (Iterable[float]): The scores to evaluate.
        concavity (Literal["convex", "concave"]): The concavity of the curve (default: "convex").
        direction (Literal["increasing", "decreasing"]): The direction of the curve (default: "decreasing").

    Returns:
        int: The number of scores above the kneedle cutoff.

    """
    scores_np = np.array(scores)
    x = np.arange(1, len(scores_np) + 1)
    knee_locator = KneeLocator(
        x,
        scores_np,
        S=1.0,
        curve=concavity,
        direction=direction,
    )
    if knee_locator.knee is None:
        return len(scores_np)
    return int(knee_locator.knee)


def kneedle_cutoff_fit(
    scores: Iterable[float],
    concavity: Literal["convex", "concave"] = "convex",
    direction: Literal["decreasing", "increasing"] = "decreasing",
) -> int:
    """
    Return the number of scores above the kneedle cutoff.

    Args:
        scores (Iterable[float]): The scores to evaluate.
        concavity (Literal["convex", "concave"]): The concavity of the curve (default: "convex").
        direction (Literal["increasing", "decreasing"]): The direction of the curve (default: "decreasing").

    Returns:
        int: The number of scores above the kneedle cutoff.

    """
    scores_np = np.array(scores)
    n = len(scores_np)
    x = np.arange(1, n + 1)

    f = cp.Variable(n)

    constraints = []
    for i in range(n - 1):
        if direction == "decreasing":
            constraints.append(f[i] >= f[i + 1])
        else:
            constraints.append(f[i] <= f[i + 1])

    for i in range(n - 2):
        if concavity == "convex":
            constraints.append(f[i] + f[i + 2] >= 2 * f[i + 1])
        else:
            constraints.append(f[i] + f[i + 2] <= 2 * f[i + 1])

    objective = cp.Minimize(cp.sum_squares(f - scores_np))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    fitted_scores = f.value
    knee_locator = KneeLocator(
        x,
        fitted_scores,
        S=1.0,
        curve=concavity,
        direction=direction,
    )
    if knee_locator.knee is None:
        return len(scores_np)
    return int(knee_locator.knee)


def merge_intersecting_sets[T](
    sets: Iterable[set[T]],
) -> list[set[T]]:
    """
    Merge intersecting sets from an iterable of sets.

    Args:
        sets (Iterable[set]): An iterable of sets to merge.

    Returns:
        list[set]: A list of merged sets.

    """
    parent = {}

    def find(item: T) -> T:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(item1: T, item2: T) -> None:
        root1 = find(item1)
        root2 = find(item2)
        if root1 != root2:
            parent[root1] = root2

    for s in sets:
        s = list(s)
        for i in range(1, len(s)):
            union(s[0], s[i])
        if len(s) == 1:
            find(s[0])

    clusters = defaultdict(set)
    for item in parent:
        root = find(item)
        clusters[root].add(item)

    return list(clusters.values())


def chunk_text(text: str, max_length: int) -> list[str]:
    """
    Chunk text into partitions not exceeding max_length.

    Args:
        text (str): The input text to chunk.
        max_length (int): The maximum length of each chunk.

    Returns:
        list[str]: A list of text chunks.

    """
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def chunk_text_balanced(text: str, max_length: int) -> list[str]:
    """
    Chunk text into balanced partitions not exceeding max_length.

    Args:
        text (str): The input text to chunk.
        max_length (int): The maximum length of each chunk.

    Returns:
        list[str]: A list of text chunks.

    """
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    if len(text) == 0:
        return []

    num_chunks = math.ceil(len(text) / max_length)
    chunk_size = math.ceil(len(text) / num_chunks)

    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def unflatten_like[T](
    flat_list: list[T],
    template_list: list[list[Any]],
) -> list[list[T]]:
    """
    Unflatten a flat list into a nested list structure based on a template.

    Args:
        flat_list (list): The flat list to unflatten.
        template_list (list): The template nested list structure.

    Returns:
        list: The unflattened nested list.

    """
    if not all(isinstance(template, list) for template in template_list):
        raise TypeError("All elements in template_list must be lists.")

    unflattened_list = []
    current_index = 0

    for template in template_list:
        unflattened_list.append(
            flat_list[current_index : current_index + len(template)]
        )
        current_index += len(template)

    if current_index != len(flat_list):
        raise ValueError("flat_list cannot be unflattened to match template_list.")

    return unflattened_list


def cluster_texts(
    texts: Iterable[str],
    max_num_texts_per_cluster: int,
    max_total_length_per_cluster: int,
) -> list[list[str]]:
    """
    Cluster texts based on maximum number of texts and total length of texts per cluster.

    Args:
        texts (Iterable[str]): The input texts to cluster.
        max_num_texts_per_cluster (int): The maximum number of texts per cluster.
        max_total_length_per_cluster (int): The maximum total length of texts per cluster.

    Returns:
        list[list[str]]: A list of text clusters.

    """
    if max_num_texts_per_cluster <= 0:
        raise ValueError("max_num_texts_per_cluster must be greater than 0")
    if max_total_length_per_cluster <= 0:
        raise ValueError("max_total_length_per_cluster must be greater than 0")

    clusters: list[list[str]] = []
    current_cluster: list[str] = []
    current_length = 0

    for text in texts:
        text_length = len(text)
        if text_length > max_total_length_per_cluster:
            raise ValueError(
                f"Text length {text_length} exceeds max_total_length_per_cluster {max_total_length_per_cluster}"
            )

        if (
            len(current_cluster) >= max_num_texts_per_cluster
            or current_length + text_length > max_total_length_per_cluster
        ):
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [text]
            current_length = text_length
        else:
            current_cluster.append(text)
            current_length += text_length

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def next_similarities(
    vectors: list[list[float]],
) -> list[list[tuple[float, list[float]]]]:
    """
    For each vector in the input list, compute and return a list of vectors
    that are most similar to it among the subsequent vectors in the list.

    Args:
        vectors (list[list[float]]): The input vectors.

    Returns:
        list[list[tuple[float, list[float]]]]:
            A list where each element corresponds to an input vector and contains
            a list of tuples of similarity scores and the corresponding similar vectors.
    """
    next_similarities: list[list[tuple[float, list[float]]]] = []
    for i, vector in enumerate(vectors):
        similarities = compute_similarity(
            vector,
            vectors[i + 1 :],
            similarity_metric=SimilarityMetric.COSINE,
        )
        similar_vectors = [
            (similarity, vectors[i + j + 1])
            for j, similarity in enumerate(similarities)
        ]
        sorted_similar_vectors = sorted(
            similar_vectors,
            key=lambda pair: pair[0],
            reverse=True,
        )
        next_similarities.append(sorted_similar_vectors)

    return next_similarities


def zero_random_n_entries(vector: list[float], n: int) -> list[float]:
    """
    Zero out n random entries in the input vector.

    Args:
        vector (list[float]): The input vector.
        n (int): The number of entries to zero out.

    Returns:
        list[float]: The modified vector with n entries set to zero.

    """
    if n < 0 or n > len(vector):
        raise ValueError("n must be between 0 and the length of the vector.")

    vector_np = np.array(vector)
    indexes_to_zero = np.random.choice(len(vector), size=n, replace=False)
    vector_np[indexes_to_zero] = 0.0

    return vector_np.astype(float).tolist()
