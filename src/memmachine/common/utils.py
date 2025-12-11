"""Common utility functions."""

import asyncio
import functools
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Any, ParamSpec, TypeVar

import numpy as np

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
                1 + np.dot(candidate_embeddings_np, query_embedding_np) / magnitude_products
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
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    similarity_metric: SimilarityMetric | None = None,
    lambda_param: float = 0.5,
) -> list[int]:
    """
    Rank candidate embeddings using Maximal Marginal Relevance (MMR).

    Args:
        query_embedding (list[float]): The embedding of the query.
        candidate_embeddings (list[list[float]]): A list of candidate embeddings to rank.
        similarity_metric (SimilarityMetric | None): The similarity metric to use (default: None).
        lambda_param (float): The trade-off parameter between relevance and diversity (default: 0.5).

    """
    query_similarities = compute_similarity(
        query_embedding,
        candidate_embeddings,
        similarity_metric,
    )

    pairwise_candidate_similarities = [
        compute_similarity(
            candidate_embedding,
            candidate_embeddings,
            similarity_metric,
        )
        for candidate_embedding in candidate_embeddings
    ]

    selected_indexes: list[int] = []
    remaining_indexes = set(range(len(candidate_embeddings)))

    while remaining_indexes:
        if not selected_indexes:
            # Select the candidate with the highest similarity to the query.
            best_index = max(
                remaining_indexes,
                key=lambda index: query_similarities[index],
            )

        else:
            # Select the candidate that maximizes the MMR score.
            best_index = max(
                remaining_indexes,
                key=lambda index: (
                    lambda_param * query_similarities[index]
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
    return int(np.argmax(logit_gaps[first_max_logit_gap:])) + first_max_logit_gap + 1 if len(logit_gaps) > first_max_logit_gap else first_max_logit_gap
