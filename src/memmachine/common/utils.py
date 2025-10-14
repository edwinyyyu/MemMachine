"""
Common utility functions.
"""

import asyncio
import functools
from collections.abc import Awaitable, Generator
from contextlib import AbstractAsyncContextManager
from typing import Any, TypeVar

from .data_types import Nested


async def async_with(
    async_context_manager: AbstractAsyncContextManager,
    awaitable: Awaitable,
) -> Any:
    """
    Helper function to use an async context manager with an awaitable.

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


def async_locked(func):
    """
    Decorator to ensure that a coroutine function is executed with a lock.
    The lock is shared across all invocations of the decorated coroutine function.
    """
    lock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        async with lock:
            return await func(*args, **kwargs)

    return wrapper

T = TypeVar("T")

def get_nested_values(nested: Nested[T]) -> Generator[T, None, None]:
    """
    Recursively yield all values from a nested dict/list structure.
    """
    if isinstance(nested, dict):
        for value in nested.values():
            yield from get_nested_values(value)
    elif isinstance(nested, list):
        for item in nested:
            yield from get_nested_values(item)
    else:
        yield nested
