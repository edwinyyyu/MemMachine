"""ASGI short-circuit for POST /api/v2/memories/search.

Serves the happy path without the framework stack: read body, validate the
spec, run the search, send pre-serialized JSON. Anything unusual - body
parse or spec validation failure, a service exception, a missing
application state - is NOT handled here: the buffered request is replayed
through the full FastAPI application so every error response stays
byte-canonical. Search is read-only, so a replay is safe.
"""

import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from fastapi import FastAPI
from memmachine_common.api.spec import SearchMemoriesSpec

from memmachine_server.server.api_v2.service import (
    _search_target_memories_response,
)

logger = logging.getLogger(__name__)

_Message = MutableMapping[str, Any]
_Receive = Callable[[], Awaitable[_Message]]
_Send = Callable[[_Message], Awaitable[None]]

def _all_memory_types() -> list[Any]:
    # Imported at call time to avoid an import cycle with the main package.
    from memmachine_server.main.memmachine import ALL_MEMORY_TYPES

    return ALL_MEMORY_TYPES


async def _read_body(receive: _Receive) -> bytes | None:
    """Drain the request body; None if the client disconnected."""
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


def _replay(body: bytes) -> _Receive:
    """A receive callable that replays the buffered body once."""
    sent = False

    async def receive() -> _Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def fast_search_asgi(
    app: FastAPI,
    scope: MutableMapping[str, Any],
    receive: _Receive,
    send: _Send,
) -> bool:
    """Serve the search request directly; always sends some response.

    Returns True in every case: either the fast path answered, or the
    buffered request was replayed through the full application (canonical
    error handling), or the client disconnected.
    """
    body = await _read_body(receive)
    if body is None:
        return True  # client went away; nothing to serve

    try:
        spec = SearchMemoriesSpec.model_validate_json(body)
        memmachine = app.state.mem_machine
        target_memories = spec.types or _all_memory_types()
        response = await _search_target_memories_response(
            target_memories=target_memories, spec=spec, memmachine=memmachine
        )
    except Exception:  # canonical handling via full-app replay
        logger.debug("fast search fell back to the full app", exc_info=True)
        await FastAPI.__call__(app, scope, _replay(body), send)
        return True

    await response(scope, _replay(b""), send)
    return True
