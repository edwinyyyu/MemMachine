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
from memmachine_common.api.spec import AddMemoriesSpec, SearchMemoriesSpec
from starlette.responses import PlainTextResponse, Response

from memmachine_server.common import fast_json
from memmachine_server.server.api_v2.service import (
    _add_messages_to,
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


async def fast_add_asgi(
    app: FastAPI,
    scope: MutableMapping[str, Any],
    receive: _Receive,
    send: _Send,
) -> bool:
    """Serve POST /api/v2/memories directly; always sends some response.

    Pre-service failures (parse/validation) replay the buffered request
    through the full application, matching canonical 422s. Unlike search,
    an add has side effects, so a SERVICE exception must NOT replay (it
    would re-execute the write); the canonical route has no handler for it
    either (it surfaces as the framework's plain 500), which is what is
    sent here.
    """
    body = await _read_body(receive)
    if body is None:
        return True

    try:
        spec = AddMemoriesSpec.model_validate_json(body)
        memmachine = app.state.mem_machine
        target_memories = spec.types or _all_memory_types()
    except Exception:  # parse/validation -> canonical 422 via replay
        logger.debug("fast add fell back to the full app", exc_info=True)
        await FastAPI.__call__(app, scope, _replay(body), send)
        return True

    try:
        results = await _add_messages_to(
            target_memories=target_memories, spec=spec, memmachine=memmachine
        )
    except Exception:  # side effects may exist; do NOT replay
        logger.exception("add_memories failed")
        await PlainTextResponse("Internal Server Error", status_code=500)(
            scope, _replay(b""), send
        )
        return True

    payload = fast_json.dumps({"results": [{"uid": result.uid} for result in results]})
    await Response(content=payload, media_type="application/json")(
        scope, _replay(b""), send
    )
    return True


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
