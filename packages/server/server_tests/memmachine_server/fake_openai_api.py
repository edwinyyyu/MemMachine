"""In-process fake of the OpenAI HTTP API for the integration tests.

Serves ``POST /v1/embeddings`` and ``POST /v1/responses`` on a loopback
port so the OpenAI-backed integration tests run without network access
or credentials. Requests are validated the way the real API validates
them (input shape, empty strings, the per-request input limit), so the
embedder's chunking and clustering are still exercised.

Embeddings are a feature-hashed bag of words: equal texts get equal
vectors, texts sharing words get positive cosine similarity, unrelated
texts get similarity near zero. Vectors are deterministic across
processes.

Structured Responses API output is generated from the request's JSON
schema: every property is filled, arrays get one element, ``anyOf`` and
``enum`` take their first alternative, and strings are the property
name. Unstructured output is a fixed sentence.
"""

import base64
import hashlib
import json
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_DEFAULT_DIMENSIONS = 1536
_MAX_INPUTS_PER_REQUEST = 2048
_UNSTRUCTURED_OUTPUT_TEXT = "This is a fake response."

_WORD_PATTERN = re.compile(r"\w+")
_SCALAR_INSTANCES: dict[str, Any] = {
    "integer": 0,
    "number": 0.0,
    "boolean": False,
    "null": None,
}


@contextmanager
def serve() -> Iterator[str]:
    """Run the fake API on an ephemeral loopback port; yields its ``/v1`` base URL."""
    server = uvicorn.Server(
        uvicorn.Config(_create_app(), host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("fake OpenAI API server failed to start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join()


def _create_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if (
            not isinstance(inputs, list)
            or not inputs
            or not all(isinstance(text, str) for text in inputs)
        ):
            return _error(400, "'$.input' is invalid.", "input")
        if "" in inputs:
            return _error(400, "'$.input' is invalid: empty string.", "input")
        if len(inputs) > _MAX_INPUTS_PER_REQUEST:
            return _error(
                400,
                f"'$.input' is invalid: at most {_MAX_INPUTS_PER_REQUEST} inputs per request.",
                "input",
            )

        dimensions = body.get("dimensions", _DEFAULT_DIMENSIONS)
        vectors = [_fake_embedding(text, dimensions) for text in inputs]
        if body.get("encoding_format") == "base64":
            encoded = [
                base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode()
                for vector in vectors
            ]
        else:
            encoded = vectors
        tokens = sum(_token_estimate(text) for text in inputs)
        return JSONResponse(
            {
                "object": "list",
                "model": body.get("model"),
                "data": [
                    {"object": "embedding", "index": index, "embedding": embedding}
                    for index, embedding in enumerate(encoded)
                ],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )

    @app.post("/v1/responses")
    async def responses(request: Request) -> JSONResponse:
        body = await request.json()
        text_format = (body.get("text") or {}).get("format") or {}
        if text_format.get("type") == "json_schema":
            schema = text_format["schema"]
            text = json.dumps(_instance_of_schema(schema, schema, text_format["name"]))
        else:
            text = _UNSTRUCTURED_OUTPUT_TEXT

        input_tokens = sum(
            _token_estimate(str(message.get("content", "")))
            for message in body.get("input", [])
        )
        output_tokens = _token_estimate(text)
        return JSONResponse(
            {
                "id": "resp_fake",
                "object": "response",
                "created_at": time.time(),
                "status": "completed",
                "model": body.get("model"),
                "output": [
                    {
                        "id": "msg_fake",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "text": body.get("text", {"format": {"type": "text"}}),
                "usage": {
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        )

    return app


def _fake_embedding(text: str, dimensions: int) -> list[float]:
    vector = np.zeros(dimensions, dtype=np.float64)
    for word in _WORD_PATTERN.findall(text.lower()) or [text]:
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    return (vector / np.linalg.norm(vector)).tolist()


def _instance_of_schema(schema: dict[str, Any], root: dict[str, Any], name: str) -> Any:
    if "$ref" in schema:
        return _instance_of_schema(_resolve_ref(schema["$ref"], root), root, name)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return _instance_of_schema(schema["anyOf"][0], root, name)
    schema_type = schema.get("type")
    if schema_type == "object":
        return {
            key: _instance_of_schema(value, root, key)
            for key, value in schema.get("properties", {}).items()
        }
    if schema_type == "array":
        return [_instance_of_schema(schema["items"], root, name)]
    if schema_type == "string":
        return name
    if schema_type in _SCALAR_INSTANCES:
        return _SCALAR_INSTANCES[schema_type]
    raise ValueError(f"Unsupported JSON schema: {schema}")


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    target = root
    for part in ref.removeprefix("#/").split("/"):
        target = target[part]
    return target


def _error(status_code: int, message: str, param: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )


def _token_estimate(text: str) -> int:
    return len(text) // 4 + 1
