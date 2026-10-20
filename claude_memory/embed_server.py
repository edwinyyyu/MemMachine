"""embeddinggemma-300m through a local llama.cpp server, started on demand.

The server (``llama-server`` from ggml-org's llama.cpp, installed by
``embedding_server/setup.sh`` and launched by its ``run.sh``; see
``embedding_server/README.md``) runs the model on the Metal GPU with every buffer reserved at startup,
so its footprint stays at ~140-225 MB under load. Running sentence-transformers
in the daemon instead cost 2 GB at rest and grew to 7.4 GB, because torch's GPU
allocator keeps what the largest batch ever needed.

Like the daemon, the server is spawned detached by its first client and outlives
it. ``run.sh`` passes ``--sleep-idle-seconds``, so after an idle period the server
unloads the model (~97 MB left) and reloads it on the next request (~0.6s).
Binding the port is its single-instance guard: a second server exits at once.

Vectors match the sentence-transformers ones already in the stores (measured
2026-10-19 on 519 real texts: cosine >= 0.999997, identical top-10 for every
query) as long as this client reproduces what sentence-transformers does around
the model:

  - prepend the model's query/document prompt, which the server never applies;
  - cut input to 2048 tokens including BOS/EOS, which the server would reject.
"""

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from memmachine_server.common.data_types import (
    ExternalServiceAPIError,
    SimilarityMetric,
)
from memmachine_server.common.embedder import Embedder

# httpx logs every request at INFO, which would add a daemon.log line per embedding call.
logging.getLogger("httpx").setLevel(logging.WARNING)

EMBEDDINGGEMMA = "google/embeddinggemma-300m"
SERVER_URL = "http://127.0.0.1:9931"

# embeddinggemma's prompts, from its config_sentence_transformers.json.
_QUERY_PROMPT = "task: search result | query: "
_DOCUMENT_PROMPT = "title: none | text: "

# sentence-transformers' max_seq_length for embeddinggemma; it keeps BOS and EOS
# and truncates the text between them.
_MAX_TOKENS = 2048
_BOS, _EOS = 2, 1

# Inputs per HTTP request. The server spreads one request across its slots.
_REQUEST_INPUTS = 32
# run.sh's -np: the server embeds this many inputs at a time. Extra concurrent
# requests only queue inside it, where the wait counts against the client timeout.
_SERVER_SLOTS = 4
_START_TIMEOUT_SECONDS = 60.0
_POLL_INTERVAL_SECONDS = 0.25


class EmbedServerEmbedder(Embedder):
    """Embed with embeddinggemma-300m via the local server, spawning it if down."""

    def __init__(
        self,
        run_script: Path,
        url: str = SERVER_URL,
        log_path: Path | None = None,
        batch_size: int = _REQUEST_INPUTS,
    ) -> None:
        """Use the server at ``url``, started by ``run_script`` on its port if down.

        ``log_path`` defaults to ``logs/server.log`` next to ``run_script``.
        """
        super().__init__(batch_size=batch_size)
        self._run_script = run_script
        self._url = url
        self._log_path = log_path or run_script.parent / "logs" / "server.log"
        self._start_lock = asyncio.Lock()
        self._in_flight = asyncio.Semaphore(_SERVER_SLOTS)
        self._process: subprocess.Popen[Any] | None = None
        # Whether any server has answered since this client last spawned one.
        self._answered = False
        self._identity_checked = False

    async def _ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        del max_attempts  # _request waits out restarts and model loads itself
        return await self._embed([_DOCUMENT_PROMPT + str(item) for item in inputs])

    async def _search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        del max_attempts  # _request waits out restarts and model loads itself
        return await self._embed([_QUERY_PROMPT + str(query) for query in queries])

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(base_url=self._url, timeout=300) as client:
            if not self._identity_checked:
                await self._check_identity(client)
            for start in range(0, len(texts), _REQUEST_INPUTS):
                batch = [
                    await self._fit(client, text)
                    for text in texts[start : start + _REQUEST_INPUTS]
                ]
                async with self._in_flight:
                    response = await self._post(
                        client, "/v1/embeddings", {"input": batch}
                    )
                data = response["data"]
                vectors += [
                    item["embedding"] for item in sorted(data, key=lambda d: d["index"])
                ]
        return vectors

    async def _fit(self, client: httpx.AsyncClient, text: str) -> str | list[int]:
        """Return the text, or BOS + its first 2046 tokens + EOS if it is longer."""
        # A token covers at least one UTF-8 byte, so a text this short fits as is.
        if len(text.encode()) <= _MAX_TOKENS - 2:
            return text
        ids = (
            await self._post(
                client, "/tokenize", {"content": text, "add_special": False}
            )
        )["tokens"]
        if len(ids) <= _MAX_TOKENS - 2:
            return text
        # Token-ID input gets no BOS/EOS from the server, so add them here.
        return [_BOS, *ids[: _MAX_TOKENS - 2], _EOS]

    async def _post(
        self, client: httpx.AsyncClient, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        response = await self._request(client, "POST", path, body)
        if response.is_error:
            raise ExternalServiceAPIError(
                f"embedding server returned HTTP {response.status_code} for {path}: "
                f"{response.text[:200]}"
            )
        return response.json()

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Send one request, starting the server and waiting for it as needed."""
        deadline = time.monotonic() + _START_TIMEOUT_SECONDS
        while True:
            try:
                response = await client.request(method, path, json=body)
            except (httpx.ConnectError, httpx.RemoteProtocolError):
                # Not running, or it died mid-request: start it and send again.
                await self._start()
            except httpx.TimeoutException as error:
                # A slow or hung server is still running; spawning would not help.
                raise ExternalServiceAPIError(
                    f"embedding server at {self._url} timed out on {path}"
                ) from error
            else:
                # 503 means the model is still loading after a (re)start.
                if response.status_code != 503:
                    self._answered = True
                    return response
            if time.monotonic() > deadline:
                raise ExternalServiceAPIError(
                    f"embedding server at {self._url} did not answer within "
                    f"{_START_TIMEOUT_SECONDS:.0f}s (log: {self._log_path})"
                )
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _check_identity(self, client: httpx.AsyncClient) -> None:
        """Make sure the port is served by embeddinggemma, not some other program."""
        response = await self._request(client, "GET", "/v1/models")
        try:
            models = response.json().get("data", [])
            model_ids = [model.get("id", "") for model in models]
        except (ValueError, AttributeError):
            model_ids = []
        if not any("embeddinggemma" in model_id for model_id in model_ids):
            raise ExternalServiceAPIError(
                f"{self._url} is not the embeddinggemma server "
                f"(HTTP {response.status_code}, models {model_ids})"
            )
        self._identity_checked = True

    async def _start(self) -> None:
        """Spawn the server detached, unless the one this client spawned is still alive.

        A duplicate would exit when it fails to bind the port, but run.sh would first
        rotate the live server's log away.
        """
        async with self._start_lock:
            if self._process is not None:
                returncode = self._process.poll()  # also reaps it once it has exited
                if returncode is None:
                    return
                if not self._answered:
                    raise ExternalServiceAPIError(
                        f"embedding server exited with code {returncode} before "
                        f"answering (log: {self._log_path})"
                    )
            self._answered = False
            # Fixed argv, no shell. run.sh writes its own log when LOG is set and
            # listens on PORT.
            env = {
                **os.environ,
                "LOG": str(self._log_path),
                "PORT": str(httpx.URL(self._url).port),
            }
            self._process = await asyncio.to_thread(
                subprocess.Popen,
                [str(self._run_script)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    @property
    def model_id(self) -> str:
        return EMBEDDINGGEMMA

    @property
    def dimensions(self) -> int:
        return 768

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE
