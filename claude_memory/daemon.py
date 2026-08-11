"""The single memory process and its IPC layer.

The daemon is the *only* process that loads the embedder, opens the stores, and
holds state. Everything else (the MCP server, the hooks) is a thin client that
connects over a Unix domain socket and exchanges one newline-delimited JSON
request/response. This module is both halves:

  - server:  ``MemoryService`` + ``run_service`` + ``main`` (run as
             ``python -m claude_memory.daemon``)
  - client:  ``call`` (connect, auto-spawn the daemon, backoff)
  - state:   per-(partition, session) novelty + the ingest high-water mark

Protocol (one JSON object per line over the socket):
    {"op":"ping"}                                              -> {"ok":true}
    {"op":"shutdown"}                                          -> {"ok":true}
    {"op":"search","partition","session_id","cue","limit","filters","use_context"}
                                                  -> {"ok":true,"result":{...}}
    {"op":"expand","partition","session_id","id","before","after","unit"}
                                                  -> {"ok":true,"result":{...}}
    {"op":"outline","partition","id","before","after"}
                                                  -> {"ok":true,"result":{...}}
    {"op":"demote","partition","memory_id","cue"}
    {"op":"annotate","partition","memory_id","note"}
                                                  -> {"ok":true,"result":{...}}

Compatibility runs one way only: an MCP client is a subprocess that lives as long as
its session, so a restarted daemon regularly serves clients running week-old code.
Adding a reply field is therefore safe (clients drop what they do not declare) while
RENAMING a request field is not — ``expand``/``outline`` still read the pre-unification
``seed`` alongside today's ``id`` for exactly that reason.
    {"op":"reflect","partition","session_id","transcript_path"}
                                                  -> {"ok":true,"memories":"..."}
    {"op":"ingest","partition","session_id","transcript_path"}
                                                  -> {"ok":true,"ingested":N}

Single instance enforced by an exclusive flock; idle-exits after
``CLAUDE_MEMORY_DAEMON_IDLE`` seconds (default 1800).
"""

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from claude_memory.daemon_client import (
    _SPAWN_MARKER,
    lock_path,
    socket_path,
)
from claude_memory.engine import (
    MemoryCore,
    MemoryStores,
    TextBlock,
    blend_context_cue,
    build_embedder,
    fold_running_context,
)
from claude_memory.transcript import (
    events_from_transcript,
    last_assistant_message_text,
    last_compaction_time,
)
from claude_memory.wire import (
    Hit,
    MemoryConfig,
    Source,
    format_memory_line,
    in_context_exclusion_filter,
    observe,
)

logger = logging.getLogger("claude_memory.daemon")

_IDLE_TIMEOUT = float(os.environ.get("CLAUDE_MEMORY_DAEMON_IDLE", "1800"))


# ==================================================================== hwm state


def _safe_name(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id) or "unknown"


def _hwm_path(config: MemoryConfig, session_id: str) -> Path:
    return config.state_dir / f"{_safe_name(session_id)}.json"


def read_high_water_mark(config: MemoryConfig, session_id: str) -> int:
    """Return the number of transcript lines already ingested for this session."""
    path = _hwm_path(config, session_id)
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("lines", 0))
    except (ValueError, OSError):
        return 0


def write_high_water_mark(config: MemoryConfig, session_id: str, lines: int) -> None:
    """Persist the high-water mark for this session."""
    config.ensure_dirs()
    _hwm_path(config, session_id).write_text(
        json.dumps({"lines": lines}), encoding="utf-8"
    )


# ======================================================================= server


class MemoryService:
    """Holds the shared stores/embedder and routes requests to per-partition cores."""

    def __init__(self, config: MemoryConfig) -> None:
        """Create the service (call ``start`` to load the embedder and stores)."""
        self._config = config
        self._stores: MemoryStores | None = None
        self._cores: dict[str, MemoryCore] = {}
        self._seen: dict[tuple[str, str], set[str]] = {}
        self._context_vectors: dict[tuple[str, str], np.ndarray] = {}
        self._latest_session: dict[str, str] = {}
        self._open_lock = asyncio.Lock()
        self.last_activity = time.monotonic()
        self.should_stop = asyncio.Event()

    async def start(self) -> None:
        embedder = build_embedder(self._config.embedding_model)
        self._stores = await MemoryStores.open(self._config, embedder)
        logger.info("memory daemon ready (embedder=%s)", self._config.embedding_model)

    async def stop(self) -> None:
        for core in self._cores.values():
            with contextlib.suppress(Exception):
                await core.close_partition()
        if self._stores is not None:
            with contextlib.suppress(Exception):
                await self._stores.aclose()

    async def _core(self, partition: str) -> MemoryCore:
        async with self._open_lock:
            if partition not in self._cores:
                if self._stores is None:
                    raise RuntimeError("service not started")
                self._cores[partition] = await MemoryCore.open_on(
                    self._stores, partition
                )
            return self._cores[partition]

    def _resolve_session(self, partition: str, session_id: str) -> str:
        if session_id:
            self._latest_session[partition] = session_id
        return session_id or self._latest_session.get(partition, "_default")

    def _seen_set(self, partition: str, session_id: str) -> set[str]:
        resolved = self._resolve_session(partition, session_id)
        return self._seen.setdefault((partition, resolved), set())

    async def _context_cue(
        self, core: MemoryCore, partition: str, session_id: str, cue: str
    ) -> list[float]:
        """Hybrid ambient cue from the per-session running context vector.

        Blends the session's decayed document-side running context with the
        current prompt's query embedding, then folds the prompt into the
        running context. Model replies fold in at capture time, so the
        context tracks where the conversation (both sides) is heading.
        """
        key = (partition, self._resolve_session(partition, session_id))
        embedder = core.stores.embedder
        previous = self._context_vectors.get(key)
        [query_vec] = await embedder.search_embed([cue])
        blended = blend_context_cue(previous, np.asarray(query_vec, dtype=float))
        [doc_vec] = await embedder.ingest_embed([cue])
        self._context_vectors[key] = fold_running_context(
            previous, np.asarray(doc_vec, dtype=float)
        )
        return blended.tolist()

    async def _fold_replies(
        self, core: MemoryCore, partition: str, session_id: str, events: list
    ) -> None:
        """Fold newly captured assistant replies into the running context.

        User prompts fold at ambient-search time; replies fold here, exactly
        once each (the capture tail is high-water-marked).
        """
        texts = [
            event.blocks[0].text
            for event in events
            if event.properties.get("source") == Source.ASSISTANT_MESSAGE
            and event.blocks
            and isinstance(event.blocks[0], TextBlock)
            and event.blocks[0].text.strip()
        ]
        if not texts:
            return
        key = (partition, self._resolve_session(partition, session_id))
        vectors = await core.stores.embedder.ingest_embed(texts)
        folded = self._context_vectors.get(key)
        for vector in vectors:
            folded = fold_running_context(folded, np.asarray(vector, dtype=float))
        if folded is not None:
            self._context_vectors[key] = folded

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route one request to the right operation and return a JSON-able reply."""
        self.last_activity = time.monotonic()
        op = request.get("op")
        if op == "ping":
            return {"ok": True}
        if op == "shutdown":
            self.should_stop.set()
            return {"ok": True}

        partition = request.get("partition") or self._config.partition
        session_id = request.get("session_id") or ""

        if op == "handles":
            core = await self._core(partition)
            return {
                "ok": True,
                "handles": await core.first_handles(request.get("sessions") or []),
            }

        if op in ("search", "expand", "outline"):
            return await self._recall(op, request, partition, session_id)

        if op in ("demote", "annotate"):
            memory_id = request.get("memory_id") or request.get("id")
            if not memory_id:
                return {"ok": False, "error": f"{op} requires a memory_id"}
            core = await self._core(partition)
            if op == "demote":
                result = await core.demote(memory_id, request.get("cue", ""))
                return {"ok": True, "result": asdict(result)}
            message = await core.annotate(memory_id, request.get("note", ""))
            return {"ok": True, "message": message}

        if op == "reflect":
            return await self._reflect(request, partition, session_id)

        if op == "ingest":
            return await self._ingest(request, partition, session_id)

        return {"ok": False, "error": f"unknown op {op!r}"}

    async def _recall(
        self, op: str, request: dict[str, Any], partition: str, session_id: str
    ) -> dict[str, Any]:
        """The three read paths: find a moment, read around one, see the shape."""
        core = await self._core(partition)
        if op == "search":
            cue = request.get("cue")
            if not cue:
                return {"ok": False, "error": "search requires a cue"}
            query_vector = None
            if request.get("use_context"):
                query_vector = await self._context_cue(core, partition, session_id, cue)
            result: Any = await core.search(
                cue,
                limit=request.get("limit", 8),
                within=request.get("within"),
                kinds=request.get("kinds"),
                since=request.get("since"),
                before=request.get("before"),
                # Internal only — the hooks' one clause the named parameters
                # cannot express. Nothing model-facing sets this.
                filter_spec=request.get("filters"),
                seen=self._seen_set(partition, session_id),
                query_vector=query_vector,
                session=self._resolve_session(partition, session_id),
            )
        elif op == "expand":
            # "seed" was this field's name before the addressing unification. A client
            # older than that change is still sending it and cannot be updated without
            # restarting its session, so the old name stays readable here.
            address = request.get("id") or request.get("seed")
            if not address:
                return {
                    "ok": False,
                    "error": "expand requires an id (a mem: handle or a segment uuid)",
                }
            result = await core.expand(
                address,
                before=request.get("before", 5),
                after=request.get("after", 5),
                unit=request.get("unit") or "segments",
                seen=self._seen_set(partition, session_id),
                kinds=request.get("kinds"),
                blocklist=bool(request.get("blocklist", False)),
                session=self._resolve_session(partition, session_id),
            )
        else:
            address = request.get("id") or request.get("seed")
            if not address:
                return {
                    "ok": False,
                    "error": "outline requires an id (a mem: handle or a segment uuid)",
                }
            result = await core.outline(
                address,
                before=request.get("before", 20),
                after=request.get("after", 20),
                session=self._resolve_session(partition, session_id),
            )
        return {"ok": True, "result": asdict(result)}

    async def _ingest(
        self, request: dict[str, Any], partition: str, session_id: str
    ) -> dict[str, Any]:
        """Capture the new transcript tail (idempotent; advances the hwm)."""
        transcript_path = request.get("transcript_path")
        if not transcript_path:
            return {"ok": False, "error": "ingest requires transcript_path"}
        session_key = session_id or "unknown"
        start_line = read_high_water_mark(self._config, session_key)
        events, total_lines = events_from_transcript(
            transcript_path, session_id=session_key, start_line=start_line
        )
        ingested = 0
        if total_lines > start_line:
            if events:
                core = await self._core(partition)
                ingested = await core.ingest(events)
                await self._fold_replies(core, partition, session_id, events)
            write_high_water_mark(self._config, session_key, total_lines)
        observe(
            self._config,
            "ingest",
            session=session_key,
            from_line=start_line,
            to_line=total_lines,
            events=len(events),
            ingested=ingested,
            behind=0,
        )
        return {"ok": True, "ingested": ingested}

    async def _reflect(
        self, request: dict[str, Any], partition: str, session_id: str
    ) -> dict[str, Any]:
        """Post-response recall: search with the model's reply, return novel hits.

        Returns a rendered ``memories`` block (one line per hit, same format as
        search/ambient) of cross-session hits that are both new to this session's
        context and at least ``reflect_threshold`` similar. The current session is
        excluded (its recent turns are already in the transcript). Only the hits
        actually returned are marked seen, so a hit gated out now can surface later.
        """
        transcript_path = request.get("transcript_path")
        if not transcript_path:
            return {"ok": False, "error": "reflect requires transcript_path"}
        if not self._config.reflect_enabled:
            return {"ok": True, "memories": ""}
        cue = last_assistant_message_text(transcript_path)
        if not cue.strip():
            return {"ok": True, "memories": ""}

        core = await self._core(partition)
        seen = self._seen_set(partition, session_id)
        limit = self._config.reflect_limit
        # Don't reflect back what's already in the model's context window: exclude
        # this session's IN-CONTEXT turns. After a compaction, this session's
        # pre-compaction turns ARE recallable (out of context) and can surface; a
        # different session is never in context.
        filter_spec = in_context_exclusion_filter(
            session_id, last_compaction_time(transcript_path)
        )
        result = await core.search(
            cue,
            limit=max(limit * 3, 8),
            filter_spec=filter_spec,
            seen=seen,
            commit_seen=False,
        )
        surfaced: list[Hit] = []
        for hit in result.hits:
            if not hit.is_new or hit.score < self._config.reflect_threshold:
                continue
            surfaced.append(hit)
            if hit.segment_uuid:
                seen.add(hit.segment_uuid)  # now in context; do not resurface it
            if len(surfaced) >= limit:
                break
        memories = "\n".join(
            format_memory_line(hit.memory_id, hit.text) for hit in surfaced
        )
        return {"ok": True, "memories": memories}


def _make_handler(
    service: MemoryService,
) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]:
    async def handle_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    request = json.loads(line)
                    response = await service.handle(request)
                except json.JSONDecodeError:
                    response = {"ok": False, "error": "invalid json"}
                except Exception as error:
                    logger.exception("request failed")
                    response = {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    return handle_connection


async def run_service(config: MemoryConfig) -> None:
    """Start the service and serve until idle-timeout or shutdown."""
    service = MemoryService(config)
    await service.start()

    sock = socket_path(config)
    with contextlib.suppress(FileNotFoundError):
        sock.unlink()
    server = await asyncio.start_unix_server(_make_handler(service), path=str(sock))

    stop_wait = asyncio.ensure_future(service.should_stop.wait())
    try:
        async with server:
            await server.start_serving()
            # We are up and listening: clear the spawn marker so the backoff in
            # ``call`` only ever suppresses a genuinely in-flight or failed spawn
            # (and a restart can re-spawn immediately).
            with contextlib.suppress(FileNotFoundError):
                (config.home / _SPAWN_MARKER).unlink()
            while not stop_wait.done():
                done, _ = await asyncio.wait({stop_wait}, timeout=60)
                if stop_wait in done:
                    break
                if time.monotonic() - service.last_activity > _IDLE_TIMEOUT:
                    logger.info("idle for %.0fs; shutting down", _IDLE_TIMEOUT)
                    break
    finally:
        stop_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_wait
        await service.stop()


def main() -> None:
    """Acquire the single-instance lock and run the service."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    config = MemoryConfig.load()
    config.ensure_dirs()

    lock_file = lock_path(config).open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info("another daemon holds the lock; exiting")
        lock_file.close()
        return

    # Record our PID while holding the lock, so a controller can address exactly
    # this process (see ``stop_daemon``) instead of matching on process name.
    lock_file.write(str(os.getpid()))
    lock_file.flush()

    try:
        asyncio.run(run_service(config))
    finally:
        with contextlib.suppress(FileNotFoundError):
            socket_path(config).unlink()
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


# ======================================================================= client


if __name__ == "__main__":
    main()
