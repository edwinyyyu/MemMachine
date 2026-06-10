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
    {"op":"expand","partition","session_id","seed","before","after"}
                                                  -> {"ok":true,"result":{...}}
    {"op":"demote","partition","memory_id","cue"}
    {"op":"annotate","partition","memory_id","note"}
                                                  -> {"ok":true,"result":{...}}
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
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from claude_memory.engine import (
    Hit,
    MemoryConfig,
    MemoryCore,
    MemoryStores,
    Source,
    TextBlock,
    blend_context_cue,
    build_embedder,
    fold_running_context,
    format_memory_line,
    in_context_exclusion_filter,
    parse_memory_id,
)
from claude_memory.transcript import (
    events_from_transcript,
    last_assistant_message_text,
    last_compaction_time,
)

logger = logging.getLogger("claude_memory.daemon")

SOCKET_NAME = "daemon.sock"
LOCK_NAME = "daemon.lock"
_SPAWN_MARKER = "daemon.spawning"
_SPAWN_BACKOFF_SECONDS = 90.0
_POLL_INTERVAL_SECONDS = 0.25
_IDLE_TIMEOUT = float(os.environ.get("CLAUDE_MEMORY_DAEMON_IDLE", "1800"))


def socket_path(config: MemoryConfig) -> Path:
    """Path to the daemon's Unix domain socket."""
    return config.home / SOCKET_NAME


def lock_path(config: MemoryConfig) -> Path:
    """Path to the daemon's single-instance lock file."""
    return config.home / LOCK_NAME


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

        if op == "search":
            core = await self._core(partition)
            query_vector = None
            if request.get("use_context"):
                query_vector = await self._context_cue(
                    core, partition, session_id, request["cue"]
                )
            result = await core.search(
                request["cue"],
                limit=request.get("limit", 8),
                filter_spec=request.get("filters"),
                seen=self._seen_set(partition, session_id),
                query_vector=query_vector,
            )
            return {"ok": True, "result": asdict(result)}

        if op == "expand":
            core = await self._core(partition)
            result = await core.expand(
                request["seed"],
                before=request.get("before", 5),
                after=request.get("after", 5),
                seen=self._seen_set(partition, session_id),
            )
            return {"ok": True, "result": asdict(result)}

        if op == "demote":
            core = await self._core(partition)
            result = await core.demote(request["memory_id"], request.get("cue", ""))
            return {"ok": True, "result": asdict(result)}

        if op == "annotate":
            core = await self._core(partition)
            message = await core.annotate(
                request["memory_id"], request.get("note", "")
            )
            return {"ok": True, "message": message}

        if op == "reflect":
            return await self._reflect(request, partition, session_id)

        if op == "ingest":
            return await self._ingest(request, partition, session_id)

        return {"ok": False, "error": f"unknown op {op!r}"}

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
            uuid = parse_memory_id(hit.memory_id)
            if uuid is not None:
                seen.add(uuid.hex)  # now in context; do not resurface it
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


class DaemonUnavailableError(RuntimeError):
    """The memory daemon could not be reached (and was not started)."""


def _connect(path: Path, timeout: float) -> socket.socket | None:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(timeout)
        connection.connect(str(path))
    except OSError:
        connection.close()
        return None
    return connection


def _spawn_daemon(config: MemoryConfig) -> None:
    log_file = (config.home / "daemon.log").open("a")
    # Fixed argv, no shell: spawn the daemon detached so it outlives this client.
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.Popen(
        [sys.executable, "-m", "claude_memory.daemon"],
        cwd=str(repo_root),
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )


def _roundtrip(
    connection: socket.socket, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    try:
        connection.settimeout(timeout)
        connection.sendall((json.dumps(payload) + "\n").encode())
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk:
                break
            buffer += chunk
        return json.loads(buffer.decode())
    finally:
        connection.close()


def call(
    payload: dict[str, Any],
    *,
    config: MemoryConfig | None = None,
    wait_for_start: float = 0.0,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Send one request to the daemon and return its reply.

    ``wait_for_start``: if > 0 and the daemon is down, spawn it (unless a startup
    is already in flight) and poll up to this many seconds for it to come up. If
    0, fast-fail when the daemon is not already running (used by ambient recall,
    which must never block a prompt).
    """
    config = config or MemoryConfig.load()
    path = socket_path(config)

    connection = _connect(path, timeout)
    if connection is not None:
        return _roundtrip(connection, payload, timeout)

    if wait_for_start <= 0:
        raise DaemonUnavailableError("memory daemon not running")

    marker = config.home / _SPAWN_MARKER
    recently_spawning = (
        marker.exists()
        and (time.time() - marker.stat().st_mtime) < _SPAWN_BACKOFF_SECONDS
    )
    if not recently_spawning:
        config.ensure_dirs()
        marker.write_text(str(time.time()))
        _spawn_daemon(config)

    deadline = time.monotonic() + wait_for_start
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)
        connection = _connect(path, timeout)
        if connection is not None:
            return _roundtrip(connection, payload, timeout)

    raise DaemonUnavailableError("memory daemon did not come up in time")


# ==================================================================== lifecycle
#
# Controlling the daemon WITHOUT matching on process name. Every primitive below
# is keyed to ``config.home``: the socket reaches only the process bound to that
# home's socket path, and the lock file's PID is one the daemon itself wrote
# while holding the home's exclusive lock. So these can only ever touch this
# home's daemon — never an unrelated process (the danger with ``pkill -f``).


def read_lock_pid(config: MemoryConfig) -> int | None:
    """Read the daemon's PID from its lock file (written at startup), or None."""
    try:
        text = lock_path(config).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def daemon_alive(config: MemoryConfig) -> bool:
    """True iff a daemon for this home is alive, decided by its instance lock.

    The live daemon holds an exclusive ``flock`` on its lock file for its whole
    life, and the OS releases that lock only when the process dies. So if we
    cannot acquire the lock, a daemon is alive; if we can, none is. The file is
    opened read-only so probing never truncates the recorded PID.
    """
    path = lock_path(config)
    if not path.exists():
        return False
    try:
        handle = path.open("r")
    except OSError:
        return False
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _pid_is_daemon(pid: int) -> bool:
    """Verify a PID is actually a claude_memory daemon before signalling it."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return "claude_memory.daemon" in result.stdout


def _wait_until_gone(config: MemoryConfig, timeout: float) -> bool:
    """Poll until no daemon holds the lock, up to ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not daemon_alive(config):
            return True
        time.sleep(_POLL_INTERVAL_SECONDS)
    return not daemon_alive(config)


def _clear_runtime_files(config: MemoryConfig) -> None:
    """Remove the socket and spawn marker left by a dead/stopped daemon."""
    for path in (socket_path(config), config.home / _SPAWN_MARKER):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def stop_daemon(config: MemoryConfig, *, timeout: float = 10.0) -> str:
    """Stop this home's daemon, addressing it by socket then verified PID.

    Order: ask the home's socket to ``shutdown`` (graceful, runs the daemon's
    cleanup); if it does not exit, signal *only* the PID the daemon wrote into
    this home's lock file, after confirming via the lock that that process is
    still the live daemon and via ``ps`` that it is a claude_memory daemon. It
    therefore cannot stop anything but this home's daemon.
    """
    if not daemon_alive(config):
        _clear_runtime_files(config)
        return "No daemon is running for this home."

    with contextlib.suppress(DaemonUnavailableError):
        call({"op": "shutdown"}, config=config, wait_for_start=0.0, timeout=5.0)
    if _wait_until_gone(config, timeout):
        _clear_runtime_files(config)
        return "Daemon stopped gracefully."

    pid = read_lock_pid(config)
    if pid is None:
        return (
            "Daemon is still running but recorded no PID; not force-killing. "
            "Inspect it before stopping it manually."
        )
    if not _pid_is_daemon(pid):
        return (
            f"PID {pid} in the lock file is not a claude_memory daemon; "
            "refusing to kill it."
        )

    os.kill(pid, signal.SIGTERM)
    if _wait_until_gone(config, timeout):
        _clear_runtime_files(config)
        return f"Daemon (pid {pid}) stopped with SIGTERM."

    os.kill(pid, signal.SIGKILL)
    _wait_until_gone(config, timeout)
    _clear_runtime_files(config)
    return f"Daemon (pid {pid}) did not exit; force-killed with SIGKILL."


def daemon_status(config: MemoryConfig) -> dict[str, Any]:
    """Report whether the daemon is running, plus its PID and socket."""
    running = daemon_alive(config)
    return {
        "running": running,
        "pid": read_lock_pid(config) if running else None,
        "home": str(config.home),
        "socket": str(socket_path(config)),
    }


if __name__ == "__main__":
    main()
