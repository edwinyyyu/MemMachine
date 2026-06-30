"""Thin client + lifecycle control for the memory daemon (standard library only).

Split out of ``daemon.py`` so the hooks and MCP server can talk to the daemon
without importing the daemon *service* (which pulls in ``engine`` -> numpy,
sqlalchemy, the memmachine_server stack). This module only opens a socket, spawns
the service as a detached subprocess (by module name, never importing it), and
reads the lock/PID files. Imports nothing heavier than ``wire.MemoryConfig``.
"""

import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from claude_memory.wire import MemoryConfig

SOCKET_NAME = "daemon.sock"
LOCK_NAME = "daemon.lock"
_SPAWN_MARKER = "daemon.spawning"
_SPAWN_BACKOFF_SECONDS = 90.0
_POLL_INTERVAL_SECONDS = 0.25


def socket_path(config: MemoryConfig) -> Path:
    """Path to the daemon's Unix domain socket."""
    return config.home / SOCKET_NAME


def lock_path(config: MemoryConfig) -> Path:
    """Path to the daemon's single-instance lock file."""
    return config.home / LOCK_NAME


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
