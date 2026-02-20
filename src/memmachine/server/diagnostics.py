"""Diagnostics for debugging asyncio deadlocks and long-running tasks.

On Unix, send SIGUSR1 to the server process to dump all asyncio task stacks
and thread stacks to the log output:

    kill -SIGUSR1 <pid>

On all platforms, task and thread stacks are dumped automatically on shutdown.
"""

import asyncio
import logging
import signal
import sys
import traceback

logger = logging.getLogger(__name__)


def dump_traceback() -> None:
    """Dump all asyncio task stacks and thread stacks to the logger."""
    loop = asyncio.get_running_loop()
    tasks = asyncio.all_tasks(loop)

    lines: list[str] = []
    lines.append(f"=== Async task dump: {len(tasks)} task(s) ===")

    for task in sorted(tasks, key=lambda t: t.get_name()):
        state = (
            "done" if task.done() else ("cancelled" if task.cancelled() else "pending")
        )
        lines.append(f"\nTask {task.get_name()!r} [{state}] {task.get_coro()!r}")

        stack = task.get_stack(limit=50)
        if stack:
            lines.append("  Stack:")
            for frame in stack:
                lines.extend(
                    "    " + line.rstrip()
                    for line in traceback.format_stack(frame, limit=1)
                )
        else:
            lines.append("  (no stack available)")

    # Also dump thread stacks to catch blocking I/O or thread deadlocks.
    # noqa: SLF001 — _current_frames() is the only API for thread stacks; stable since Python 2.5
    thread_frames = sys._current_frames()  # noqa: SLF001
    lines.append(f"\n=== Thread stacks: {len(thread_frames)} thread(s) ===")
    for thread_id, frame in thread_frames.items():
        lines.append(f"\nThread {thread_id:#x}:")
        lines.extend("  " + line.rstrip() for line in traceback.format_stack(frame))

    logger.warning("\n".join(lines))


def install_sigusr1_handler() -> None:
    """Register SIGUSR1 to dump traceback. No-op on Windows."""
    if not hasattr(signal, "SIGUSR1"):
        return
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, dump_traceback)
