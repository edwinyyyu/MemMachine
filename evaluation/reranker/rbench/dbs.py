"""SQLite settings for every database rbench opens: EventMemory's segment and vector stores
(memory.Memory) and the score and embedding caches."""

from __future__ import annotations

import sqlite3

# Every SQLite database rbench opens (the user, 2026-11-10: "can't you make sure it works using the
# right pragmas?"): wait up to 30 s for a lock instead of failing, write-ahead log, NORMAL sync
# (a power cut may lose the last few commits, never corrupt the file), and full fsync, since
# macOS's fsync does not flush the drive without it. MemMachine's #1605 / PR #1823 sets the
# first three for its vector stores; fullfsync is set here.
PRAGMAS = ("PRAGMA busy_timeout=30000", "PRAGMA journal_mode=WAL", "PRAGMA synchronous=NORMAL",
           "PRAGMA fullfsync=ON", "PRAGMA checkpoint_fullfsync=ON")


def sqlite_pragmas(connection) -> None:
    """Apply PRAGMAS to a DB-API connection (sqlite3, or aiosqlite's adapted one)."""
    cursor = connection.cursor()
    for pragma in PRAGMAS:
        cursor.execute(pragma)
    cursor.close()


def connect(path) -> sqlite3.Connection:
    """sqlite3.connect with PRAGMAS (the score and embedding caches)."""
    db = sqlite3.connect(path, timeout=30)
    sqlite_pragmas(db)
    return db


def replace_atomically(path, write) -> None:
    """Write a file whole or not at all: write(temporary path) writes it beside the target (its
    suffix kept, so numpy and pandas name it as asked), then it is fsynced and moved into place.
    An interrupted write leaves the old file (or none), never a truncated one."""
    import os
    from pathlib import Path

    path = Path(path)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    write(temporary)
    with open(temporary, "rb+") as file:
        os.fsync(file.fileno())
    os.replace(temporary, path)
