"""
Atomic on-disk index persistence helpers, shared by vector search engines.

Each engine owns its index file layout, so the atomic-swap logic lives here
(shared by the engines) rather than in the vector store: the index save
location and the number of files written differ across engine implementations.
An engine whose backend implements this protocol natively can satisfy
`VectorSearchEngine.save` by delegating to it and skip these helpers.

The two guarantees fail differently, which is why they are separated below:

- **Atomic.** A reader never observes a partially written index at the target
  path. The index is written to a sibling temp file and swapped into place with
  ``Path.replace`` (``os.replace``), which is atomic on POSIX and Windows when
  source and destination share a filesystem (guaranteed here, since the temp
  file is a sibling of the target). A save that is interrupted (crash,
  exception) leaves the previous index untouched rather than corrupting it,
  which matters because the vector store treats a saved-but-unloadable index as
  a hard error rather than silently rebuilding it empty.
- **Durable.** A save that returns has put the new index on stable storage, and
  on POSIX the swap itself too. This is not a bonus on top of the atomicity:
  the vector store trims its pending-operation log — the only other copy of
  these vectors — once ``save`` returns, so a swap that quietly fails to reach
  disk costs data rather than performance.

Durability is platform-limited in one place. On POSIX the swap is made durable
by fsyncing the parent directory, since ``rename(2)`` leaves the new directory
entry in the page cache. Windows has no equivalent operation: the swap is
atomic through NTFS metadata journaling, but that journal record is not
necessarily flushed when the call returns, so power loss (a process crash is
not enough) within a sub-second window can roll it back. The residue is bounded
— the previous index reappears, and the records whose vectors that checkpoint
added are still in SQLite but no longer in the index, so they stop matching
queries rather than matching them wrongly.
"""

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

if sys.platform == "darwin":
    import fcntl

    def _fsync(fd: int) -> None:
        """
        Flush `fd` to stable storage, through the drive's write cache.

        macOS ``fsync`` only pushes the data to the drive, which may hold it in
        a volatile write cache; ``F_FULLFSYNC`` is the documented request for
        the stronger flush. Not every filesystem implements it, so a refusal
        falls back to ``fsync`` rather than failing the caller.
        """
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        except OSError:
            os.fsync(fd)

else:

    def _fsync(fd: int) -> None:
        """Flush `fd` to stable storage, as far as the platform allows."""
        os.fsync(fd)


# Deterministic suffix so a crash leaves at most one stale temp per index file.
# The next save overwrites it and load clears it, rather than accumulating
# uniquely named leftovers.
_TEMP_SUFFIX = ".tmp"


def _temp_path(path: str) -> str:
    """Return the sibling temp path used while writing the index at `path`."""
    return f"{path}{_TEMP_SUFFIX}"


def clear_stale_index_temp(path: str) -> None:
    """
    Remove a temp file left behind by a previously interrupted save.

    Call this on load/startup so a save that crashed before the atomic swap
    does not leak a temp file across restarts. A missing temp file is a no-op.

    Args:
        path (str):
            The index file path whose sibling temp file should be cleared.
    """
    Path(_temp_path(path)).unlink(missing_ok=True)


@contextlib.contextmanager
def atomic_index_write(path: str) -> Iterator[str]:
    """
    Write an index to a temp file, then atomically swap it into `path`.

    Yields a sibling temp path for the caller to write the index to. On normal
    exit the temp file is flushed, atomically renamed onto `path`, and the
    rename itself flushed where the platform allows — so a reader sees either
    the old index or the new one, never a partial write, and a return means the
    new index is on disk. If the body raises, the temp file is removed and the
    exception propagates, leaving any existing index at `path` intact.

    The rename is the commit point: nothing after it may raise, or a caller
    would roll back against an index that has already been replaced.

    Args:
        path (str):
            The final index file path to swap the written index into.

    Yields:
        str:
            The temp path to write the index to.
    """
    temp = _temp_path(path)
    # Clear any temp left by a previously interrupted save before reusing it.
    Path(temp).unlink(missing_ok=True)
    try:
        yield temp
        _flush_to_disk(temp)
        Path(temp).replace(path)
        _flush_parent_dir(path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _flush_to_disk(path: str) -> None:
    """
    Flush the temp file's bytes to disk before the swap. Errors propagate.

    The vector store trims its pending-operation log — its only other copy of
    these vectors — once the save returns, so a failed flush has to fail the
    save rather than be reported as success. ``EIO`` here is not hypothetical:
    a writeback error is reported to ``fsync`` once and the dirty pages are
    then dropped, which is exactly when the save must not be treated as
    committed. `atomic_index_write` removes the temp file and leaves the
    previous index in place, so the next save retries. (SQLite draws the same
    line: a file fsync failure raises ``SQLITE_IOERR_FSYNC``, while a directory
    fsync failure is ignored — see `_flush_parent_dir`.)
    """
    # O_RDWR rather than O_RDONLY: os.fsync is `_commit` on Windows, which
    # needs a writable handle. Opening for write does not truncate.
    fd = os.open(path, os.O_RDWR)
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def _flush_parent_dir(path: str) -> None:
    """
    Flush the directory holding `path`, making the swap itself durable.

    Without this the swap is atomic but not durable: POSIX ``rename(2)`` leaves
    the new directory entry in the page cache, so power loss can roll it back to
    the previous file.

    Best-effort, matching SQLite's ``unixSync``, which fsyncs the directory
    holding a journal for exactly this reason and treats a directory it cannot
    open as success. A filesystem that refuses this gives the SQLite database
    beside the index no guarantee either, so hardening the index alone would
    buy nothing, and failing the save would instead grow the pending log
    without bound. Windows cannot open a directory this way at all, so this is
    a no-op there and the swap's durability follows NTFS metadata journaling.
    """
    with contextlib.suppress(OSError):
        # `.parent` before `.resolve()`: the entry the swap created lives in the
        # directory the path names, not in a symlink target's directory.
        fd = os.open(Path(path).parent.resolve(), os.O_RDONLY)
        try:
            _fsync(fd)
        finally:
            os.close(fd)
