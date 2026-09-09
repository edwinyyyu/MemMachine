"""
Atomic on-disk index publication, shared by vector search engines.

The index is written to a sibling temp file, flushed, and swapped into place
with ``os.replace``. Because the flush completes before the rename is issued,
a crash leaves either the previous index or the new one -- never the new name
over incomplete bytes, which is the case that matters: the vector store treats
a saved-but-unloadable index as a hard error, while an index that reverts is
only missing vectors until they are upserted again.

The rename itself is not made durable -- that needs an fsync on a directory
fd, which POSIX offers unevenly and Windows not at all -- so a power failure
can still roll a save back to the previously published index after ``save``
returned, while the SQLite side stays committed. Records applied since the
last checkpoint then resolve by uuid but cannot be found by search. That is
the direction this store tolerates; #1588 records why.

Why a rename at all: the stronger answer is to put the commit point inside the
file, where an fsync reaches it portably. SQLite never renames -- it commits by
truncating or zeroing its rollback journal, and in WAL mode by appending frames
whose checksums make a torn tail self-identifying, so recovery keeps everything
up to the last frame that verifies. Both need the writer to own the file
format. A search engine owns its own and exposes ``save(path)``, so above that
call a rename is the only atomicity primitive left. An engine whose format
already commits that way needs none of this.
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

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
    exit the temp file is flushed and atomically renamed onto `path`, so a
    reader sees either the old index or the new one, never a partial write. If
    the body raises, the temp file is removed and the exception propagates,
    leaving any existing index at `path` intact.

    The swap is atomic, not durable: after a power failure the index at `path`
    may be the previous one.

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
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _flush_to_disk(path: str) -> None:
    """
    Fsync the temp file so its bytes are on disk before the swap.

    This is what rules out publishing the new name over incomplete bytes: the
    data is durable before the rename is issued, and a durable write does not
    un-happen. A failed fsync therefore fails the save -- the caller discards
    the temp and the previously published index stands -- because a failure
    here is exactly the evidence that the bytes are not safe to publish.

    The descriptor is a fresh one, because the engine writes through its own
    and closes it before returning. Flushing works regardless: dirty pages
    belong to the file, not to the descriptor that dirtied them. Error
    reporting does not. Linux hands a writeback error to descriptors that were
    open when it was recorded, so one recorded in the gap between the engine's
    close and this open is never reported here and the save proceeds. What
    closes that window is fsyncing the descriptor the bytes were written
    through, which needs an engine that writes through a caller-supplied
    handle rather than to a path.
    """
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
