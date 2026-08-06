"""
Crash-safe index publication, shared by vector search engines.

An engine's `save` may only return once a later `load` would produce that index
even if the machine lost power on the next instruction, because the vector store
trims its pending-operation log — the only other copy of those vectors — as soon
as it returns. This module implements that guarantee for engines whose backend
writes an index to a path.

Why not the obvious protocol
----------------------------

Writing a temp file and renaming it over the destination is atomic, but the
rename cannot be made durable. A rename changes a *directory entry*, and the two
kinds of state have very different guarantees:

===================  ==========================================  ==============
state                make it durable                             available
===================  ==========================================  ==============
file contents        ``fsync`` / ``F_FULLFSYNC`` / FlushFileBuffers  everywhere
directory entry      ``fsync`` on a directory file descriptor     POSIX only
===================  ==========================================  ==============

On POSIX a parent-directory fsync closes the gap, but only best-effort — network
and FUSE filesystems refuse it. Windows has no equivalent at all: you cannot open
a directory as a file, and `MOVEFILE_WRITE_THROUGH` does not help, since its
documented guarantee is scoped to moves performed as a copy and delete. The
decisive evidence is SQLite's own: its VFS threads a directory-sync flag through
every commit-relevant directory operation, honors it in ``unixDelete``, and
declares it ``/* Not used on win32 */`` in ``winDelete``. So the rename could
return, the store's trim could commit durably behind it, and a power cut could
still roll the rename back — leaving the mapping forward, the index back, and no
copy of the difference anywhere.

What we do instead
------------------

SQLite's answer was not to harden the directory operation but to stop using one
as a commit point: ``PERSIST`` commits by zeroing a journal header, ``TRUNCATE``
by truncating, WAL by appending frames. This module does the same.

A base path expands into four files — two index slots and a generation record
for each — created once and thereafter only overwritten. A checkpoint is:

1. Write the index over the inactive slot and flush it. In-place file data,
   which every platform can make durable.
2. Write that slot's generation record and flush it. **This is the commit
   point**, and it is a write into a file that already exists.
3. Empty the retired slot and its record, so the spare costs no disk at rest.

`load` reads both records, keeps the ones whose halves agree, and takes the
higher generation. That comparison is the entire bookkeeping: no pointer,
manifest, or generation is kept anywhere else, so nothing about this layout
leaks into the vector store.

The record is 16 bytes — a generation and its bitwise complement. That is what
makes step 2 atomic without asking the hardware for single-sector-write
atomicity: every byte of a torn write comes from either the new record or the
slot's previous one, so the result is the new generation, the old one (which
loses to the live slot, correctly, since the caller had not trimmed), or a mix
whose halves disagree and is rejected. A torn record reads as *absent*, never as
some other generation.

Every crash window is benign:

- **during step 1** — the damaged file is the inactive slot, which nothing
  reads. The older record still wins and the log is intact.
- **between steps 1 and 2** — a complete new index exists and is invisible,
  because nothing published it. Correct: the caller has not trimmed either.
- **during step 2** — torn record, reads as absent, previous generation wins.
- **after step 2** — the index it names was flushed before it was written.
- **between steps 2 and 3** — two valid records; the higher generation wins.

There is no directory operation anywhere in this after the four files exist,
which is the whole point.
"""

import contextlib
import os
import struct
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


_SLOTS = (0, 1)

# Generation, then its bitwise complement.
_RECORD_FORMAT = "<QQ"
_RECORD_SIZE = struct.calcsize(_RECORD_FORMAT)
_COMPLEMENT_MASK = 0xFFFFFFFFFFFFFFFF


def _slot_path(base: str, slot: int) -> Path:
    """Path of an index slot under `base`."""
    return Path(f"{base}.{slot}")


def _record_path(base: str, slot: int) -> Path:
    """Path of a slot's generation record."""
    return Path(f"{base}.{slot}.gen")


def index_artifact_paths(base: str) -> list[Path]:
    """
    Every file `base` expands into.

    The vector store uses this to discard a collection's index without having to
    know how many files an engine keeps or what they are called.

    Args:
        base (str):
            The index base path.

    Returns:
        list[Path]:
            The slot and generation-record paths, in no particular order.
    """
    return [
        path
        for slot in _SLOTS
        for path in (_slot_path(base, slot), _record_path(base, slot))
    ]


def published_index_path(base: str) -> str | None:
    """
    The index a `load` should read, or None if nothing has been published.

    Reads both generation records, discards any whose halves disagree — a torn
    or absent record — and returns the slot holding the higher generation.

    A caller must **not** fall back to the other slot when the returned index
    fails to parse. The log was trimmed against the published one, so the other
    is stale by exactly the operations that can no longer be replayed; failing
    loudly is right, and quietly serving the previous generation would be data
    loss dressed as success.

    Args:
        base (str):
            The index base path.

    Returns:
        str | None:
            Path of the published index slot, or None if there is none.
    """
    published = _published_slot(base)
    return None if published is None else str(_slot_path(base, published[0]))


@contextlib.contextmanager
def publish_index(base: str) -> Iterator[str]:
    """
    Write an index into the free slot and publish it.

    Yields the path of the slot that is not currently published, for the caller
    to write the index to. On normal exit that slot is flushed, its generation
    record is written and flushed — the commit — and the retired slot is
    emptied.

    If the body raises, nothing is published and the exception propagates: the
    previously published index stays live, and the half-written slot is
    invisible until the next checkpoint overwrites it. The same holds for a
    failed flush, which is why flush errors are not suppressed — the store trims
    its log on the strength of this call, and ``EIO`` from ``fsync`` means the
    writeback already failed and the dirty pages were dropped.

    Args:
        base (str):
            The index base path.

    Yields:
        str:
            Path of the slot to write the index to.
    """
    live = _published_slot(base)
    if live is None:
        target, generation = _SLOTS[0], 1
    else:
        live_slot, live_generation = live
        target, generation = 1 - live_slot, live_generation + 1

    _create_artifacts(base)

    yield str(_slot_path(base, target))

    # 1. The index itself, durable before anything names it.
    _flush_file(_slot_path(base, target))
    # 2. The commit point.
    _write_generation(base, target, generation)
    # 3. Reclaim the retired pair. Best-effort: this runs after the commit, so
    #    it must not fail the save, and a slot left full is harmless — its
    #    record holds the lower generation and loses the comparison.
    if live is not None:
        _empty_slot(base, live[0])


def _published_slot(base: str) -> tuple[int, int] | None:
    """The (slot, generation) with the highest believable generation, if any."""
    published = [
        (generation, slot)
        for slot in _SLOTS
        if (generation := _read_generation(base, slot)) is not None
    ]
    if not published:
        return None
    generation, slot = max(published)
    return slot, generation


def _read_generation(base: str, slot: int) -> int | None:
    """The generation published in `slot`, or None if no believable record."""
    try:
        record = _record_path(base, slot).read_bytes()
    except OSError:
        return None
    if len(record) != _RECORD_SIZE:
        return None
    generation, complement = struct.unpack(_RECORD_FORMAT, record)
    # Generations start at 1, so an all-zero record is never a publication.
    if generation == 0 or complement != generation ^ _COMPLEMENT_MASK:
        return None
    return generation


def _write_generation(base: str, slot: int, generation: int) -> None:
    """Publish `slot` by writing and flushing its generation record."""
    record = struct.pack(_RECORD_FORMAT, generation, generation ^ _COMPLEMENT_MASK)
    fd = os.open(_record_path(base, slot), os.O_RDWR)
    try:
        os.write(fd, record)
        _fsync(fd)
    finally:
        os.close(fd)


def _empty_slot(base: str, slot: int) -> None:
    """Unpublish a slot and reclaim its space, best-effort."""
    # The record first: no valid record may ever name an emptied slot.
    for path in (_record_path(base, slot), _slot_path(base, slot)):
        with contextlib.suppress(OSError):
            os.truncate(path, 0)


def _flush_file(path: Path) -> None:
    """Flush a file's contents to stable storage. Errors propagate."""
    # O_RDWR rather than O_RDONLY: os.fsync is `_commit` on Windows, which needs
    # a writable handle. Opening for write does not truncate.
    fd = os.open(path, os.O_RDWR)
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def _create_artifacts(base: str) -> None:
    """
    Create any missing slot or record files, once per collection.

    This is the only directory operation in the protocol, and it happens when
    there is nothing to lose: before anything has been published, the store's
    log still holds every vector. The parent directory is flushed afterwards so
    that even this is durable where the platform allows it — a POSIX-only,
    best-effort step, and the one place where a filesystem that refuses it costs
    nothing more than a loud missing-index error at the next open.
    """
    created = False
    for path in index_artifact_paths(base):
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
        except FileExistsError:
            continue
        created = True

    if not created:
        return

    with contextlib.suppress(OSError):
        fd = os.open(Path(base).parent.resolve(), os.O_RDONLY)
        try:
            _fsync(fd)
        finally:
            os.close(fd)
