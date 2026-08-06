"""
Tests for crash-safe index publication.

The anomaly tests walk the crash points in the publish sequence by constructing
the on-disk state each one would leave — stricter than killing a process, since
every state is produced deterministically. The one property construction cannot
observe is the *ordering* itself: a generation record written before its index
would satisfy every state-based test here, so
`test_a_failed_index_write_publishes_nothing` pins it separately.
"""

import errno
import os
import stat
import struct
import sys
from pathlib import Path

import pytest

from memmachine_server.common.vector_store.vector_search_engine import (
    index_persistence,
)
from memmachine_server.common.vector_store.vector_search_engine.index_persistence import (
    index_artifact_paths,
    publish_index,
    published_index_path,
)

_MASK = 0xFFFFFFFFFFFFFFFF


def _write_index(base: str, contents: str) -> str:
    """Publish `contents` as the next index generation, returning its slot."""
    with publish_index(base) as slot:
        Path(slot).write_text(contents)
    return slot


def _record(base: str, slot: int) -> Path:
    return Path(f"{base}.{slot}.gen")


def _slot(base: str, slot: int) -> Path:
    return Path(f"{base}.{slot}")


def _generation(base: str, slot: int) -> int | None:
    """The generation stored in a record, ignoring the complement check."""
    data = _record(base, slot).read_bytes()
    if len(data) != 16:
        return None
    return struct.unpack("<QQ", data)[0]


class TestPublish:
    def test_nothing_is_published_before_the_first_save(self, tmp_path: Path):
        assert published_index_path(str(tmp_path / "index.idx")) is None

    def test_publishes_and_reads_back(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")

        _write_index(base, "FIRST")

        published = published_index_path(base)
        assert published is not None
        assert Path(published).read_text() == "FIRST"

    def test_alternates_slots_across_checkpoints(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")

        first = _write_index(base, "FIRST")
        second = _write_index(base, "SECOND")
        third = _write_index(base, "THIRD")

        assert first != second
        assert third == first
        published = published_index_path(base)
        assert published is not None
        assert Path(published).read_text() == "THIRD"

    def test_empties_the_retired_slot(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")

        retired = _write_index(base, "FIRST")
        _write_index(base, "SECOND")

        # The spare costs no disk at rest, and its record no longer publishes.
        assert Path(retired).stat().st_size == 0
        retired_slot = int(retired[-1])
        assert _record(base, retired_slot).stat().st_size == 0

    def test_creates_its_files_once(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")

        _write_index(base, "FIRST")

        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
            p.name for p in index_artifact_paths(base)
        )


class TestCrashPoints:
    def test_a_torn_index_write_is_invisible(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")
        _write_index(base, "GOOD")
        live = published_index_path(base)
        assert live is not None

        # Crash during step 1: the damaged file is the slot nobody reads.
        free = 1 - int(live[-1])
        _slot(base, free).write_text("HALF-WRITTEN")

        assert published_index_path(base) == live
        assert Path(live).read_text() == "GOOD"

    def test_an_unpublished_index_is_invisible(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")
        _write_index(base, "GOOD")
        live = published_index_path(base)
        assert live is not None

        # Crash between steps 1 and 2: a complete new index that nothing named.
        free = 1 - int(live[-1])
        _slot(base, free).write_text("COMPLETE BUT UNPUBLISHED")

        assert published_index_path(base) == live

    def test_a_torn_record_reads_as_absent(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")
        _write_index(base, "GOOD")
        live = published_index_path(base)
        assert live is not None
        live_slot = int(live[-1])
        free = 1 - live_slot

        # Crash during step 2: halves from different writes. The higher
        # generation is present but unbelievable, so the live slot still wins.
        _slot(base, free).write_text("NEWER")
        torn = struct.pack("<QQ", 2, 1 ^ _MASK)
        _record(base, free).write_bytes(torn)

        assert published_index_path(base) == live
        assert Path(live).read_text() == "GOOD"

    def test_a_short_record_reads_as_absent(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")
        _write_index(base, "GOOD")
        live = published_index_path(base)
        assert live is not None
        free = 1 - int(live[-1])

        _slot(base, free).write_text("NEWER")
        _record(base, free).write_bytes(struct.pack("<Q", 2))

        assert published_index_path(base) == live

    def test_the_higher_generation_wins(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")
        first = _write_index(base, "FIRST")
        second = _write_index(base, "SECOND")

        # Crash between steps 2 and 3: both records valid, retired not yet
        # emptied. Reconstruct that state by republishing the older record.
        first_slot = int(first[-1])
        _slot(base, first_slot).write_text("FIRST")
        _record(base, first_slot).write_bytes(struct.pack("<QQ", 1, 1 ^ _MASK))

        assert _generation(base, first_slot) == 1
        assert published_index_path(base) == second

    def test_a_failed_index_write_publishes_nothing(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")
        _write_index(base, "GOOD")
        live = published_index_path(base)
        assert live is not None

        # The ordering itself: no state-based test can see a record written
        # before its index, because the end state is identical either way.
        def write_then_fail() -> None:
            with publish_index(base) as slot:
                Path(slot).write_text("PARTIAL")
                raise RuntimeError("engine exploded")

        with pytest.raises(RuntimeError, match="engine exploded"):
            write_then_fail()

        assert published_index_path(base) == live
        assert Path(live).read_text() == "GOOD"

    def test_a_failed_flush_publishes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        base = str(tmp_path / "index.idx")
        _write_index(base, "GOOD")
        live = published_index_path(base)
        assert live is not None

        def fail(_fd: int) -> None:
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(index_persistence, "_fsync", fail)

        # A flush that fails must fail the save: the caller trims its log on the
        # strength of it, and EIO means the writeback already failed.
        def write_index() -> None:
            with publish_index(base) as slot:
                Path(slot).write_text("NEWER")

        with pytest.raises(OSError, match="Input/output error"):
            write_index()

        assert published_index_path(base) == live


class TestDurability:
    def test_flushes_the_index_before_the_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        base = str(tmp_path / "index.idx")
        real_fsync = index_persistence._fsync
        flushed: list[str] = []

        def record(fd: int) -> None:
            mode = os.fstat(fd).st_mode
            flushed.append("dir" if stat.S_ISDIR(mode) else f"{os.fstat(fd).st_size}")
            real_fsync(fd)

        monkeypatch.setattr(index_persistence, "_fsync", record)

        _write_index(base, "FIRST")

        # A one-time parent flush for the created files (POSIX only), then the
        # index, then the 16-byte record that publishes it.
        assert flushed[-2:] == ["5", "16"]
        if os.name != "nt":
            assert flushed[0] == "dir"

    @pytest.mark.skipif(
        sys.platform != "darwin", reason="F_FULLFSYNC only exists on macOS"
    )
    def test_falls_back_when_full_fsync_is_unsupported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        fcntl = pytest.importorskip("fcntl")
        calls: list[str] = []

        def unsupported(_fd: int, _cmd: int) -> None:
            calls.append("full_fsync")
            raise OSError(errno.ENOTSUP, "Operation not supported")

        monkeypatch.setattr(fcntl, "fcntl", unsupported)
        monkeypatch.setattr(os, "fsync", lambda fd: calls.append("fsync"))

        path = tmp_path / "index.idx"
        path.write_text("DATA")
        fd = os.open(str(path), os.O_RDWR)
        try:
            index_persistence._fsync(fd)
        finally:
            os.close(fd)

        assert calls == ["full_fsync", "fsync"]


class TestIndexArtifactPaths:
    def test_lists_every_file_the_base_expands_into(self, tmp_path: Path):
        base = str(tmp_path / "index.idx")

        assert sorted(p.name for p in index_artifact_paths(base)) == [
            "index.idx.0",
            "index.idx.0.gen",
            "index.idx.1",
            "index.idx.1.gen",
        ]
