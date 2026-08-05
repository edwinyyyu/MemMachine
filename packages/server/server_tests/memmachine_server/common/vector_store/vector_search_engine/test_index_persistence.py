"""Tests for the shared atomic index persistence helpers."""

import errno
import os
import stat
import sys
from pathlib import Path

import pytest

from memmachine_server.common.vector_store.vector_search_engine import (
    index_persistence,
)
from memmachine_server.common.vector_store.vector_search_engine.index_persistence import (
    atomic_index_write,
    clear_stale_index_temp,
)


class TestAtomicIndexWrite:
    def test_swaps_temp_into_place_on_success(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        path.write_text("OLD")

        with atomic_index_write(str(path)) as temp:
            Path(temp).write_text("NEW")
            # Until the context exits, the target still holds the old contents.
            assert path.read_text() == "OLD"

        assert path.read_text() == "NEW"
        assert not (tmp_path / "index.idx.tmp").exists()

    def test_creates_target_when_missing(self, tmp_path: Path):
        path = tmp_path / "index.idx"

        with atomic_index_write(str(path)) as temp:
            Path(temp).write_text("NEW")

        assert path.read_text() == "NEW"

    def test_preserves_existing_index_on_failure(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        path.write_text("GOOD")

        def write_then_fail() -> None:
            with atomic_index_write(str(path)) as temp:
                Path(temp).write_text("PARTIAL")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            write_then_fail()

        # The existing index is untouched and the temp file is cleaned up.
        assert path.read_text() == "GOOD"
        assert not (tmp_path / "index.idx.tmp").exists()

    def test_does_not_create_target_on_failure(self, tmp_path: Path):
        path = tmp_path / "index.idx"

        def write_then_fail() -> None:
            with atomic_index_write(str(path)) as temp:
                Path(temp).write_text("PARTIAL")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            write_then_fail()

        assert not path.exists()
        assert not (tmp_path / "index.idx.tmp").exists()

    def test_clears_stale_temp_before_writing(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        # A temp file left by a previously interrupted save.
        (tmp_path / "index.idx.tmp").write_text("STALE")

        with atomic_index_write(str(path)) as temp:
            assert not Path(temp).exists()
            Path(temp).write_text("NEW")

        assert path.read_text() == "NEW"


class TestDurability:
    def test_flushes_the_file_then_the_parent_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        real_fsync = index_persistence._fsync
        # True for a directory fd, False for a regular file.
        flushed: list[bool] = []

        def record(fd: int) -> None:
            flushed.append(stat.S_ISDIR(os.fstat(fd).st_mode))
            real_fsync(fd)

        monkeypatch.setattr(index_persistence, "_fsync", record)

        path = tmp_path / "index.idx"
        with atomic_index_write(str(path)) as temp:
            Path(temp).write_text("NEW")

        if os.name == "nt":
            # Windows cannot open a directory to flush it; the swap's
            # durability follows NTFS metadata journaling instead.
            assert flushed == [False]
        else:
            # The bytes first, then the directory entry the swap created.
            assert flushed == [False, True]

    def test_file_flush_failure_fails_the_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def fail(fd: int) -> None:
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(index_persistence, "_fsync", fail)

        path = tmp_path / "index.idx"
        path.write_text("GOOD")

        def write_index() -> None:
            with atomic_index_write(str(path)) as temp:
                Path(temp).write_text("NEW")

        # Reported as a failure rather than swallowed: the caller trims its
        # pending log on the strength of this save.
        with pytest.raises(OSError, match="Input/output error"):
            write_index()

        assert path.read_text() == "GOOD"
        assert not (tmp_path / "index.idx.tmp").exists()

    def test_parent_directory_flush_failure_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        real_fsync = index_persistence._fsync

        def fail_for_directories(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EINVAL, "Invalid argument")
            real_fsync(fd)

        monkeypatch.setattr(index_persistence, "_fsync", fail_for_directories)

        path = tmp_path / "index.idx"
        # Filesystems that refuse a directory fsync must not fail every save.
        with atomic_index_write(str(path)) as temp:
            Path(temp).write_text("NEW")

        assert path.read_text() == "NEW"


class TestFsync:
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


class TestClearStaleIndexTemp:
    def test_removes_leftover_temp(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        temp = tmp_path / "index.idx.tmp"
        temp.write_text("STALE")

        clear_stale_index_temp(str(path))

        assert not temp.exists()

    def test_no_temp_is_noop(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        # Must not raise when there is nothing to clear.
        clear_stale_index_temp(str(path))

    def test_leaves_index_file_untouched(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        path.write_text("GOOD")

        clear_stale_index_temp(str(path))

        assert path.read_text() == "GOOD"
