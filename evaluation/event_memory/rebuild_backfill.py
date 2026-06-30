"""Clean rebuild of a claude_memory space with the idempotent (record-uuid) ingest.

Fixes the duplicate-memory problem at the source by re-ingesting every transcript
into a FRESH home, where:

  * event uuids derive from the transcript record uuid (transcript._with_stable_uuids),
    so forks / resumed prefixes / re-ingests collapse to one copy
    (MemoryCore.ingest skips already-present events);
  * compaction summaries (isCompactSummary) are skipped by the parser;
  * eviction runs over DEDUPED data, so its decisions are correct (the old space
    evicted on duplicated, denser-than-real clusters);
  * the per-session high-water-mark is SEEDED = each file's line count, so after
    the symlink swap the live Stop hook continues incrementally instead of
    re-ingesting from line 0 (the original backfill skipped this — the root cause).

Resumable: completed files are recorded in <home>/rebuild_done.txt.

    CLAUDE_MEMORY_HOME=~/.claude/claude_memory.rebuild_evict \
    PYTHONPATH=<repo> uv run python evaluation/event_memory/rebuild_backfill.py
"""

import asyncio
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

from claude_memory.daemon import write_high_water_mark
from claude_memory.engine import MemoryConfig, MemoryCore
from claude_memory.transcript import _as_dict, _Builder, _events_from_record

CHUNK = 1000


def _prepare_home() -> Path:
    home = Path(os.environ["CLAUDE_MEMORY_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CLAUDE_MEMORY_EMBEDDING", "embeddinggemma")
    os.environ.setdefault("CLAUDE_MEMORY_VECTOR_BACKEND", "turbovec")
    os.environ.pop("CLAUDE_MEMORY_PARTITION", None)  # shared
    # Self-describing space: eviction + reflection config (matches the live space).
    config_path = home / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(
                {
                    "eviction_threshold": 0.9,
                    "eviction_target_size": 5,
                    "reflect_enabled": True,
                }
            ),
            encoding="utf-8",
        )
    return home


def stream_chunks(
    path: Path, session_id: str, project: str
) -> Iterator[tuple[list, int]]:
    """Yield (events_chunk, lines_seen_so_far); the final yield carries total lines."""
    builder = _Builder(session_id=session_id, project=project)
    batch: list = []
    lines = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            lines += 1
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            record_dict = _as_dict(record)
            if record_dict is None:
                continue
            batch.extend(_events_from_record(record_dict, builder))
            if len(batch) >= CHUNK:
                yield batch, lines
                batch = []
    yield batch, lines


async def _ingest_file(core: MemoryCore, path: Path) -> tuple[int, int, int]:
    """Ingest one transcript file; returns (parsed, newly_ingested, total_lines)."""
    parsed = 0
    added = 0
    total_lines = 0
    for chunk, lines in stream_chunks(path, path.stem, path.parent.name):
        total_lines = lines
        if chunk:
            added += await core.ingest(chunk)  # record-uuid dedup happens here
            parsed += len(chunk)
    return parsed, added, total_lines


async def main() -> None:
    home = _prepare_home()
    config = MemoryConfig.load()
    projects = Path.home() / ".claude" / "projects"
    files = sorted(projects.glob("*/*.jsonl"), key=lambda p: p.stat().st_size)
    done_path = home / "rebuild_done.txt"
    done = (
        set(done_path.read_text(encoding="utf-8").split("\n"))
        if done_path.exists()
        else set()
    )

    core = await MemoryCore.open(config)
    total_files = len(files)
    t0 = time.time()
    seen = 0
    new = 0
    print(
        f"START rebuild: {total_files} files -> {home} "
        f"(eviction_threshold={config.eviction_threshold}, "
        f"target={config.eviction_target_size})",
        flush=True,
    )
    for idx, path in enumerate(files, 1):
        if str(path) in done:
            continue
        size_mb = path.stat().st_size / 1048576
        file_t = time.time()
        try:
            parsed, added, total_lines = await _ingest_file(core, path)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            print(
                f"[{idx}/{total_files}] ERROR {path.name}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        # Seed the high-water-mark so the live hook continues, not restarts.
        write_high_water_mark(config, path.stem, total_lines)
        with done_path.open("a", encoding="utf-8") as handle:
            handle.write(str(path) + "\n")
        seen += parsed
        new += added
        elapsed = time.time() - t0
        print(
            f"[{idx}/{total_files}] {size_mb:6.1f}MB {added:6d}new/{parsed:6d} "
            f"{time.time() - file_t:5.0f}s | cum {new:7d}new/{seen:7d} "
            f"({100 * (1 - new / max(seen, 1)):.0f}% dup) "
            f"{elapsed / 60:5.1f}min {new / max(elapsed, 1):4.0f}new/s",
            flush=True,
        )
    await core.aclose()
    print(
        f"DONE rebuild: {new} new of {seen} parsed "
        f"({100 * (1 - new / max(seen, 1)):.1f}% duplicates), "
        f"{(time.time() - t0) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
