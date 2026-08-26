#!/usr/bin/env python3
"""Rebuild the vector index as a turbovec v7 container.

The deployed index is TVIM format version 3, written by turbovec 0.7.0. turbovec
1.0.0 reads and writes v7 only, and refuses a v3 file by name::

    ... is a version 3 turbovec index, which predates the v5 rotation change and
    cannot be decoded by any current build; rebuild it from the source vectors.

``turbovec.convert`` brings v5 and v6 files forward, but v3 predates the v5
rotation change, which altered every encoded byte, so there is nothing for a
converter to decode. Neither can the vectors be read back out of the old index:
the records table has no vector column and turbovec keeps only TurboQuant codes,
which is why ``TurboVecVectorSearchEngine.get_vectors`` raises. The index
therefore has to be rebuilt from the one source that survives -- the segment
text -- by re-deriving each record's embedding anchor and embedding it again.

What is preserved
-----------------
**Row ids.** The index keys are the vector store's ``row_id`` values, and every
one is carried over unchanged, so ``vector.db``, ``segment.db``, ``state/`` and
``demotions.json`` keep pointing at exactly what they pointed at before. Nothing
outside ``vector_index/`` is written.

**Demotions.** A demoted derivative's stored vector is ``normalize(d_orig -
delta)`` rather than ``d_orig`` (see ``MemoryCore.demote``), so the deltas in
``demotions.json`` are subtracted again here. Skipping them would silently undo
every demotion the user has made.

**The embedding anchor.** The text is rebuilt with the same
``MessageOnlyDeriver(WholeTextDeriver())`` and the same ``ingest_embed`` prompt
the ingest path uses, so a rebuilt vector is the vector ingest would have
written, up to the quantizer.

How it runs
-----------
Embedding ~100k records takes tens of minutes -- too long to hold the daemon
down. So the build is resumable and runs against a scratch path while the daemon
keeps serving the old index::

    python3 -m claude_memory.migrate_index_v7            # plan only, no model load
    python3 -m claude_memory.migrate_index_v7 --build    # daemon may keep running
    <stop the daemon>
    python3 -m claude_memory.migrate_index_v7 --build    # catch up what it added
    python3 -m claude_memory.migrate_index_v7 --install  # swap it in

A second ``--build`` embeds only the records the first pass did not cover, drops
the ids that have since been deleted, and commits with ``IdMapIndex.sync``, which
appends that delta to the existing container instead of restating the whole
index. That is the property v7 exists for, and it is what the catch-up pass costs
rather than a second full write.

``--install`` refuses unless the built index covers the live record set exactly,
so a swap can never publish an index that is quietly missing rows.

turbovec >= 1.0.0 must be importable, since only it writes v7. If the venv still
carries an older build, run these passes against an isolated install of 1.0.0
(``uv pip install --target <dir> turbovec==1.0.0``, then put ``<dir>`` first on
``PYTHONPATH``) so the daemon keeps serving the old container until ``--install``.
"""

import argparse
import asyncio
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.properties_json import decode_properties
from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.episodic_memory.event_memory.data_types import (
    NullContext,
    Segment,
    decode_block,
    decode_context,
)

from claude_memory.engine import MessageOnlyDeriver, WholeTextDeriver, build_embedder
from claude_memory.wire import MemoryConfig

# Records handed to the model per call. Big enough that sentence-transformers'
# own batching dominates, small enough that the progress line stays frequent.
_EMBED_BATCH = 512

# Segment fetch batch. SQLite's default parameter limit is well above this, but a
# smaller IN list keeps the query planner on the primary key.
_FETCH_BATCH = 500


def _block_text(block: object) -> str | None:
    """The text of a TextBlock, or None for any other block type."""
    return getattr(block, "text", None)


def _l2norm(vector: np.ndarray) -> np.ndarray:
    """Unit-normalize, leaving a zero vector alone."""
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _find_index_path(home: Path) -> Path:
    """The one index file in ``<home>/vector_index``."""
    directory = home / "vector_index"
    candidates = sorted(directory.glob("*.idx"))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one .idx in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def _find_records_table(vector_db: Path) -> str:
    """The one records table in the vector database."""
    with sqlite3.connect(f"file:{vector_db}?mode=ro", uri=True) as conn:
        names = [
            row[0]
            for row in conn.execute(
                "select name from sqlite_master "
                "where type = 'table' and name like '%\\_rc' escape '\\'"
            )
        ]
    if len(names) != 1:
        raise SystemExit(f"expected exactly one records table, found {names}")
    return names[0]


def _index_format_version(path: Path) -> str:
    """What container is at ``path``: 'v7', 'v<n>' for a legacy .tvim, or 'unknown'."""
    head = path.read_bytes()[:5]
    if head[:4] == b"TV7\0":
        return "v7"
    if head[:4] in (b"TVIM", b"TVPI"):
        return f"v{head[4]}"
    return "unknown"


def read_records(vector_db: Path, table: str) -> list[tuple[int, str, str]]:
    """Every (row_id, derivative uuid, segment uuid) in the records table.

    Both uuids come back in the dashless hex the segment store's ``CHAR(32)``
    columns use. The record's own uuid is already stored that way, but
    ``_segment_uuid`` is a user property serialized from a ``UUID``, so it carries
    dashes and will not join against ``segment_store_sg.uuid`` until normalized.
    """
    with sqlite3.connect(f"file:{vector_db}?mode=ro", uri=True) as conn:
        return [
            (int(row_id), str(uuid), UUID(str(segment_uuid)).hex)
            for row_id, uuid, segment_uuid in conn.execute(
                f'select row_id, uuid, json_extract(properties, "$._segment_uuid.v") '
                f'from "{table}" order by row_id'
            )
            if segment_uuid is not None
        ]


def read_segments(segment_db: Path, uuids: list[str]) -> dict[str, Segment]:
    """Load segments by uuid, decoded exactly as the segment store decodes them.

    The payload codec on this store is plaintext, so ``context`` and ``block`` are
    the UTF-8 JSON the encoders wrote; anything else would need the codec the
    partition row records.
    """
    found: dict[str, Segment] = {}
    with sqlite3.connect(f"file:{segment_db}?mode=ro", uri=True) as conn:
        for start in range(0, len(uuids), _FETCH_BATCH):
            chunk = uuids[start : start + _FETCH_BATCH]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                'select uuid, event_uuid, "index", "offset", timestamp, '
                "timestamp_timezone_offset, context, block, properties "
                f"from segment_store_sg where uuid in ({placeholders})",
                chunk,
            )
            for (
                uuid,
                event_uuid,
                index,
                offset,
                timestamp,
                tz_offset,
                context_blob,
                block_blob,
                properties_json,
            ) in rows:
                context = decode_context(json.loads(context_blob)) or NullContext()
                original_tz = timezone(timedelta(seconds=tz_offset))
                found[uuid] = Segment(
                    uuid=UUID(uuid),
                    event_uuid=UUID(event_uuid),
                    index=index,
                    offset=offset,
                    timestamp=ensure_tz_aware(
                        datetime.fromisoformat(timestamp)
                        if isinstance(timestamp, str)
                        else timestamp
                    ).astimezone(original_tz),
                    context=context,
                    block=decode_block(json.loads(block_blob)),
                    properties=decode_properties(json.loads(properties_json)),
                )
    return found


async def derive_texts(
    segments: dict[str, Segment], wanted: list[tuple[int, str, str]]
) -> tuple[list[int], list[str], list[str]]:
    """Rebuild each record's embedding anchor.

    Returns the row ids, their derivative uuids, and the texts, dropping any
    record whose segment is gone or whose block is not text -- both of which are
    unsearchable rather than an error.
    """
    deriver = MessageOnlyDeriver(WholeTextDeriver())
    row_ids: list[int] = []
    derivative_uuids: list[str] = []
    texts: list[str] = []
    for row_id, derivative_uuid, segment_uuid in wanted:
        segment = segments.get(segment_uuid)
        if segment is None:
            continue
        derivatives = await deriver.derive(segment, format_options=None)
        if len(derivatives) != 1:
            continue
        text = _block_text(derivatives[0].block)
        if text is None:
            continue
        row_ids.append(row_id)
        derivative_uuids.append(derivative_uuid)
        texts.append(text)
    return row_ids, derivative_uuids, texts


async def embed_chunk(
    chunk: list[tuple[int, str, str]],
    segment_db: Path,
    embedder: Embedder,
    demotions: dict[str, list[float]],
) -> tuple[list[int], np.ndarray]:
    """Re-derive and embed one chunk, returning its row ids and unit vectors.

    A record whose segment is gone, or whose block is not text, is not indexable
    and is simply absent from the result; the caller counts the shortfall.
    """
    segments = read_segments(segment_db, [seg for _, _, seg in chunk])
    row_ids, derivative_uuids, texts = await derive_texts(segments, chunk)
    vectors = np.zeros((len(texts), embedder.dimensions), dtype=np.float32)
    if not texts:
        return row_ids, vectors

    embedded = await embedder.ingest_embed(texts)
    for slot, (derivative_uuid, vector) in enumerate(
        zip(derivative_uuids, embedded, strict=True)
    ):
        array = np.asarray(vector, dtype=np.float64)
        delta = demotions.get(derivative_uuid)
        if delta is not None:
            # `demote` stores normalize(d_orig - delta), so a rebuild that
            # skipped this would silently undo every demotion the user made.
            array = _l2norm(array) - np.asarray(delta, dtype=np.float64)
        vectors[slot] = _l2norm(array).astype(np.float32)
    return row_ids, vectors


def _require_v7_writer() -> None:
    """Refuse to run unless the importable turbovec can write a v7 container."""
    version = turbovec_version()
    if version < (1, 0, 0):
        raise SystemExit(
            f"turbovec {'.'.join(map(str, version))} is importable, "
            "but writing a v7 container needs 1.0.0 or later"
        )


def _partition_work(
    records: list[tuple[int, str, str]], covered: set[int], limit: int | None
) -> tuple[list[tuple[int, str, str]], list[int]]:
    """Split the record set into what still needs embedding and what to drop.

    A `limit` run drops nothing: the ids it has not reached yet are
    indistinguishable from ids that have left the records table.
    """
    todo = [record for record in records if record[0] not in covered]
    if limit is not None:
        return todo[:limit], []
    live_ids = {row_id for row_id, _, _ in records}
    return todo, sorted(covered - live_ids)


def _report_shortfall(wanted: int, produced: int, skipped: int, dropped: int) -> None:
    """Account for what the pass did not index, and stop if it is most of it.

    A record whose segment is gone, or whose block is not text, is not indexable
    and is dropped. Silently dropping the whole set, though, is how a join bug
    ships as a successful run, so a majority loss raises instead of syncing.
    """
    if dropped:
        print(f"dropped {dropped:,} ids no longer in the records table")
    if skipped:
        print(f"skipped {skipped:,} records with no derivable text")
    if wanted and produced < wanted / 2:
        raise SystemExit(
            f"only {produced:,} of {wanted:,} records produced a vector; "
            "refusing to sync a mostly-empty index"
        )


async def build(args: argparse.Namespace) -> int:
    """Embed every uncovered record and commit it into the v7 container."""
    from turbovec import IdMapIndex

    _require_v7_writer()

    config = MemoryConfig.load()
    table = _find_records_table(config.vector_db)
    records = read_records(config.vector_db, table)
    demotions = json.loads((config.home / "demotions.json").read_text())

    out = args.out
    resuming = out.exists()
    covered = read_ledger(out) if resuming else set()
    index = IdMapIndex.load(str(out)) if resuming else None
    if resuming:
        print(
            f"resuming {out.name} ({_index_format_version(out)}, {len(covered):,} ids)"
        )

    todo, stale = _partition_work(records, covered, args.limit)
    print(
        f"records {len(records):,} | already covered {len(covered):,} | "
        f"to embed {len(todo):,} | to drop {len(stale):,}"
    )

    embedder = build_embedder(config.embedding_model)
    # `build_embedder` sizes the batch for interactive ingest, where a turn
    # embeds a handful of segments and latency is the whole cost. A bulk pass
    # wants the opposite: hand the model a whole chunk in one call and let
    # sentence-transformers do its own batching, which is several times faster
    # per record.
    embedder.batch_size = None
    dimensions = embedder.dimensions
    if index is None:
        index = IdMapIndex(dim=dimensions, bit_width=args.bits)

    # Buffer one checkpoint's worth, not the whole run: this array is the run's
    # memory floor, and checkpointing bounds both it and the work a crash costs.
    checkpoint = min(args.checkpoint, max(len(todo), 1))
    vectors = np.zeros((checkpoint, dimensions), dtype=np.float32)
    keys = np.zeros(checkpoint, dtype=np.uint64)
    buffered = 0
    committed = 0
    skipped = 0
    started = time.monotonic()

    def flush() -> None:
        """Add the buffer to the index and commit it, incrementally after the first."""
        nonlocal buffered, committed
        if buffered:
            index.add_with_ids(vectors[:buffered], keys[:buffered])
            covered.update(int(key) for key in keys[:buffered])
            committed += buffered
            buffered = 0
        # `sync` rather than `write`: the first sync of a fresh path writes the
        # file whole, and every later one appends only what changed since the
        # last commit -- so a checkpoint costs what changed, not what the index
        # holds. The ledger goes after it, so a ledger never claims more than
        # the container has committed.
        index.sync(str(out))
        write_ledger(out, covered)

    for start_at in range(0, len(todo), _EMBED_BATCH):
        chunk = todo[start_at : start_at + _EMBED_BATCH]
        row_ids, chunk_vectors = await embed_chunk(
            chunk, config.segment_db, embedder, demotions
        )
        skipped += len(chunk) - len(row_ids)
        for row_id, vector in zip(row_ids, chunk_vectors, strict=True):
            vectors[buffered] = vector
            keys[buffered] = row_id
            buffered += 1
            if buffered == checkpoint:
                flush()
        done = committed + buffered
        elapsed = time.monotonic() - started
        print(
            f"  {done:,}/{len(todo):,}  {elapsed / 60:.1f}min  "
            f"{done / max(elapsed, 1e-9):.0f}/s  committed {committed:,}",
            flush=True,
        )

    for key in stale:
        index.remove(key)
        covered.discard(key)

    _report_shortfall(len(todo), committed + buffered, skipped, len(stale))
    flush()
    print(
        f"synced {out} ({_index_format_version(out)}, "
        f"{out.stat().st_size / 1e6:.1f}MB, {len(covered):,} ids) in "
        f"{(time.monotonic() - started) / 60:.1f}min"
    )
    return 0


def _ledger_path(out: Path) -> Path:
    """Sidecar recording which row ids the built index holds."""
    return out.with_suffix(out.suffix + ".ids.json")


def read_ledger(out: Path) -> set[int]:
    """Row ids the built index is known to hold, or an empty set for a fresh build.

    ``IdMapIndex`` cannot enumerate its ids in 1.0.0 (``iter_ids`` lands after it),
    and membership can only be asked one id at a time. Keeping the ledger beside
    the container is what lets a resume know which ids to *drop* -- an id deleted
    from the records table between passes is in the index and in no live query, so
    without the ledger there is nothing left to name it.
    """
    path = _ledger_path(out)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def write_ledger(out: Path, ids: set[int]) -> None:
    """Record the row ids the built index now holds."""
    _ledger_path(out).write_text(json.dumps(sorted(ids)))


def turbovec_version() -> tuple[int, ...]:
    """The importable turbovec's version as a tuple."""
    import turbovec

    return tuple(int(part) for part in turbovec.__version__.split(".")[:3])


def install(args: argparse.Namespace) -> int:
    """Swap the built index in, once it is proven to cover the live record set."""
    from turbovec import IdMapIndex

    config = MemoryConfig.load()
    table = _find_records_table(config.vector_db)
    records = read_records(config.vector_db, table)
    live = _find_index_path(config.home)

    if not args.out.exists():
        raise SystemExit(f"{args.out} does not exist; run --build first")
    index = IdMapIndex.load(str(args.out))
    missing = [row_id for row_id, _, _ in records if not index.contains(row_id)]
    if missing:
        raise SystemExit(
            f"{args.out.name} is missing {len(missing):,} of {len(records):,} live "
            "records; stop the daemon and re-run --build to catch up"
        )

    backup = live.with_suffix(live.suffix + ".v3.bak")
    shutil.copy2(live, backup)
    print(f"backed up {live.name} -> {backup.name}")
    shutil.move(str(args.out), str(live))
    print(f"installed {live} ({_index_format_version(live)})")
    return 0


def plan(args: argparse.Namespace) -> int:
    """Report what a rebuild would cover, without loading the embedding model."""
    config = MemoryConfig.load()
    table = _find_records_table(config.vector_db)
    records = read_records(config.vector_db, table)
    live = _find_index_path(config.home)
    print(f"home           {config.home}")
    print(f"live index     {live.name} ({_index_format_version(live)})")
    print(f"records        {len(records):,} in {table}")
    print(f"build target   {args.out}")
    if args.out.exists():
        print(f"built so far   {args.out.name} ({_index_format_version(args.out)})")
    print("\nnothing written; pass --build to embed, then --install to swap in")
    return 0


def main() -> int:
    """Parse arguments and run the requested pass."""
    default_out = Path.home() / ".claude" / "claude_memory" / "vector_index.v7build"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="embed and sync the delta")
    parser.add_argument(
        "--install", action="store_true", help="swap the built index in"
    )
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=20_000,
        help="commit an incremental sync every this many embedded records",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="embed at most this many uncovered records (smoke runs)",
    )
    args = parser.parse_args()

    if args.build and args.install:
        raise SystemExit("run --build and --install as separate passes")
    if args.build:
        return asyncio.run(build(args))
    if args.install:
        return install(args)
    return plan(args)


if __name__ == "__main__":
    sys.exit(main())
