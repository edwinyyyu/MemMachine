#!/usr/bin/env python3
"""Reclassify already-stored injected text from ``user_message`` to ``injected``.

Ingest now separates text the user typed from text that was loaded into the session
around them (``wire.user_text_source``). Everything captured before that change is
still filed as ``user_message``, which means it is embedded and competing in search.
This pass fixes the existing rows.

It is a metadata migration, not a rebuild. Nothing is re-embedded.

There are TWO stores to update, and missing the second one is why an earlier version
of this script appeared to work while search kept returning the same boilerplate:

* ``segment.db`` holds the timeline. Its ``properties`` column is what expansion
  reads and filters on.
* ``vector.db`` holds the search surface, and keeps its OWN copy of each record's
  properties (``source``, ``session_id``, ``_segment_uuid``). Search filters against
  that copy and resolves hits back to segments through ``_segment_uuid`` — it never
  consults ``segment.db``, so retyping there alone changes nothing it can see.

The text itself is untouched in both, so everything stays reachable by expand.

Classification is per EVENT, not per segment. Only an event's first chunk carries the
marker that identifies it — a later chunk of a task notification looks like ordinary
prose — so the first chunk decides, and the verdict applies to all of that event's
segments. Ingest classifies the whole text before chunking, so this matches it.

    python3 -m claude_memory.migrate_injected            # dry run, prints counts
    python3 -m claude_memory.migrate_injected --apply    # backs up, then writes
"""

import argparse
import json
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from uuid import UUID

from claude_memory.wire import Source, user_text_source

DEFAULT_DB = Path.home() / ".claude" / "claude_memory" / "segment.db"


def _source_of(properties: str) -> str:
    """The stored source value, which is wrapped in a typed envelope."""
    try:
        return str(json.loads(properties)["source"]["v"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""


def _retyped(properties: str, source: Source) -> str:
    payload = json.loads(properties)
    payload["source"] = {"v": str(source), "t": "str"}
    return json.dumps(payload)


def plan(conn: sqlite3.Connection) -> tuple[list[tuple[str, str, str]], Counter]:
    """Segments to retype, as (partition_key, uuid, new properties), plus counts.

    ``partition_key`` is carried because the primary key is (partition_key, uuid).
    Addressing a row by ``uuid`` alone cannot use that index, so each update would
    scan the whole table — 575k rows per statement.
    """
    events: dict[str, list[tuple[int, int, str, str, str, str]]] = defaultdict(list)
    rows = conn.execute(
        'select partition_key, uuid, event_uuid, "index", "offset", block, properties '
        "from segment_store_sg"
    )
    for partition_key, uuid, event_uuid, index, offset, block, properties in rows:
        if _source_of(properties) != Source.USER_MESSAGE:
            continue
        events[event_uuid].append(
            (index, offset, partition_key, uuid, block, properties)
        )

    changes: list[tuple[str, str, str]] = []
    counts: Counter = Counter()
    for segments in events.values():
        segments.sort()
        try:
            head = json.loads(segments[0][4]).get("text", "")
        except (json.JSONDecodeError, TypeError):
            head = ""
        if user_text_source(head) is not Source.INJECTED:
            counts["kept as user_message"] += len(segments)
            continue
        counts["retyped to injected"] += len(segments)
        for _, _, partition_key, uuid, _, properties in segments:
            changes.append((partition_key, uuid, _retyped(properties, Source.INJECTED)))
    counts["events examined"] = len(events)
    return changes, counts


def _record_tables(conn: sqlite3.Connection) -> list[str]:
    """The vector store's per-collection record tables."""
    return [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type='table' and name like ?",
            ("vector_store_sqlite_%_rc",),
        )
    ]


def _vector_rows(vector_db: Path, retyped: set[str]) -> list[tuple[str, int, str]]:
    """(table, row_id, new properties) for vector records of retyped segments."""
    out: list[tuple[str, int, str]] = []
    conn = sqlite3.connect(f"file:{vector_db}?mode=ro", uri=True)
    try:
        for table in _record_tables(conn):
            for row_id, properties in conn.execute(
                f'select row_id, properties from "{table}"'
            ):
                try:
                    payload = json.loads(properties)
                    segment = payload["_segment_uuid"]["v"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                if segment not in retyped:
                    continue
                payload["source"] = {"v": str(Source.INJECTED), "t": "str"}
                out.append((table, row_id, json.dumps(payload)))
    finally:
        conn.close()
    return out


def main(argv: list[str] | None = None) -> int:
    """Report the plan, and write it when --apply is given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--apply", action="store_true", help="write the change (default: dry run)"
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"No store at {args.db}", file=sys.stderr)
        return 1

    uri = f"file:{args.db}?mode=ro"
    read_conn = sqlite3.connect(uri, uri=True)
    try:
        changes, counts = plan(read_conn)
        # Every segment that IS injected, not merely the ones this run retypes, so a
        # re-run repairs a partial migration rather than finding nothing left to do.
        already = {
            uuid
            for uuid, properties in read_conn.execute(
                "select uuid, properties from segment_store_sg"
            )
            if _source_of(properties) == Source.INJECTED
        }
    finally:
        read_conn.close()

    # The vector store keys its records by _segment_uuid in hyphenated form, while
    # the segment store stores bare hex.
    retyped = {str(UUID(uuid)) for uuid in already | {u for _, u, _ in changes}}
    vector_db = args.db.with_name("vector.db")
    vector_rows = _vector_rows(vector_db, retyped) if vector_db.exists() else []

    print(f"events examined        : {counts['events examined']}")
    print(f"segments -> injected   : {counts['retyped to injected']}")
    print(f"segments unchanged     : {counts['kept as user_message']}")
    print(f"vector records retyped : {len(vector_rows)}  (in {vector_db.name})")
    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        return 0

    backup = args.db.with_suffix(args.db.suffix + ".pre-injected.bak")
    print(f"\nbacking up to {backup} …", flush=True)
    shutil.copy2(args.db, backup)
    vector_backup = vector_db.with_suffix(vector_db.suffix + ".pre-injected.bak")
    if vector_db.exists():
        shutil.copy2(vector_db, vector_backup)

    with sqlite3.connect(args.db) as conn:
        # Addressed through (partition_key, uuid), the primary key. Matching on
        # uuid alone turns each statement into a full table scan.
        conn.executemany(
            "update segment_store_sg set properties = ? "
            "where partition_key = ? and uuid = ?",
            [(properties, part, uuid) for part, uuid, properties in changes],
        )

    if vector_rows:
        with sqlite3.connect(vector_db) as conn:
            for table in {table for table, _, _ in vector_rows}:
                conn.executemany(
                    f'update "{table}" set properties = ? where row_id = ?',
                    [
                        (properties, row_id)
                        for tbl, row_id, properties in vector_rows
                        if tbl == table
                    ],
                )

    print(
        f"done: {len(changes)} segments and {len(vector_rows)} vector records retyped"
    )
    print(f"to undo: cp {backup} {args.db}")
    if vector_rows:
        print(f"         cp {vector_backup} {vector_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
