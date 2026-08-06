#!/usr/bin/env python3
"""Reclassify already-stored injected text from ``user_message`` to ``injected``.

Ingest now separates text the user typed from text that was loaded into the session
around them (``wire.user_text_source``). Everything captured before that change is
still filed as ``user_message``, which means it is embedded and competing in search.
This pass fixes the existing rows.

It is a metadata migration, not a rebuild: ``properties`` is a JSON column on
``segment_store_sg``, and the embeddings live in a separate table keyed by segment
uuid. Nothing is re-embedded. Reclassified segments have their vector rows deleted,
which is what removes them from the search surface; their text stays on the timeline
and is still reachable by expand.

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


def plan(conn: sqlite3.Connection) -> tuple[list[tuple[str, str]], Counter]:
    """Segments to retype, as (uuid, new properties), plus per-event counts."""
    events: dict[str, list[tuple[int, int, str, str, str]]] = defaultdict(list)
    rows = conn.execute(
        'select uuid, event_uuid, "index", "offset", block, properties '
        "from segment_store_sg"
    )
    for uuid, event_uuid, index, offset, block, properties in rows:
        if _source_of(properties) != Source.USER_MESSAGE:
            continue
        events[event_uuid].append((index, offset, uuid, block, properties))

    changes: list[tuple[str, str]] = []
    counts: Counter = Counter()
    for segments in events.values():
        segments.sort()
        try:
            head = json.loads(segments[0][3]).get("text", "")
        except (json.JSONDecodeError, TypeError):
            head = ""
        if user_text_source(head) is not Source.INJECTED:
            counts["kept as user_message"] += len(segments)
            continue
        counts["retyped to injected"] += len(segments)
        for _, _, uuid, _, properties in segments:
            changes.append((uuid, _retyped(properties, Source.INJECTED)))
    counts["events examined"] = len(events)
    return changes, counts


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
    with sqlite3.connect(uri, uri=True) as read_conn:
        changes, counts = plan(read_conn)
        vectors = 0
        for start in range(0, len(changes), 900):
            batch = [uuid for uuid, _ in changes[start : start + 900]]
            placeholders = ",".join("?" * len(batch))
            vectors += read_conn.execute(
                f"select count(*) from segment_store_dv_ln "
                f"where segment_uuid in ({placeholders})",
                batch,
            ).fetchone()[0]

    print(f"events examined      : {counts['events examined']}")
    print(f"segments -> injected : {counts['retyped to injected']}")
    print(f"segments unchanged   : {counts['kept as user_message']}")
    print(f"embeddings dropped   : {vectors}")
    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        return 0

    backup = args.db.with_suffix(args.db.suffix + ".pre-injected.bak")
    print(f"\nbacking up to {backup} …", flush=True)
    shutil.copy2(args.db, backup)

    with sqlite3.connect(args.db) as conn:
        conn.executemany(
            "update segment_store_sg set properties = ? where uuid = ?",
            [(properties, uuid) for uuid, properties in changes],
        )
        for start in range(0, len(changes), 900):
            batch = [uuid for uuid, _ in changes[start : start + 900]]
            placeholders = ",".join("?" * len(batch))
            conn.execute(
                f"delete from segment_store_dv_ln "
                f"where segment_uuid in ({placeholders})",
                batch,
            )
    print(f"done: {len(changes)} segments retyped, {vectors} embeddings removed")
    print(f"to undo: cp {backup} {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
