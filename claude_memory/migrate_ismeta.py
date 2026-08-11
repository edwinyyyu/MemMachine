#!/usr/bin/env python3
"""Retype already-stored host-injected text from ``user_message`` to ``injected``.

Ingest now believes the transcript when it says a record is something the host put
into the session — ``isMeta`` — rather than sniffing the text for markers. The sniff
missed every large class: skill bodies, slash-command bodies, and the prompts behind
/commands all open with ordinary prose. This pass fixes the rows captured before that
change.

**It reads the transcripts, not the text.** Event uuids are derived from the source
record (``transcript._derive_event_uuid``), so a record flagged ``isMeta`` names its
stored events exactly. Nothing is guessed from content, which is the whole point:
the previous classifier's failure was that it guessed.

Metadata only. The text is untouched and nothing is re-embedded — search excludes
these because ``searchable_only`` admits only message sources, so retyping is
sufficient to take them off the search surface while leaving them on the timeline
where they belong.

    python3 -m claude_memory.migrate_ismeta            # dry run, prints counts
    python3 -m claude_memory.migrate_ismeta --apply    # backs up, then writes
"""

import argparse
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from claude_memory.transcript import _derive_event_uuid
from claude_memory.wire import Source

PROJECTS = Path.home() / ".claude" / "projects"
DEFAULT_DB = Path.home() / ".claude" / "claude_memory" / "segment.db"
DEFAULT_VECTOR_DB = Path.home() / ".claude" / "claude_memory" / "vector.db"


def injected_event_uuids() -> set[str]:
    """Every event uuid derivable from a transcript record flagged ``isMeta``.

    A record can produce several events (one per content block), and the derivation
    is by block index, so a generous range covers any record without needing to
    re-parse its content.
    """
    wanted: set[str] = set()
    for path in PROJECTS.glob("*/*.jsonl"):
        try:
            handle = path.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                if '"isMeta"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                record_uuid = record.get("uuid")
                if not record.get("isMeta") or not isinstance(record_uuid, str):
                    continue
                wanted.update(
                    _derive_event_uuid(record_uuid, index).hex for index in range(8)
                )
    return wanted


def _retyped(properties: str) -> str:
    payload = json.loads(properties)
    payload["source"] = {"v": str(Source.INJECTED), "t": "str"}
    return json.dumps(payload)


def plan_segments(
    conn: sqlite3.Connection, wanted: set[str]
) -> tuple[list[tuple[str, str, str]], Counter]:
    """Segments to retype, as (partition_key, uuid, new properties), plus counts."""
    changes: list[tuple[str, str, str]] = []
    counts: Counter = Counter()
    rows = conn.execute(
        "select partition_key, uuid, event_uuid, properties from segment_store_sg"
    )
    for partition_key, uuid, event_uuid, properties in rows:
        if event_uuid not in wanted:
            continue
        try:
            source = str(json.loads(properties)["source"]["v"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if source != Source.USER_MESSAGE:
            counts[f"already {source}"] += 1
            continue
        counts["retyped"] += 1
        changes.append((partition_key, uuid, _retyped(properties)))
    return changes, counts


def plan_vectors(
    conn: sqlite3.Connection, segment_uuids: set[str]
) -> list[tuple[str, int, str]]:
    """(table, row_id, new properties) for the vector records of retyped segments.

    The vector store keeps its OWN copy of each record's properties and search
    filters against that copy, never consulting segment.db — so retyping there
    alone would change nothing search can see.
    """
    changes: list[tuple[str, int, str]] = []
    for (table,) in conn.execute(
        "select name from sqlite_master where type='table' and name like '%_rc'"
    ):
        for row_id, properties in conn.execute(
            f'select row_id, properties from "{table}"'
        ):
            try:
                payload = json.loads(properties)
                segment_uuid = payload["_segment_uuid"]["v"].replace("-", "")
                source = payload["source"]["v"]
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                continue
            if segment_uuid in segment_uuids and source == Source.USER_MESSAGE:
                payload["source"] = {"v": str(Source.INJECTED), "t": "str"}
                changes.append((table, row_id, json.dumps(payload)))
    return changes


def main() -> int:
    """Report or apply the retype."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--vector-db", type=Path, default=DEFAULT_VECTOR_DB)
    args = parser.parse_args()

    wanted = injected_event_uuids()
    print(f"transcript records flagged isMeta -> {len(wanted):,} candidate event uuids")

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        segment_changes, counts = plan_segments(conn, wanted)
    segment_uuids = {uuid for _, uuid, _ in segment_changes}
    with sqlite3.connect(f"file:{args.vector_db}?mode=ro", uri=True) as conn:
        vector_changes = plan_vectors(conn, segment_uuids)

    print(f"  segments to retype : {len(segment_changes):,}")
    print(f"  vector records     : {len(vector_changes):,}")
    for label, count in sorted(counts.items()):
        print(f"  {label:18s} : {count:,}")
    if not args.apply:
        print("\ndry run; pass --apply to write")
        return 0
    if not segment_changes and not vector_changes:
        print("nothing to do")
        return 0

    for path in (args.db, args.vector_db):
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"backed up {path.name} -> {backup.name}")

    with sqlite3.connect(args.db) as conn:
        conn.executemany(
            "update segment_store_sg set properties = ? "
            "where partition_key = ? and uuid = ?",
            [(props, part, uuid) for part, uuid, props in segment_changes],
        )
    with sqlite3.connect(args.vector_db) as conn:
        for table in {table for table, _, _ in vector_changes}:
            conn.executemany(
                f'update "{table}" set properties = ? where row_id = ?',
                [(props, row) for tbl, row, props in vector_changes if tbl == table],
            )
    print(
        f"applied: {len(segment_changes):,} segments, {len(vector_changes):,} vectors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
