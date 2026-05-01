"""Utterance-anchor store. One row per doc representing the doc's own
creation/utterance time. Parallel to the `intervals` table — indexes
*when a doc was written*, not what referents the doc contains.

Schema:
    utterance_anchors(
        doc_id      TEXT PRIMARY KEY,
        earliest_us INTEGER NOT NULL,
        latest_us   INTEGER NOT NULL,
        best_us     INTEGER,
        granularity TEXT NOT NULL
    )

Granularity default = "day" (bracket [start-of-day, end-of-day)). If a
doc carries an explicit `granularity` metadata field, use it:

- "second"  -> ±0.5s bracket around ref_time
- "minute"  -> minute bracket
- "hour"    -> hour bracket
- "day"     -> day bracket (DEFAULT)
- "week"    -> ISO week bracket
- "month"   -> month bracket
- "year"    -> year bracket
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from schema import to_us

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS utterance_anchors (
    doc_id      TEXT PRIMARY KEY,
    earliest_us INTEGER NOT NULL,
    latest_us   INTEGER NOT NULL,
    best_us     INTEGER,
    granularity TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ua_earliest ON utterance_anchors(earliest_us);
CREATE INDEX IF NOT EXISTS idx_ua_latest   ON utterance_anchors(latest_us);
"""


def bracket_for(
    ref_time: datetime, granularity: str
) -> tuple[datetime, datetime, datetime]:
    """Return (earliest, latest, best) for a ref_time at a given granularity."""
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc)
    ref_time = ref_time.astimezone(timezone.utc)

    if granularity == "second":
        e = ref_time.replace(microsecond=0)
        l = e + timedelta(seconds=1)
    elif granularity == "minute":
        e = ref_time.replace(second=0, microsecond=0)
        l = e + timedelta(minutes=1)
    elif granularity == "hour":
        e = ref_time.replace(minute=0, second=0, microsecond=0)
        l = e + timedelta(hours=1)
    elif granularity == "day":
        e = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        l = e + timedelta(days=1)
    elif granularity == "week":
        # ISO week start = Monday
        weekday = ref_time.weekday()  # Monday=0
        e = ref_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=weekday
        )
        l = e + timedelta(days=7)
    elif granularity == "month":
        e = ref_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if e.month == 12:
            l = e.replace(year=e.year + 1, month=1)
        else:
            l = e.replace(month=e.month + 1)
    elif granularity == "year":
        e = ref_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        l = e.replace(year=e.year + 1)
    else:
        # Fallback: day
        e = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        l = e + timedelta(days=1)
    best = ref_time
    return e, l, best


class UtteranceAnchorStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        self.conn.executescript("DROP TABLE IF EXISTS utterance_anchors;")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def upsert_anchor(
        self,
        doc_id: str,
        ref_time: datetime,
        granularity: str = "day",
    ) -> None:
        """Insert/replace a doc's utterance anchor."""
        e, l, best = bracket_for(ref_time, granularity)
        self.conn.execute(
            "INSERT OR REPLACE INTO utterance_anchors "
            "(doc_id, earliest_us, latest_us, best_us, granularity) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, to_us(e), to_us(l), to_us(best), granularity),
        )
        self.conn.commit()

    def bulk_insert(
        self,
        items: Iterable[tuple[str, datetime, str]],
    ) -> None:
        """items: (doc_id, ref_time, granularity)."""
        rows = []
        for doc_id, ref_time, gran in items:
            e, l, best = bracket_for(ref_time, gran)
            rows.append((doc_id, to_us(e), to_us(l), to_us(best), gran))
        self.conn.executemany(
            "INSERT OR REPLACE INTO utterance_anchors "
            "(doc_id, earliest_us, latest_us, best_us, granularity) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def query_overlap(
        self, earliest_us: int, latest_us: int
    ) -> list[tuple[str, int, int, int | None, str]]:
        """Return (doc_id, earliest_us, latest_us, best_us, granularity)."""
        cur = self.conn.execute(
            "SELECT doc_id, earliest_us, latest_us, best_us, granularity "
            "FROM utterance_anchors WHERE earliest_us < ? AND latest_us > ?",
            (latest_us, earliest_us),
        )
        return cur.fetchall()

    def all_doc_ids(self) -> list[str]:
        cur = self.conn.execute("SELECT doc_id FROM utterance_anchors")
        return [r[0] for r in cur.fetchall()]
