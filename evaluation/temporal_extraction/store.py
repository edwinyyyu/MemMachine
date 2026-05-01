"""SQLite-backed interval store for TimeExpression retrieval."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from expander import expand
from schema import (
    TimeExpression,
    time_expression_to_dict,
    to_us,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS expressions (
    expr_id        INTEGER PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    kind           TEXT NOT NULL,
    surface        TEXT NOT NULL,
    ref_time       INTEGER NOT NULL,
    confidence     REAL NOT NULL,
    rrule          TEXT,
    dtstart_us     INTEGER,
    until_us       INTEGER,
    duration_us    INTEGER,
    payload        TEXT
);

CREATE TABLE IF NOT EXISTS intervals (
    iv_id          INTEGER PRIMARY KEY,
    expr_id        INTEGER NOT NULL REFERENCES expressions(expr_id),
    doc_id         TEXT NOT NULL,
    earliest_us    INTEGER NOT NULL,
    latest_us      INTEGER NOT NULL,
    best_us        INTEGER,
    granularity    TEXT NOT NULL,
    is_instance    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_iv_earliest ON intervals(earliest_us);
CREATE INDEX IF NOT EXISTS idx_iv_latest   ON intervals(latest_us);
CREATE INDEX IF NOT EXISTS idx_iv_doc      ON intervals(doc_id);
"""


class IntervalStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        self.conn.executescript(
            "DROP TABLE IF EXISTS intervals; DROP TABLE IF EXISTS expressions;"
        )
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def insert_expression(
        self,
        doc_id: str,
        te: TimeExpression,
        recurrence_window_years: int = 10,
    ) -> int:
        """Insert a TimeExpression; fan out intervals; return expr_id."""
        rrule = (
            te.recurrence.rrule if te.kind == "recurrence" and te.recurrence else None
        )
        dtstart_us = (
            to_us(te.recurrence.dtstart.best or te.recurrence.dtstart.earliest)
            if te.kind == "recurrence" and te.recurrence
            else None
        )
        until_us = (
            to_us(
                (te.recurrence.until.latest or te.recurrence.until.earliest)
                if te.recurrence.until
                else None
            )
            if te.kind == "recurrence" and te.recurrence
            else None
        )
        duration_us = (
            int(te.duration.total_seconds() * 1_000_000)
            if te.kind == "duration" and te.duration is not None
            else None
        )
        payload = json.dumps(time_expression_to_dict(te))

        cur = self.conn.execute(
            "INSERT INTO expressions (doc_id, kind, surface, ref_time, "
            "confidence, rrule, dtstart_us, until_us, duration_us, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                te.kind,
                te.surface,
                to_us(te.reference_time),
                te.confidence,
                rrule,
                dtstart_us,
                until_us,
                duration_us,
                payload,
            ),
        )
        expr_id = cur.lastrowid

        # Fan out intervals.
        intervals_to_insert: list[tuple[int, str, int, int, int | None, str, int]] = []
        if te.kind == "instant" and te.instant:
            intervals_to_insert.append(
                (
                    expr_id,
                    doc_id,
                    to_us(te.instant.earliest),
                    to_us(te.instant.latest),
                    to_us(te.instant.best),
                    te.instant.granularity,
                    0,
                )
            )
        elif te.kind == "interval" and te.interval:
            e = te.interval.start.earliest
            l = te.interval.end.latest
            best = te.interval.start.best or te.interval.start.earliest
            # granularity: coarsest of start/end
            from schema import GRANULARITY_ORDER

            g = (
                te.interval.start.granularity
                if GRANULARITY_ORDER[te.interval.start.granularity]
                >= GRANULARITY_ORDER[te.interval.end.granularity]
                else te.interval.end.granularity
            )
            intervals_to_insert.append(
                (
                    expr_id,
                    doc_id,
                    to_us(e),
                    to_us(l),
                    to_us(best),
                    g,
                    0,
                )
            )
        elif te.kind == "recurrence" and te.recurrence:
            now = datetime.now(tz=timezone.utc)
            dtstart = te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
            anchor = dtstart or now
            window_start = min(
                now - timedelta(days=365 * recurrence_window_years),
                anchor - timedelta(days=365),
            )
            window_end = now + timedelta(days=365 * 2)
            if te.recurrence.until is not None:
                window_end = min(
                    window_end,
                    te.recurrence.until.latest or te.recurrence.until.earliest,
                )
            instances = expand(te.recurrence, window_start, window_end)
            for inst in instances:
                intervals_to_insert.append(
                    (
                        expr_id,
                        doc_id,
                        to_us(inst.earliest),
                        to_us(inst.latest),
                        to_us(inst.best),
                        inst.granularity,
                        1,
                    )
                )
        # duration: no interval inserted (per DESIGN)

        if intervals_to_insert:
            self.conn.executemany(
                "INSERT INTO intervals (expr_id, doc_id, earliest_us, "
                "latest_us, best_us, granularity, is_instance) VALUES "
                "(?, ?, ?, ?, ?, ?, ?)",
                intervals_to_insert,
            )
        self.conn.commit()
        return expr_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query_overlap(
        self, earliest_us: int, latest_us: int
    ) -> list[tuple[int, str, int, int, int | None, str]]:
        """Return (expr_id, doc_id, earliest_us, latest_us, best_us, granularity)."""
        cur = self.conn.execute(
            "SELECT expr_id, doc_id, earliest_us, latest_us, best_us, granularity "
            "FROM intervals WHERE earliest_us < ? AND latest_us > ?",
            (latest_us, earliest_us),
        )
        return cur.fetchall()

    def all_doc_ids(self) -> list[str]:
        cur = self.conn.execute("SELECT DISTINCT doc_id FROM intervals")
        return [r[0] for r in cur.fetchall()]
