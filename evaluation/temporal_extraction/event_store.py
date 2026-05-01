"""E1 — SQLite storage for event-time bindings.

Table: event_time_bindings(binding_id, doc_id, event_span, event_vec,
earliest_us, latest_us, best_us, granularity). event_vec stored as
JSON-serialized float list (pragmatic, not production).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_time_bindings (
    binding_id     INTEGER PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    event_span     TEXT,
    event_vec      TEXT,
    earliest_us    INTEGER NOT NULL,
    latest_us      INTEGER NOT NULL,
    best_us        INTEGER,
    granularity    TEXT
);

CREATE INDEX IF NOT EXISTS idx_etb_doc ON event_time_bindings(doc_id);
CREATE INDEX IF NOT EXISTS idx_etb_earliest ON event_time_bindings(earliest_us);
CREATE INDEX IF NOT EXISTS idx_etb_latest ON event_time_bindings(latest_us);
"""


class EventStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        self.conn.executescript("DROP TABLE IF EXISTS event_time_bindings;")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def insert(
        self,
        doc_id: str,
        event_span: str | None,
        event_vec: np.ndarray | None,
        earliest_us: int,
        latest_us: int,
        best_us: int | None,
        granularity: str | None,
    ) -> int:
        vec_str = (
            json.dumps([float(x) for x in event_vec.tolist()])
            if event_vec is not None
            else None
        )
        cur = self.conn.execute(
            "INSERT INTO event_time_bindings(doc_id, event_span, event_vec, "
            "earliest_us, latest_us, best_us, granularity) VALUES "
            "(?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                event_span,
                vec_str,
                earliest_us,
                latest_us,
                best_us,
                granularity,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def all_bindings(self) -> list[tuple]:
        cur = self.conn.execute(
            "SELECT binding_id, doc_id, event_span, event_vec, earliest_us, "
            "latest_us, best_us, granularity FROM event_time_bindings"
        )
        return cur.fetchall()

    @staticmethod
    def parse_vec(v: str | None) -> np.ndarray | None:
        if not v:
            return None
        return np.array(json.loads(v), dtype=np.float32)
