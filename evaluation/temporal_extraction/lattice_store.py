"""SQLite inverted index for lattice tags.

Schema:

    lattice_tags(doc_id TEXT, tag TEXT, precision TEXT, axis_type TEXT)
    lattice_doc_tags_view(doc_id TEXT, tags TEXT)  -- JSON of tags per doc, for debug

Indexes on (tag) for fast lookup by cell and (doc_id) for per-doc tag
fetches. This store is pure SQLite — NO LLM at retrieval time.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS lattice_tags (
        doc_id TEXT NOT NULL,
        tag TEXT NOT NULL,
        precision TEXT NOT NULL,
        axis_type TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lattice_tag ON lattice_tags(tag)",
    "CREATE INDEX IF NOT EXISTS idx_lattice_doc ON lattice_tags(doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_lattice_axis ON lattice_tags(axis_type)",
]


class LatticeStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL;")
        for ddl in _SCHEMA:
            self.con.execute(ddl)

    def clear(self) -> None:
        self.con.execute("DELETE FROM lattice_tags")

    def insert(
        self,
        doc_id: str,
        absolute_tags: Iterable[tuple[str, str]],  # (precision, tag)
        cyclical_tags: Iterable[str],
    ) -> None:
        rows: list[tuple[str, str, str, str]] = []
        for prec, tag in absolute_tags:
            rows.append((doc_id, tag, prec, "absolute"))
        for tag in cyclical_tags:
            head = tag.split(":", 1)[0]
            rows.append((doc_id, tag, head, "cyclical"))
        if rows:
            self.con.executemany(
                "INSERT INTO lattice_tags(doc_id, tag, precision, axis_type) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def query_by_tags(self, tags: Iterable[str]) -> dict[str, set[str]]:
        """Return a dict {doc_id -> matched_tag_set} for all rows whose
        tag is in ``tags``.
        """
        tlist = list(tags)
        if not tlist:
            return {}
        out: dict[str, set[str]] = {}
        # SQLite has a parameter-count limit (~999). Batch.
        CHUNK = 500
        for i in range(0, len(tlist), CHUNK):
            chunk = tlist[i : i + CHUNK]
            placeholders = ",".join(["?"] * len(chunk))
            q = f"SELECT doc_id, tag FROM lattice_tags WHERE tag IN ({placeholders})"
            for doc_id, tag in self.con.execute(q, chunk):
                out.setdefault(doc_id, set()).add(tag)
        return out

    def tags_for_doc(self, doc_id: str) -> list[dict]:
        cur = self.con.execute(
            "SELECT tag, precision, axis_type FROM lattice_tags WHERE doc_id = ?",
            (doc_id,),
        )
        return [
            {"tag": t, "precision": p, "axis_type": a} for t, p, a in cur.fetchall()
        ]

    def all_doc_ids(self) -> list[str]:
        cur = self.con.execute("SELECT DISTINCT doc_id FROM lattice_tags")
        return [r[0] for r in cur.fetchall()]

    def tag_frequencies(self) -> list[tuple[str, int]]:
        cur = self.con.execute(
            "SELECT tag, COUNT(*) c FROM lattice_tags GROUP BY tag ORDER BY c DESC"
        )
        return cur.fetchall()

    def stats(self) -> dict:
        n_docs = len(self.all_doc_ids())
        n_tag_rows = self.con.execute("SELECT COUNT(*) FROM lattice_tags").fetchone()[0]
        n_uniq = self.con.execute(
            "SELECT COUNT(DISTINCT tag) FROM lattice_tags"
        ).fetchone()[0]
        per_doc = {}
        for doc_id, n in self.con.execute(
            "SELECT doc_id, COUNT(*) FROM lattice_tags GROUP BY doc_id"
        ):
            per_doc[doc_id] = n
        vals = list(per_doc.values()) or [0]
        return {
            "n_docs_tagged": n_docs,
            "n_rows": n_tag_rows,
            "n_unique_tags": n_uniq,
            "avg_tags_per_doc": sum(vals) / len(vals),
            "max_tags_per_doc": max(vals),
            "min_tags_per_doc": min(vals),
        }

    def close(self) -> None:
        self.con.close()
