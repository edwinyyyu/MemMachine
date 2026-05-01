"""SQLite-backed tag index + in-memory inverted index for hierarchical tags.

Schema:
    time_tags(doc_id TEXT, expr_id INTEGER, tag TEXT)
    INDEX on (tag).

Also exposes an in-memory ``inverted`` dict[tag -> list[(doc_id, expr_id)]]
and a forward dict[(doc_id, expr_id) -> set[tag]] for fast per-pair
scoring during evaluation.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from hierarchical_tags import tags_for_expression
from schema import TimeExpression

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS time_tags (
    doc_id   TEXT NOT NULL,
    expr_id  INTEGER NOT NULL,
    tag      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_time_tags_tag ON time_tags(tag);
CREATE INDEX IF NOT EXISTS idx_time_tags_doc ON time_tags(doc_id);
"""


class TagStore:
    def __init__(self, path: str | Path | None = None):
        """If ``path`` is None, run fully in-memory (no SQLite)."""
        self.path = str(path) if path is not None else ":memory:"
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

        # In-memory indices
        self.inverted: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.forward: dict[tuple[str, int], set[str]] = {}
        # Document frequency (number of distinct doc_ids carrying the tag).
        self._doc_tags: dict[str, set[str]] = defaultdict(set)

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        self.conn.executescript("DROP TABLE IF EXISTS time_tags;")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self.inverted.clear()
        self.forward.clear()
        self._doc_tags.clear()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def insert_expression(
        self,
        doc_id: str,
        expr_id: int,
        te: TimeExpression,
    ) -> set[str]:
        """Generate tags for ``te`` and insert into the store. Returns tag set."""
        tags = tags_for_expression(te)
        if not tags:
            return tags
        rows = [(doc_id, expr_id, t) for t in tags]
        self.conn.executemany(
            "INSERT INTO time_tags (doc_id, expr_id, tag) VALUES (?, ?, ?)",
            rows,
        )
        key = (doc_id, expr_id)
        self.forward[key] = set(tags)
        for t in tags:
            self.inverted[t].append(key)
            self._doc_tags[doc_id].add(t)
        return tags

    def bulk_insert(
        self,
        exprs_by_doc: dict[str, list[TimeExpression]],
    ) -> dict[tuple[str, int], set[str]]:
        """Insert every expression; expr_id auto-assigned per doc."""
        self.conn.execute("BEGIN")
        try:
            for doc_id, tes in exprs_by_doc.items():
                for i, te in enumerate(tes):
                    self.insert_expression(doc_id, i, te)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return dict(self.forward)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def candidates_for_tags(self, tags: Iterable[str]) -> set[tuple[str, int]]:
        """Return every (doc_id, expr_id) carrying any of the given tags."""
        out: set[tuple[str, int]] = set()
        for t in tags:
            out.update(self.inverted.get(t, ()))
        return out

    def num_docs(self) -> int:
        return len(self._doc_tags)

    def tag_doc_frequency(self, tag: str) -> int:
        """Number of distinct doc_ids carrying this tag."""
        docs = {d for d, _ in self.inverted.get(tag, ())}
        return len(docs)

    def all_doc_ids(self) -> list[str]:
        return sorted(self._doc_tags.keys())
