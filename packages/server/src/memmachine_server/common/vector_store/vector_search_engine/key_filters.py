"""KeyFilter implementations for vector search engines."""

import sqlite3

from .vector_search_engine import KeyFilter


class SQLKeyFilter(KeyFilter):
    """KeyFilter backed by a live SQLite query via a raw sync connection.

    Each ``__contains__`` call executes an indexed rowid lookup.
    Results are cached for the lifetime of the filter.

    The connection is created lazily on the first access, so it runs
    on the engine's worker thread (via ``asyncio.to_thread``).
    """

    def __init__(
        self,
        db_path: str,
        table_name: str,
        filter_sql: str,
    ) -> None:
        """Initialize with the database path and a SQL filter fragment."""
        self._db_path = db_path
        self._table_name = table_name
        self._filter_sql = filter_sql
        self._cache: dict[int, bool] = {}
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def __contains__(self, key: object) -> bool:
        """Return whether the key passes the SQL filter."""
        if not isinstance(key, int):
            return False
        if key in self._cache:
            return self._cache[key]
        row = (
            self._get_conn()
            .execute(
                f"SELECT 1 FROM [{self._table_name}] "
                f"WHERE rowid = ? AND {self._filter_sql}",
                (key,),
            )
            .fetchone()
        )
        result = row is not None
        self._cache[key] = result
        return result

    def close(self) -> None:
        """Close the underlying sync connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
