"""KeyFilter backed by a sync SQLAlchemy session."""

from sqlalchemy import Table, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement


class SQLKeyFilter:
    """Per-candidate SQL filter using a sync SQLAlchemy session.

    Each ``__contains__`` call executes an indexed rowid lookup.
    Results are cached for the lifetime of the filter.
    """

    def __init__(
        self,
        sync_engine: Engine,
        records_table: Table,
        filter_expression: ColumnElement[bool],
    ) -> None:
        """Initialize with a sync engine, records table, and filter expression."""
        self._sync_engine = sync_engine
        self._records_table = records_table
        self._filter_expression = filter_expression
        self._cache: dict[int, bool] = {}
        self._session: Session | None = None

    def _get_session(self) -> Session:
        if self._session is None:
            self._session = Session(self._sync_engine)
        return self._session

    def __contains__(self, key: object) -> bool:
        """Return whether the key passes the SQL filter."""
        if not isinstance(key, int):
            return False
        if key in self._cache:
            return self._cache[key]
        row = (
            self._get_session()
            .execute(
                select(self._records_table.c.rowid).where(
                    self._records_table.c.rowid == key,
                    self._filter_expression,
                )
            )
            .scalar()
        )
        result = row is not None
        self._cache[key] = result
        return result

    def close(self) -> None:
        """Close the underlying sync session."""
        if self._session is not None:
            self._session.close()
            self._session = None
