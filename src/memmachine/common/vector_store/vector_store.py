"""
Abstract base class for a vector store.

Defines the interface for adding, searching,
and deleting records.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from uuid import UUID

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)

from .data_types import PropertyValue, Record


class VectorStore(ABC):
    """Abstract base class for a vector store."""

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        *,
        vector_dimensions: int,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        properties_schema: Mapping[str, type[PropertyValue]] | None = None,
    ) -> None:
        """
        Create a collection in the vector store.

        Args:
            name (str):
                Name of the collection to create.
            vector_dimensions (int):
                Number of dimensions for the vectors.
            similarity_metric (SimilarityMetric):
                Similarity metric to use for vector comparisons
                (default: SimilarityMetric.COSINE).
            properties_schema (Mapping[str, type] | None):
                Mapping of property names to their types
                (default: None).

        """
        raise NotImplementedError

    @abstractmethod
    async def add(
        self,
        *,
        collection: str,
        records: Iterable[Record],
    ) -> None:
        """
        Add records to the vector store.

        Args:
            collection (str):
                Collection that the records belong to.
            records (Iterable[Record]):
                Iterable of records to add.

        """
        raise NotImplementedError

    @abstractmethod
    async def query(
        self,
        *,
        collection: str,
        query_vector: Sequence[float],
        similarity_threshold: float | None = None,
        limit: int | None = None,
        property_filter: FilterExpr | None = None,
    ) -> Iterable[Record]:
        """
        Query for records matching the criteria by vector similarity.

        Args:
            collection (str):
                Collection that the records belong to.
            query_vector (Sequence[float] | None):
                The vector to compare against.
            similarity_threshold (float | None):
                Minimum similarity score to consider a match
                (default: None).
            limit (int | None):
                Maximum number of matching records to return.
                If None, return as many matching records as possible
                (default: None).
            property_filter (FilterExpr | None):
                Filter expression tree.
                If None or empty, no property filtering is applied
                (default: None).

        Returns:
            Iterable[Record]:
                List of records matching the search criteria,
                ordered by similarity score descending.

        """
        raise NotImplementedError

    @abstractmethod
    async def get(
        self,
        *,
        collection: str,
        record_uuids: Iterable[UUID],
    ) -> Iterable[Record]:
        """
        Get records from the collection by their UUIDs.

        Args:
            collection (str):
                Name of the collection containing the records.
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to retrieve.

        Returns:
            Iterable[Record]:
                List of records with the specified UUIDs,
                ordered as in the input iterable.

        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        *,
        collection: str,
        record_uuids: Iterable[UUID],
    ) -> None:
        """
        Delete records from the collection.

        Args:
            collection (str):
                Name of the collection containing the records.
            record_uuids (Iterable[UUID]):
                Iterable of UUIDs of the records to delete.

        """
        raise NotImplementedError

    @abstractmethod
    async def drop_collection(self, collection: str) -> None:
        """Drop everything from the collection."""
        raise NotImplementedError

    @abstractmethod
    async def startup(self) -> None:
        """Startup."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown."""
        raise NotImplementedError
