"""Adapter between :class:`SemanticAttribute` and the vector store record shape.

This module is pure helpers (no I/O).  It encodes the reserved-prefix
convention for system metadata and the filter-field translation
between user-facing names and vector-store property keys.

System vs user metadata
-----------------------
System fields are the hierarchy slots (``partition_id``, ``topic``,
``category``, ``attribute``, ``value``).  In the :class:`SemanticStore`
they are first-class columns; in the vector store they are stored in
the flat property map with a ``_`` prefix (``_partition_id`` etc.).

User-supplied metadata sits in ``SemanticAttribute.properties``.  In
both stores it is stored under the user-chosen key, *without* the
``_`` prefix.  In filter expressions it is addressed as ``m.<key>`` or
``metadata.<key>``; the adapter strips that prefix when rewriting for
the vector store.  Keys beginning with ``_`` are reserved;
:func:`validate_attribute_properties` rejects them at input.
"""

from collections.abc import Mapping
from typing import Any

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.vector_store import Record
from memmachine_server.semantic_memory.attribute_memory.semantic_store.semantic_store import (
    SemanticAttribute,
)

RESERVED_PREFIX = "_"

SYSTEM_FIELDS: tuple[str, ...] = (
    "partition_id",
    "topic",
    "category",
    "attribute",
    "value",
)

SYSTEM_PROPERTIES_SCHEMA: dict[str, type[PropertyValue]] = {
    f"{RESERVED_PREFIX}{name}": str for name in SYSTEM_FIELDS
}


def is_reserved_field(name: str) -> bool:
    """Return True if the given property key is reserved for system use."""
    return name.startswith(RESERVED_PREFIX)


def validate_attribute_properties(
    properties: Mapping[str, Any] | None,
) -> None:
    """Raise :class:`ValueError` if any user-supplied property key is reserved.

    The ``_`` prefix is reserved for system metadata; user keys that
    start with it are rejected before the attribute reaches the stores.
    """
    if properties is None:
        return
    reserved = sorted(k for k in properties if is_reserved_field(k))
    if reserved:
        raise ValueError(
            f"Property keys {reserved!r} are reserved; keys starting with "
            f"{RESERVED_PREFIX!r} may not be used by user-supplied metadata"
        )


def build_vector_record_properties(
    attribute: SemanticAttribute,
) -> dict[str, PropertyValue]:
    """Build the flat property map for the paired vector store record.

    System fields are written with a ``_`` prefix; user properties are
    copied through unchanged.  Call
    :func:`validate_attribute_properties` before this (the coordinator
    does so on the add path).
    """
    props: dict[str, PropertyValue] = {
        f"{RESERVED_PREFIX}partition_id": attribute.partition_id,
        f"{RESERVED_PREFIX}topic": attribute.topic,
        f"{RESERVED_PREFIX}category": attribute.category,
        f"{RESERVED_PREFIX}attribute": attribute.attribute,
        f"{RESERVED_PREFIX}value": attribute.value,
    }
    if attribute.properties:
        props.update(attribute.properties)
    return props


def build_vector_record(
    attribute: SemanticAttribute,
    vector: list[float],
) -> Record:
    """Build a :class:`Record` for the given attribute.

    Uses the attribute's UUID as the record's UUID, establishing the
    identity link between the relational and vector stores.
    """
    return Record(
        uuid=attribute.id,
        vector=vector,
        properties=build_vector_record_properties(attribute),
    )


def translate_filter_for_vector_store(
    filter_expr: FilterExpr | None,
) -> FilterExpr | None:
    """Rewrite a user-facing :class:`FilterExpr` for the vector store.

    * System-field bare names are rewritten to their ``_``-prefixed form.
    * User metadata references (``m.X`` / ``metadata.X``) are rewritten
      to the bare key ``X``.
    """
    if filter_expr is None:
        return None
    return map_filter_fields(filter_expr, _translate_field_for_vector_store)


def _translate_field_for_vector_store(field: str) -> str:
    internal, is_user = normalize_filter_field(field)
    if is_user:
        return demangle_user_metadata_key(internal)
    return f"{RESERVED_PREFIX}{internal}"
