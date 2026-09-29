"""Enforcement of a collection's declared property types at the write boundary."""

from collections.abc import Mapping

from memmachine_server.common.data_types import PropertyValue

from .data_types import PropertyTypeMismatchError


def require_declared_types(
    properties: Mapping[str, PropertyValue],
    indexed_properties_schema: Mapping[str, type[PropertyValue]],
) -> None:
    """
    Raise unless every declared property holds a value of its declared type.

    Called before anything is sent, so a declared key never holds a value
    its field or index would have to coerce.
    """
    for key, value in properties.items():
        declared_type = indexed_properties_schema.get(key)
        # `bool` is an `int` at runtime and is its own property type here.
        if declared_type is not None and type(value) is not declared_type:
            raise PropertyTypeMismatchError(key, declared_type, value)
