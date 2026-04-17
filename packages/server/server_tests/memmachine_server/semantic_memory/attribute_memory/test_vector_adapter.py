"""Tests for :mod:`attribute_memory.vector_adapter`."""

from uuid import uuid4

import pytest

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store import (
    SemanticAttribute,
)
from memmachine_server.semantic_memory.attribute_memory.vector_adapter import (
    RESERVED_PREFIX,
    SYSTEM_FIELDS,
    SYSTEM_PROPERTIES_SCHEMA,
    build_vector_record,
    build_vector_record_properties,
    is_reserved_field,
    translate_filter_for_vector_store,
    validate_attribute_properties,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_reserved_prefix_is_underscore() -> None:
    assert RESERVED_PREFIX == "_"


def test_system_fields_match_model_hierarchy() -> None:
    assert SYSTEM_FIELDS == (
        "partition_id",
        "topic",
        "category",
        "attribute",
        "value",
    )


def test_system_properties_schema_prefixes_names_and_types_str() -> None:
    assert {
        "_partition_id": str,
        "_topic": str,
        "_category": str,
        "_attribute": str,
        "_value": str,
    } == SYSTEM_PROPERTIES_SCHEMA


# ---------------------------------------------------------------------------
# is_reserved_field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("_partition_id", True),
        ("_", True),
        ("partition_id", False),
        ("m.foo", False),
        ("foo_bar", False),
    ],
)
def test_is_reserved_field(name: str, expected: bool) -> None:
    assert is_reserved_field(name) is expected


# ---------------------------------------------------------------------------
# validate_attribute_properties
# ---------------------------------------------------------------------------


def test_validate_accepts_none() -> None:
    validate_attribute_properties(None)


def test_validate_accepts_unprefixed_user_keys() -> None:
    validate_attribute_properties({"color": "red", "score": 42})


def test_validate_rejects_reserved_key() -> None:
    with pytest.raises(ValueError, match="reserved"):
        validate_attribute_properties({"_partition_id": "x"})


def test_validate_rejects_any_underscore_prefixed_key() -> None:
    with pytest.raises(ValueError, match="reserved"):
        validate_attribute_properties({"_custom": "x"})


def test_validate_reports_all_reserved_keys_sorted() -> None:
    with pytest.raises(ValueError, match="reserved") as exc_info:
        validate_attribute_properties({"_zzz": 1, "ok": 2, "_aaa": 3})
    message = str(exc_info.value)
    # Both reported; sorted so "_aaa" comes before "_zzz".
    assert message.index("_aaa") < message.index("_zzz")
    assert "ok" not in message


# ---------------------------------------------------------------------------
# build_vector_record_properties / build_vector_record
# ---------------------------------------------------------------------------


def _attribute(**overrides: object) -> SemanticAttribute:
    defaults: dict[str, object] = {
        "id": uuid4(),
        "partition_id": "org_acme/user_42",
        "topic": "Profile",
        "category": "food",
        "attribute": "favorite_pizza",
        "value": "margherita",
    }
    defaults.update(overrides)
    return SemanticAttribute(**defaults)  # type: ignore[arg-type]


def test_build_vector_record_properties_has_all_system_fields_prefixed() -> None:
    props = build_vector_record_properties(_attribute())
    assert props["_partition_id"] == "org_acme/user_42"
    assert props["_topic"] == "Profile"
    assert props["_category"] == "food"
    assert props["_attribute"] == "favorite_pizza"
    assert props["_value"] == "margherita"


def test_build_vector_record_properties_includes_unprefixed_user_metadata() -> None:
    attribute = _attribute(properties={"source": "doc.txt", "confidence": 0.9})
    props = build_vector_record_properties(attribute)
    assert props["source"] == "doc.txt"
    assert props["confidence"] == 0.9
    # System fields still present, unaffected.
    assert props["_topic"] == "Profile"


def test_build_vector_record_properties_no_user_metadata() -> None:
    props = build_vector_record_properties(_attribute(properties=None))
    assert set(props.keys()) == {
        "_partition_id",
        "_topic",
        "_category",
        "_attribute",
        "_value",
    }


def test_build_vector_record_uses_attribute_id_as_uuid() -> None:
    attribute = _attribute()
    record = build_vector_record(attribute, [0.1, 0.2, 0.3])
    assert record.uuid == attribute.id
    assert record.vector == [0.1, 0.2, 0.3]
    assert record.properties is not None
    assert record.properties["_topic"] == "Profile"


# ---------------------------------------------------------------------------
# translate_filter_for_vector_store
# ---------------------------------------------------------------------------


def test_translate_filter_system_fields_get_prefixed() -> None:
    expr = Comparison(field="topic", op="=", value="Profile")
    translated = translate_filter_for_vector_store(expr)
    assert isinstance(translated, Comparison)
    assert translated.field == "_topic"
    assert translated.value == "Profile"


def test_translate_filter_user_metadata_m_prefix_stripped() -> None:
    expr = Comparison(field="m.color", op="=", value="red")
    translated = translate_filter_for_vector_store(expr)
    assert isinstance(translated, Comparison)
    assert translated.field == "color"


def test_translate_filter_user_metadata_metadata_prefix_stripped() -> None:
    expr = Comparison(field="metadata.color", op="=", value="red")
    translated = translate_filter_for_vector_store(expr)
    assert isinstance(translated, Comparison)
    assert translated.field == "color"


def test_translate_filter_returns_none_for_none() -> None:
    assert translate_filter_for_vector_store(None) is None


def test_translate_filter_recurses_into_and_or_not() -> None:
    expr = And(
        left=Comparison(field="topic", op="=", value="Profile"),
        right=Or(
            left=Comparison(field="category", op="=", value="food"),
            right=Not(
                expr=Comparison(field="m.color", op="=", value="red"),
            ),
        ),
    )
    translated = translate_filter_for_vector_store(expr)
    assert isinstance(translated, And)
    left = translated.left
    assert isinstance(left, Comparison)
    assert left.field == "_topic"
    right = translated.right
    assert isinstance(right, Or)
    cat = right.left
    assert isinstance(cat, Comparison)
    assert cat.field == "_category"
    color_not = right.right
    assert isinstance(color_not, Not)
    color_cmp = color_not.expr
    assert isinstance(color_cmp, Comparison)
    assert color_cmp.field == "color"


def test_translate_filter_in_expression() -> None:
    expr = In(field="partition_id", values=["p1", "p2"])
    translated = translate_filter_for_vector_store(expr)
    assert isinstance(translated, In)
    assert translated.field == "_partition_id"
    assert translated.values == ["p1", "p2"]


def test_translate_filter_is_null_expression() -> None:
    expr = IsNull(field="m.color")
    translated = translate_filter_for_vector_store(expr)
    assert isinstance(translated, IsNull)
    assert translated.field == "color"
