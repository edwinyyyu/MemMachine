"""Tests for :class:`AttributeMemory`'s pure helpers and public schema."""

from uuid import uuid4

from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    In,
    IsNull,
    Not,
    Or,
)
from memmachine_server.semantic_memory.attribute_memory import (
    AttributeMemory,
    SemanticAttribute,
)
from memmachine_server.semantic_memory.attribute_memory.attribute_memory import (
    _is_context_length_exceeded_error,
)

# ---------------------------------------------------------------------------
# Constants and public schema
# ---------------------------------------------------------------------------


def test_reserved_prefix_is_underscore() -> None:
    assert AttributeMemory._RESERVED_PREFIX == "_"


def test_system_fields_match_hierarchy() -> None:
    assert AttributeMemory._SYSTEM_FIELDS == (
        "topic",
        "category",
        "attribute",
        "value",
    )


def test_reserved_property_keys_are_empty() -> None:
    assert frozenset() == AttributeMemory._RESERVED_PROPERTY_KEYS


def test_expected_vector_store_collection_schema() -> None:
    assert AttributeMemory.expected_vector_store_collection_schema() == {
        "_topic": str,
        "_category": str,
        "_attribute": str,
        "_value": str,
    }


# ---------------------------------------------------------------------------
# _validate_attribute_properties: narrow reservation
# ---------------------------------------------------------------------------


def test_validate_accepts_none() -> None:
    AttributeMemory._validate_attribute_properties(None)


def test_validate_accepts_user_keys() -> None:
    AttributeMemory._validate_attribute_properties({"color": "red", "score": 42})


def test_validate_accepts_application_underscore_keys() -> None:
    """Applications can set their own ``_``-prefixed keys — reservation
    is narrow (specific names only), not the whole prefix."""
    AttributeMemory._validate_attribute_properties(
        {
            "_app_internal": "x",
            "_session_started_at": 1,
            "regular": "ok",
        }
    )


# ---------------------------------------------------------------------------
# _build_vector_record
# ---------------------------------------------------------------------------


def _attribute(**overrides: object) -> SemanticAttribute:
    defaults: dict[str, object] = {
        "id": uuid4(),
        "topic": "Profile",
        "category": "food",
        "attribute": "favorite_pizza",
        "value": "margherita",
    }
    defaults.update(overrides)
    return SemanticAttribute(**defaults)  # type: ignore[arg-type]


def test_build_vector_record_system_fields_prefixed() -> None:
    attribute = _attribute()
    record = AttributeMemory._build_vector_record(attribute, [0.1, 0.2, 0.3])
    assert record.properties is not None
    assert record.properties["_topic"] == "Profile"
    assert record.properties["_category"] == "food"
    assert record.properties["_attribute"] == "favorite_pizza"
    assert record.properties["_value"] == "margherita"


def test_build_vector_record_merges_user_metadata() -> None:
    attribute = _attribute(properties={"source": "doc.txt", "confidence": 0.9})
    record = AttributeMemory._build_vector_record(attribute, [0.1, 0.2, 0.3])
    assert record.properties is not None
    assert record.properties["source"] == "doc.txt"
    assert record.properties["confidence"] == 0.9
    assert record.properties["_topic"] == "Profile"


def test_build_vector_record_passes_through_underscore_property_keys() -> None:
    """Underscore metadata is preserved in the vector record."""
    attribute = _attribute(properties={"_app_internal": "value"})
    record = AttributeMemory._build_vector_record(attribute, [0.1, 0.2, 0.3])
    assert record.properties is not None
    assert record.properties["_app_internal"] == "value"


def test_build_vector_record_uses_attribute_id_as_uuid() -> None:
    attribute = _attribute()
    record = AttributeMemory._build_vector_record(attribute, [0.1, 0.2, 0.3])
    assert record.uuid == attribute.id
    assert record.vector == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# _translate_filter_for_vector_store
# ---------------------------------------------------------------------------


def test_translate_filter_none_returns_none() -> None:
    assert AttributeMemory._translate_filter_for_vector_store(None) is None


def test_translate_system_field_prefixes() -> None:
    expr = Comparison(field="topic", op="=", value="Profile")
    translated = AttributeMemory._translate_filter_for_vector_store(expr)
    assert isinstance(translated, Comparison)
    assert translated.field == "_topic"


def test_translate_user_metadata_m_prefix_stripped() -> None:
    expr = Comparison(field="m.color", op="=", value="red")
    translated = AttributeMemory._translate_filter_for_vector_store(expr)
    assert isinstance(translated, Comparison)
    assert translated.field == "color"


def test_translate_user_metadata_underscore_key() -> None:
    """``m._app_internal`` demangles to ``_app_internal`` like any user key."""
    expr = Comparison(field="m._app_internal", op="=", value="value")
    translated = AttributeMemory._translate_filter_for_vector_store(expr)
    assert isinstance(translated, Comparison)
    assert translated.field == "_app_internal"


def test_translate_recurses_into_combinators() -> None:
    expr = And(
        left=In(field="topic", values=["A"]),
        right=Or(
            left=Comparison(field="category", op="=", value="x"),
            right=Not(expr=IsNull(field="m.source")),
        ),
    )
    translated = AttributeMemory._translate_filter_for_vector_store(expr)
    assert isinstance(translated, And)
    assert isinstance(translated.left, In)
    assert translated.left.field == "_topic"
    right = translated.right
    assert isinstance(right, Or)
    left_cmp = right.left
    assert isinstance(left_cmp, Comparison)
    assert left_cmp.field == "_category"
    not_expr = right.right
    assert isinstance(not_expr, Not)
    null_expr = not_expr.expr
    assert isinstance(null_expr, IsNull)
    assert null_expr.field == "source"


# ---------------------------------------------------------------------------
# _is_context_length_exceeded_error
# ---------------------------------------------------------------------------


def test_context_length_detects_message_fragment() -> None:
    err = RuntimeError("model response failed: context_length_exceeded on input")
    assert _is_context_length_exceeded_error(err)


def test_context_length_detects_alternative_phrasing() -> None:
    err = RuntimeError("Input exceeds the context window of 128k tokens")
    assert _is_context_length_exceeded_error(err)


def test_context_length_detects_code_attribute() -> None:
    err = RuntimeError("generic failure")
    err.code = "context_length_exceeded"  # type: ignore[attr-defined]
    assert _is_context_length_exceeded_error(err)


def test_context_length_walks_exception_chain() -> None:
    inner = RuntimeError("context_length_exceeded")
    outer = RuntimeError("wrapper")
    outer.__cause__ = inner
    assert _is_context_length_exceeded_error(outer)


def test_context_length_other_errors_return_false() -> None:
    assert not _is_context_length_exceeded_error(RuntimeError("unrelated"))
    assert not _is_context_length_exceeded_error(ValueError("bad input"))
