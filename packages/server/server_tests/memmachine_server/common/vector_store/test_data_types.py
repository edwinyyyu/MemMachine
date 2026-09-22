"""The vector store's data types: the rule a vector store name follows."""

import pytest

from memmachine_server.common.vector_store.data_types import (
    VECTOR_STORE_NAME_MAX_BYTES,
    validate_vector_store_name,
)


@pytest.mark.parametrize(
    "name", ["long_term_memory__openai_small", "x" * VECTOR_STORE_NAME_MAX_BYTES]
)
def test_a_vector_store_name_of_lowercase_letters_digits_and_underscores(name):
    validate_vector_store_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Upper",
        "with-hyphen",
        "trailing_newline\n",
        "x" * (VECTOR_STORE_NAME_MAX_BYTES + 1),
    ],
)
def test_any_other_vector_store_name_is_refused(name):
    with pytest.raises(ValueError, match="Vector store name"):
        validate_vector_store_name(name)
