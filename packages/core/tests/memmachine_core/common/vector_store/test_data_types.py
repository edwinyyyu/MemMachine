"""Tests for vector store data types."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from memmachine_core.common.vector_store.data_types import Record


class TestRecord:
    def test_properties_default_to_empty(self):
        assert Record(uuid=uuid4(), vector=[1.0, 0.0]).properties == {}

    def test_invalid_property_key_rejected(self):
        with pytest.raises(ValidationError):
            Record(uuid=uuid4(), vector=[1.0, 0.0], properties={"Invalid-Key": 1})
