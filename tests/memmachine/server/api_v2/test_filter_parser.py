from memmachine.server.api_v2.filter_parser import (
    FILTER_CONTRADICTION_KEY,
    parse_filter,
)


def test_parse_empty_filter():
    filter_str = ""
    assert parse_filter(filter_str) == {}


def test_parse_simple_eq_filter():
    filter_str = "theta = -30"
    assert parse_filter(filter_str) == {"theta": -30}

    filter_str = "degrees = 30"
    assert parse_filter(filter_str) == {"degrees": 30}

    filter_str = "displacement = -1.33"
    assert parse_filter(filter_str) == {"displacement": -1.33}

    filter_str = "distance = 1.33"
    assert parse_filter(filter_str) == {"distance": 1.33}

    filter_str = "name = 'Alice'"
    assert parse_filter(filter_str) == {"name": "Alice"}

    filter_str = "test = true"
    assert parse_filter(filter_str) == {"test": True}

    filter_str = "test = false"
    assert parse_filter(filter_str) == {"test": False}


def test_parse_simple_is_null_filter():
    filter_str = "name IS NULL"
    assert parse_filter(filter_str) == {"name": None}


def test_parse_and_filter():
    filter_str = "theta = -30 AND name = 'Alice'"
    assert parse_filter(filter_str) == {"theta": -30, "name": "Alice"}

    filter_str = "distance = 1.33 AND test = true AND name IS NULL"
    assert parse_filter(filter_str) == {
        "distance": 1.33,
        "test": True,
        "name": None,
    }

    filter_str = "a = 1 AND a = 1 AND a = 1 AND b = 2"
    assert parse_filter(filter_str) == {"a": 1, "b": 2}

    filter_str = "a = 1 AND a = 2"
    assert FILTER_CONTRADICTION_KEY in parse_filter(filter_str)


def test_parse_filter_dots_in_keys():
    filter_str = "table.column = 'value'"
    assert parse_filter(filter_str) == {"table.column": "value"}

    filter_str = "db.table.column = 'value'"
    assert parse_filter(filter_str) == {"db.table.column": "value"}

    filter_str = "extra1.db.table.column = 'value'"
    assert parse_filter(filter_str) == {"extra1.db.table.column": "value"}

    filter_str = "extra2.extra1.db.table.column = 'value'"
    assert parse_filter(filter_str) == {"extra2.extra1.db.table.column": "value"}

    filter_str = "extra3.extra2.extra1.db.table.column = 'value'"
    assert parse_filter(filter_str) == {"extra3.extra2.extra1.db.table.column": "value"}
