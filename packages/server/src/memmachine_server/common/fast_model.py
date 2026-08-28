"""Construct pydantic models from trusted fields without any bookkeeping.

`build` is model_construct minus its defaults loop and per-field checks: it
blesses a plain dict as the instance __dict__. Use it only where every field
is supplied and the values are already the declared types - data this
process produced or its own stores wrote. Equality, hashing, and
serialization behave identically to a validated instance.
"""

from pydantic import BaseModel

_set = object.__setattr__


def build[M: BaseModel](cls: type[M], fields: dict[str, object]) -> M:
    """Bless `fields` as an instance of `cls`. Every field must be present."""
    obj = cls.__new__(cls)
    _set(obj, "__dict__", fields)
    _set(obj, "__pydantic_fields_set__", set(fields))
    _set(obj, "__pydantic_extra__", None)
    _set(obj, "__pydantic_private__", None)
    return obj
