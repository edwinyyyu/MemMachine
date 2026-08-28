"""JSON helpers that use orjson when available, stdlib json otherwise.

orjson is an optional dependency: absent, these degrade to the stdlib with
identical semantics. `safe_loads` additionally falls back to the stdlib
parser per call, because orjson rejects a few inputs the stdlib accepts
(NaN/Infinity literals, integers beyond 64 bits) and stored data must keep
decoding exactly as it always has.
"""

import json
from typing import cast

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None


if _orjson is not None:

    def dumps(obj: dict[str, object]) -> bytes:
        """Serialize to compact JSON bytes."""
        return _orjson.dumps(obj)

    def safe_loads(data: bytes | str) -> object:
        """Parse JSON, falling back to the stdlib for inputs orjson rejects."""
        try:
            return _orjson.loads(data)
        except Exception:  # orjson-specific strictness -> stdlib semantics
            return json.loads(data)

    def loads(data: bytes | str) -> dict[str, object]:
        """Parse a JSON object."""
        return cast(dict[str, object], _orjson.loads(data))

else:  # pragma: no cover

    def dumps(obj: dict[str, object]) -> bytes:
        """Serialize to compact JSON bytes."""
        return json.dumps(obj).encode()

    def safe_loads(data: bytes | str) -> object:
        """Parse JSON."""
        return json.loads(data)

    def loads(data: bytes | str) -> dict[str, object]:
        """Parse a JSON object."""
        return cast(dict[str, object], json.loads(data))
