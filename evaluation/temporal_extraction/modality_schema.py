"""Modality extension to TimeExpression (v2'' addition).

Adds a ``modality`` classification to each extracted TimeExpression so that
fictional / hypothetical / quoted-embedded temporal references can be
filtered out at retrieval time.

This module does NOT modify ``schema.TimeExpression`` (per task constraints).
Instead we:

1. Provide a ``Modality`` Literal + validator.
2. Provide ``attach_modality`` / ``get_modality`` helpers that stash/read the
   modality on a TimeExpression via a conventional attribute name
   (``_modality``). Back-compat: if the field is missing, treat as ``actual``.

Rationale: TimeExpression is a @dataclass; we cannot freely add fields
without touching schema.py. A conventional ``_modality`` attribute is
compatible with the dataclass — dataclasses allow dynamic attribute
assignment on instances — and round-trips through JSON via a companion
key.
"""

from __future__ import annotations

from typing import Any, Literal

Modality = Literal["actual", "fictional", "hypothetical", "quoted_embedded"]

ALLOWED_MODALITIES: tuple[str, ...] = (
    "actual",
    "fictional",
    "hypothetical",
    "quoted_embedded",
)

DEFAULT_MODALITY: str = "actual"


def normalize_modality(value: Any) -> str:
    """Coerce a possibly-None / unknown modality into the allowed set."""
    if not value:
        return DEFAULT_MODALITY
    v = str(value).strip().lower()
    if v in ALLOWED_MODALITIES:
        return v
    # Common LLM aliases
    alias = {
        "real": "actual",
        "fact": "actual",
        "factual": "actual",
        "fiction": "fictional",
        "novel": "fictional",
        "story": "fictional",
        "imagined": "fictional",
        "imaginary": "fictional",
        "conditional": "hypothetical",
        "counterfactual": "hypothetical",
        "subjunctive": "hypothetical",
        "what-if": "hypothetical",
        "embedded": "quoted_embedded",
        "quoted": "quoted_embedded",
        "quote": "quoted_embedded",
    }
    return alias.get(v, DEFAULT_MODALITY)


def attach_modality(te: Any, modality: str) -> None:
    """Attach a modality tag to a TimeExpression (or any obj)."""
    te._modality = normalize_modality(modality)


def get_modality(te: Any) -> str:
    """Read the modality of a TimeExpression, defaulting to ``actual``."""
    return normalize_modality(getattr(te, "_modality", DEFAULT_MODALITY))


def is_actual(te: Any) -> bool:
    return get_modality(te) == "actual"
