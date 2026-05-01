"""Polarity schema: attaches a polarity label to each TimeExpression.

Polarity values:
- "affirmed"    : event happened (default)
- "negated"     : explicitly negated ("didn't", "wasn't", "never")
- "hypothetical": conditional / aspirational / unrealized ("if I had gone",
                  "would have been", "planning to")
- "uncertain"   : hedged ("maybe", "probably", "I think")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Polarity = Literal["affirmed", "negated", "hypothetical", "uncertain"]

POLARITY_VALUES: tuple[str, ...] = (
    "affirmed",
    "negated",
    "hypothetical",
    "uncertain",
)

DEFAULT_POLARITY: Polarity = "affirmed"


@dataclass
class PolarizedTimeExpression:
    """Thin wrapper carrying (surface, polarity) for extraction output."""

    surface: str
    polarity: Polarity = DEFAULT_POLARITY
    evidence: str = ""  # cue word(s) that justified the polarity call


def is_positive(polarity: str) -> bool:
    """True if polarity should count as an affirmed assertion."""
    return polarity == "affirmed"


def is_negative(polarity: str) -> bool:
    return polarity == "negated"


def is_nonfactual(polarity: str) -> bool:
    """True if polarity is hypothetical or uncertain (not factually affirmed)."""
    return polarity in ("hypothetical", "uncertain")
