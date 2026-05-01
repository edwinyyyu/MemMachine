"""F5 — Allen-relation schema extension.

Extends the base TimeExpression with:
- ``anchor``: an optional reference to an event or an absolute time that
  anchors a qualitative relation (e.g. "the meeting" in "before the
  meeting").  ``anchor.resolved`` is the absolute TimeExpression the
  anchor points to; it can be null at extraction time and filled in later
  by :mod:`event_resolver`.
- ``relation``: one of Allen's interval relations, restricted to the five
  most useful (``before``, ``after``, ``during``, ``overlaps``,
  ``contains``). ``None`` means the expression is purely absolute.

When a temporal reference is purely absolute ("March 15, 2026"),
``relation`` is ``None`` and ``anchor`` is ``None`` — the object is
functionally identical to a plain TimeExpression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from schema import (
    TimeExpression,
    time_expression_from_dict,
    time_expression_to_dict,
)

AllenRelation = Literal["before", "after", "during", "overlaps", "contains"]
AnchorKind = Literal["event", "time"]


@dataclass
class AllenAnchor:
    """An anchor for a qualitative temporal relation.

    - ``kind`` = "event" for named-event anchors ("the meeting").
    - ``kind`` = "time" for temporal-reference anchors ("before 2020").
    - ``span`` is the surface string describing the anchor.
    - ``resolved`` is the absolute TimeExpression the anchor points to,
      if known. ``None`` at extraction time for event anchors; filled in
      by the event-resolver. For ``kind="time"`` anchors, ``resolved``
      SHOULD be populated by the extractor since time anchors are
      resolvable without an external lookup.
    """

    kind: AnchorKind
    span: str
    resolved: TimeExpression | None = None


@dataclass
class AllenExpression:
    """A TimeExpression augmented with an Allen relation + anchor."""

    time: TimeExpression
    relation: AllenRelation | None = None
    anchor: AllenAnchor | None = None


# ---------------------------------------------------------------------------
# (De)serialization
# ---------------------------------------------------------------------------
def allen_anchor_to_dict(a: AllenAnchor) -> dict[str, Any]:
    return {
        "kind": a.kind,
        "span": a.span,
        "resolved": (
            time_expression_to_dict(a.resolved) if a.resolved is not None else None
        ),
    }


def allen_anchor_from_dict(d: dict[str, Any]) -> AllenAnchor:
    resolved = (
        time_expression_from_dict(d["resolved"])
        if d.get("resolved") is not None
        else None
    )
    return AllenAnchor(kind=d["kind"], span=d["span"], resolved=resolved)


def allen_expression_to_dict(ae: AllenExpression) -> dict[str, Any]:
    return {
        "time": time_expression_to_dict(ae.time),
        "relation": ae.relation,
        "anchor": (allen_anchor_to_dict(ae.anchor) if ae.anchor is not None else None),
    }


def allen_expression_from_dict(d: dict[str, Any]) -> AllenExpression:
    return AllenExpression(
        time=time_expression_from_dict(d["time"]),
        relation=d.get("relation"),
        anchor=(
            allen_anchor_from_dict(d["anchor"]) if d.get("anchor") is not None else None
        ),
    )
