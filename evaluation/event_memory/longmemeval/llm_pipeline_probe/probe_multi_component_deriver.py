"""MultiComponentDeriver: emit one derivative per embed-component.

The matching SidecarMultiSegmenter stores each embed-component as its
own segment.properties key ("embed_0", "embed_1", ...). This deriver
reads those keys and emits ONE Derivative per component, all sharing
the parent segment's uuid.

Vector store ends up with ~3-4 rows per segment instead of 1. Retrieval
already deduplicates by segment_uuid (keeping best score), so per-
component matches max-pool naturally with no further downstream change.

Falls back to the standard WholeTextDeriver behavior (single derivative
= the joined text_to_embed) if the segment has no "embed_n" property.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    DecoupledRetrievalContext,
    Derivative,
    FirstPersonDecoupledRetrievalContext,
    NullContext,
    ProducerContext,
    RawSegmentEventContext,
    RewriteContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)

EMBED_COMPONENT_COUNT_KEY = "embed_n"
EMBED_COMPONENT_KEY_PREFIX = "embed_"


def _fallback_embed_string(segment: Segment) -> str:
    """Replicate WholeTextDeriver's text for a segment (fallback path)."""
    text = segment.block.text if isinstance(segment.block, TextBlock) else ""
    match segment.context:
        case (
            ProducerContext(producer=p)
            | SurroundingEventsContext(producer=p)
            | RawSegmentEventContext(producer=p)
        ):
            return f"{p}: {text}"
        case NullContext():
            return text
        case (
            RewriteContext(text_to_embed=t)
            | DecoupledRetrievalContext(text_to_embed=t)
            | FirstPersonDecoupledRetrievalContext(text_to_embed=t)
        ):
            return t
        case _:
            return text


def _components_from_properties(segment: Segment) -> list[str] | None:
    """Read embed_0..embed_{n-1} from properties; None if not present."""
    n_raw = segment.properties.get(EMBED_COMPONENT_COUNT_KEY)
    if n_raw is None:
        return None
    try:
        n = int(str(n_raw))
    except ValueError:
        return None
    components: list[str] = []
    for i in range(n):
        comp = segment.properties.get(f"{EMBED_COMPONENT_KEY_PREFIX}{i}")
        if comp is None:
            return None
        s = str(comp).strip()
        if s:
            components.append(s)
    return components or None


class MultiComponentDeriver(Deriver):
    """Emit one derivative per embed-component property.

    Reads structured ``embed_0..embed_{n-1}`` keys from
    segment.properties. Falls back to a single derivative
    (= WholeTextDeriver behavior) if those keys are absent.
    """

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        components = _components_from_properties(segment)
        if components is None:
            # Fallback: single derivative with the full embed string.
            components = [_fallback_embed_string(segment)]
        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=component),
                properties=segment.properties,
            )
            for component in components
            if component
        ]
