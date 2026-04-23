"""Adapter layer: BEAM turns → EventMemory Events.

This layer enforces the rule that **content blocks contain only the turn text**.
Timestamps go on `Event.timestamp` (and flow through to segments automatically).
Speaker identity goes on `MessageContext.source`. Turn IDs and the original
BEAM `time_anchor` string go on `Event.properties` so they're available for
retrieval-time filtering without being embedded in the block text.

Retrieval-time formatting is delegated entirely to
`EventMemory.string_from_segment_context`, which honors `FormatOptions` for
locale-aware timestamp prefixes. Callers that want non-default formatting
(e.g., to suppress timestamps entirely for BEAM) pass the appropriate
`FormatOptions` to that function directly.
"""

from __future__ import annotations

from uuid import uuid4

from beam_models import BEAMConversation, BEAMTurn
from memmachine_server.episodic_memory.event_memory.data_types import (
    Content,
    Event,
    MessageContext,
    Text,
)


def _normalize_role(role: str) -> str:
    """Map BEAM role strings to a canonical `MessageContext.source` value.

    BEAM uses "user" / "assistant" in lowercase; we capitalize for display
    (retrieval formatters print the source verbatim).
    """
    r = (role or "").strip().lower()
    if r in ("assistant", "ai", "bot", "model"):
        return "Assistant"
    return "User"


def turn_to_event(turn: BEAMTurn, conversation_id: str) -> Event:
    """Convert a BEAMTurn into an EventMemory Event.

    Deliberately does NOT embed the timestamp, role, or turn_id in the block
    text. Those live on Event.timestamp, MessageContext.source, and
    Event.properties respectively.
    """
    properties: dict[str, object] = {
        "beam_conversation_id": conversation_id,
        "beam_session_index": turn.session_index,
        "beam_turn_id": turn.turn_id,
        "beam_time_anchor": turn.time_anchor or "",
    }
    if turn.question_type:
        properties["beam_question_type"] = turn.question_type
    if turn.index_str:
        properties["beam_index"] = turn.index_str
    if turn.batch_number is not None:
        properties["beam_batch_number"] = turn.batch_number
    if turn.plan_name:
        properties["beam_plan_name"] = turn.plan_name

    return Event(
        uuid=uuid4(),
        timestamp=turn.timestamp,
        body=Content(
            context=MessageContext(source=_normalize_role(turn.role)),
            items=[Text(text=turn.content.strip())],
        ),
        properties=properties,
    )


def conversation_to_events_by_session(
    conversation: BEAMConversation,
) -> list[list[Event]]:
    """Return one list of Events per session.

    Ingestion typically calls `EventMemory.encode_events` once per session so
    that eviction / batch-predecessor logic operates on chronologically
    consistent batches.
    """
    return [
        [turn_to_event(turn, conversation.conversation_id) for turn in session]
        for session in conversation.sessions
    ]
