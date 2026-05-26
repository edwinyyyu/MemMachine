"""Adapter layer: BEAM turns → EventMemory Events.

Content blocks contain only the turn text. Timestamps go on `Event.timestamp`,
speaker identity on `ProducerContext.producer`, and turn IDs / time anchors on
`Event.properties`.
"""

from uuid import uuid4

from beam_models import BEAMConversation, BEAMTurn
from memmachine_server.common.data_types import PropertyValue
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    TextBlock,
)


def _normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("assistant", "ai", "bot", "model"):
        return "Assistant"
    return "User"


def turn_to_event(turn: BEAMTurn, conversation_id: str) -> Event:
    """Convert a BEAMTurn into an EventMemory Event."""
    properties: dict[str, PropertyValue] = {
        "beam_conv_id": conversation_id,
        "beam_session_idx": turn.session_index,
        "beam_turn_id": turn.turn_id,
        "beam_time_anchor": turn.time_anchor or "",
    }
    if turn.question_type:
        properties["beam_q_type"] = turn.question_type
    if turn.index_str:
        properties["beam_index"] = turn.index_str
    if turn.batch_number is not None:
        properties["beam_batch_num"] = turn.batch_number
    if turn.plan_name:
        properties["beam_plan_name"] = turn.plan_name

    return Event(
        uuid=uuid4(),
        timestamp=turn.timestamp,
        context=ProducerContext(producer=_normalize_role(turn.role)),
        blocks=[TextBlock(text=turn.content.strip())],
        properties=properties,
    )


def conversation_to_events_by_session(
    conversation: BEAMConversation,
) -> list[list[Event]]:
    """Return one list of Events per session."""
    return [
        [turn_to_event(turn, conversation.conversation_id) for turn in session]
        for session in conversation.sessions
    ]
