"""BEAM dataset models.

Handles the standard BEAM format (100K / 500K / 1M):

    [
      {
        "conversation_id": "...",
        "chat": [[turn, turn, ...], [turn, ...], ...],
        "probing_questions": {
          "abstention": [{"question": "...", "rubric": [...], ...}, ...],
          "event_ordering": [...],
          ...
        },
        "user_profile": {"user_info": "..."},
        ...
      },
      ...
    ]
"""

import ast
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum


class QuestionCategory(str, Enum):
    ABSTENTION = "abstention"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    EVENT_ORDERING = "event_ordering"
    INFORMATION_EXTRACTION = "information_extraction"
    INSTRUCTION_FOLLOWING = "instruction_following"
    KNOWLEDGE_UPDATE = "knowledge_update"
    MULTI_SESSION_REASONING = "multi_session_reasoning"
    PREFERENCE_FOLLOWING = "preference_following"
    SUMMARIZATION = "summarization"
    TEMPORAL_REASONING = "temporal_reasoning"


ALL_CATEGORIES: list[str] = [c.value for c in QuestionCategory]

_ANSWER_FIELDS = [
    "ideal_response",
    "answer",
    "expected_answer",
    "expected",
    "gold_answer",
    "reference",
]


@dataclass
class BEAMTurn:
    role: str
    content: str
    turn_id: int
    time_anchor: str | None = None
    session_index: int = 0
    question_type: str | None = None
    index_str: str | None = None
    batch_number: int | None = None
    plan_name: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime(1970, 1, 1, tzinfo=UTC)
    )


@dataclass
class BEAMQuestion:
    category: str
    index: int
    question: str
    answer: str
    rubric: list[str]
    ordering_tested: list[str] = field(default_factory=list)
    preference_being_tested: str = ""
    instruction_being_tested: str = ""
    compliance_indicators: list[str] = field(default_factory=list)
    time_points: list[str] = field(default_factory=list)
    calculation_required: str = ""
    why_unanswerable: str = ""
    tests_for: str = ""
    total_mentions: int | None = None
    difficulty: str = ""
    abstention_type: str = ""
    contradiction_type: str = ""
    ordering_type: str = ""


@dataclass
class BEAMConversation:
    conversation_id: str
    sessions: list[list[BEAMTurn]]
    questions: list[BEAMQuestion]
    user_info: str = ""


def _extract_answer(question_obj: dict) -> str:
    for field_name in _ANSWER_FIELDS:
        if field_name in question_obj:
            val = question_obj[field_name]
            return str(val) if val is not None else ""
    return ""


def _parse_probing_questions(item: dict) -> dict[str, list[dict]]:
    pq = item.get("probing_questions", {})
    if isinstance(pq, str):
        try:
            pq = json.loads(pq)
        except (json.JSONDecodeError, ValueError):
            try:
                pq = ast.literal_eval(pq)
            except (ValueError, SyntaxError):
                pq = {}
    return pq if isinstance(pq, dict) else {}


@dataclass
class _RawSession:
    raw_turns: list[dict]
    batch_number: int | None = None
    plan_name: str | None = None


def _flatten_turn_groups(turn_groups: list) -> list[dict]:
    out: list[dict] = []
    for tg in turn_groups:
        if isinstance(tg, list):
            out.extend(t for t in tg if isinstance(t, dict) and "role" in t)
        elif isinstance(tg, dict) and "role" in tg:
            out.append(tg)
    return out


def _batch_to_session(batch: dict, plan_name: str) -> _RawSession | None:
    turns = _flatten_turn_groups(batch.get("turns") or [])
    if not turns:
        return None
    bn = batch.get("batch_number")
    try:
        batch_number = int(bn) if bn is not None else None
    except (TypeError, ValueError):
        batch_number = None
    return _RawSession(
        raw_turns=turns,
        batch_number=batch_number,
        plan_name=plan_name,
    )


def _plan_group_to_sessions(plan_group: dict) -> list[_RawSession]:
    sessions: list[_RawSession] = []
    for plan_name, batches in plan_group.items():
        if not isinstance(batches, list):
            continue
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            session = _batch_to_session(batch, str(plan_name))
            if session is not None:
                sessions.append(session)
    return sessions


def _iter_sessions(chat: list) -> list[_RawSession]:
    if not chat:
        return []

    if any(isinstance(item, dict) and "role" in item for item in chat):
        turns = [t for t in chat if isinstance(t, dict) and "role" in t]
        return [_RawSession(raw_turns=turns)] if turns else []

    sessions: list[_RawSession] = []
    for outer in chat:
        if isinstance(outer, list):
            turns = [t for t in outer if isinstance(t, dict) and "role" in t]
            if turns:
                sessions.append(_RawSession(raw_turns=turns))
        elif isinstance(outer, dict):
            sessions.extend(_plan_group_to_sessions(outer))
    return sessions


def _session_start_time(session_index: int, base: datetime) -> datetime:
    return base + timedelta(days=session_index)


def _build_turns(sessions_raw: list[_RawSession]) -> list[list[BEAMTurn]]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    out: list[list[BEAMTurn]] = []
    for s_idx, session in enumerate(sessions_raw):
        session_start = _session_start_time(s_idx, base)
        session_turns: list[BEAMTurn] = []
        for t_idx, raw in enumerate(session.raw_turns):
            turn_id_raw = raw.get("id", t_idx)
            try:
                turn_id = int(turn_id_raw)
            except (TypeError, ValueError):
                turn_id = t_idx
            session_turns.append(
                BEAMTurn(
                    role=str(raw.get("role", "user")),
                    content=str(raw.get("content", "")),
                    turn_id=turn_id,
                    time_anchor=raw.get("time_anchor"),
                    session_index=s_idx,
                    question_type=raw.get("question_type"),
                    index_str=raw.get("index"),
                    batch_number=session.batch_number,
                    plan_name=session.plan_name,
                    timestamp=session_start + timedelta(seconds=t_idx),
                )
            )
        out.append(session_turns)
    return out


def _build_questions(pq: dict[str, list[dict]]) -> list[BEAMQuestion]:
    questions: list[BEAMQuestion] = []
    for cat in ALL_CATEGORIES:
        for idx, raw in enumerate(pq.get(cat, [])):
            if not isinstance(raw, dict):
                continue
            q_text = raw.get("question", "")
            if not q_text:
                continue
            rubric = raw.get("rubric", [])
            if rubric and not isinstance(rubric, list):
                rubric = [str(rubric)]
            questions.append(
                BEAMQuestion(
                    category=cat,
                    index=idx,
                    question=str(q_text),
                    answer=_extract_answer(raw),
                    rubric=[str(r) for r in rubric],
                    ordering_tested=[
                        str(x) for x in raw.get("ordering_tested", []) or []
                    ],
                    preference_being_tested=str(raw.get("preference_being_tested", "")),
                    instruction_being_tested=str(
                        raw.get("instruction_being_tested", "")
                    ),
                    compliance_indicators=[
                        str(x) for x in raw.get("compliance_indicators", []) or []
                    ],
                    time_points=[str(x) for x in raw.get("time_points", []) or []],
                    calculation_required=str(raw.get("calculation_required", "")),
                    why_unanswerable=str(raw.get("why_unanswerable", "")),
                    tests_for=str(raw.get("tests_for", "")),
                    total_mentions=raw.get("total_mentions"),
                    difficulty=str(raw.get("difficulty", "")),
                    abstention_type=str(raw.get("abstention_type", "")),
                    contradiction_type=str(raw.get("contradiction_type", "")),
                    ordering_type=str(raw.get("ordering_type", "")),
                )
            )
    return questions


def load_beam_dataset(path: str) -> list[BEAMConversation]:
    """Load a BEAM dataset file from disk."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise TypeError(f"Expected a list at top level of {path}, got {type(raw)}")

    conversations: list[BEAMConversation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        conv_id = str(item.get("conversation_id", f"conv_{len(conversations)}"))
        sessions_raw = _iter_sessions(item.get("chat", []))
        sessions = _build_turns(sessions_raw)
        questions = _build_questions(_parse_probing_questions(item))
        user_profile = item.get("user_profile") or {}
        user_info = (
            str(user_profile.get("user_info", ""))
            if isinstance(user_profile, dict)
            else ""
        )
        conversations.append(
            BEAMConversation(
                conversation_id=conv_id,
                sessions=sessions,
                questions=questions,
                user_info=user_info,
            )
        )
    return conversations


def questions_by_category(
    conversations: list[BEAMConversation],
) -> dict[str, list[tuple[BEAMConversation, BEAMQuestion]]]:
    by_cat: dict[str, list[tuple[BEAMConversation, BEAMQuestion]]] = defaultdict(list)
    for conv in conversations:
        for q in conv.questions:
            by_cat[q.category].append((conv, q))
    return by_cat
