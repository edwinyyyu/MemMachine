"""Turn the Claude Code transcript JSONL into EventMemory events.

The transcript at ``~/.claude/projects/<hash>/<session>.jsonl`` is the source of
truth for verbatim capture. Each line is one record; this module converts the
records into timeline ``Event``s:

  * user text                -> source=user_message
  * assistant text           -> source=assistant_message
  * assistant thinking       -> source=reasoning      (only if present on disk)
  * assistant tool_use       -> source=tool_call      (name + full JSON input)
  * tool_result (user turn)  -> source=tool_result    (full result content)

Only message sources are embedded (see sources.py); tool calls / results are
stored on the timeline and reached by expansion. Capturing tool_use with its
arguments verbatim is what makes a past procedure replayable on request.

Known limitation: extended-thinking blocks are not always written to the
transcript, so reasoning capture is best-effort. Hooks cannot see the model's
hidden reasoning directly.
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    TextBlock,
)

from claude_memory.engine import Source


def _as_dict(value: object) -> dict[str, Any] | None:
    """Return value as a str-keyed dict if it is a dict, else None (JSON is dynamic)."""
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return None


@dataclass
class _Builder:
    """Assigns strictly increasing timestamps as events are produced."""

    session_id: str
    project: str = ""
    last_timestamp: datetime.datetime | None = None

    def _next_timestamp(self, parsed: datetime.datetime | None) -> datetime.datetime:
        candidate = parsed or self.last_timestamp or _epoch()
        if self.last_timestamp is not None and candidate <= self.last_timestamp:
            candidate = self.last_timestamp + datetime.timedelta(microseconds=1)
        self.last_timestamp = candidate
        return candidate

    def build(
        self,
        *,
        source: Source,
        producer: str,
        text: str,
        parsed_timestamp: datetime.datetime | None,
        extra_properties: dict[str, str] | None = None,
    ) -> Event:
        properties: dict[str, str] = {
            "source": str(source),
            "producer": producer,
            "session_id": self.session_id,
            "project": self.project,
        }
        if extra_properties:
            properties.update(extra_properties)
        return Event(
            uuid=uuid4(),
            timestamp=self._next_timestamp(parsed_timestamp),
            context=ProducerContext(producer=producer),
            blocks=[TextBlock(text=text)],
            properties=dict(properties),
        )


def _epoch() -> datetime.datetime:
    return datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)


def _parse_timestamp(record: dict[str, Any]) -> datetime.datetime | None:
    raw = record.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _text_from_content_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        block_dict = _as_dict(block)
        if block_dict is not None and block_dict.get("type") == "text":
            parts.append(str(block_dict.get("text", "")))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(part for part in parts if part)


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            block_dict = _as_dict(block)
            if block_dict is None:
                parts.append(str(block))
            elif block_dict.get("type") == "text":
                parts.append(str(block_dict.get("text", "")))
            else:
                parts.append(json.dumps(block_dict, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _string_message_event(
    role: str | None,
    content: str,
    parsed: datetime.datetime | None,
    builder: _Builder,
) -> list[Event]:
    if not content.strip():
        return []
    if role == "user":
        return [
            builder.build(
                source=Source.USER_MESSAGE,
                producer="user",
                text=content,
                parsed_timestamp=parsed,
            )
        ]
    if role == "assistant":
        return [
            builder.build(
                source=Source.ASSISTANT_MESSAGE,
                producer="assistant",
                text=content,
                parsed_timestamp=parsed,
            )
        ]
    return []


def _user_turn_events(
    content: list[Any],
    parsed: datetime.datetime | None,
    builder: _Builder,
) -> list[Event]:
    """A user turn is either a real user message or a batch of tool results."""
    tool_results = [
        block_dict
        for block in content
        if (block_dict := _as_dict(block)) is not None
        and block_dict.get("type") == "tool_result"
    ]
    if not tool_results:
        text = _text_from_content_blocks(content)
        return _string_message_event("user", text, parsed, builder)

    events: list[Event] = []
    for block_dict in tool_results:
        text = _tool_result_text(block_dict.get("content"))
        if not text.strip():
            continue
        events.append(
            builder.build(
                source=Source.TOOL_RESULT,
                producer="tool",
                text=text,
                parsed_timestamp=parsed,
                extra_properties={
                    "tool_use_id": str(block_dict.get("tool_use_id", ""))
                },
            )
        )
    return events


def _tool_call_event(
    block: dict[str, Any],
    parsed: datetime.datetime | None,
    builder: _Builder,
) -> Event:
    name = str(block.get("name", "tool"))
    tool_input = block.get("input", {})
    text = f"{name} {json.dumps(tool_input, ensure_ascii=False)}"
    extra = {"tool_name": name}
    input_dict = _as_dict(tool_input)
    if input_dict is not None:
        path = input_dict.get("file_path") or input_dict.get("path")
        if isinstance(path, str):
            extra["path"] = path
    return builder.build(
        source=Source.TOOL_CALL,
        producer="assistant",
        text=text,
        parsed_timestamp=parsed,
        extra_properties=extra,
    )


def _assistant_turn_events(
    content: list[Any],
    parsed: datetime.datetime | None,
    builder: _Builder,
) -> list[Event]:
    """An assistant turn: text, thinking, and tool_use blocks, in order."""
    events: list[Event] = []
    for block in content:
        block_dict = _as_dict(block)
        if block_dict is None:
            continue
        block_type = block_dict.get("type")
        if block_type == "text":
            events.extend(
                _string_message_event(
                    "assistant", str(block_dict.get("text", "")), parsed, builder
                )
            )
        elif block_type == "thinking":
            text = str(block_dict.get("thinking", ""))
            if text.strip():
                events.append(
                    builder.build(
                        source=Source.REASONING,
                        producer="assistant",
                        text=text,
                        parsed_timestamp=parsed,
                    )
                )
        elif block_type == "tool_use":
            events.append(_tool_call_event(block_dict, parsed, builder))
    return events


# Event uuids are derived from the transcript record's stable ``uuid`` so that
# re-ingesting the same record produces the SAME uuid and is recognized as a
# duplicate (see MemoryCore.ingest) instead of written again with a fresh id.
# This is what makes ingest idempotent across backfill/live overlap, forks, and
# resumed sessions that copy earlier records. One record can yield several events
# (e.g. assistant text + tool_use), so the block index disambiguates.
_EVENT_NAMESPACE = uuid5(NAMESPACE_URL, "claude_memory.event")


def _derive_event_uuid(record_uuid: str, block_index: int) -> UUID:
    return uuid5(_EVENT_NAMESPACE, f"{record_uuid}#{block_index}")


def _with_stable_uuids(events: list[Event], record_uuid: object) -> list[Event]:
    """Re-key events to deterministic uuids derived from the source record.

    Records lacking a usable ``uuid`` keep their fallback ``uuid4`` — without a
    stable source identity there is nothing to dedup against, so they are always
    ingested.
    """
    if not isinstance(record_uuid, str) or not record_uuid:
        return events
    return [
        event.model_copy(update={"uuid": _derive_event_uuid(record_uuid, index)})
        for index, event in enumerate(events)
    ]


def _events_from_record(record: dict[str, Any], builder: _Builder) -> list[Event]:
    if record.get("type") not in ("user", "assistant"):
        return []
    # Skip synthetic compaction summaries. After /compact (or auto-compact), which
    # stays in the SAME session, Claude Code appends the summary as a type="user"
    # record flagged ``isCompactSummary``. It is a condensed restatement of earlier
    # turns we have already captured verbatim, so ingesting it would only create
    # near-duplicate memories (the main source of verbatim-duplicate recall).
    if record.get("isCompactSummary"):
        return []
    message = _as_dict(record.get("message"))
    if message is None:
        return []

    role = message.get("role")
    content = message.get("content")
    parsed = _parse_timestamp(record)

    if isinstance(content, str):
        events = _string_message_event(role, content, parsed, builder)
    elif not isinstance(content, list):
        events = []
    elif role == "user":
        events = _user_turn_events(content, parsed, builder)
    else:
        events = _assistant_turn_events(content, parsed, builder)
    return _with_stable_uuids(events, record.get("uuid"))


def last_assistant_message_text(
    transcript_path: str | Path, *, max_chars: int = 4000
) -> str:
    """Return the text of the most recent assistant message, capped to max_chars.

    Used as the cue for reflective post-response recall: the model's own reply is
    what we re-evoke memory against. Tool-use blocks are ignored (only ``text``
    blocks contribute); records are scanned newest-first so we stop at the final
    spoken message of the turn.
    """
    path = Path(transcript_path)
    if not path.exists():
        return ""
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        record_dict = _as_dict(record)
        if record_dict is None or record_dict.get("type") != "assistant":
            continue
        message = _as_dict(record_dict.get("message"))
        if message is None:
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = _text_from_content_blocks(content).strip()
        else:
            continue
        if text:
            return text[:max_chars]
    return ""


def last_compaction_time(
    transcript_path: str | Path,
) -> datetime.datetime | None:
    """Timestamp of the most recent compaction in this session, or None.

    ``/compact`` and auto-compact stay in the same session and append the summary
    as a ``type:"user"`` record flagged ``isCompactSummary``. Everything in the
    session BEFORE that time was compacted out of the context window (only its
    summary remains visible) and so is legitimately recallable; everything at/after
    it is still in context. Scans newest-first and returns at the last summary.
    """
    path = Path(transcript_path)
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if '"isCompactSummary"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        record_dict = _as_dict(record)
        if record_dict is not None and record_dict.get("isCompactSummary"):
            return _parse_timestamp(record_dict)
    return None


def events_from_transcript(
    transcript_path: str | Path,
    *,
    session_id: str,
    start_line: int = 0,
) -> tuple[list[Event], int]:
    """Parse new transcript records into events.

    Returns ``(events, total_lines)``. Pass ``total_lines`` back as
    ``start_line`` next time to ingest only what is new (the high-water mark).
    """
    path = Path(transcript_path)
    if not path.exists():
        return [], start_line

    # The Claude project slug is the transcript file's parent directory name
    # (e.g. ``-Users-eyu-...-agentic-expansion``); kept as a filterable property
    # so the shared search space can be scoped by project as well as by session.
    project = path.parent.name
    lines = path.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)
    builder = _Builder(session_id=session_id, project=project)
    events: list[Event] = []
    for line in lines[start_line:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        record_dict = _as_dict(record)
        if record_dict is not None:
            events.extend(_events_from_record(record_dict, builder))
    return events, total_lines
