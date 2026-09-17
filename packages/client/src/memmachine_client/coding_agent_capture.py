"""Post what a coding agent session did to a MemMachine server.

The agents' `Stop` hooks run `memmachine agent capture <agent>` at the
end of every turn. It reads the transcript from where the last run left
off, turns the new entries into events, and posts them to
`<server>/v1/tenants/<tenant>/events` in batches.

How far a session has been posted is kept in a state file under the
agent's own directory, `~/.claude/memmachine/capture-state.json` for
Claude Code and `$CODEX_HOME/memmachine/capture-state.json` for Codex.
That mark is a shortcut rather than a record: a batch is stored whole or
rejected whole under the event ids the transcript fixes, so a lost mark
costs one batch the server answers `event_exists` to, never a second
copy of anything. A batch the server answers `event_exists` to is posted
again an event at a time, and the mark moves only once the server holds
every one of them, so a batch that mixes events already held with events
that are new loses neither. A failure leaves the mark where it was and
answers non-zero, so the next `Stop` posts the same entries again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from memmachine_client import claude_code_transcript, codex_transcript
from memmachine_client.coding_agent import (
    CLAUDE_DIRECTORY_NAME,
    CodingAgentError,
    codex_home_directory,
    validated_server,
    validated_tenant,
)

_STATE_DIRECTORY_NAME = "memmachine"
_STATE_FILE_NAME = "capture-state.json"
DEFAULT_BUDGET_SECONDS = 5.0

_BATCH_SIZE = 200
_OK = 200
_CONFLICT = 409
_REMEMBERED_SESSIONS = 500
_GIT_DIRECTORY_NAME = ".git"


def run_capture_command(args: argparse.Namespace) -> int:
    """Run an `agent capture` command and return the process exit code.

    Nothing is written to standard output: an agent reads a hook's
    output as context, so what happened is reported on standard error.
    """
    try:
        transcript_path, session_id, working_directory = _what_to_capture(args)
        count = _capture_session(
            agent_name=args.agent_name,
            server=validated_server(args.server),
            tenant=validated_tenant(args.tenant),
            transcript_path=transcript_path,
            session_id=session_id,
            working_directory=working_directory,
            budget_seconds=args.budget,
        )
        noun = "event" if count == 1 else "events"
        sys.stderr.write(f"captured {count} {noun} of session {session_id}\n")
    except CodingAgentError as error:
        sys.stderr.write(f"{args.prog}: error: {error}\n")
        return 1
    return 0


def _capture_session(
    *,
    agent_name: str,
    server: str,
    tenant: str,
    transcript_path: Path,
    session_id: str,
    working_directory: Path,
    budget_seconds: float,
) -> int:
    """Post a session's new entries and move its mark past them.

    The mark moves after each batch the server holds, so a failure part
    way through keeps what was posted and repeats only the rest. It
    moves past entries that produce no events as well, so a session that
    ends in them is not read again.

    Args:
        agent_name: `claude-code` or `codex`, which picks the reader.
        server: Base URL of the MemMachine server.
        tenant: The tenant whose memory the events land in.
        transcript_path: The transcript the hook named.
        session_id: The session the hook named.
        working_directory: The directory the session runs in, whose
            repository root is the events' project.
        budget_seconds: How long the whole capture may take, after which
            it fails without moving the mark.

    Returns:
        How many events were posted.

    Raises:
        CodingAgentError: If the transcript cannot be read, or the
            server does not take a batch within the budget.
    """
    read_entries = (
        claude_code_transcript.read_entries
        if agent_name == "claude-code"
        else codex_transcript.read_entries
    )
    state_path = _state_path(agent_name)
    mark = _loaded_marks(state_path).get(session_id, _Mark(0, 0))
    try:
        size = transcript_path.stat().st_size
    except OSError as error:
        raise CodingAgentError(f"{transcript_path} cannot be read: {error}") from error
    if size < mark.offset:
        # The transcript was replaced by a shorter one, so what the mark
        # counted is gone and the file is read from its start again.
        mark = _Mark(0, 0)

    url = f"{server.rstrip('/')}/v1/tenants/{quote(tenant, safe='')}/events"
    properties = {"agent": agent_name, "project": _project(working_directory)}
    entries = read_entries(
        transcript_path,
        session_id=session_id,
        properties=properties,
        start_offset=mark.offset,
        start_index=mark.index,
    )
    deadline = time.monotonic() + budget_seconds
    posted = 0
    with requests.Session() as http:
        for events, batch_mark in _batched(entries, mark):
            if events:
                posted += _post_batch(http, url, events, deadline=deadline)
            _save_mark(state_path, session_id, batch_mark)
    return posted


def _batched(
    entries: Iterator[tuple[list[dict[str, Any]], int, int]],
    mark: _Mark,
) -> Iterator[tuple[list[dict[str, Any]], _Mark]]:
    """Group the entries' events into batches, each with the mark past it.

    No entry is split across two batches, so the mark a batch carries is
    past every event in it. The last batch carries the mark past the
    whole read, including entries that produced nothing.
    """
    batch: list[dict[str, Any]] = []
    batch_mark = mark
    read_anything = False
    for events, offset, index in entries:
        read_anything = True
        if batch and len(batch) + len(events) > _BATCH_SIZE:
            yield batch, batch_mark
            batch = []
        batch.extend(events)
        batch_mark = _Mark(offset, index)
    if batch or read_anything:
        yield batch, batch_mark


def _post_batch(
    http: requests.Session,
    url: str,
    events: list[dict[str, Any]],
    *,
    deadline: float,
) -> int:
    """Post one batch and return once the server holds every event of it.

    A batch is stored whole or rejected whole, so `event_exists` for a
    batch says the server holds one of these events, not that it holds
    them all: a session resumed or forked from another repeats entries
    stored under the ids they already had. The batch is then posted an
    event at a time, in the order it was read, where `event_exists` is
    that one event and says it is held. The caller moves its mark only
    when this returns, so nothing is stored twice and nothing is lost.

    Returns:
        How many of the events the server stored now.

    Raises:
        CodingAgentError: If the budget ran out, the server could not be
            reached, or it answered anything else.
    """
    if _stored(http, url, events, deadline=deadline):
        return len(events)
    return sum(1 for event in events if _stored(http, url, [event], deadline=deadline))


def _stored(
    http: requests.Session,
    url: str,
    events: list[dict[str, Any]],
    *,
    deadline: float,
) -> bool:
    """Whether the server stored what was posted, against already holding it.

    Raises:
        CodingAgentError: If the budget ran out, the server could not be
            reached, or it answered anything but stored or held.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CodingAgentError(f"the capture budget ran out before posting to {url}")
    try:
        response = http.post(url, json=events, timeout=remaining)
    except requests.RequestException as error:
        raise CodingAgentError(f"{url} could not be reached: {error}") from error
    if response.status_code == _OK:
        return True
    code, message = _error_of(response)
    if response.status_code == _CONFLICT and code == "event_exists":
        return False
    raise CodingAgentError(f"{url} answered {response.status_code}: {message}")


def _error_of(response: requests.Response) -> tuple[str, str]:
    """The code and message of a v1 error envelope.

    A body that is not one is reported as it arrived, under no code.
    """
    try:
        document = response.json()
    except ValueError:
        return "", response.text[:200]
    error = document.get("error") if isinstance(document, dict) else None
    if not isinstance(error, dict):
        return "", str(document)[:200]
    return str(error.get("code", "")), str(error.get("message", ""))


@dataclass(frozen=True)
class _Mark:
    """How far one session's transcript has been posted.

    `offset` is where the next read starts and `index` is how many
    entries lie before it, which is what an entry carrying no id of its
    own is named after.
    """

    offset: int
    index: int


def _state_path(agent_name: str) -> Path:
    """The state file holding an agent's marks, in the agent's own directory."""
    if agent_name == "claude-code":
        home = Path.home() / CLAUDE_DIRECTORY_NAME
    else:
        home = codex_home_directory()
    return home / _STATE_DIRECTORY_NAME / _STATE_FILE_NAME


def _loaded_marks(state_path: Path) -> dict[str, _Mark]:
    """Every session's mark, in the order they were last written.

    A file that cannot be read is treated as holding no marks, which
    costs the sessions in it one rejected batch each.
    """
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sessions = document.get("sessions") if isinstance(document, dict) else None
    if not isinstance(sessions, dict):
        return {}
    marks: dict[str, _Mark] = {}
    for session_id, mark in sessions.items():
        offset = mark.get("offset") if isinstance(mark, dict) else None
        index = mark.get("index") if isinstance(mark, dict) else None
        if isinstance(offset, int) and isinstance(index, int):
            marks[session_id] = _Mark(offset, index)
    return marks


def _save_mark(state_path: Path, session_id: str, mark: _Mark) -> None:
    """Write one session's mark, keeping the most recent other sessions'.

    The file is read again first, so a session that advanced while this
    one ran keeps its mark, and it is replaced in one step, so a reader
    never sees half of it. Only the most recently written sessions are
    kept, which bounds the file; a session whose mark is dropped is read
    from its start again and rejected as already held.
    """
    marks = _loaded_marks(state_path)
    marks.pop(session_id, None)
    marks[session_id] = mark
    kept = list(marks.items())[-_REMEMBERED_SESSIONS:]
    document = {
        "sessions": {
            held_session: {"offset": held.offset, "index": held.index}
            for held_session, held in kept
        }
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_name(f"{state_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(f"{json.dumps(document, indent=2)}\n", encoding="utf-8")
    temporary_path.replace(state_path)


def _project(working_directory: Path) -> str:
    """The repository root of a directory, or the directory itself.

    A project is the agent's concept, so it is carried on every event as
    a property rather than being a memory of its own.
    """
    for candidate in (working_directory, *working_directory.parents):
        if (candidate / _GIT_DIRECTORY_NAME).exists():
            return str(candidate)
    return str(working_directory)


def _what_to_capture(args: argparse.Namespace) -> tuple[Path, str, Path]:
    """The transcript, session and directory to capture.

    The hook states them on standard input; `--transcript` and
    `--session-id` state them by hand and are read first, so a
    transcript can be posted without an agent running.

    Raises:
        CodingAgentError: If neither names a transcript and a session.
    """
    stated = (
        {}
        if args.transcript is not None and args.session_id is not None
        else _hook_input()
    )
    transcript = args.transcript or stated.get("transcript_path")
    session_id = args.session_id or stated.get("session_id")
    if not isinstance(transcript, str) or not transcript:
        raise CodingAgentError(
            "no transcript to read: the hook named none and --transcript was not given"
        )
    if not isinstance(session_id, str) or not session_id:
        raise CodingAgentError(
            "no session to capture: the hook named none and --session-id was not given"
        )
    working_directory = stated.get("cwd")
    return (
        Path(transcript).expanduser(),
        session_id,
        Path(working_directory) if isinstance(working_directory, str) else Path.cwd(),
    )


def _hook_input() -> dict[str, Any]:
    """The JSON object a hook writes to standard input, empty for a terminal.

    Raises:
        CodingAgentError: If what arrived is not a JSON object.
    """
    if sys.stdin.isatty():
        return {}
    text = sys.stdin.read().strip()
    if not text:
        return {}
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise CodingAgentError(f"the hook input is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise CodingAgentError("the hook input is not a JSON object")
    return document
