"""Light client/wire surface for claude_memory (standard library only).

The thin clients (CLI hooks, MCP server, the daemon client) need the result
dataclasses, their wire codecs, rendering, ``MemoryConfig``, and the filter
builders — but NOT the embedder/stores/segmenter machinery in ``engine.py``
(which drags in numpy, sqlalchemy, and the memmachine_server event-memory stack,
~hundreds of MB of import footprint). Keeping these here lets a client import the
surface without paying that cost. ``engine.py`` re-exports everything here, so
daemon-side code is unchanged.
"""

import datetime
import json
import os
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

# ======================================================================== config


def _home() -> Path:
    override_path = os.environ.get("CLAUDE_MEMORY_HOME")
    if override_path:
        return Path(override_path)
    return Path.home() / ".claude" / "claude_memory"


def _home_config(home: Path) -> dict[str, Any]:
    """Read optional per-home settings from ``<home>/config.json``.

    Makes a memory space self-describing: any daemon serving it picks up its
    eviction config no matter how it was spawned. Environment variables still
    take precedence over these values.
    """
    try:
        data = json.loads((home / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# All sessions/projects live in ONE shared vector search space; ``session_id``
# and ``project`` are filterable properties for scoping (see
# ``_EXTRA_INDEXED_PROPERTIES``). ``CLAUDE_MEMORY_PARTITION`` overrides it.
_SHARED_PARTITION = "shared"

# Eviction on by default at the embeddinggemma-calibrated cosine threshold (the
# value our deployment runs). Disable with CLAUDE_MEMORY_EVICTION_THRESHOLD="".
_DEFAULT_EVICTION_THRESHOLD = 0.9


@dataclass(frozen=True)
class MemoryConfig:
    """Resolved paths and model selection for one process."""

    home: Path
    vector_db: Path
    segment_db: Path
    state_dir: Path
    namespace: str
    partition: str
    embedding_model: str
    # Eviction (semantic compaction at ingest): cosine threshold for near-dup
    # clusters whose temporal middle is deleted. None = off. Defaults to
    # _DEFAULT_EVICTION_THRESHOLD (on; calibrated for embeddinggemma) — set
    # CLAUDE_MEMORY_EVICTION_THRESHOLD="" (or config null) to disable. See EventMemory.
    eviction_threshold: float | None
    eviction_target_size: int
    eviction_search_limit: int
    # Vector store backend (turbovec | turbovecdisk | sqlitevec). Lives in the
    # home config because the on-disk index formats differ per backend: a
    # daemon serving this home must use the backend that wrote its files.
    vector_backend: str
    # Reflective post-response recall (Stop hook). off by default; when on, the
    # Stop hook searches memory with the model's own reply and surfaces novel,
    # sufficiently-relevant hits so the model can follow up before the turn ends.
    reflect_enabled: bool
    reflect_threshold: float
    reflect_limit: int
    # Manual demotion: per-call geometric decay of a memory's cosine to the cue.
    # Each demote multiplies the current cosine by this factor, so repeated demotes
    # for the same cue decay it geometrically. No relevance floor / pool target /
    # per-call strength is used. See MemoryCore.demote.
    demote_decay: float

    @classmethod
    def load(cls) -> "MemoryConfig":
        """Build the config. Precedence: env var > <home>/config.json > default."""
        home = _home()
        home_config = _home_config(home)

        env_threshold = os.environ.get("CLAUDE_MEMORY_EVICTION_THRESHOLD")
        if env_threshold is not None:
            eviction_threshold = float(env_threshold) if env_threshold != "" else None
        elif "eviction_threshold" in home_config:
            raw = home_config["eviction_threshold"]
            eviction_threshold = float(raw) if raw is not None else None
        else:
            eviction_threshold = _DEFAULT_EVICTION_THRESHOLD

        env_reflect = os.environ.get("CLAUDE_MEMORY_REFLECT")
        if env_reflect is not None:
            reflect_enabled = env_reflect.strip().lower() in ("1", "true", "yes", "on")
        else:
            reflect_enabled = bool(home_config.get("reflect_enabled", False))

        return cls(
            home=home,
            vector_db=home / "vector.db",
            segment_db=home / "segment.db",
            state_dir=home / "state",
            namespace=os.environ.get("CLAUDE_MEMORY_NAMESPACE", "claude"),
            partition=os.environ.get("CLAUDE_MEMORY_PARTITION") or _SHARED_PARTITION,
            embedding_model=os.environ.get("CLAUDE_MEMORY_EMBEDDING", "embeddinggemma"),
            vector_backend=(
                os.environ.get("CLAUDE_MEMORY_VECTOR_BACKEND")
                or home_config.get("vector_backend", "turbovec")
            ),
            eviction_threshold=eviction_threshold,
            eviction_target_size=int(
                os.environ.get("CLAUDE_MEMORY_EVICTION_TARGET")
                or home_config.get("eviction_target_size", 5)
            ),
            eviction_search_limit=int(
                os.environ.get("CLAUDE_MEMORY_EVICTION_SEARCH_LIMIT")
                or home_config.get("eviction_search_limit", 20)
            ),
            reflect_enabled=reflect_enabled,
            reflect_threshold=float(
                os.environ.get("CLAUDE_MEMORY_REFLECT_THRESHOLD")
                or home_config.get("reflect_threshold", 0.6)
            ),
            reflect_limit=int(
                os.environ.get("CLAUDE_MEMORY_REFLECT_LIMIT")
                or home_config.get("reflect_limit", 3)
            ),
            demote_decay=float(
                os.environ.get("CLAUDE_MEMORY_DEMOTE_DECAY")
                or home_config.get("demote_decay", 0.9)
            ),
        )

    def ensure_dirs(self) -> None:
        """Create the home and state directories if they do not exist."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)


# ======================================================================= sources


class Source(StrEnum):
    """Where a timeline event came from.

    Only message sources are embedded (the search surface); the rest live on the
    timeline and are reached by expansion around a message seed. This keeps
    natural-language messages from being out-ranked by command strings / file
    blobs, and avoids embedding large low-value content you would never search for
    by content but do want to replay verbatim.
    """

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INJECTED = "injected"


SEARCHABLE_SOURCES: frozenset[str] = frozenset(
    {Source.USER_MESSAGE, Source.ASSISTANT_MESSAGE}
)


def is_searchable(source: str) -> bool:
    """Whether events from this source are embedded (and thus directly searchable)."""
    return source in SEARCHABLE_SOURCES


# Text injected INTO a session rather than typed in it: hook context, skill bodies,
# system reminders, slash-command echoes, background-task notifications, and the
# compaction summary a session is handed about itself. All of it arrives on the user
# turn, so role alone cannot tell it from something the user wrote.
#
# Measured over the local corpus: 37% of user-role segments and 42% of their
# characters. It is classified at INGEST rather than at read time because expansion
# fills a LIMIT-bounded window — a filter has to push down into that walk to leave
# the budget intact, and post-filtering a fetched window returns short instead.
#
# Kept off the search surface because it is not what happened in a conversation, it
# is what was loaded into one. A compaction summary is the sharpest case: it stays in
# the same session as the turns it describes (compaction does not fork a session),
# and those turns are still present verbatim, so embedding the summary only lets a
# paraphrase outrank its own source.
_INJECTED_HEAD = re.compile(
    r"""^\s*(?:
          <(?:system-reminder|task-notification|local-command|command-name
            |command-message|command-args)
        | Caveat:\s+The\s+messages\s+below
        | This\s+session\s+is\s+being\s+continued
        | The\s+following\s+skills\s+were\s+invoked
        | UserPromptSubmit\s+hook\s+additional\s+context
        )""",
    re.VERBOSE | re.IGNORECASE,
)
# Only the head is examined: these markers open the text they introduce, and
# scanning further would misclassify a real message that merely quotes one.
_INJECTED_HEAD_CHARS = 400


def user_text_source(text: str) -> Source:
    """USER_MESSAGE for something the user typed, INJECTED for anything loaded in."""
    if _INJECTED_HEAD.match(text[:_INJECTED_HEAD_CHARS]):
        return Source.INJECTED
    return Source.USER_MESSAGE


# ======================================================================= mem ids

_MEM_PREFIX = "mem:"


def memory_id_for_segment_uuid(segment_uuid: UUID) -> str:
    """The stable handle the model sees for a segment."""
    return f"{_MEM_PREFIX}{segment_uuid.hex}"


def parse_memory_id(memory_id: str) -> UUID | None:
    """Resolve a ``mem:<hex>`` handle (or a bare uuid) back to a UUID, or None."""
    candidate = memory_id.strip().removeprefix(_MEM_PREFIX)
    try:
        return UUID(hex=candidate)
    except ValueError:
        return None


# ============================================================== result dataclasses


@dataclass
class Hit:
    """A single search result."""

    memory_id: str
    score: float
    text: str
    is_new: bool


@dataclass
class SearchResult:
    """The outcome of one search, with the per-session novelty accounting."""

    hits: list[Hit]
    new_count: int
    saturated: bool
    note: str | None = None


@dataclass
class ExpandResult:
    """The seed's merged event-timeline window plus edge ids for navigation."""

    seed_id: str
    window_text: str = ""
    earliest_id: str | None = None
    latest_id: str | None = None
    note: str | None = None
    found: bool = True


@dataclass
class DemoteResult:
    """Outcome of a manual demotion; ``message`` is the model-facing summary."""

    ok: bool
    verdict: str  # demoted|saturated|not_found|not_searchable|invalid
    message: str
    memory_id: str = ""
    cue: str = ""
    before: float = 0.0
    after: float = 0.0


# ============================================================ model-facing rendering

_SATURATED_MESSAGE = (
    "This memory cannot be demoted any lower for this cue. Stop demoting it."
)


def format_memory_line(
    memory_id: str, text: str, *, max_chars: int | None = None
) -> str:
    """One surfaced memory as a single line: ``[mem:id] <whitespace-collapsed text>``.

    The uniform format for auto-surfacing and search (no scores, no headers).
    """
    one_line = " ".join(text.split())
    if max_chars is not None and len(one_line) > max_chars:
        one_line = one_line[:max_chars].rstrip() + " …"
    return f"[{memory_id}] {one_line}"


def _demote_message(pool: list[Hit]) -> str:
    """Model-facing demote summary: the cue's resulting top matches + how to judge.

    The model decides by reading the ranking, not a number: a mostly-irrelevant
    tail means there is likely no better memory for the cue.
    """
    lines = [
        "Demoted for that cue (and similar cues); demoting it again would push "
        "it lower."
    ]
    if not pool:
        lines.append(
            "Nothing else matches this cue - there is likely no better memory; "
            "stop demoting."
        )
        return "\n".join(lines)
    lines.append("This cue's top matches are now:")
    lines.extend(
        "  " + format_memory_line(hit.memory_id, hit.text, max_chars=120)
        for hit in pool
    )
    lines.append(
        "Demote another of these only if it too is wrong or misleading for this "
        "cue - never just to push something buried further down the list upward "
        "(sharpen the cue instead). If these are mostly off-topic, there is "
        "probably no good memory for this cue - stop."
    )
    return "\n".join(lines)


def render_search_result(result: SearchResult, *, cue: str) -> str:
    """Format search hits: one memory per line, no scores, no headers."""
    if result.note:
        return result.note
    if not result.hits:
        return f'No memories matched "{cue}".'
    return "\n".join(format_memory_line(h.memory_id, h.text) for h in result.hits)


def render_expand_result(result: ExpandResult) -> str:
    """Format the merged event-timeline window around the seed, plus edge nav.

    Expand deliberately deviates from the one-line format: same-event segments
    are merged into whole events so a chunked message reads as one coherent
    block (one header/timestamp), which is what replaying a procedure needs.
    """
    if not result.found:
        return result.note or "Nothing to expand."
    if not result.window_text.strip():
        return "(no surrounding context — this seed is at a timeline edge.)"

    parts: list[str] = [result.window_text.rstrip()]
    hints: list[str] = []
    if result.earliest_id:
        hints.append(
            "To see earlier context, expand the first item above: "
            f"memory_expand {result.earliest_id} before=<count> after=0"
        )
    if result.latest_id:
        hints.append(
            "To see later context, expand the last item above: "
            f"memory_expand {result.latest_id} before=0 after=<count>"
        )
    if hints:
        parts.append("\n".join(hints))
    return "\n\n".join(parts)


# ============================================================= filter builders


def _filter_str(value: str) -> str:
    """Render a value as a filter string literal.

    The filter grammar only tokenizes SINGLE-quoted strings; a double-quoted
    value silently fails to tokenize (a hyphenated UUID then parses as several
    bare identifiers and raises), so always single-quote and drop any embedded
    single quotes.
    """
    return "'" + value.replace("'", "") + "'"


def session_scope_filter(session_id: str) -> str:
    """Filter keeping only this session (scopes expansion to one conversation)."""
    return f"m.session_id = {_filter_str(session_id)}"


def in_context_exclusion_filter(
    session_id: str, compaction_cutoff: datetime.datetime | None
) -> str | None:
    """Filter excluding the current session's IN-CONTEXT turns from recall.

    The involuntary channels (ambient recall, reflection) should not surface what
    is already in the model's context window, but SHOULD reach this session's
    pre-compaction turns (compacted out of context) and any other session. With no
    compaction yet the whole session is in context, so exclude all of it; after a
    compaction at time ``T`` only turns at/after ``T`` remain in context, so keep
    same-session turns before ``T``. Deliberate search/expand do NOT apply this —
    they can reach anything, including this session's pre-compaction turns.
    """
    if not session_id:
        return None
    not_this_session = f"m.session_id != {_filter_str(session_id)}"
    if compaction_cutoff is None:
        return not_this_session
    return f"{not_this_session} OR timestamp < date({_filter_str(compaction_cutoff.isoformat())})"


# ================================================== wire (de)serialization codecs


def demote_result_to_dict(result: DemoteResult) -> dict[str, Any]:
    """Serialize a DemoteResult for the daemon wire protocol."""
    return asdict(result)


def demote_result_from_dict(data: dict[str, Any]) -> DemoteResult:
    """Rebuild a DemoteResult from its wire dict."""
    return DemoteResult(**data)


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    """Serialize a SearchResult for transport over the daemon socket."""
    return asdict(result)


def search_result_from_dict(data: dict[str, Any]) -> SearchResult:
    """Rebuild a SearchResult received over the daemon socket."""
    return SearchResult(
        hits=[Hit(**hit) for hit in data["hits"]],
        new_count=data["new_count"],
        saturated=data["saturated"],
        note=data.get("note"),
    )


def expand_result_to_dict(result: ExpandResult) -> dict[str, Any]:
    """Serialize an ExpandResult for transport over the daemon socket."""
    return asdict(result)


def expand_result_from_dict(data: dict[str, Any]) -> ExpandResult:
    """Rebuild an ExpandResult received over the daemon socket."""
    return ExpandResult(**data)
