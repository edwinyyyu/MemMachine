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
import hashlib
import json
import os
import re
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

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
    # Vector store backend (turbovec | sqlitevec). Lives in the
    # home config because the on-disk index formats differ per backend: a
    # daemon serving this home must use the backend that wrote its files.
    vector_backend: str
    # Reflective post-response recall (Stop hook). off by default; when on, the
    # Stop hook searches memory with the model's own reply and surfaces novel,
    # sufficiently-relevant hits so the model can follow up before the turn ends.
    reflect_enabled: bool
    reflect_threshold: float
    reflect_limit: int
    # Ambient recall (UserPromptSubmit hook): memories retrieved on the user's
    # raw prompt and injected before the model answers. OFF by default. Measured
    # on real traffic 2026-08: fired on ~92% of turns behind a similarity gate
    # that could not separate a one-word reply from a substantive question;
    # ~24% of injections detectably influenced the reply; ~1.3% produced a claim
    # the user then had to correct. Kept for experiments — enable per install
    # with `ambient_enabled: true` in <home>/config.json or CLAUDE_MEMORY_AMBIENT=1.
    ambient_enabled: bool
    # Manual demotion: per-call geometric decay of a memory's cosine to the cue.
    # Each demote multiplies the current cosine by this factor, so repeated demotes
    # for the same cue decay it geometrically. No relevance floor / pool target /
    # per-call strength is used. See MemoryCore.demote.
    demote_decay: float
    # Observability: append one JSONL record per retrieval decision. Off by
    # default (it writes on every prompt); see the observability section below.
    observe: bool

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

        env_ambient = os.environ.get("CLAUDE_MEMORY_AMBIENT")
        if env_ambient is not None:
            ambient_enabled = env_ambient.strip().lower() in ("1", "true", "yes", "on")
        else:
            ambient_enabled = bool(home_config.get("ambient_enabled", False))

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
            ambient_enabled=ambient_enabled,
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
            observe=(
                os.environ.get("CLAUDE_MEMORY_OBSERVE", "").strip().lower()
                in ("1", "true", "yes", "on")
                or bool(home_config.get("observe", False))
            ),
        )

    def ensure_dirs(self) -> None:
        """Create the home and state directories if they do not exist."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)


def _display_timezone() -> datetime.tzinfo:
    """The machine's local timezone for rendering stored UTC timestamps.

    Claude Code writes transcript timestamps in UTC and the segment store keeps
    them UTC; without converting at display time everything reads in UTC (e.g. a
    PDT user sees times 7h late). Prefer the OS IANA zone (DST-correct for each
    timestamp's own date); ``CLAUDE_MEMORY_TIMEZONE`` overrides; fall back to the
    current fixed local offset.
    """
    name = os.environ.get("CLAUDE_MEMORY_TIMEZONE")
    if name:
        with suppress(KeyError, ValueError):
            return ZoneInfo(name)
    with suppress(OSError, ValueError, KeyError):
        link = str(Path("/etc/localtime").readlink())
        if "zoneinfo/" in link:
            return ZoneInfo(link.split("zoneinfo/", 1)[1])
    return datetime.datetime.now().astimezone().tzinfo or datetime.UTC


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
#
# A handle is a PREFIX of a segment uuid, not the whole thing. Whole uuids are 32
# hex digits and tokenize badly — measured with tiktoken over 3,000 real ids, a
# `mem:<uuid>` costs 20.3 tokens against 5.3 for an abbreviated one, and the
# handles the ambient hook injects on every prompt (plus the session ids in the
# roster) came to roughly 57k tokens a day. Nothing else about them changes: they
# are still opaque, still stable, and a whole uuid is still accepted everywhere.
#
# The length is per id, not global: each is rendered just long enough to be unique
# among what is stored right now (`MemoryCore.short_ids`), so a corpus that grows
# lengthens only the ids that actually start to collide. Resolution is a prefix
# range query, and an ambiguous handle answers with candidates rather than
# guessing.

_MEM_PREFIX = "mem:"
_HEX_DIGITS = frozenset("0123456789abcdef")

#: The shortest handle ever rendered. A minimal unique prefix is exactly long
#: enough for the store as it is at the moment of rendering, so it can go
#: ambiguous later as segments arrive — and a handle can outlive the turn that
#: produced it, in an annotation or in the user's own notes. Each extra digit
#: divides the chance that a future arrival collides with it by 16. At ~600k
#: segments the median id is already unique at five digits, so a floor of six is
#: one digit of headroom and costs about one token.
ID_FLOOR_CHARS = 6

#: How many candidates an ambiguous handle reports before saying "and more".
ID_CANDIDATE_LIMIT = 5

#: Written after the event holding the segment an expansion was anchored on. With
#: context on both sides there is otherwise nothing in a window saying which of
#: its turns you expanded from — the seed's own text is in there, but so is
#: everything else, and the reader has no way to tell them apart.
ANCHOR_MARKER = "  ← expanded from here"

#: How much of a session id is shown, in the roster and wherever one is echoed
#: back. Eight hex digits is 4.3e9 of space against a few thousand conversations,
#: and ``memory_search``/``memory_outline`` resolve a prefix against the sessions
#: actually in memory — answering with candidates rather than a guess if it is
#: ever ambiguous.
SESSION_ID_CHARS = 8


def memory_id_for_segment_uuid(segment_uuid: UUID, *, chars: int | None = None) -> str:
    """The handle the model sees for a segment; whole uuid unless ``chars`` says less.

    This does no lookup, so it cannot tell whether a prefix is unique — callers
    that abbreviate get their length from ``MemoryCore.short_ids``, which does.
    """
    hex_digits = segment_uuid.hex
    if chars is not None:
        hex_digits = hex_digits[: max(chars, 1)]
    return f"{_MEM_PREFIX}{hex_digits}"


def parse_memory_ref(memory_id: str) -> str | None:
    """Normalize a handle to a bare lowercase hex prefix, or None if it is not one.

    Deliberately permissive about the wrapping, because all of these come back to
    us: with or without ``mem:``, with or without a uuid's hyphens, any length from
    one digit to a whole uuid. Only the digits themselves have to be hex.
    """
    candidate = memory_id.strip().removeprefix(_MEM_PREFIX).replace("-", "").lower()
    if not candidate or len(candidate) > 32 or set(candidate) - _HEX_DIGITS:
        return None
    return candidate


def parse_memory_id(memory_id: str) -> UUID | None:
    """Resolve a WHOLE handle back to a UUID, or None.

    A prefix cannot be resolved without consulting the store; that is
    ``MemoryCore.resolve_memory_id``. This stays exact-only for the callers that
    have a full uuid in hand and want no I/O.
    """
    candidate = memory_id.strip().removeprefix(_MEM_PREFIX)
    try:
        return UUID(hex=candidate)
    except ValueError:
        return None


def common_prefix_length(left: str, right: str) -> int:
    """How many leading characters two strings share."""
    shared = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        shared += 1
    return shared


def abbreviation_length(hex_digits: str, neighbours: Iterable[str]) -> int:
    """Digits of ``hex_digits`` needed to distinguish it from its nearest neighbours.

    Only the immediate predecessor and successor in sorted order matter: whatever
    prefix separates an id from those two separates it from everything else, since
    anything sharing a longer prefix would have sorted between them.
    """
    longest = max(
        (common_prefix_length(hex_digits, other) for other in neighbours), default=0
    )
    return max(min(longest + 1, len(hex_digits)), ID_FLOOR_CHARS)


def ambiguous_id_note(memory_id: str, candidates: list[str]) -> str:
    """What to say when a handle names more than one memory.

    Answering with the candidates rather than the first match: picking one would be
    a guess, and a wrong guess here is invisible — it returns a real memory that
    simply is not the one that was asked for.
    """
    shown = candidates[:ID_CANDIDATE_LIMIT]
    lines = [
        f"{memory_id} matches {_count_phrase(candidates, shown, 'memories')}. "
        "Retry with one of these in full:"
    ]
    lines += [f"  {_MEM_PREFIX}{hex_digits}" for hex_digits in shown]
    return "\n".join(lines)


def _count_phrase(candidates: list[str], shown: list[str], noun: str) -> str:
    """How many matched, honest about the cases where we stopped counting.

    Resolution fetches one more candidate than it will show, so it knows whether
    the list is the whole of it — but not how much bigger the whole is.
    """
    if len(candidates) <= len(shown):
        return f"{len(candidates)} {noun}"
    return f"more than {len(shown)} {noun}"


# ============================================================== result dataclasses


@dataclass
class Hit:
    """A single search result."""

    memory_id: str
    score: float
    text: str
    is_new: bool
    # The whole segment uuid, carried alongside the abbreviated handle. It costs
    # nothing here (this crosses a socket, not a context window) and it keeps
    # daemon-side bookkeeping — the per-session novelty set — off the display
    # form, which is a prefix and no longer resolvable without the store.
    segment_uuid: str = ""


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
    # Which conversation this window is from, how much of it is here, and whether
    # it reaches either end. The last two are what stop the navigation hints
    # offering to walk further off a conversation that has already run out.
    session_id: str = ""
    events: int = 0
    at_start: bool = False
    at_end: bool = False


@dataclass
class Beat:
    """One user turn in a session outline, with how much followed it."""

    memory_id: str
    when: str
    text: str
    events_after: int


@dataclass
class OutlineResult:
    """A conversation's spine: its user turns, in order, with handles to expand."""

    session_id: str
    project: str = ""
    total_events: int = 0
    span: str = ""
    beats: list[Beat] = field(default_factory=list)
    earlier_id: str | None = None
    later_id: str | None = None
    note: str | None = None
    found: bool = True


@dataclass
class DemoteResult:
    """Outcome of a manual demotion; ``message`` is the model-facing summary."""

    ok: bool
    verdict: str  # demoted|saturated|not_found|not_searchable|unresolved
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


def render_outline_result(result: OutlineResult) -> str:
    """Format a session outline: a header, one line per user turn, and how to page back.

    Deliberately one line per turn with no assistant text. What an outline is for is
    choosing WHERE to look, and a conversation's shape is carried by what was asked;
    the answers are what ``memory_expand`` is for once a place has been picked. The
    event count after each turn is the density signal — it says where the work
    happened, which a list of subjects alone does not.

    A column header names the count, because a bare "31" beside a sentence reads as
    anything: a score, a line number, a number of matches. It costs one line for
    the whole outline rather than a word on every row. The count sits left of the
    text it measures past, which is the wrong way round to read as a sentence — so
    the header names COLUMNS ("events until the next turn") rather than describing
    a relationship, and the eye reads down the column instead of along the row.
    """
    if not result.found:
        return result.note or "No such conversation."
    # The id is echoed at roster length, not whole. It is here to confirm WHICH
    # conversation a prefix resolved to, and this is enough to recognise it —
    # printing all 36 characters would spend more on the header than on a turn.
    short_session = result.session_id[:SESSION_ID_CHARS]
    header = f"Session {short_session}"
    if result.project:
        header += f" · {result.project}"
    header += f" · {result.total_events:,} events"
    if result.span:
        header += f" · {result.span}"
    lines = [header]
    if not result.beats:
        lines.append("  (no user turns in this range)")
    lines.append("  handle · when · events until the next turn · what was asked")
    lines += [
        f"  [{beat.memory_id}] {beat.when} · {beat.events_after:>4} · {beat.text}"
        for beat in result.beats
    ]
    # A handle names its own conversation, so paging needs nothing else.
    if result.earlier_id:
        lines.append(f"Earlier turns: memory_outline {result.earlier_id} after=0")
    if result.later_id:
        lines.append(f"Later turns: memory_outline {result.later_id} before=0")
    return "\n".join(lines)


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

    parts: list[str] = []
    if result.session_id:
        parts.append(_expand_header(result))
    parts.append(result.window_text.rstrip())
    hints: list[str] = []
    # Offered only where there is something to reach. Inviting a walk off the end
    # of a conversation costs a call to learn what the header already knew.
    if result.earliest_id and not result.at_start:
        hints.append(
            "To see earlier context, expand the first item above: "
            f"memory_expand {result.earliest_id} before=<count> after=0"
        )
    if result.latest_id and not result.at_end:
        hints.append(
            "To see later context, expand the last item above: "
            f"memory_expand {result.latest_id} before=0 after=<count>"
        )
    if hints:
        parts.append("\n".join(hints))
    return "\n\n".join(parts)


def _expand_header(result: ExpandResult) -> str:
    """Which conversation this window is from.

    Only that. The event count was recoverable by reading the output, and the
    start/end-of-conversation flags told the reader something they could not act
    on — the navigation hints are already withheld at an edge, so there is no call
    left to save. The session id is the one thing here that is NOT in the body: a
    window otherwise names no conversation, leaving nothing to hand to
    memory_outline or to a session-scoped search.
    """
    return f"[session {result.session_id[:SESSION_ID_CHARS]}]"


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


def _iso_literal(value: str) -> str | None:
    """An ISO 8601 datetime as a filter ``date()`` literal, or None if unreadable.

    Plain ISO 8601, meaning exactly what it says: a bound is the instant it names,
    and a bare date is that day's midnight, because that is what a date denotes.
    An earlier version quietly promoted a bare upper bound to 23:59:59 so that a
    single date would read as a whole day. That made the same string mean two
    different instants depending on which end it was passed to. The half-open
    range in ``scope_filter`` gets the whole day without the trick.

    Zones are handled at both ends, and both halves are load-bearing:

    * A datetime with NO offset names a wall clock, not an instant. It is read in
      the caller's own zone — the one every timestamp is displayed in — because
      reading it as UTC would place a Pacific evening on the following day.
    * The literal is then emitted in UTC, whatever zone it arrived in. The two
      stores compare it differently: the vector store normalizes to UTC before
      comparing, but the segment store's timestamps are a naive-UTC text column and
      the filter compiles to a STRING comparison — so a bound carrying '-07:00' is
      matched on its wall clock, seven hours off, and silently returns a shifted
      window. Emitting UTC makes the wall clock and the instant the same thing, so
      both stores agree no matter which path a filter takes.
    """
    try:
        stamp = datetime.datetime.fromisoformat(value.strip())
    except (AttributeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=_display_timezone())
    return stamp.astimezone(datetime.UTC).isoformat()


def scope_filter(
    *,
    session: str | None = None,
    kinds: list[str] | None = None,
    since: str | None = None,
    before: str | None = None,
) -> tuple[str | None, str]:
    """Assemble a search scope from named parameters. Returns (filter, problem).

    The filter grammar used to be exposed to the model as a string to write. Named
    parameters replace it: they cannot be mistyped into something that parses but
    means nothing, they document what is actually filterable, and the two that
    people reach for — one conversation, one date range — no longer require
    knowing that ``m.`` prefixes a user property while ``timestamp`` is bare.

    ``session`` is a whole session id; callers name a conversation by a memory
    handle and the engine resolves it to one, because a handle is the only kind of
    address these tools take.
    """
    clauses: list[str] = []
    if session:
        clauses.append(session_scope_filter(session))
    if kinds:
        wanted = sorted({str(kind) for kind in kinds} & SEARCHABLE_SOURCES)
        if not wanted:
            searchable = ", ".join(sorted(SEARCHABLE_SOURCES))
            return None, (
                f"Search only reaches messages, so kinds must name some of: "
                f"{searchable}. Use memory_expand to reach anything else."
            )
        if len(wanted) < len(SEARCHABLE_SOURCES):
            clauses.append(f"m.source IN ({', '.join(_filter_str(k) for k in wanted)})")
    # Half-open [since, before): closed on the left, open on the right. Adjacent
    # ranges then tile without overlapping or leaving a gap, and a whole day is
    # since="2026-08-08", before="2026-08-09" — no reaching for 23:59:59, and no
    # rule about what a bare date means on one end but not the other.
    for value, operator, label in (
        (since, ">=", "since"),
        (before, "<", "before"),
    ):
        if not value:
            continue
        literal = _iso_literal(value)
        if literal is None:
            return None, (
                f"Could not read {label}={value!r} as an ISO 8601 datetime. Use "
                "'2026-08-08T09:30:00' (read in your local zone), "
                "'2026-08-08T16:30:00Z' or '2026-08-08T09:30:00-07:00' to name a "
                "zone, or '2026-08-08' for that day's midnight."
            )
        clauses.append(f"timestamp {operator} date({_filter_str(literal)})")
    return (" AND ".join(clauses) if clauses else None), ""


#: What expansion blocks when the caller names no kinds. Injected text stays on the
#: timeline for fidelity — it is genuinely what the session saw — but it is not what
#: happened in the conversation, and it arrives in runs: 12,664 task notifications
#: cluster around tool activity, so a window landing in one is entirely boilerplate.
DEFAULT_BLOCKED_KINDS: tuple[str, ...] = (str(Source.INJECTED),)


def kind_scope_filter(
    kinds: list[str] | None,
    *,
    blocklist: bool = False,
) -> str | None:
    """Filter selecting which sources an expansion window may spend its budget on.

    One list, read as an allowlist or — with ``blocklist`` — as a blocklist. A list
    given by the caller REPLACES the default outright, so what a call does is read
    off its own arguments and nothing is silently added or dropped behind them.

    The seed is NOT exempted here. Re-admitting the seed's whole kind would widen
    the filter for every other segment too — and when that completes the set it
    removes the filter altogether, so blocking a kind you happened to seed into
    would quietly return everything.  ``MemoryCore.expand`` re-attaches the seed
    segment itself instead, which is the narrow thing that was actually wanted.

    Returns None when everything is allowed. This has to be a filter the store
    applies rather than a pass over its result: the window is LIMIT-bounded, so
    dropping segments afterwards returns fewer than were asked for, and the unwanted
    text has already spent the budget.
    """
    every = {str(source) for source in Source}
    if kinds is None:
        kept = every - set(DEFAULT_BLOCKED_KINDS)
    elif blocklist:
        kept = every - {str(kind) for kind in kinds}
    else:
        kept = {str(kind) for kind in kinds} & every
    if kept == every:
        return None
    if not kept:
        # An empty allowlist, taken literally. The grammar has no empty IN list, so
        # match a value no segment carries; expansion then returns the seed alone,
        # which is what asking for no kinds at all means.
        return "m.source = ''"
    return f"m.source IN ({', '.join(_filter_str(k) for k in sorted(kept))})"


def searchable_only(filter_spec: str | None) -> str:
    """Add the search surface's own constraint to a caller's filter.

    ``derive`` refuses to embed a non-searchable source, so nothing new can reach the
    vector store from one. Records captured before a source became non-searchable are
    already there, though, and the vector store keeps its own copy of the properties —
    so without this the only thing removing them would be a full re-index.

    Scoping the query is also simply more honest than trusting the index's contents:
    the filter now states what search is *for*, rather than assuming write-side gating
    was always in place.
    """
    sources = ", ".join(_filter_str(source) for source in sorted(SEARCHABLE_SOURCES))
    mine = f"m.source IN ({sources})"
    return f"({filter_spec}) AND {mine}" if filter_spec else mine


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

# These decode a NEWER daemon's replies inside an OLDER client, and that is the only
# direction the skew ever runs. An MCP client is a subprocess that lives as long as its
# session — days — holding whatever code it started with, while the daemon is restarted
# freely. So a client routinely meets results carrying fields its dataclasses have never
# heard of, and a strict ``Cls(**data)`` turns each one into a TypeError that breaks
# every read tool in that session until it is restarted. Dropping unknown keys makes
# ADDING a field a compatible change, which is what adding a field ought to be.


def _declared(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """``data`` less any key ``cls`` does not declare as a field."""
    allowed = {f.name for f in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def demote_result_to_dict(result: DemoteResult) -> dict[str, Any]:
    """Serialize a DemoteResult for the daemon wire protocol."""
    return asdict(result)


def demote_result_from_dict(data: dict[str, Any]) -> DemoteResult:
    """Rebuild a DemoteResult from its wire dict."""
    return DemoteResult(**_declared(DemoteResult, data))


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    """Serialize a SearchResult for transport over the daemon socket."""
    return asdict(result)


def search_result_from_dict(data: dict[str, Any]) -> SearchResult:
    """Rebuild a SearchResult received over the daemon socket."""
    return SearchResult(
        hits=[Hit(**_declared(Hit, hit)) for hit in data["hits"]],
        new_count=data["new_count"],
        saturated=data["saturated"],
        note=data.get("note"),
    )


def outline_result_to_dict(result: OutlineResult) -> dict[str, Any]:
    """Serialize an OutlineResult for transport over the daemon socket."""
    return asdict(result)


def outline_result_from_dict(data: dict[str, Any]) -> OutlineResult:
    """Rebuild an OutlineResult received over the daemon socket."""
    return OutlineResult(
        **_declared(
            OutlineResult,
            {
                **data,
                "beats": [
                    Beat(**_declared(Beat, beat)) for beat in data.get("beats", [])
                ],
            },
        )
    )


def expand_result_to_dict(result: ExpandResult) -> dict[str, Any]:
    """Serialize an ExpandResult for transport over the daemon socket."""
    return asdict(result)


def expand_result_from_dict(data: dict[str, Any]) -> ExpandResult:
    """Rebuild an ExpandResult received over the daemon socket."""
    return ExpandResult(**_declared(ExpandResult, data))


# ================================================================ observability

# Every retrieval decision this system makes is currently invisible: what a cue
# scored, how much was injected, whether a turn got memories it had no use for.
# Thresholds cannot be set from a guess, so the log comes first and any gate comes
# after — recording costs nothing and changes no behaviour.
#
# One JSONL record per event under <home>/observability.jsonl. Off unless the home
# config says otherwise, because it writes to disk on every prompt.

_OBSERVE_FILE = "observability.jsonl"


def observing(config: "MemoryConfig") -> bool:
    """Whether this home records observability events."""
    return config.observe


def observe(config: "MemoryConfig", event: str, **fields: object) -> None:
    """Append one observability record. Never raises, never blocks a turn."""
    if not config.observe:
        return
    try:
        record = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        with (config.home / _OBSERVE_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Observability must never be the reason a turn fails.
        return


def cue_fingerprint(cue: str) -> str:
    """A short stable digest of a cue, for spotting repeats without storing text.

    Whether a search was USEFUL is not directly observable, but reformulation is a
    usable proxy in the negative direction: a second search in one conversation
    means the first did not settle the question. That needs cue identity and
    nothing else, so the log carries a digest rather than the cue itself — the
    conversation text is already in the store, and duplicating it here would make
    a debugging log into a second copy of the corpus.
    """
    normalized = " ".join(cue.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def score_shape(scores: list[float]) -> dict[str, float]:
    """Compact description of a result set's score distribution.

    The top score alone cannot distinguish "one strong match" from "everything is
    equally mediocre" — which is the distinction a relevance gate would need, and
    the reason the spread is recorded rather than just the maximum.
    """
    if not scores:
        return {"n": 0}
    ordered = sorted(scores, reverse=True)
    return {
        "n": len(ordered),
        "top": round(ordered[0], 4),
        "median": round(ordered[len(ordered) // 2], 4),
        "min": round(ordered[-1], 4),
        "spread": round(ordered[0] - ordered[-1], 4),
    }
