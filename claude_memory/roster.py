#!/usr/bin/env python3
"""Cross-session conversation roster for Claude Code.

Two emitters on two independent axes:

  WHEN            what it answers
  ----            ---------------
  session-start   "what else has been going on?"  (top-N by recency)
  delta           "what moved since I last looked?" (only what is new)
  describe        re-describes THIS session, so the other two can cite it

Each line has two halves, because one cannot do both jobs. The identity half says
what a conversation IS and changes only when its subject does; the state half says
where it has got to and changes every turn:

    [when] project · id — <identity> → now: <state>

  --line          which halves to render
  ------          ---------------------
  both            both of them (DEFAULT)
  first           identity only: states the topic, but goes stale as work drifts
  latest          state only: says what is live, but later turns are anaphoric
                  ("this", "our runs", "the third point") because their referents
                  live in that conversation's context, not in the roster

The state half is always a verbatim EXCERPT of a user turn. The identity half is a
one-line *description* written by a small model where one exists, and otherwise an
excerpt of the opening turn — see identity_of and the describer section for why it
is not extracted from the conversation's own compaction summary.

Reads ~/.claude/projects/**/*.jsonl directly, so it does not depend on the
memory daemon and has no ingestion lag. The index is incremental: each
transcript is tail-read from a stored byte offset, so a per-turn hook touches
only the bytes appended since the last run.

Hook usage (stdin = the hook's JSON, stdout = additionalContext):
    roster.py session-start      SessionStart
    roster.py delta              UserPromptSubmit
    roster.py describe           Stop — forks and returns; emits nothing

Turning it on and off (independent of claude_memory — see HOOK_MARKER):
    roster.py enable [--scope project] [--settings PATH] [--dry-run]
    roster.py disable
    roster.py status

Offline usage:
    roster.py update                        # refresh the index
    roster.py show --session <sid>          # what a map would look like now
    roster.py replay --session <sid> --as-of '2026-08-04T21:12:39'
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"
# Claude Code names a project directory after its cwd with "/" replaced by "-", so
# every one of them starts with the home directory spelled the same way. Stripping
# that leaves the part that actually distinguishes one checkout from another.
HOME_TAG = str(Path.home()).replace("/", "-")
STATE = Path.home() / ".claude" / "roster"
INDEX = STATE / "index.json"
CURSORS = STATE / "cursors"
LINES = STATE / "lines"  # one describer-written subject per session

WINDOW_DAYS = 14  # conversations older than this never appear
MAP_N = 12  # lines in a session-start map
DELTA_N = 6  # lines in one delta
MIN_USER_TURNS = 2  # below this it is a headless/one-shot run, not a conversation
EXCERPT_CHARS = 400  # stored; render decides how much to show
DESCRIBE_TURNS = 3  # turns from each end fed to the describer
SUMMARY_CHARS = 4000  # of a compaction summary's intent section, fed whole
SUBJECT_CHARS = 80  # the describer's budget, stated in its prompt
DESCRIBE_TIMEOUT = 120  # seconds; also bounds how long a describe claim can be held
DESCRIBER = "claude-haiku-4-5"

SKIP_PROJ = re.compile(r"private-tmp|scratchpad|neutral|fresh-test", re.IGNORECASE)
# A compacting session is handed a summary of itself as a user message. It is not a
# turn the user typed, so it is skipped outright — see identity_of for why nothing is
# harvested out of it either.
COMPACTION = re.compile(r"This session is being continued", re.IGNORECASE)


def section(text, name, n=400):
    """Body of a named compaction-summary section, or "".

    Anchored to the start of a line (allowing markdown bullets, bold markers and
    numbering), and the colon is REQUIRED. Both matter: a summary's own prose
    names its sections in passing — "Compaction summaries with 9 sections
    (Primary Request and Intent, ..., Optional Next Step)" — and a floating,
    optional-colon pattern matches there, capturing whatever text happens to
    follow the mention instead of the section itself.
    """
    # The colon may fall inside or outside the bold markers ("**Name:**" / "**Name**:"),
    # so bold is consumed on both sides of it.
    m = re.search(
        r"^[\s>*#-]*(?:\d+\.\s*)?\**"
        + re.escape(name)
        + r"\**\s*:\s*\**\s*(.{20,"
        + str(n)
        + r"})",
        text,
        re.DOTALL | re.MULTILINE,
    )
    return m.group(1) if m else ""


NOISE = re.compile(
    r"^(This session is being continued|Caveat: The messages below|<task-notification"
    r"|<system-reminder|<command-name|<local-command|A user is talking to an AI"
    r"|You maintain learned standing rules|A user corrected an AI assistant"
    r"|# Pre-registered|A judgment rubric was WRITTEN"
    # Interrupts and bare continuations describe no task, so they make useless
    # gists; skipping them leaves the previous real turn standing as the gist.
    r"|\[Request interrupted|Continue from where you left off|\[Request cancelled"
    # A bare slash command states no task; skipping it leaves the last real turn.
    r"|/[a-z][a-z-]*\s*$)",
    re.IGNORECASE,
)

# Two DIFFERENT decisions, deliberately given different thresholds.
#
#   SEARCHING is cheap and its payoff is unknowable in advance — whether prior work
#   changes the answer can only be judged after reading it. So the bar to look is
#   low: plausible bearing on the task is enough. (This wording is the one the
#   144-run benchmark measured: 92% on positives, 0% on controls.)
#
#   SURFACING is expensive — it spends the reader's attention — and its payoff IS
#   knowable, because by then you have read the thing. So the bar to mention is
#   high: it must change the answer, not decorate it.
#
# Putting the high bar on searching would be circular, and is the failure this
# whole mechanism exists to prevent.
_INDEX_NOTE = (
    "This is an index, not content. If the current task touches any conversation "
    "listed, search memory before answering — memory_search(cue), or "
    "memory_search(cue, filters=\"m.session_id='<id>'\") to read one, copying the "
    "id exactly as printed: that filter is equality-only, so a shortened id "
    "matches nothing rather than matching loosely. "
    "Then use what you find only where it changes the answer — the recommendation, "
    "the assumptions, or how the question should be read. Do not mention a past "
    "conversation merely to show continuity."
)
LEAD_MAP = "Other recent conversations, most recent first. " + _INDEX_NOTE
LEAD_DELTA = (
    "Conversations that took a new request since your last turn — most likely the one "
    "just switched away from. " + _INDEX_NOTE
)


# ---------------------------------------------------------------- transcript parsing


def typed_user_text(rec):
    """The text of a genuine typed user turn, or None."""
    if rec.get("type") != "user" or rec.get("isSidechain") or rec.get("isMeta"):
        return None
    c = (rec.get("message") or {}).get("content")
    if isinstance(c, str):
        s = c
    elif isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return None
        s = "\n".join(
            b.get("text", "")
            for b in c
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        return None
    s = s.strip()
    if not s or NOISE.match(s):
        return None
    return re.sub(r"\s+", " ", s)


def scan_tail(path, offset, sess):
    """Read appended bytes, folding new turns into `sess`. Returns the new offset."""
    try:
        size = path.stat().st_size
    except OSError:
        return offset
    if size < offset:  # truncated or rotated: start over
        offset, sess["n_user"] = 0, 0
    if size == offset:
        return offset
    with path.open("rb") as fh:
        fh.seek(offset)
        blob = fh.read()
    # Only consume whole lines; leave a partial trailing line for the next run.
    cut = blob.rfind(b"\n")
    if cut == -1:
        return offset
    for raw in blob[:cut].split(b"\n"):
        if not raw.strip():
            continue
        # cheap prefilter before paying for json.loads
        if b'"user"' not in raw and b'"timestamp"' not in raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        ts = rec.get("timestamp") or ""
        if ts and ts > (sess["last_ts"] or ""):
            sess["last_ts"] = ts
        if not sess["proj"] and rec.get("cwd"):
            sess["cwd"] = rec["cwd"]
        blob = (rec.get("message") or {}).get("content")
        flat = (
            blob
            if isinstance(blob, str)
            else (
                " ".join(b.get("text", "") for b in blob if isinstance(b, dict))
                if isinstance(blob, list)
                else ""
            )
        )
        if flat and COMPACTION.search(flat[:200]):
            # Not a turn in the session — but the describer reads this section whole,
            # unparsed, which is the one use a summary supports (see identity_of).
            sess["summary_raw"] = section(
                flat, "Primary Request and Intent", SUMMARY_CHARS
            )
            continue
        t = typed_user_text(rec)
        if t is None:
            continue
        sess["n_user"] += 1
        if not sess["excerpt_first"]:
            sess["excerpt_first"] = t[:EXCERPT_CHARS]
            sess["first_ts"] = ts
        sess["excerpt_last"] = t[:EXCERPT_CHARS]
        sess["excerpt_last_ts"] = ts
        # The describer's input, accumulated here so it costs nothing extra: this
        # loop already visits every turn, and both ends stay bounded.
        if len(sess["head"]) < DESCRIBE_TURNS:
            sess["head"].append(t[:EXCERPT_CHARS])
        sess["tail"] = (sess["tail"] + [t[:EXCERPT_CHARS]])[-DESCRIBE_TURNS:]
    return offset + cut + 1


def load_index():
    try:
        return json.loads(INDEX.read_text())
    except Exception:
        return {"files": {}, "sessions": {}}


def update_index(idx):
    """Incrementally fold every recently-touched transcript into the index."""
    cutoff = time.time() - WINDOW_DAYS * 86400
    for path in PROJECTS.rglob("*.jsonl"):
        proj = path.parent.name
        if SKIP_PROJ.search(proj):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_mtime < cutoff:
            continue
        key = str(path)
        rec = idx["files"].get(key)
        if rec and rec["size"] == st.st_size and rec["mtime"] == st.st_mtime:
            continue
        sid = (rec or {}).get("sid") or path.stem
        sess = idx["sessions"].setdefault(
            sid,
            {
                "proj": proj,
                "cwd": "",
                "first_ts": "",
                "last_ts": "",
                "n_user": 0,
                "excerpt_first": "",
                "excerpt_last": "",
                "excerpt_last_ts": "",
                "head": [],
                "tail": [],
                "summary_raw": "",
            },
        )
        sess.setdefault("head", [])  # sessions indexed before these fields existed
        sess.setdefault("tail", [])
        sess["proj"] = sess["proj"] or proj
        off = scan_tail(path, (rec or {}).get("off", 0), sess)
        idx["files"][key] = {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "off": off,
            "sid": sid,
        }
    return idx


def save_index(idx):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx))
    tmp.replace(INDEX)


# ---------------------------------------------------------------- rendering


def eligible(idx, self_sid, as_of=None):
    """Interactive conversations other than this one, most recently active first."""
    out = []
    horizon = None
    if as_of:
        horizon = as_of
    for sid, s in idx["sessions"].items():
        if sid == self_sid or s["n_user"] < MIN_USER_TURNS or not s["excerpt_last"]:
            continue
        last = s["excerpt_last_ts"] or s["last_ts"]
        if horizon and last >= horizon:
            continue
        if not last:
            continue
        out.append((last, sid, s))
    out.sort(reverse=True)
    return out


def when_label(ts):
    """Render a timestamp the way claude-memory does.

    Medium date, medium time, local zone, so both injected blocks read the same way.
    """
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
    except Exception:
        return ts[:16].replace("T", " ")
    return f"{t:%b %-d, %Y, %-I:%M:%S %p}"


def clip(text, n, keep_tail=False):
    """Shorten with a VISIBLE marker; a silent clip reads as a whole thought.

    keep_tail elides the MIDDLE rather than the end, because a long request often
    carries its operative ask last ("...also consider X"). Cutting the end keeps
    the setup and throws away the request.
    """
    if len(text) <= n:
        return text
    if keep_tail:
        mark = " […] "
        head_n = int((n - len(mark)) * 0.6)
        return (
            text[:head_n].rstrip() + mark + text[-(n - len(mark) - head_n) :].lstrip()
        )
    head = text[:n]
    # Never cut mid-word. Ending on a clause boundary reads better, but "[cut off]"
    # already says the line is incomplete, so a boundary is only worth taking when it
    # costs almost nothing — an earlier one throws away usable width. It cannot be
    # trusted to be a real boundary either: "." matches abbreviations ("SPFresh vs.
    # other indexes"), and telling those apart is not something punctuation supports.
    word = head.rfind(" ")
    brk = max(head.rfind(". "), head.rfind("; "), head.rfind(" — "), head.rfind(", "))
    if brk > 0 and brk >= word - 15:
        head = head[:brk]
    elif word > n * 0.7:
        head = head[:word]
    # Otherwise a single token runs past the budget — a URL, nearly always. Hard-cut
    # it: the leading part still identifies the repo and path, whereas retreating to
    # the last space would drop most of the line.
    return head.rstrip(" ,;—") + "[cut off]"


def identity_of(s):
    """What a conversation IS — the half that only changes when the subject does.

    The first typed message, verbatim. A compaction summary's "Primary Request and
    Intent" was tried here and dropped: it needs parsing, and the shapes it comes in
    cannot be told apart mechanically. Sometimes it opens by describing the
    conversation rather than the work ("This is a long sequence of precise,
    incremental refactoring requests on..."), which is only recognisable
    semantically. And because a summary generalises, sibling sessions on one system
    compact to the SAME sentence — measured on this corpus, 4 of 10 summaries were
    byte-identical to another session's, which is exactly the collision that makes
    two conversations indistinguishable in the map.
    """
    return s["excerpt_first"]


def line(sid, s, line_mode):
    when = when_label(s["excerpt_last_ts"] or s["last_ts"])
    proj = s["proj"].replace(HOME_TAG + "-", "").replace(HOME_TAG, "")
    # Identity half: what this conversation IS. The describer's line if that session
    # has written one, else the opening turn.
    first = described(sid) or identity_of(s)
    # State half: always the newest typed turn. Short follow-ups ("is that right?")
    # are anaphoric on their own, but they are not read on their own — the identity
    # half to their left supplies the topic. Substituting summary text here instead
    # would trade a live signal for one frozen at the last compaction.
    now = s["excerpt_last"]
    if line_mode == "first":
        body = first
    elif line_mode == "latest":
        body = now
    # Default: the opening turn anchors the topic, the current turn says what is
    # live. Neither alone is sufficient — measured on the real corpus, about half
    # of latest turns are anaphoric and carry no recognisable topic on their own.
    elif first[:60] == now[:60]:
        body = clip(first, 110)  # single-turn conversation: do not repeat it
    else:
        body = f"{clip(first, 80)} → now: {clip(now, 110, keep_tail=True)}"
    # The id is printed WHOLE, not as a readable prefix. It exists to be pasted into
    # m.session_id = '<id>', and that filter grammar has only equality — no LIKE, no
    # prefix operator — so a truncated id does not narrow the search, it silently
    # matches nothing at all. A short id is the more dangerous failure: it looks like
    # a working filter and returns an empty result that reads as "no such memory".
    return f"[{when}] {proj[:28]} · {sid} — {body}"


def emit(text):
    if not text:
        return
    sys.stdout.write(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": os.environ.get("ROSTER_EVENT", "UserPromptSubmit"),
                    "additionalContext": text,
                }
            }
        )
    )


# ---------------------------------------------------------------- cursors (delta state)


def cursor_path(sid):
    CURSORS.mkdir(parents=True, exist_ok=True)
    return CURSORS / f"{sid}.json"


def load_cursor(sid):
    try:
        return json.loads(cursor_path(sid).read_text())
    except Exception:
        return {"reported": {}}


def save_cursor(sid, cur):
    tmp = cursor_path(sid).with_suffix(".tmp")
    tmp.write_text(json.dumps(cur))
    tmp.replace(cursor_path(sid))


# ---------------------------------------------------------------- describer
#
# Each session describes ITSELF, on its own Stop hook. Sessions therefore never
# duplicate each other's work, and a session that is not running does not need
# describing — its last line still stands.
#
# Measured against the free fallback on 12 real conversations (subject_eval/):
# it fixed 3 identity lines and broke 1. Too small to resolve at that sample
# size, but the direction is consistent and there is no sign of net harm.

ASK = (
    f"Write ONE line of at most {SUBJECT_CHARS} characters naming what this conversation "
    "is about — enough that someone working in a different conversation could tell "
    "whether this one might bear on their task.\n\n"
    "Other conversations cover the same systems, so naming the component is not enough. "
    "Say what distinguishes THIS conversation: the specific question being settled, "
    "decision being made, or problem being fixed. If the conversation moved on from "
    "where it started, describe where it ended up, not where it began.\n\n"
    "Output only that line: no quotes, no label, no explanation."
)


def describe_prompt(s):
    """V7: an authored account of what the conversation IS, plus where it got to.

    The recent turns are load-bearing twice over — they carry the state a summary
    frozen at the last compaction cannot, and they restore the conversation's own
    wording, which a summary has already paraphrased away.
    """
    turns = "\n".join(f"user: {t}" for t in s.get("tail") or [])
    if s.get("summary_raw"):
        body = (
            f"<summary_of_conversation>\n{s['summary_raw']}\n</summary_of_conversation>\n"
            f"<most_recent_messages>\n{turns}\n</most_recent_messages>"
        )
    else:
        head = "\n".join(f"user: {t}" for t in s.get("head") or [])
        body = f"<conversation>\n{head}\n...\n{turns}\n</conversation>"
    return f"{body}\n\n{ASK}"


def describe(sid, s):
    """Call the describer and store its line. Any failure leaves the last one."""
    try:
        p = subprocess.run(
            [
                "claude",
                "-p",
                describe_prompt(s),
                "--model",
                DESCRIBER,
                # No settings, no MCP, no tools: this runs INSIDE a Stop hook, so
                # inheriting the user's hooks would have it re-enter itself.
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--disallowedTools",
                "Bash",
                "Read",
                "Grep",
                "Glob",
                "Write",
                "Edit",
                "WebSearch",
                "WebFetch",
                "Task",
            ],
            cwd=str(Path.home()),
            capture_output=True,
            text=True,
            timeout=DESCRIBE_TIMEOUT,
        )
    except Exception:
        return ""
    got = re.sub(r"\s+", " ", p.stdout).strip().strip('"')
    # A describer that rambles has misunderstood the task; its output is not a
    # subject line, and the free fallback is better than a paragraph.
    if not got or len(got) > SUBJECT_CHARS * 2:
        return ""
    LINES.mkdir(parents=True, exist_ok=True)
    tmp = LINES / f"{sid}.tmp"
    tmp.write_text(got)
    tmp.replace(LINES / f"{sid}.txt")
    return got


def described(sid):
    try:
        return (LINES / f"{sid}.txt").read_text().strip()
    except OSError:
        return ""


# ---------------------------------------------------------------- commands


def cmd_session_start(line_mode):
    data = read_hook_input()
    sid = data.get("session_id") or ""
    idx = update_index(load_index())
    save_index(idx)
    rows = eligible(idx, sid)[:MAP_N]
    # Prime the cursor so the first delta reports only what moves AFTER this map.
    cur = load_cursor(sid)
    for _, other, s in eligible(idx, sid):
        cur["reported"][other] = s["excerpt_last_ts"] or s["last_ts"]
    save_cursor(sid, cur)
    if not rows:
        return
    os.environ["ROSTER_EVENT"] = "SessionStart"
    emit(LEAD_MAP + "\n\n" + "\n".join(line(sid_, s, line_mode) for _, sid_, s in rows))


def cmd_delta(line_mode):
    data = read_hook_input()
    sid = data.get("session_id") or ""
    idx = update_index(load_index())
    save_index(idx)
    cur = load_cursor(sid)
    fresh = []
    for last, other, s in eligible(idx, sid):
        # Fire when another conversation has taken a NEW USER TURN since this session's
        # last prompt. That is the switch signal: measured on real history, 61% of
        # prompt-to-prompt transitions are a switch to a different session (median gap
        # ~5 min), and this rule fires at ~53% — so it is tracking the thing it should.
        #
        # Firing often is correct here and is not redundancy: the requirement is that a
        # session know enough to REFERENCE the one just switched away from, and each
        # fire carries the request just typed there, which is new every time. Keep the
        # LINE cheap rather than the trigger rare.
        stamp = s["excerpt_last_ts"] or s["last_ts"]
        if stamp and stamp > cur["reported"].get(other, ""):
            fresh.append((last, other, s))
            cur["reported"][other] = stamp
    save_cursor(sid, cur)
    if not fresh:
        return  # silent by construction: nothing new, nothing said
    fresh = fresh[:DELTA_N]
    emit(LEAD_DELTA + "\n\n" + "\n".join(line(o, s, line_mode) for _, o, s in fresh))


def cmd_show(sid, line_mode, as_of=None):
    idx = update_index(load_index())
    save_index(idx)
    rows = eligible(idx, sid, as_of=as_of)[:MAP_N]
    print(LEAD_MAP + "\n")
    for _, other, s in rows:
        print("  " + line(other, s, line_mode))


def cmd_replay(self_sid, as_of, line_mode):
    """Offline: reconstruct the roster as it would have been at `as_of`.

    The live index keeps only each conversation's CURRENT gist, so replaying a
    past moment needs a full scan restricted to turns before the horizon. Slow
    and offline by design — this is the acceptance test, not the hot path.
    """
    sessions = {}
    for path in PROJECTS.rglob("*.jsonl"):
        proj = path.parent.name
        if SKIP_PROJ.search(proj):
            continue
        try:
            # Opened outside the `with` so an unreadable transcript is skipped
            # rather than aborting the replay; the handle is closed by the `with`
            # immediately below.
            fh = path.open("rb")
        except OSError:
            continue
        with fh:
            for raw in fh:
                if b'"user"' not in raw:
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                ts = rec.get("timestamp") or ""
                if not ts or ts >= as_of:  # nothing from the future
                    continue
                t = typed_user_text(rec)
                if t is None:
                    continue
                sid = rec.get("sessionId") or path.stem
                s = sessions.setdefault(
                    sid,
                    {
                        "proj": proj,
                        "cwd": rec.get("cwd", ""),
                        "first_ts": ts,
                        "last_ts": ts,
                        "n_user": 0,
                        "excerpt_first": t[:EXCERPT_CHARS],
                        "excerpt_last": "",
                        "excerpt_last_ts": "",
                        "head": [],
                        "tail": [],
                        "summary_raw": "",
                    },
                )
                s["n_user"] += 1
                if ts > s["excerpt_last_ts"]:
                    s["excerpt_last"], s["excerpt_last_ts"], s["last_ts"] = (
                        t[:EXCERPT_CHARS],
                        ts,
                        ts,
                    )
    rows = eligible({"sessions": sessions}, self_sid)[:MAP_N]
    print(f"# roster for session {self_sid[:8]} as of {as_of}\n")
    for _, other, s in rows:
        print("  " + line(other, s, line_mode))


def cmd_describe():
    """Stop hook: hand this session's own re-description to a detached child.

    Nothing is emitted and nothing is awaited. A Stop hook runs between the reply
    and the next prompt, so a blocking model call here would be latency the user
    pays on every single turn for a line only OTHER sessions ever read.
    """
    data = read_hook_input()
    sid = data.get("session_id") or ""
    if not sid:
        return
    subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "describe-now",
            "--session",
            sid,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def cmd_describe_now(sid):
    """The detached half: re-describe this session if it has moved since last time."""
    idx = update_index(load_index())
    save_index(idx)
    s = idx["sessions"].get(sid)
    if not s or s.get("n_user", 0) < MIN_USER_TURNS:
        return
    # Re-describing an unchanged conversation would spend a call to reproduce the
    # line already on disk, so the gate is movement, not age.
    cur = load_cursor(sid)
    stamp = s.get("excerpt_last_ts") or s.get("last_ts") or ""
    if cur.get("described_at") == stamp and described(sid):
        return
    # Claim BEFORE the call, not after. A describe takes seconds, and turns sent
    # inside that window would otherwise each spawn their own describe of the same
    # session — the failure mode of firing off requests in quick succession. The
    # claim expires on its own so a killed child cannot wedge the session shut.
    if time.time() - cur.get("describing_since", 0) < DESCRIBE_TIMEOUT + 30:
        return
    cur["describing_since"] = time.time()
    save_cursor(sid, cur)
    got = describe(sid, s)
    # Re-read: the map and delta commands write this same cursor on every prompt.
    cur = load_cursor(sid)
    cur["describing_since"] = 0
    if got:
        cur["described_at"] = stamp
    save_cursor(sid, cur)


def read_hook_input():
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


# ---------------------------------------------------------------- enable / disable
#
# The roster owns its hook groups and recognises them by the path to THIS file.
# claude_memory/cli.py marks its own groups with "claude_memory.cli", so neither
# installer can match the other's entries: the two attach to the same three events
# and are still switched on and off independently. Turning the roster off leaves
# memory running, and uninstalling memory leaves the roster running.

HOOK_MARKER = "claude_memory/roster.py"
HOOK_EVENTS = {
    "SessionStart": "session-start",
    "UserPromptSubmit": "delta",
    "Stop": "describe",
}


def settings_file(scope, override):
    if override:
        return Path(override)
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    return Path.home() / ".claude" / "settings.json"


def _ours(group):
    return any(HOOK_MARKER in h.get("command", "") for h in group.get("hooks", []))


def _read_settings(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Could not parse {path}: {error}") from error


def cmd_enable(path, *, disable, dry_run):
    """Add or remove ONLY this tool's hook groups, leaving every other one alone."""
    settings = _read_settings(path)
    hooks = settings.setdefault("hooks", {})
    for event in list(hooks):
        hooks[event] = [g for g in hooks[event] if not _ours(g)]
    if not disable:
        me = str(Path(__file__).resolve())
        for event, sub in HOOK_EVENTS.items():
            hooks.setdefault(event, []).append(
                {
                    "hooks": [
                        {"type": "command", "command": f"{sys.executable} {me} {sub}"}
                    ]
                }
            )
    for event in list(hooks):
        if not hooks[event]:
            del hooks[event]
    if not hooks:
        settings.pop("hooks", None)
    if dry_run:
        print(json.dumps(settings, indent=2))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        print(f"(backup: {path}.bak)")
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"{'Disabled' if disable else 'Enabled'} roster hooks in {path}")


def cmd_status(path):
    hooks = _read_settings(path).get("hooks", {})
    for event in HOOK_EVENTS:
        on = any(_ours(g) for g in hooks.get(event, []))
        print(f"  {event:17s} {'on' if on else 'off'}")
    others = [g for groups in hooks.values() for g in groups if not _ours(g)]
    print(f"\n{path}")
    print(f"{len(others)} hook group(s) belong to something else and are untouched.")


def main():
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "update"
    line_mode = "both"
    if "--line" in argv:
        line_mode = argv[argv.index("--line") + 1]
    sid = argv[argv.index("--session") + 1] if "--session" in argv else ""
    as_of = argv[argv.index("--as-of") + 1] if "--as-of" in argv else None
    scope = argv[argv.index("--scope") + 1] if "--scope" in argv else "user"
    settings_arg = argv[argv.index("--settings") + 1] if "--settings" in argv else ""
    try:
        if cmd == "session-start":
            cmd_session_start(line_mode)
        elif cmd == "delta":
            cmd_delta(line_mode)
        elif cmd == "describe":
            cmd_describe()
        elif cmd == "describe-now":
            cmd_describe_now(sid)
        elif cmd == "replay":
            cmd_replay(sid, as_of, line_mode)
        elif cmd == "show":
            cmd_show(sid, line_mode, as_of)
        elif cmd in ("enable", "disable"):
            cmd_enable(
                settings_file(scope, settings_arg),
                disable=(cmd == "disable"),
                dry_run="--dry-run" in argv,
            )
        elif cmd == "status":
            cmd_status(settings_file(scope, settings_arg))
        elif cmd == "update":
            idx = update_index(load_index())
            save_index(idx)
            n = sum(
                1 for s in idx["sessions"].values() if s["n_user"] >= MIN_USER_TURNS
            )
            print(
                f"indexed {len(idx['files'])} transcripts, {n} conversations",
                file=sys.stderr,
            )
        else:
            print(__doc__, file=sys.stderr)
    except Exception:
        # A hook must never break a turn. Fail silent, emit nothing.
        if cmd in ("session-start", "delta", "describe", "describe-now"):
            return
        raise


if __name__ == "__main__":
    main()
