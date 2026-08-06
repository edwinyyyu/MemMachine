"""Single entry point for every claude_memory command.

All of Claude Code's integration points dispatch here, so there is one file
instead of one per hook/server:

    python -m claude_memory.cli mcp        # MCP server (stdio) — deliberate recall
    python -m claude_memory.cli warm       # SessionStart hook — spawn+warm daemon
    python -m claude_memory.cli ambient    # UserPromptSubmit hook — ambient recall
    python -m claude_memory.cli stop       # Stop hook — reflective recall + capture
    python -m claude_memory.cli ingest     # capture only (verbatim transcript tail)
    python -m claude_memory.cli daemon ... # status/start/stop/restart the daemon
    python -m claude_memory.cli install    # enable globally (or --disable/--dry-run)

The hook/MCP subcommands are thin clients of the daemon (see daemon.py); they
hold no model and no state. ``install`` writes the Claude Code config that wires
these up. All commands read their hook payload as JSON on stdin where relevant.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

from claude_memory.daemon_client import (
    DaemonUnavailableError,
    call,
    daemon_status,
    stop_daemon,
)
from claude_memory.transcript import last_compaction_time
from claude_memory.wire import (
    Hit,
    MemoryConfig,
    demote_result_from_dict,
    expand_result_from_dict,
    format_memory_line,
    in_context_exclusion_filter,
    render_expand_result,
    render_search_result,
    search_result_from_dict,
)

# ------------------------------------------------------------------ hook helpers


def _stdin_json() -> dict:
    raw = sys.stdin.read()
    return json.loads(raw) if raw.strip() else {}


_AMBIENT_LIMIT = 4
_SNIPPET_CHARS = 400

# Standing curation affordance carried on the ambient block (the surfaced
# mem:<id>s are already in front of me at the moment I judge their relevance,
# so this is the cheapest seam to act on a wrong/stale one — no extra hook, no
# blocking, no extra retrieval). Closes the recognition->action gap that leaves
# the write tools dormant.
_AMBIENT_CURATION_NOTE = (
    "If a memory below is wrong, stale, or superseded for what you're doing, "
    "curate it as you go rather than just skipping it: "
    "memory_annotate(mem:<id>, note) to attach a lasting correction "
    "(e.g. 'superseded by mem:<id>', 'this was abandoned'), or "
    "memory_demote(mem:<id>, cue) if it shouldn't keep surfacing for a cue "
    "you'll search again."
)


def _render_ambient(hits: list[Hit]) -> str:
    # One line per hit (no scores/headers; snippet-capped because it's injected
    # every prompt), led by the standing curation affordance.
    if not hits:
        return ""
    lines = [
        format_memory_line(hit.memory_id, hit.text, max_chars=_SNIPPET_CHARS)
        for hit in hits
    ]
    return _AMBIENT_CURATION_NOTE + "\n\n" + "\n".join(lines)


def cmd_ambient() -> None:
    """UserPromptSubmit: inject ambient recall if the daemon is already warm."""
    try:
        data = _stdin_json()
        prompt = data.get("prompt") or ""
        if not prompt.strip():
            return
        # Don't re-surface what's already in the context window: exclude this
        # session's in-context turns (everything pre-compaction stays reachable).
        transcript_path = data.get("transcript_path")
        cutoff = last_compaction_time(transcript_path) if transcript_path else None
        response = call(
            {
                "op": "search",
                "cue": prompt,
                "limit": _AMBIENT_LIMIT,
                "session_id": data.get("session_id") or "",
                "cwd": data.get("cwd"),
                "filters": in_context_exclusion_filter(
                    data.get("session_id") or "", cutoff
                ),
                # Hybrid cue: per-session running context + this prompt.
                "use_context": True,
            },
            wait_for_start=0.0,
            timeout=10.0,
        )
        if not response.get("ok"):
            return
        result = search_result_from_dict(response["result"])
        # Only inject genuinely-new memories: already-surfaced ones are still in
        # the conversation, so re-injecting them just bloats context.
        new_hits = [hit for hit in result.hits if hit.is_new]
        if not new_hits:
            return
        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": _render_ambient(new_hits),
                    }
                }
            )
        )
    except Exception:
        # Recall is best-effort; never fail a turn because memory was unavailable.
        return


# When reflection is on and surfaces something, this leads the blocked Stop so
# the model knows the memories are optional follow-up material, not a directive.
_REFLECT_LEAD = (
    "Past memories surfaced by similarity to your reply (below) — they may or may "
    "not be relevant. Use any that genuinely apply; otherwise ignore them and "
    "wrap up."
)


def _capture(data: dict) -> None:
    """Forward the new transcript tail to the daemon for verbatim capture."""
    transcript_path = data.get("transcript_path")
    if not transcript_path:
        return
    with contextlib.suppress(Exception):
        # Capture is best-effort; never fail a turn because ingest broke.
        call(
            {
                "op": "ingest",
                "transcript_path": transcript_path,
                "session_id": data.get("session_id") or "unknown",
                "cwd": data.get("cwd"),
            },
            wait_for_start=45.0,
            timeout=60.0,
        )


def _reflect_and_maybe_block(data: dict) -> bool:
    """Search memory with the model's own reply; block with novel hits if any.

    Returns True iff it emitted a block (so the caller defers capture to the
    final stop). Best-effort: any failure, or the daemon being down, just means
    no reflection — never block the turn from ending on memory trouble.
    """
    if not MemoryConfig.load().reflect_enabled:
        return False
    transcript_path = data.get("transcript_path")
    if not transcript_path:
        return False
    try:
        response = call(
            {
                "op": "reflect",
                "transcript_path": transcript_path,
                "session_id": data.get("session_id") or "",
                "cwd": data.get("cwd"),
            },
            wait_for_start=0.0,  # never spawn the daemon just to reflect
            timeout=10.0,
        )
    except DaemonUnavailableError:
        return False
    memories = response.get("memories") if response.get("ok") else ""
    if not memories:
        return False
    sys.stdout.write(
        json.dumps({"decision": "block", "reason": f"{_REFLECT_LEAD}\n\n{memories}"})
    )
    return True


def cmd_stop() -> None:
    """Stop hook: optional reflective recall, then verbatim capture.

    Two phases per turn. On the first stop, if reflection is enabled, search
    memory with the model's reply as the cue and *block* when novel, relevant
    memories surface so the model can follow up before the turn ends. On the
    final stop (``stop_hook_active`` — the turn is really over) capture the
    transcript tail, including any follow-up. When reflection surfaces nothing
    (or is off), capture happens on the first stop and the turn ends at once.
    """
    try:
        data = _stdin_json()
    except Exception:
        return
    if data.get("stop_hook_active"):
        _capture(data)  # final pass after our own block
        return
    if _reflect_and_maybe_block(data):
        return  # blocked; capture is deferred to the final stop
    _capture(data)


def cmd_ingest() -> None:
    """Stop hook (capture only): forward the transcript tail to the daemon."""
    try:
        _capture(_stdin_json())
    except Exception:
        return


def cmd_warm() -> None:
    """SessionStart: spawn + warm the daemon so the first prompt's recall is instant."""
    try:
        data = _stdin_json()
        call({"op": "ping", "cwd": data.get("cwd")}, wait_for_start=90.0, timeout=90.0)
    except Exception:
        return


# -------------------------------------------------------------------- MCP server


def _call_daemon(payload: dict, wait: float) -> dict | str:
    """Call the daemon; return its response, or a user-facing error string."""
    try:
        response = call(payload, wait_for_start=wait)
    except DaemonUnavailableError as error:
        return f"Memory unavailable: {error}."
    if not response.get("ok"):
        return f"Memory error: {response.get('error', 'unknown')}."
    return response


# The curation tools fire opportunistically off the ambient affordance, so their
# schemas must already be in context the moment a stale memory is noticed — not
# behind a ToolSearch hop. alwaysLoad exempts a tool from MCP deferral (the wire
# flag Claude Code reads). The read tools stay deferred: they're invoked
# deliberately, so loading them is part of the intent to search.
_ALWAYS_LOAD = {"anthropic/alwaysLoad": True}


def _register_memory_tools(mcp: "FastMCP", wait: float) -> None:
    """Attach the memory tools to a FastMCP server (shared by stdio and HTTP)."""

    @mcp.tool()
    async def memory_search(
        cue: str, limit: int = 8, filters: str | None = None
    ) -> str:
        """Recall memories associatively related to a cue.

        A cue is what you'd deliberately bring to mind to surface a *specific*
        memory. Per the Temporal Context Model, you retrieve by re-evoking the
        context the target was encoded in, not by naming the item — so give the cue
        enough context to pin the episode. A bare entity ("cat") is too diffuse (it
        pulls cat facts, a shell command, anything). Prefer an event description
        ("user shared a cat picture") or the verbatim line with its speaker
        ("User: do you remember that cat picture I shared?"), and add the *why* /
        surrounding topic when you can. A question or a statement both work; the
        user's original wording is often a fine cue; trying both it and a sharper
        rephrasing can help.

        When you can't pin the target directly, search for its *surrounding
        context* instead — what was being discussed, roughly when — and then
        memory_expand from a nearby hit to reach the target (recall the path to the
        item, not the item in isolation; cf. method of loci). If even that is too
        vague, ask the user for the surrounding context (how long ago, what you
        were discussing) and search that. Multi-hop = call again with the next
        lead; stop when nothing new surfaces.

        Args:
            cue: A context-bearing cue — an event description, a verbatim line with
                speaker, or a question; not a bare keyword.
            limit: Maximum memories to return (default 8).
            filters: Optional metadata filter to narrow the results.
                String values use SINGLE quotes. User properties take an `m.`
                (or `metadata.`) prefix; system fields are bare. Examples:
                  m.session_id = '<uuid>'      m.project = '<slug>'
                  m.producer = 'assistant'     m.source IN ('user_message')
                  timestamp >= date('2026-01-01')   (system field; date() literal)
                Ops: = != > < >= <= , IN (...), NOT IN (...), IS NULL. Combine
                with AND / OR / NOT. One memory per line: `[mem:id] <text>`.
        """
        response = _call_daemon(
            {"op": "search", "cue": cue, "limit": limit, "filters": filters}, wait
        )
        if isinstance(response, str):
            return response
        return render_search_result(
            search_result_from_dict(response["result"]), cue=cue
        )

    @mcp.tool()
    async def memory_expand(
        seed: str,
        before: int = 5,
        after: int = 5,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> str:
        """Recall the surrounding timeline around a memory you already have.

        Pass a `mem:<id>` from memory_search. This walks the conversation around it
        — same session, in order — and reaches what a search won't surface on its
        own: tool calls and their results, file contents, the turns on either side.
        Use it to replay how something was done, not just that it was mentioned.
        `before`/`after` say how much context to pull on each side (asymmetric is
        fine); if it isn't enough, expand again from the first or last `mem:<id>`
        shown to reach further.

        `include`/`exclude` choose which kinds of thing the window spends itself on.
        They compose as a set difference — start from `include` (or everything), then
        subtract `exclude` — so passing both narrows twice rather than one overriding
        the other. The kinds are:
        `user_message`, `assistant_message`, `reasoning`, `tool_call`, `tool_result`,
        `injected`. They are applied while the window is gathered, so the budget buys
        only what you asked for. Reach for them when a plain expand comes back thin:
        `exclude=["tool_result"]` to read the argument in a session full of long
        command output, `include=["tool_call","tool_result"]` to replay just the
        procedure. `injected` (hook context, skill bodies, system reminders,
        slash-command echoes, the session's own compaction summary) is dropped by
        default because it arrives in runs and can fill a window on its own — pass
        `include=["injected", ...]` to see exactly what the session was shown.

        Args:
            seed: A `mem:<id>` from a prior search or expand.
            before: How much context to pull before the seed (default 5).
            after: How much context to pull after the seed (default 5).
            include: Start from only these kinds (default: all of them).
            exclude: Subtract these kinds. `injected` is subtracted anyway unless you
                name it in `include`, or pass `exclude=[]` to keep everything.
        """
        response = _call_daemon(
            {
                "op": "expand",
                "seed": seed,
                "before": before,
                "after": after,
                "include": include,
                "exclude": exclude,
            },
            wait,
        )
        if isinstance(response, str):
            return response
        return render_expand_result(expand_result_from_dict(response["result"]))

    @mcp.tool(meta=_ALWAYS_LOAD)
    async def memory_demote(memory_id: str, cue: str) -> str:
        """Make a memory rank lower for a cue (and similar cues) in future recall.

        Use this after a memory_search when a returned memory was wrong or unhelpful
        for that cue — "this shouldn't have come up for this." Pass the memory's
        `mem:<id>` and the cue you searched. Demoting once nudges it down; call it
        again on the same memory to push it lower.

        Demote a memory only because it itself was wrong or misleading for the
        cue — never to clear a path for an answer ranked far below it. If the
        right memory is buried deep, the cue is the problem: sharpen it (or expand
        from a nearby hit) instead of demoting everything above. The result shows
        the cue's top matches afterward; if they are mostly off-topic, there is
        probably no good memory for this cue — stop, rather than demoting toward
        an answer that isn't there.

        Args:
            memory_id: the `mem:<id>` to deprioritize.
            cue: the cue you searched (similar cues are affected too).
        """
        response = _call_daemon(
            {"op": "demote", "memory_id": memory_id, "cue": cue}, wait
        )
        if isinstance(response, str):
            return response
        return demote_result_from_dict(response["result"]).message

    @mcp.tool(meta=_ALWAYS_LOAD)
    async def memory_annotate(memory_id: str, note: str) -> str:
        """Attach a one-line note to a memory, visible on every future retrieval.

        Use it to record what you later learned about a memory, right where it
        will resurface: a correction ("this was fixed in a later turn"), an
        outcome ("this approach was abandoned"), or a pointer ("see mem:<id> for
        the final version"). Whoever retrieves that memory next — in this session
        or any future one — sees the note attached to it, labeled as a note so it
        won't be mistaken for the original content.

        Notes are append-only: you can add another, but never edit or remove one,
        so write notes that stay true (what happened, not what to do next).
        Annotating does not change when the memory surfaces — use memory_demote
        for that; it changes what is known when it does.

        Args:
            memory_id: the `mem:<id>` to annotate.
            note: a single line, appended to the memory as [note: ...].
        """
        response = _call_daemon(
            {"op": "annotate", "memory_id": memory_id, "note": note}, wait
        )
        if isinstance(response, str):
            return response
        return str(response.get("message", ""))


def cmd_mcp() -> None:
    """Run the MCP server (stdio): search / expand / demote / annotate tools."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("claude-memory")
    _register_memory_tools(mcp, wait=90.0)
    mcp.run()


def cmd_mcp_http(port: int) -> None:
    """Run the MCP server over streamable HTTP on 127.0.0.1 (URL-based clients).

    A stateless transport bridge, independent of the per-session stdio shims:
    it forwards every tool call to the same daemon, so URL clients (e.g. the
    Desktop connectors UI) share one memory with CLI sessions. Localhost-only
    by construction - the daemon has no auth, so this must never bind wider.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("claude-memory", host="127.0.0.1", port=port)
    _register_memory_tools(mcp, wait=90.0)
    mcp.run(transport="streamable-http")


# ---------------------------------------------------------------------- install

MCP_NAME = "claude-memory"
_HOOK_MARKER = "claude_memory.cli"
_HOOK_MODULES = {
    "SessionStart": "warm",
    "UserPromptSubmit": "ambient",
    "Stop": "stop",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


_SKILL_NAME = "episodic-recall"


def _skill_paths(repo: Path, scope: str) -> tuple[Path, Path]:
    """(source skill dir in the repo, destination skill dir for the scope)."""
    source = repo / "claude_memory" / "skills" / _SKILL_NAME
    root = Path.cwd() / ".claude" if scope == "project" else Path.home() / ".claude"
    return source, root / "skills" / _SKILL_NAME


def _install_skill(repo: Path, scope: str) -> None:
    """Copy the episodic-recall skill into the scope's skills dir (idempotent)."""
    source, dest = _skill_paths(repo, scope)
    if not source.exists():
        print(f"\nSkill source missing at {source}; skipping skill install.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)
    print(f"Installed skill '{_SKILL_NAME}' to {dest}")


def _remove_skill(repo: Path, scope: str) -> None:
    """Remove the installed episodic-recall skill for the scope."""
    _, dest = _skill_paths(repo, scope)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
        print(f"Removed skill '{_SKILL_NAME}' from {dest}")


def _apply_skill(repo: Path, scope: str, *, disable: bool) -> None:
    """Install or remove the skill for the scope."""
    if disable:
        _remove_skill(repo, scope)
    else:
        _install_skill(repo, scope)


def _apply_mcp(scope: str, mcp_argv: list[str], *, disable: bool) -> None:
    """Register or remove the MCP server for the scope."""
    _run(["claude", "mcp", "remove", MCP_NAME, "-s", scope])
    if disable:
        print(f"Removed MCP server '{MCP_NAME}'.")
    elif _run(mcp_argv) != 0:
        print(
            "\nCould not register the MCP server automatically. Run this "
            "yourself:\n  " + " ".join(mcp_argv)
        )


def _hook_command(sub: str, *, repo: Path, venv_python: Path, db_home: Path) -> str:
    return (
        f"CLAUDE_MEMORY_HOME={db_home} PYTHONPATH={repo} "
        f"{venv_python} -m claude_memory.cli {sub}"
    )


def _group_is_ours(group: dict) -> bool:
    return any(
        _HOOK_MARKER in hook.get("command", "") for hook in group.get("hooks", [])
    )


def _merge_hooks(settings: dict, commands: dict[str, str]) -> None:
    hooks = settings.setdefault("hooks", {})
    for event, command in commands.items():
        kept = [group for group in hooks.get(event, []) if not _group_is_ours(group)]
        kept.append({"hooks": [{"type": "command", "command": command}]})
        hooks[event] = kept


def _remove_hooks(settings: dict) -> None:
    hooks = settings.get("hooks", {})
    for event in list(hooks):
        hooks[event] = [g for g in hooks[event] if not _group_is_ours(g)]
        if not hooks[event]:
            del hooks[event]
    if not hooks:
        settings.pop("hooks", None)


def _load_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Could not parse {path}: {error}") from error


def _write_settings(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _mcp_command(
    *, scope: str, repo: Path, venv_python: Path, db_home: Path
) -> list[str]:
    return [
        "claude",
        "mcp",
        "add",
        MCP_NAME,
        "-s",
        scope,
        "-e",
        f"CLAUDE_MEMORY_HOME={db_home}",
        "-e",
        f"PYTHONPATH={repo}",
        "--",
        str(venv_python),
        "-m",
        "claude_memory.cli",
        "mcp",
    ]


def _run(argv: list[str]) -> int:
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr.strip():
        print(result.stderr.strip())
    return result.returncode


def _stop_daemon(db_home: Path) -> None:
    """Best-effort: stop the daemon for this db_home (if running)."""
    os.environ["CLAUDE_MEMORY_HOME"] = str(db_home)
    print(stop_daemon(MemoryConfig.load()))


def _print_install_plan(
    settings: dict,
    *,
    settings_path: Path,
    db_home: Path,
    repo: Path,
    venv_python: Path,
    scope: str,
    mcp_argv: list[str],
    skip_skill: bool,
) -> None:
    """Print what `install` would write, changing nothing (dry run)."""
    print("--- DRY RUN: nothing written ---\n")
    print(f"scope         : {scope}")
    print(f"settings file : {settings_path}")
    print(f"database home : {db_home}")
    print(f"repo / venv   : {repo}\n                {venv_python}\n")
    print("settings.json would become:\n")
    print(json.dumps(settings, indent=2))
    print("\nMCP registration command:\n  " + " ".join(mcp_argv))
    if not skip_skill:
        _, skill_dest = _skill_paths(repo, scope)
        print(f"\nSkill '{_SKILL_NAME}' would be installed to:\n  {skill_dest}")


def cmd_install(args: argparse.Namespace) -> None:
    """Enable, preview, or remove the global (or project) claude_memory config."""
    repo = _repo_root()
    venv_python = _venv_python(repo)
    db_home = (args.db_home or (Path.home() / ".claude" / "claude_memory")).resolve()
    scope = args.scope
    settings_path = args.settings or (
        Path.cwd() / ".claude" / "settings.json"
        if scope == "project"
        else Path.home() / ".claude" / "settings.json"
    )

    if not venv_python.exists():
        raise SystemExit(
            f"Repo venv not found at {venv_python}. Run `uv sync` in {repo} first."
        )

    commands = {
        event: _hook_command(sub, repo=repo, venv_python=venv_python, db_home=db_home)
        for event, sub in _HOOK_MODULES.items()
    }
    mcp_argv = _mcp_command(
        scope=scope, repo=repo, venv_python=venv_python, db_home=db_home
    )

    settings = _load_settings(settings_path)
    if args.disable:
        _remove_hooks(settings)
    else:
        _merge_hooks(settings, commands)

    if args.dry_run:
        _print_install_plan(
            settings,
            settings_path=settings_path,
            db_home=db_home,
            repo=repo,
            venv_python=venv_python,
            scope=scope,
            mcp_argv=mcp_argv,
            skip_skill=args.skip_skill,
        )
        return

    _write_settings(settings_path, settings)
    print(f"{'Removed' if args.disable else 'Wrote'} hooks in {settings_path}")
    print(f"(backup: {settings_path}.bak)")

    if not args.skip_mcp:
        _apply_mcp(scope, mcp_argv, disable=args.disable)

    if not args.skip_skill:
        _apply_skill(repo, scope, disable=args.disable)

    if args.disable:
        _stop_daemon(db_home)
        if args.purge:
            shutil.rmtree(db_home, ignore_errors=True)
            print(f"Deleted memory databases at {db_home}.")
        else:
            print(f"\nMemory data kept at {db_home} (add --purge to delete it).")
        print("Restart Claude Code for the removal to take effect.")
        return

    print(f"\nDatabases will live in: {db_home}")
    print("Restart Claude Code for it to take effect.")
    try:
        __import__("sentence_transformers")
    except ImportError:
        print(
            "\nNote: sentence-transformers is not importable from this "
            "interpreter. embeddinggemma-300m (the only embedder) needs it:\n"
            f"  {venv_python} -m pip install sentence-transformers"
        )


# ----------------------------------------------------------------- daemon control


def cmd_daemon(args: argparse.Namespace) -> None:
    """Inspect or control the daemon for the current CLAUDE_MEMORY_HOME.

    Targets the daemon by its home-keyed socket and lock — never by process
    name — so it can only ever touch this home's daemon (see daemon.stop_daemon).
    """
    config = MemoryConfig.load()
    action = args.action

    if action == "status":
        status = daemon_status(config)
        print(f"Daemon: {'running' if status['running'] else 'not running'}")
        if status["pid"] is not None:
            print(f"  pid    : {status['pid']}")
        print(f"  home   : {status['home']}")
        print(f"  socket : {status['socket']}")
        return

    if action in {"stop", "restart"}:
        print(stop_daemon(config))

    if action in {"start", "restart"}:
        try:
            call({"op": "ping"}, config=config, wait_for_start=90.0, timeout=90.0)
        except DaemonUnavailableError:
            print("Daemon did not come up; check $CLAUDE_MEMORY_HOME/daemon.log.")
            return
        status = daemon_status(config)
        print(f"Daemon running (pid {status['pid']}).")


# -------------------------------------------------------------------- dispatch


def main() -> None:
    """Parse the subcommand and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mcp", help="Run the MCP server (stdio).")
    mcp_http = sub.add_parser(
        "mcp-http",
        help="Run the MCP server over HTTP on 127.0.0.1 (URL clients, "
        "e.g. Desktop connectors). Same daemon, same memory.",
    )
    mcp_http.add_argument(
        "--port", type=int, default=8765, help="Localhost port (default 8765)."
    )
    sub.add_parser("warm", help="SessionStart hook: warm the daemon.")
    sub.add_parser("ambient", help="UserPromptSubmit hook: ambient recall.")
    sub.add_parser("stop", help="Stop hook: reflective recall + verbatim capture.")
    sub.add_parser("ingest", help="Capture only (verbatim transcript tail).")

    daemon_parser = sub.add_parser(
        "daemon", help="Inspect or control the daemon (status/start/stop/restart)."
    )
    daemon_parser.add_argument(
        "action",
        choices=["status", "start", "stop", "restart"],
        help="What to do with the daemon for the current CLAUDE_MEMORY_HOME.",
    )

    install = sub.add_parser(
        "install", help="Enable globally (or --disable/--dry-run)."
    )
    install.add_argument(
        "--dry-run", action="store_true", help="Preview, write nothing."
    )
    install.add_argument(
        "--disable", action="store_true", help="Remove the integration."
    )
    install.add_argument(
        "--skip-mcp", action="store_true", help="Skip MCP registration."
    )
    install.add_argument(
        "--skip-skill", action="store_true", help="Skip episodic-recall skill install."
    )
    install.add_argument(
        "--purge",
        action="store_true",
        help="With --disable, also delete the memory databases.",
    )
    install.add_argument(
        "--scope",
        choices=["user", "project"],
        default="user",
        help="user = all projects (default); project = current repo only.",
    )
    install.add_argument(
        "--settings", type=Path, default=None, help="settings.json path."
    )
    install.add_argument(
        "--db-home", type=Path, default=None, help="Database directory."
    )

    args = parser.parse_args()
    if args.command == "mcp":
        cmd_mcp()
    elif args.command == "mcp-http":
        cmd_mcp_http(args.port)
    elif args.command == "warm":
        cmd_warm()
    elif args.command == "ambient":
        cmd_ambient()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "ingest":
        cmd_ingest()
    elif args.command == "daemon":
        cmd_daemon(args)
    elif args.command == "install":
        cmd_install(args)


if __name__ == "__main__":
    main()
