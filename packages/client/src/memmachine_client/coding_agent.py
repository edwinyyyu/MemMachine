"""Point coding agents at a MemMachine server's memory tools.

Claude Code and Codex both speak MCP over streamable HTTP with static
headers, so one endpoint (`<server>/v1/mcp`) and one header
(`X-MemMachine-Tenant`) serve both, and the tools (`memory_search`,
`memory_expand`) live on the server. This module writes and removes that
entry in each agent's own configuration.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_NAMES = ("claude-code", "codex")
SCOPE_NAMES = ("user", "project")

MCP_SERVER_NAME = "memmachine"
MCP_ENDPOINT_PATH = "/v1/mcp"
TENANT_HEADER_NAME = "X-MemMachine-Tenant"

CLAUDE_EXECUTABLE_NAME = "claude"
CLAUDE_PROJECT_CONFIG_NAME = ".mcp.json"
CODEX_HOME_VARIABLE = "CODEX_HOME"
CODEX_DIRECTORY_NAME = ".codex"
CODEX_CONFIG_NAME = "config.toml"
BACKUP_SUFFIX = ".bak"

CODEX_TABLE_NAMES = (
    ("mcp_servers", MCP_SERVER_NAME),
    ("mcp_servers", MCP_SERVER_NAME, "http_headers"),
)


class CodingAgentError(Exception):
    """Raised when an agent's configuration cannot be read or edited."""


def run_agent_command(args: argparse.Namespace) -> int:
    """Run an `agent` subcommand and return the process exit code."""
    try:
        if args.agent_command == "install":
            server = _validated_server(args.server)
            tenant = _validated_tenant(args.tenant)
            if args.agent_name == "claude-code":
                return install_claude_code(
                    server=server,
                    tenant=tenant,
                    scope=args.scope,
                    dry_run=args.dry_run,
                )
            return install_codex(
                server=server,
                tenant=tenant,
                scope=args.scope,
                dry_run=args.dry_run,
            )
        if args.agent_name == "claude-code":
            return disable_claude_code(scope=args.scope)
        return disable_codex(scope=args.scope)
    except CodingAgentError as error:
        sys.stderr.write(f"{args.prog}: error: {error}\n")
        return 1


def install_claude_code(*, server: str, tenant: str, scope: str, dry_run: bool) -> int:
    """Register the MemMachine MCP server with Claude Code.

    User scope is delegated to the `claude` executable, which owns that
    file; project scope writes `.mcp.json` in the current directory.
    """
    url = _mcp_endpoint_url(server)
    if scope == "project":
        entry = {
            "type": "http",
            "url": url,
            "headers": {TENANT_HEADER_NAME: tenant},
        }
        return _edit_claude_project_config(entry, dry_run=dry_run)
    # `claude mcp add` refuses a name it already holds, so a reinstall
    # removes the earlier entry first and leaves exactly one behind.
    return _run_claude_executable(
        [
            "mcp",
            "add",
            "--transport",
            "http",
            "--scope",
            "user",
            MCP_SERVER_NAME,
            url,
            "--header",
            f"{TENANT_HEADER_NAME}: {tenant}",
        ],
        preceded_by=["mcp", "remove", "--scope", "user", MCP_SERVER_NAME],
        dry_run=dry_run,
    )


def disable_claude_code(*, scope: str) -> int:
    """Remove the MemMachine MCP server entry from Claude Code."""
    if scope == "project":
        return _edit_claude_project_config(None, dry_run=False)
    return _run_claude_executable(
        ["mcp", "remove", "--scope", "user", MCP_SERVER_NAME],
        preceded_by=None,
        dry_run=False,
    )


def _run_claude_executable(
    arguments: list[str],
    *,
    preceded_by: list[str] | None,
    dry_run: bool,
) -> int:
    """Run `claude` with the given arguments, reporting the command run.

    `preceded_by` runs first and its exit code is ignored, which is what
    makes a reinstall idempotent. Without a `claude` executable on PATH
    the error names the command to run by hand.
    """
    command_line = shlex.join([CLAUDE_EXECUTABLE_NAME, *arguments])
    executable = shutil.which(CLAUDE_EXECUTABLE_NAME)
    if executable is None:
        raise CodingAgentError(
            f"no {CLAUDE_EXECUTABLE_NAME} executable on PATH. Run this where "
            f"Claude Code is installed:\n  {command_line}"
        )
    if dry_run:
        sys.stdout.write(f"would run: {command_line}\n")
        return 0
    if preceded_by is not None:
        subprocess.run(
            [executable, *preceded_by],
            capture_output=True,
            check=False,
            text=True,
        )
    result = subprocess.run(
        [executable, *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    sys.stdout.write(f"ran: {command_line}\n")
    return 0


def _edit_claude_project_config(
    entry: dict[str, object] | None,
    *,
    dry_run: bool,
) -> int:
    """Set or remove the MemMachine server in `.mcp.json` in this directory.

    Every other server and every other key of the file is carried over.
    """
    path = Path.cwd() / CLAUDE_PROJECT_CONFIG_NAME
    original_text = path.read_text(encoding="utf-8") if path.exists() else ""
    document: dict[str, object] = {}
    if original_text.strip():
        try:
            loaded = json.loads(original_text)
        except json.JSONDecodeError as error:
            raise CodingAgentError(f"{path} is not valid JSON: {error}") from error
        if not isinstance(loaded, dict):
            raise CodingAgentError(f"{path} does not hold a JSON object")
        document = loaded

    servers = document.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise CodingAgentError(f'{path} has an "mcpServers" key that is not an object')
    servers = dict(servers)
    if entry is None:
        servers.pop(MCP_SERVER_NAME, None)
    else:
        servers[MCP_SERVER_NAME] = entry
    document["mcpServers"] = servers

    new_text = f"{json.dumps(document, indent=2)}\n"
    return _apply_config_change(path, original_text, new_text, dry_run=dry_run)


def install_codex(*, server: str, tenant: str, scope: str, dry_run: bool) -> int:
    """Write the MemMachine MCP server tables into Codex's `config.toml`."""
    path = _codex_config_path(scope)
    url = _mcp_endpoint_url(server)
    original_text = path.read_text(encoding="utf-8") if path.exists() else ""
    document = _loaded_toml(original_text, path)
    spans = _memmachine_table_spans(original_text.splitlines(keepends=True))
    if not spans and _codex_entry(document) is not None:
        raise CodingAgentError(
            f"{path} already defines mcp_servers.{MCP_SERVER_NAME} in a form this "
            "installer does not edit (an inline table or a dotted key). "
            "Remove it by hand and run this again."
        )

    # TOML basic strings take JSON's escapes, and JSON escapes every
    # character that a basic string may not carry raw.
    tables = (
        f"[mcp_servers.{MCP_SERVER_NAME}]\n"
        f"url = {json.dumps(url)}\n"
        "\n"
        f"[mcp_servers.{MCP_SERVER_NAME}.http_headers]\n"
        f"{TENANT_HEADER_NAME} = {json.dumps(tenant)}\n"
    )
    new_text = _codex_text_with_tables(original_text, spans, tables)
    written = _codex_entry(_loaded_toml(new_text, path))
    if written != {"url": url, "http_headers": {TENANT_HEADER_NAME: tenant}}:
        raise CodingAgentError(
            f"the edit of {path} would not have produced the intended "
            f"mcp_servers.{MCP_SERVER_NAME} table. The file is unchanged."
        )
    return _apply_config_change(path, original_text, new_text, dry_run=dry_run)


def disable_codex(*, scope: str) -> int:
    """Remove the MemMachine MCP server tables from Codex's `config.toml`."""
    path = _codex_config_path(scope)
    if not path.exists():
        sys.stdout.write(f"{path} does not exist\n")
        return 0
    original_text = path.read_text(encoding="utf-8")
    document = _loaded_toml(original_text, path)
    spans = _memmachine_table_spans(original_text.splitlines(keepends=True))
    if not spans and _codex_entry(document) is not None:
        raise CodingAgentError(
            f"{path} defines mcp_servers.{MCP_SERVER_NAME} in a form this installer "
            "does not edit (an inline table or a dotted key). Remove it by hand."
        )
    new_text = _codex_text_without_tables(original_text, spans)
    if _codex_entry(_loaded_toml(new_text, path)) is not None:
        raise CodingAgentError(
            f"the edit of {path} would have left mcp_servers.{MCP_SERVER_NAME} "
            "behind. The file is unchanged."
        )
    return _apply_config_change(path, original_text, new_text, dry_run=False)


def _codex_config_path(scope: str) -> Path:
    """Return the `config.toml` a scope owns.

    User scope follows `$CODEX_HOME`, which defaults to `~/.codex`;
    project scope is the `.codex` directory of the current directory.
    """
    if scope == "project":
        return Path.cwd() / CODEX_DIRECTORY_NAME / CODEX_CONFIG_NAME
    codex_home = os.environ.get(CODEX_HOME_VARIABLE)
    if codex_home:
        return Path(codex_home) / CODEX_CONFIG_NAME
    return Path.home() / CODEX_DIRECTORY_NAME / CODEX_CONFIG_NAME


def _codex_text_with_tables(
    original_text: str,
    spans: list[tuple[int, int]],
    tables: str,
) -> str:
    """Return the file's text with our tables in it, once.

    The first table this installer owns is replaced where it stands, any
    later one is dropped, and a file without one gets the tables appended
    after a blank line. Every line outside those spans is carried over
    unchanged.
    """
    if not spans:
        if not original_text:
            return tables
        text = original_text if original_text.endswith("\n") else f"{original_text}\n"
        if not text.endswith("\n\n"):
            text += "\n"
        return text + tables
    lines = original_text.splitlines(keepends=True)
    for start, end in reversed(spans[1:]):
        del lines[start:end]
    first_start, first_end = spans[0]
    # A span ends where the next table's header begins, so tables that
    # something follows keep the blank line that separates them from it.
    separated = tables if first_end >= len(lines) else f"{tables}\n"
    lines[first_start:first_end] = [separated]
    return "".join(lines)


def _codex_text_without_tables(
    original_text: str,
    spans: list[tuple[int, int]],
) -> str:
    """Return the file's text with the tables this installer owns removed.

    Every line outside those spans is carried over unchanged, except the
    blank line an append put before tables that run to the end of the
    file, which goes with them.
    """
    lines = original_text.splitlines(keepends=True)
    removed_to_end = bool(spans) and spans[-1][1] == len(lines)
    for start, end in reversed(spans):
        del lines[start:end]
    if removed_to_end and lines and not lines[-1].strip():
        # The blank line an append left in front of them goes too.
        del lines[-1]
    return "".join(lines)


def _memmachine_table_spans(lines: list[str]) -> list[tuple[int, int]]:
    """Return the half-open line ranges of the tables this installer owns.

    A table runs from its header to the line before the next header, so a
    span carries the table's own comments and blank lines with it. Lines
    inside a multi-line string are not read as headers.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    string_delimiter: str | None = None
    for index, line in enumerate(lines):
        if string_delimiter is not None:
            if line.count(string_delimiter) % 2 == 1:
                string_delimiter = None
            continue
        stripped = line.lstrip()
        if stripped.startswith("["):
            if start is not None:
                spans.append((start, index))
                start = None
            if _table_header_parts(stripped) in CODEX_TABLE_NAMES:
                start = index
            continue
        for delimiter in ('"""', "'''"):
            if line.count(delimiter) % 2 == 1:
                string_delimiter = delimiter
                break
    if start is not None:
        spans.append((start, len(lines)))
    return spans


def _table_header_parts(line: str) -> tuple[str, ...]:
    """Return the dotted parts of a table header, or nothing for other lines.

    Quoted key parts are returned unquoted. An array-of-tables header is
    never one of ours, so it returns nothing.
    """
    text = line.strip()
    if not text.startswith("[") or text.startswith("[["):
        return ()
    parts: list[str] = []
    current = ""
    quote: str | None = None
    for character in text[1:]:
        if quote is not None:
            if character == quote:
                quote = None
            else:
                current += character
        elif character in {'"', "'"}:
            quote = character
        elif character == ".":
            parts.append(current.strip())
            current = ""
        elif character == "]":
            parts.append(current.strip())
            return tuple(parts)
        else:
            current += character
    return ()


def _codex_entry(document: dict[str, object]) -> object:
    """Return the `mcp_servers.memmachine` value of a parsed config, or None."""
    servers = document.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    return servers.get(MCP_SERVER_NAME)


def _loaded_toml(text: str, path: Path) -> dict[str, object]:
    """Parse TOML text, naming the file that failed to parse."""
    if sys.version_info < (3, 11):
        raise CodingAgentError(
            "reading a Codex configuration needs Python 3.11 or newer for "
            "tomllib; install this client on 3.11+ to configure Codex"
        )
    import tomllib

    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CodingAgentError(f"{path} is not valid TOML: {error}") from error


def _apply_config_change(
    path: Path,
    original_text: str,
    new_text: str,
    *,
    dry_run: bool,
) -> int:
    """Write a configuration file, keeping `<file>.bak` of what it replaced.

    A file that already holds the intended text is left alone, so a
    second install neither backs up nor writes.
    """
    if new_text == original_text:
        sys.stdout.write(f"{path} already holds this entry\n")
        return 0
    if dry_run:
        difference = "".join(
            difflib.unified_diff(
                original_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path),
            )
        )
        if not difference.endswith("\n"):
            difference += "\n"
        sys.stdout.write(f"would write {path}\n{difference}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup_path = path.with_name(path.name + BACKUP_SUFFIX)
        backup_path.write_text(original_text, encoding="utf-8")
        sys.stdout.write(f"backed up {path} to {backup_path}\n")
    path.write_text(new_text, encoding="utf-8")
    sys.stdout.write(f"wrote {path}\n")
    return 0


def _mcp_endpoint_url(server: str) -> str:
    """Return the MCP endpoint of a server's base URL."""
    return server.rstrip("/") + MCP_ENDPOINT_PATH


def _validated_server(server: str) -> str:
    """Return the server base URL, rejecting one no agent could reach."""
    if not server.startswith(("http://", "https://")):
        raise CodingAgentError(f"--server must be an http or https URL: {server}")
    return server


def _validated_tenant(tenant: str) -> str:
    """Return the tenant name, rejecting one no HTTP header could carry."""
    if not tenant or any(
        character < " " or character == "\x7f" for character in tenant
    ):
        raise CodingAgentError(
            "--tenant must be a non-empty name without control characters"
        )
    return tenant
