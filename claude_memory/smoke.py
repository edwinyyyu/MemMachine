"""Dependency-free smoke tests — no API key, no network (uses the hash embedder).

Two suites, run both by default:

  core   — the engine invariants directly (MemoryCore): messages are the search
           surface, expansion reaches non-embedded tool events, stable mem:<id>
           handles round-trip, novelty/diminishing-returns is reported, and the
           transcript parser produces the right timeline events.
  daemon — the full client<->daemon socket path: auto-spawn, ingest, search,
           expand, shared per-session novelty, connection reuse, clean shutdown.

Run:  uv run python -m claude_memory.smoke            # both
      uv run python -m claude_memory.smoke --daemon   # one suite
"""

import argparse
import asyncio
import contextlib
import datetime
import json
import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from memmachine_server.episodic_memory.event_memory.data_types import (
    AnnotationContext,
    CompositeContext,
    Event,
    ProducerContext,
    Segment,
    TextBlock,
)


class _Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        print(f"  [{'ok  ' if condition else 'FAIL'}] {message}")
        if not condition:
            self.failures.append(message)


def _event(offset_seconds: int, source: str, producer: str, text: str) -> Event:
    base = datetime.datetime(2026, 3, 1, 9, 0, 0, tzinfo=datetime.UTC)
    return Event(
        uuid=uuid4(),
        timestamp=base + datetime.timedelta(seconds=offset_seconds),
        context=ProducerContext(producer=producer),
        blocks=[TextBlock(text=text)],
        properties={"source": source, "producer": producer, "session_id": "smoke"},
    )


async def _core_suite(checker: _Checker) -> None:
    from memmachine_server.common.filter.filter_parser import (
        FilterParseError,
        parse_filter,
    )

    from claude_memory.engine import (
        MemoryConfig,
        MemoryCore,
        MessageOnlyDeriver,
        Source,
        WholeTextDeriver,
        in_context_exclusion_filter,
        render_expand_result,
        render_search_result,
        session_scope_filter,
    )
    from claude_memory.transcript import events_from_transcript

    print("== core suite ==")
    core = await MemoryCore.open(MemoryConfig.load())
    try:
        events = [
            _event(0, Source.USER_MESSAGE, "user", "I want to plan a trip to Sweden"),
            _event(
                1, Source.ASSISTANT_MESSAGE, "assistant", "I'll find flights to Sweden."
            ),
            _event(
                2,
                Source.TOOL_CALL,
                "assistant",
                'WebSearch {"query": "flights to Stockholm"}',
            ),
            _event(
                3,
                Source.TOOL_RESULT,
                "tool",
                "SAS direct flights to Stockholm from $420.",
            ),
            _event(
                4,
                Source.ASSISTANT_MESSAGE,
                "assistant",
                "I tentatively chose SAS; confirm to book.",
            ),
            _event(
                10,
                Source.USER_MESSAGE,
                "user",
                "Caroline mentioned Sweden is her home country",
            ),
        ]
        checker.check(await core.ingest(events) == len(events), "ingested events")

        trip = await core.search("planning a trip to Sweden", limit=8)
        checker.check(len(trip.hits) > 0, "search returns message hits")
        joined = "\n".join(hit.text for hit in trip.hits)
        checker.check(
            "WebSearch" not in joined and "SAS direct" not in joined,
            "tool call/result are NOT direct search hits (messages are the surface)",
        )

        booking = await core.search(
            "tentatively chose flights confirm to book", limit=3
        )
        checker.check(len(booking.hits) > 0, "found the booking message")
        seed = booking.hits[0].memory_id
        checker.check(seed.startswith("mem:"), f"stable handle ({seed})")

        expanded = await core.expand(seed, before=4, after=1)
        rendered = render_expand_result(expanded)
        checker.check(expanded.found, "expand resolved the seed")
        checker.check(
            "WebSearch" in rendered or "SAS direct" in rendered,
            "expansion reaches the non-embedded tool call/result",
        )

        booking_score = booking.hits[0].score
        noted = await core.annotate(seed, "the SAS booking was later cancelled")
        checker.check(
            "[note: the SAS booking was later cancelled]" in noted,
            "annotate echoes the note inline",
        )
        noted_again = await core.annotate(seed, "rebooked  with\nFinnair")
        checker.check(
            "[note: the SAS booking was later cancelled] [note: rebooked with "
            "Finnair]" in noted_again,
            "annotations are append-only and render in order",
        )
        rebooked = await core.search(
            "tentatively chose flights confirm to book",
            limit=3,
            seen=set(),
            commit_seen=False,
        )
        checker.check(
            rebooked.hits[0].memory_id == seed
            and "[note: rebooked with Finnair]" in rebooked.hits[0].text,
            "notes surface on future retrievals of the same segment",
        )
        checker.check(
            abs(rebooked.hits[0].score - booking_score) < 1e-9,
            "annotation does not change the embedding (same retrieval score)",
        )

        plain_segment = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=datetime.datetime(2026, 3, 1, 9, 0, 0, tzinfo=datetime.UTC),
            context=ProducerContext(producer="user"),
            block=TextBlock(text="hello there"),
            properties={"source": Source.USER_MESSAGE},
        )
        deriver = MessageOnlyDeriver(WholeTextDeriver())
        before_texts = [
            d.block.text
            for d in await deriver.derive(plain_segment, format_options=None)
            if isinstance(d.block, TextBlock)
        ]
        plain_segment.context = CompositeContext(
            contexts=[
                ProducerContext(producer="user"),
                AnnotationContext(note="a later note"),
            ]
        )
        after_texts = [
            d.block.text
            for d in await deriver.derive(plain_segment, format_options=None)
            if isinstance(d.block, TextBlock)
        ]
        checker.check(
            bool(before_texts) and before_texts == after_texts,
            "deriver embed text ignores annotations (embedding frozen)",
        )

        try:
            AnnotationContext(note="bad\nnote")
            checker.check(False, "AnnotationContext rejects newlines")
        except ValueError:
            checker.check(True, "AnnotationContext rejects newlines")

        again = await core.search("planning a trip to Sweden", limit=8)
        checker.check(
            again.saturated and again.new_count == 0,
            "repeating a cue reports 0 new (recall saturating)",
        )

        hop = await core.search("Caroline home country", limit=3)
        checker.check(len(hop.hits) > 0, "follow-the-lead is just another search")
        _ = render_search_result(hop, cue="Caroline home country")

        from claude_memory.cli import _AMBIENT_CURATION_NOTE, _render_ambient
        from claude_memory.engine import Hit

        amb = _render_ambient(
            [Hit(memory_id="mem:deadbeef", score=0.5, text="a deploy note", is_new=True)]
        )
        checker.check(
            amb.startswith(_AMBIENT_CURATION_NOTE) and "[mem:deadbeef]" in amb,
            "ambient render leads with the curation affordance, then the memories",
        )
        checker.check(
            _render_ambient([]) == "",
            "ambient render is empty (no affordance) when nothing surfaced",
        )

        from mcp.server.fastmcp import FastMCP

        from claude_memory.cli import _register_memory_tools

        probe_mcp = FastMCP("probe")
        _register_memory_tools(probe_mcp, wait=0.0)
        meta_by = {t.name: (t.meta or {}) for t in await probe_mcp.list_tools()}
        checker.check(
            meta_by.get("memory_demote", {}).get("anthropic/alwaysLoad") is True
            and meta_by.get("memory_annotate", {}).get("anthropic/alwaysLoad") is True,
            "curation tools are marked alwaysLoad (pre-loaded, not deferred)",
        )
        checker.check(
            "anthropic/alwaysLoad" not in meta_by.get("memory_search", {})
            and "anthropic/alwaysLoad" not in meta_by.get("memory_expand", {}),
            "read tools stay deferred (loaded on deliberate use)",
        )

        [cue_vec] = await core.stores.embedder.search_embed(
            ["planning a trip to Sweden"]
        )
        manual = await core.search(
            "planning a trip to Sweden",
            limit=8,
            seen=set(),
            commit_seen=False,
            query_vector=list(cue_vec),
        )
        checker.check(
            [h.memory_id for h in manual.hits] == [h.memory_id for h in trip.hits],
            "precomputed query_vector reproduces the plain search",
        )

        transcript = Path(core.stores.config.home) / "fake.jsonl"
        records = [
            {
                "type": "user",
                "uuid": "rec-1",
                "timestamp": "2026-03-02T10:00:00Z",
                "message": {"role": "user", "content": "fix the bug"},
            },
            {
                "type": "assistant",
                "uuid": "rec-2",
                "timestamp": "2026-03-02T10:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Looking now."},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "/a.py"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "rec-3",
                "timestamp": "2026-03-02T10:00:02Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "line1\nline2",
                        },
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "rec-4",
                "isCompactSummary": True,
                "timestamp": "2026-03-02T10:00:03Z",
                "message": {
                    "role": "user",
                    "content": "SUMMARY: earlier the user asked to fix the bug.",
                },
            },
        ]
        transcript.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        parsed, total = events_from_transcript(transcript, session_id="t", start_line=0)
        sources = [e.properties.get("source") for e in parsed]
        checker.check(total == 4, f"counted transcript lines ({total})")
        checker.check(
            sources
            == [
                Source.USER_MESSAGE,
                Source.ASSISTANT_MESSAGE,
                Source.TOOL_CALL,
                Source.TOOL_RESULT,
            ],
            f"compaction summary skipped; sources in order ({sources})",
        )
        empty, total2 = events_from_transcript(
            transcript, session_id="t", start_line=total
        )
        checker.check(not empty and total2 == 4, "incremental parse past hwm is empty")

        # Filters must use single-quoted strings; a hyphenated UUID in double
        # quotes silently fails to tokenize (regression guard for that bug).
        hyphen_sid = "11111111-2222-3333-4444-555555555555"
        cutoff = datetime.datetime(2026, 3, 2, 10, 0, tzinfo=datetime.UTC)

        def _parses(spec: str | None) -> bool:
            try:
                parse_filter(spec)
            except FilterParseError:
                return False
            return True

        checker.check(
            all(
                _parses(spec)
                for spec in (
                    session_scope_filter(hyphen_sid),
                    in_context_exclusion_filter(hyphen_sid, None),
                    in_context_exclusion_filter(hyphen_sid, cutoff),
                )
            ),
            "session / in-context filters parse for a hyphenated id",
        )

        # Idempotent ingest: event uuids derive from the record uuid, so
        # re-ingesting the same transcript writes nothing the second time.
        events_a, _ = events_from_transcript(transcript, session_id="t2", start_line=0)
        first = await core.ingest(events_a)
        events_b, _ = events_from_transcript(transcript, session_id="t2", start_line=0)
        second = await core.ingest(events_b)
        checker.check(
            first == 4 and second == 0,
            f"re-ingest is idempotent (first {first} new, second {second} new)",
        )
    finally:
        await core.aclose()


def _daemon_suite(checker: _Checker) -> None:
    from claude_memory.daemon import DaemonUnavailableError, call
    from claude_memory.engine import (
        demote_result_from_dict,
        expand_result_from_dict,
        render_expand_result,
        search_result_from_dict,
    )

    print("== daemon suite ==")
    home = Path(os.environ["CLAUDE_MEMORY_HOME"])
    transcript = home / "session.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "type": "user",
                    "timestamp": "2026-03-02T10:00:00Z",
                    "message": {
                        "role": "user",
                        "content": "The deploy script is scripts/deploy.sh needing AWS_PROFILE=prod",
                    },
                },
                {
                    "type": "assistant",
                    "timestamp": "2026-03-02T10:00:01Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Noted the deploy script and prod profile.",
                            },
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Read",
                                "input": {"file_path": "scripts/deploy.sh"},
                            },
                        ],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    try:
        start = time.monotonic()
        ingest = call(
            {"op": "ingest", "transcript_path": str(transcript), "session_id": "sess-1111-2222"},
            wait_for_start=90.0,
        )
        print(f"  (daemon cold start + ingest: {time.monotonic() - start:.2f}s)")
        checker.check(
            bool(ingest.get("ok")) and ingest.get("ingested", 0) >= 2,
            "ingest via daemon",
        )

        warm = time.monotonic()
        response = call(
            {"op": "search", "cue": "how do I deploy to production", "session_id": "sess-1111-2222"}
        )
        print(f"  (warm search round trip: {time.monotonic() - warm:.3f}s)")
        checker.check(bool(response.get("ok")), "search via daemon (warm, no re-spawn)")
        result = search_result_from_dict(response["result"])
        joined = "\n".join(hit.text for hit in result.hits)
        checker.check("deploy script" in joined, "deploy memory retrieved")
        checker.check("Read " not in joined, "tool call is not a direct search hit")

        seed = result.hits[0].memory_id
        exp = call(
            {"op": "expand", "seed": seed, "before": 3, "after": 3, "session_id": "sess-1111-2222"}
        )
        expanded = expand_result_from_dict(exp["result"])
        checker.check(
            expanded.found and "Read " in render_expand_result(expanded),
            "expansion reaches the non-embedded tool call",
        )

        again = call(
            {"op": "search", "cue": "how do I deploy to production", "session_id": "sess-1111-2222"}
        )
        again_result = search_result_from_dict(again["result"])
        checker.check(
            again_result.saturated and again_result.new_count == 0,
            "shared per-session novelty: repeat search reports 0 new",
        )

        # Reflective recall: a fresh session re-evokes memory from the model's
        # own last reply (the transcript's final assistant message).
        refl = call(
            {"op": "reflect", "transcript_path": str(transcript), "session_id": "sess-3333-4444"}
        )
        checker.check(
            bool(refl.get("ok")) and "deploy script" in (refl.get("memories") or ""),
            "reflect surfaces related memory from the model's last reply",
        )

        bad = call({"op": "demote", "memory_id": "not-a-mem-id", "cue": "x"})
        checker.check(
            bool(bad.get("ok"))
            and demote_result_from_dict(bad["result"]).verdict == "invalid",
            "demote rejects an invalid memory id",
        )
        dm = call(
            {"op": "demote", "memory_id": seed, "cue": "how do I deploy to production"}
        )
        dm_verdict = demote_result_from_dict(dm["result"]).verdict if dm.get("ok") else ""
        checker.check(
            dm_verdict in {"demoted", "saturated"},
            f"demote op returns a verdict ({dm_verdict})",
        )

        an = call(
            {"op": "annotate", "memory_id": seed, "note": "deploy script replaced"}
        )
        checker.check(
            bool(an.get("ok"))
            and "[note: deploy script replaced]" in str(an.get("message", "")),
            "annotate op appends and echoes the note",
        )

        ctx1 = call(
            {
                "op": "search",
                "cue": "how do I deploy to production",
                "session_id": "sess-ctx",
                "use_context": True,
            }
        )
        ctx2 = call(
            {
                "op": "search",
                "cue": "and the next step?",
                "session_id": "sess-ctx",
                "use_context": True,
            }
        )
        checker.check(
            bool(ctx1.get("ok"))
            and bool(ctx2.get("ok"))
            and len(ctx2["result"]["hits"]) > 0,
            "running-context blended search serves a terse follow-up",
        )

        try:
            call({"op": "ping"}, wait_for_start=0.0, timeout=5.0)
            checker.check(True, "second connection reuses the live daemon")
        except DaemonUnavailableError:
            checker.check(False, "daemon stayed up between calls")
    finally:
        with contextlib.suppress(Exception):
            call({"op": "shutdown"}, wait_for_start=0.0, timeout=5.0)


def main() -> None:
    """Run the selected smoke suites against a throwaway hash-embedder store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", action="store_true", help="Run only the core suite.")
    parser.add_argument(
        "--daemon", action="store_true", help="Run only the daemon suite."
    )
    args = parser.parse_args()
    run_core = args.core or not args.daemon
    run_daemon = args.daemon or not args.core

    base = Path(tempfile.mkdtemp(prefix="claude_memory_smoke_"))
    os.environ["CLAUDE_MEMORY_EMBEDDING"] = "hash"
    os.environ["CLAUDE_MEMORY_DAEMON_IDLE"] = "120"
    # Exercise the reflect op; threshold -1 lets hash-embedder scores through so
    # the check tests the mechanism (novelty + render), not semantic relevance.
    os.environ["CLAUDE_MEMORY_REFLECT"] = "1"
    os.environ["CLAUDE_MEMORY_REFLECT_THRESHOLD"] = "-1"

    checker = _Checker()
    if run_core:
        # Each suite gets its own home + partition so they cannot contaminate.
        (base / "core").mkdir(parents=True, exist_ok=True)
        os.environ["CLAUDE_MEMORY_HOME"] = str(base / "core")
        os.environ["CLAUDE_MEMORY_PARTITION"] = "core"
        asyncio.run(_core_suite(checker))
    if run_daemon:
        (base / "daemon").mkdir(parents=True, exist_ok=True)
        os.environ["CLAUDE_MEMORY_HOME"] = str(base / "daemon")
        os.environ["CLAUDE_MEMORY_PARTITION"] = "daemon"
        _daemon_suite(checker)

    print()
    if checker.failures:
        print(f"SMOKE FAILED: {len(checker.failures)} check(s) failed")
        raise SystemExit(1)
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
