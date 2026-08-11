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
        MemoryCore,
        MessageOnlyDeriver,
        WholeTextDeriver,
    )
    from claude_memory.transcript import events_from_transcript
    from claude_memory.wire import (
        ANCHOR_MARKER as _ANCHOR,
    )
    from claude_memory.wire import (
        ID_FLOOR_CHARS,
        MemoryConfig,
        Source,
        expand_result_from_dict,
        in_context_exclusion_filter,
        render_expand_result,
        render_search_result,
        search_result_from_dict,
        session_scope_filter,
    )

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
        seed_uuid = booking.hits[0].segment_uuid
        checker.check(seed.startswith("mem:"), f"stable handle ({seed})")
        # Handles are prefixes: short enough to be worth the change, long enough
        # that nothing else in the store answers to them, and the whole uuid still
        # works so ids captured before this — in annotations, in the user's notes —
        # keep resolving.
        checker.check(
            len(seed) < len("mem:") + 32 and len(seed) >= len("mem:") + ID_FLOOR_CHARS,
            f"handle is an abbreviated prefix ({len(seed) - len('mem:')} digits)",
        )
        resolved, _ = await core.resolve_memory_id(seed)
        whole, _ = await core.resolve_memory_id(f"mem:{seed_uuid}")
        checker.check(
            resolved is not None and resolved.hex == seed_uuid == (whole and whole.hex),
            "short handle and whole uuid resolve to the same segment",
        )
        bad_uuid, bad_note = await core.resolve_memory_id("mem:zzzz")
        checker.check(
            bad_uuid is None and "not a valid memory id" in bad_note,
            "a non-hex handle is rejected as malformed",
        )
        # An MCP client is a subprocess that outlives many daemon restarts, so it
        # decodes replies from daemons newer than itself. Adding a reply field must
        # not break it — this once took down every read tool in a running session.
        checker.check(
            expand_result_from_dict(
                {"seed_id": seed, "window_text": "w", "invented_later": 1}
            ).seed_id
            == seed
            and search_result_from_dict(
                {
                    "hits": [
                        {
                            "memory_id": seed,
                            "score": 0.5,
                            "text": "t",
                            "is_new": True,
                            "invented_later": 1,
                        }
                    ],
                    "new_count": 1,
                    "saturated": False,
                }
            )
            .hits[0]
            .memory_id
            == seed,
            "a reply field the client has never heard of is ignored, not fatal",
        )
        # Half-open [since, before): one day is that day and the next, and the two
        # bounds never mean different things for the same string.
        day = await core.search(
            "planning a trip to Sweden",
            limit=8,
            since="2026-03-01",
            before="2026-03-02",
            seen=set(),
        )
        empty = await core.search(
            "planning a trip to Sweden",
            limit=8,
            since="2026-03-01",
            before="2026-03-01",
            seen=set(),
        )
        bad_date = await core.search(
            "planning a trip to Sweden", limit=8, since="last tuesday", seen=set()
        )
        checker.check(
            len(day.hits) > 0
            and not empty.hits
            and "ISO 8601" in (bad_date.note or ""),
            "since/before is a half-open range and says so when it cannot parse",
        )
        # Scoping is by handle too: any memory names its own conversation. There is
        # no session id to pass anywhere, so there is no second kind of address to
        # get wrong.
        scoped = await core.search(
            "planning a trip to Sweden", limit=8, within=seed, seen=set()
        )
        unknown = await core.search(
            "planning a trip to Sweden", limit=8, within="mem:zzzz", seen=set()
        )
        checker.check(
            len(scoped.hits) > 0
            and not unknown.hits
            and "not a valid memory id" in (unknown.note or ""),
            "search scopes to the conversation a handle names",
        )

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
        from claude_memory.wire import Hit

        amb = _render_ambient(
            [
                Hit(
                    memory_id="mem:deadbeef",
                    score=0.5,
                    text="a deploy note",
                    is_new=True,
                )
            ]
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
                # The compaction summary is kept on the timeline and typed
                # INJECTED, which keeps it off the search surface without losing
                # the one record of where the session lost its context.
                Source.INJECTED,
            ],
            f"compaction summary typed injected; sources in order ({sources})",
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
            first == 5 and second == 0,
            f"re-ingest is idempotent (first {first} new, second {second} new)",
        )

        # Ambiguity, made certain rather than likely. Seventeen segments cannot fit
        # in the sixteen buckets a single hex digit has, so at least one digit names
        # more than one memory no matter how the uuids fall — and guessing which was
        # meant would be an invisible error, a real memory that simply is not the one
        # that was asked for.
        filler = [
            _event(100 + n, Source.USER_MESSAGE, "user", f"filler turn number {n}")
            for n in range(20)
        ]
        await core.ingest(filler)
        collisions = [
            note
            for digit in "0123456789abcdef"
            for found, note in [await core.resolve_memory_id(f"mem:{digit}")]
            if found is None and "matches" in note
        ]
        checker.check(
            bool(collisions)
            and all("Retry with one of these in full" in note for note in collisions),
            f"an ambiguous prefix reports candidates instead of guessing "
            f"({len(collisions)} of 16 digits ambiguous)",
        )

        # A long command output must not be able to spend a whole window on its own
        # opening. Asking for four events around the prompt has to reach the reply
        # on the far side of a 33-chunk result, and what is shown of that result is
        # its two ends with the hole between them marked.
        bulk = "\n".join(f"output line {n} " + "x" * 120 for n in range(120))
        await core.ingest(
            [
                _event(200, Source.USER_MESSAGE, "user", "run the long build please"),
                _event(201, Source.TOOL_CALL, "assistant", "Bash(cmd='make all')"),
                _event(202, Source.TOOL_RESULT, "tool", bulk),
                _event(203, Source.ASSISTANT_MESSAGE, "assistant", "Build passed."),
            ]
        )
        build = await core.search("run the long build please", limit=1, seen=set())
        around = render_expand_result(
            await core.expand(build.hits[0].memory_id, before=0, after=3, unit="events")
        )
        checker.check(
            "output line 0" in around and "output line 119" in around,
            "a bulky result is shown by its two ends",
        )
        checker.check(
            "more characters — memory_expand from" in around,
            "the hole between those ends is marked where it happens",
        )
        checker.check(
            "Build passed." in around,
            "the turn after the bulky result still fits in the window",
        )
        checker.check(
            "[...]" not in around and around.count("more characters") == 1,
            "ONE mark, not a bare [...] plus a separate line saying what it meant",
        )
        checker.check(
            "output line 0" in around and "output line 119" in around,
            "the two ends are shown as content, not just as handles",
        )
        # The advice has to work AS PRINTED: the marker names a handle and nothing
        # else, so run it with the tool's own defaults. Making that event the seed
        # is what raises its cap from three chunks to forty.
        advice = around.split("— memory_expand from ", 1)[1].split("]")[0].split()
        more = render_expand_result(await core.expand(advice[0], before=0, after=30))
        checker.check(
            len(advice) >= 2 and all(a.startswith("mem:") for a in advice),
            f"the marker hands back every surviving segment as a seed ({advice})",
        )
        checker.check(
            "output line 60" in more and "output line 60" not in around,
            f"expanding one of them reaches the middle the sample skipped "
            f"({len(around):,} -> {len(more):,} chars)",
        )

        # Stepping by segments is the default and is a flat budget; stepping by
        # events buys whole turns however long they ran.
        by_segments = await core.expand(build.hits[0].memory_id, before=0, after=3)
        by_events = await core.expand(
            build.hits[0].memory_id, before=0, after=3, unit="events"
        )
        checker.check(
            by_segments.events < by_events.events,
            f"segments is the default unit; events reaches further "
            f"({by_segments.events} vs {by_events.events} events)",
        )
        checker.check(
            render_expand_result(by_events).startswith("[session smoke]"),
            "the window names its conversation, and nothing else in the header",
        )
        await core.ingest(
            [
                _event(300, Source.USER_MESSAGE, "user", "and now deploy it"),
                _event(
                    301,
                    Source.INJECTED,
                    "user",
                    "<system-reminder>noise</system-reminder>",
                ),
                _event(
                    302,
                    Source.INJECTED,
                    "user",
                    "<system-reminder>more</system-reminder>",
                ),
                _event(303, Source.ASSISTANT_MESSAGE, "assistant", "Deploying now."),
            ]
        )
        # A seed is an address: the store locates it whether or not it passes the
        # filter and never returns it among the neighbours, and expand renders it
        # itself. So `kinds` selects the SURROUNDINGS, and the turn you named is
        # shown either way — uniformly, not depending on what it happens to be.
        deploy = await core.search("and now deploy it", limit=1, seen=set())
        only_injected = render_expand_result(
            await core.expand(
                deploy.hits[0].memory_id, before=0, after=2, kinds=["injected"]
            )
        )
        checker.check(
            "system-reminder" in only_injected and "and now deploy it" in only_injected,
            "kinds selects the surroundings; the seed you named is shown anyway",
        )
        checker.check(
            only_injected.count(_ANCHOR) == 1,
            "exactly one anchor marker says which segment was expanded from",
        )
        nothing = await core.expand(
            deploy.hits[0].memory_id, before=0, after=0, kinds=["reasoning"]
        )
        checker.check(
            nothing.found and "and now deploy it" in render_expand_result(nothing),
            "asking for no surroundings still returns the memory you named",
        )
        default_kinds = render_expand_result(
            await core.expand(deploy.hits[0].memory_id, before=0, after=1)
        )
        everything = render_expand_result(
            await core.expand(
                deploy.hits[0].memory_id,
                before=0,
                after=1,
                kinds=[],
                blocklist=True,
            )
        )
        checker.check(
            "system-reminder" not in default_kinds and "system-reminder" in everything,
            "injected text is blocked by default and reachable on request",
        )
    finally:
        await core.aclose()


def _daemon_suite(checker: _Checker) -> None:
    from claude_memory.daemon_client import DaemonUnavailableError, call
    from claude_memory.wire import (
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
            {
                "op": "ingest",
                "transcript_path": str(transcript),
                "session_id": "sess-1111-2222",
            },
            wait_for_start=90.0,
        )
        print(f"  (daemon cold start + ingest: {time.monotonic() - start:.2f}s)")
        checker.check(
            bool(ingest.get("ok")) and ingest.get("ingested", 0) >= 2,
            "ingest via daemon",
        )

        warm = time.monotonic()
        response = call(
            {
                "op": "search",
                "cue": "how do I deploy to production",
                "session_id": "sess-1111-2222",
            }
        )
        print(f"  (warm search round trip: {time.monotonic() - warm:.3f}s)")
        checker.check(bool(response.get("ok")), "search via daemon (warm, no re-spawn)")
        result = search_result_from_dict(response["result"])
        joined = "\n".join(hit.text for hit in result.hits)
        checker.check("deploy script" in joined, "deploy memory retrieved")
        checker.check("Read " not in joined, "tool call is not a direct search hit")

        seed = result.hits[0].memory_id
        exp = call(
            {
                "op": "expand",
                "id": seed,
                "before": 3,
                "after": 3,
                "session_id": "sess-1111-2222",
            }
        )
        expanded = expand_result_from_dict(exp["result"])
        checker.check(
            expanded.found and "Read " in render_expand_result(expanded),
            "expansion reaches the non-embedded tool call",
        )

        again = call(
            {
                "op": "search",
                "cue": "how do I deploy to production",
                "session_id": "sess-1111-2222",
            }
        )
        again_result = search_result_from_dict(again["result"])
        checker.check(
            again_result.saturated and again_result.new_count == 0,
            "shared per-session novelty: repeat search reports 0 new",
        )

        # Reflective recall: a fresh session re-evokes memory from the model's
        # own last reply (the transcript's final assistant message).
        refl = call(
            {
                "op": "reflect",
                "transcript_path": str(transcript),
                "session_id": "sess-3333-4444",
            }
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
        dm_verdict = (
            demote_result_from_dict(dm["result"]).verdict if dm.get("ok") else ""
        )
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
    # Pin eviction OFF: the suite asserts exact ingest/novelty counts, which the
    # lossy eviction path (now on by default) would perturb — and hash-embedder
    # near-dup behavior isn't representative anyway.
    os.environ["CLAUDE_MEMORY_EVICTION_THRESHOLD"] = ""
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
