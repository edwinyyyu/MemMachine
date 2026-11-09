"""EventMemory's segmentation, embedding text, rendering and context assembly, copied.

Copied from MemMachine at bbcc8c3da (2026-11-06, episodic_memory/event_memory/) so the
benchmark measures exactly what the service does. tests/test_eventmemory.py checks these
copies against MemMachine's own functions through fixtures that
scripts/eventmemory_fixtures.py generates with MemMachine installed.

  segment_text            TextSegmenter: LangChain's RecursiveCharacterTextSplitter with
                          EventMemory's separators, no overlap, separators kept at the end
                          (the production splitter; PR #1855's replacement is not merged)
  derivative_text         WholeTextDeriver: what is embedded for a segment,
                          '[<date>] <producer>: "<JSON text>"', date only
  string_from_segment_context   one context rendered: a '[<date and time>] <producer>: '
                          header per run of one event block, then the run's text as JSON
  string_from_segment_contexts  several contexts: deduplicated, joined into connected
                          components, ordered chronologically, components separated by a
                          blank line (what the LLM receives)
  build_query_result_context    which segments a segment limit admits: contexts in score
                          order, whole while they fit, then the partly fitting one's
                          segments nearest its seed (forward first)
  string_from_query_result      the string for a segment limit: every context of the query
                          result restricted to the admitted segments, then joined
Segments here are light records; `key` orders them as EventMemory does:
(timestamp, event, index, offset).
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache

from babel.dates import format_date, format_time, get_datetime_format

_STYLE_LEVELS = ("short", "medium", "long", "full")
SEPARATORS = ["\n\n", "],\n", "},\n", "),\n", "]\n", "}\n", ")\n", ",\n", "？\n", "?\n", "！\n", "!\n", "。\n",
              ".\n", "？", "? ", "！", "! ", "。", ". ", "; ", ": ", "—", "--", "，", "、", ", ",
              "​", " ", ""]
EMBED_FORMAT = ("full", None)  # (date style, time style) of derivative text: WholeTextDeriver's default
RENDER_FORMAT = ("full", "short")  # of rendered contexts: query()'s and string_from_segment_context's default
LOCALE = "en_US"


@dataclass(frozen=True)
class Segment:
    id: int  # position in its partition's segment table
    event: int  # event (turn) id; unique and ordered within a partition
    index: int  # block index within the event
    offset: int  # segment number within the block
    timestamp: datetime.datetime
    producer: str | None  # ProducerContext's producer; None for NullContext
    text: str

    @property
    def key(self) -> tuple:
        return (self.timestamp, self.event, self.index, self.offset)


@cache
def _splitter(max_chunk_length: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(chunk_size=max_chunk_length, chunk_overlap=0, separators=SEPARATORS,
                                          keep_separator="end")


def segment_text(text: str, max_chunk_length: int = 500) -> list[str]:
    """TextSegmenter's chunks of one text block."""
    return _splitter(max_chunk_length).split_text(text)


def format_timestamp(timestamp: datetime.datetime, date_style: str | None, time_style: str | None,
                     locale: str = LOCALE) -> str:
    if date_style is None and time_style is None:
        return ""
    date_string = format_date(timestamp, format=date_style, locale=locale) if date_style else ""
    time_string = format_time(timestamp, format=time_style, locale=locale) if time_style else ""
    if not time_string:
        return date_string
    if not date_string:
        return time_string
    connector = _STYLE_LEVELS[max(_STYLE_LEVELS.index(date_style), _STYLE_LEVELS.index(time_style))]
    template = str(get_datetime_format(connector, locale=locale))
    return template.replace("{1}", date_string).replace("{0}", time_string)


def derivative_text(segment: Segment, fmt: tuple[str | None, str | None] = EMBED_FORMAT) -> str:
    body = json.dumps(segment.text, ensure_ascii=False)
    if segment.producer is not None:
        body = f"{segment.producer}: {body}"
    stamp = format_timestamp(segment.timestamp, *fmt)
    return f"[{stamp}] {body}" if stamp else body


def _header(segment: Segment, fmt: tuple[str | None, str | None]) -> str:
    stamp = format_timestamp(segment.timestamp, *fmt)
    prefix = f"[{stamp}] " if stamp else ""
    return f"{prefix}{segment.producer}: " if segment.producer is not None else prefix


def string_from_segment_context(segments: Iterable[Segment], fmt: tuple[str | None, str | None] = RENDER_FORMAT) -> str:
    out, last, text, first = "", None, "", True
    for segment in segments:
        continuation = last is not None and segment.event == last.event and segment.index == last.index
        if not continuation:
            if not first:
                out += json.dumps(text, ensure_ascii=False) + "\n"
            first, text = False, ""
            out += _header(segment, fmt)
        text += segment.text
        last = segment
    if not first:
        out += json.dumps(text, ensure_ascii=False) + "\n"
    return out.strip()


def components(contexts: Iterable[Sequence[Segment]]) -> list[list[Segment]]:
    """Deduplicated segments grouped into connected components (contexts sharing a segment
    are connected), each sorted, the components in chronological order."""
    by_id: dict[int, Segment] = {}
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        parent.setdefault(i, i)
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for context in contexts:
        first_root = None
        for segment in context:
            by_id.setdefault(segment.id, segment)
            if first_root is None:
                first_root = find(segment.id)
            else:
                parent[find(segment.id)] = first_root
    groups: dict[int, list[Segment]] = {}
    for i, segment in by_id.items():
        groups.setdefault(find(i), []).append(segment)
    out = [sorted(group, key=lambda s: s.key) for group in groups.values()]
    out.sort(key=lambda group: group[0].key)
    return out


def string_from_segment_contexts(contexts: Iterable[Sequence[Segment]],
                                 fmt: tuple[str | None, str | None] = RENDER_FORMAT) -> str:
    return "\n\n".join(string_from_segment_context(c, fmt) for c in components(contexts))


def seed_proximity(position: int, seed_position: int) -> float:
    """Lower is closer; forward context is preferred over backward."""
    offset = position - seed_position
    return (offset - 0.5) / 2 if offset >= 0 else -offset


def admission_order(contexts: Sequence[Sequence[Segment]], seeds: Sequence[int]) -> list[Segment]:
    """Segments in the order build_query_result_context admits them as its limit grows by one:
    contexts in the given (score) order, each one's new segments nearest its seed first.
    The admitted set at limit L is the first L of this list (a context that fits whole is
    admitted whole either way, so the within-context order only matters for the last one)."""
    by_id = {s.id: s for context in contexts for s in context}
    return [by_id[i] for i in admission_order_ids([[s.id for s in context] for context in contexts], seeds)]


def build_query_result_context(contexts: Sequence[Sequence[Segment]], seeds: Sequence[int],
                               max_num_segments: int) -> list[Segment]:
    """MemMachine's build_query_result_context, line for line (its fit check counts the whole
    context, already admitted segments included). The simulator uses admission_order instead;
    tests check that the first L of admission_order is always this set."""
    unified: dict[int, Segment] = {}
    for context, seed in zip(contexts, seeds, strict=True):
        if len(unified) >= max_num_segments:
            break
        if len(unified) + len(context) <= max_num_segments:
            unified.update((s.id, s) for s in context)
        else:
            seed_position = next(i for i, s in enumerate(context) if s.id == seed)
            for i in sorted(range(len(context)), key=lambda i: seed_proximity(i, seed_position)):
                if len(unified) >= max_num_segments:
                    break
                unified.setdefault(context[i].id, context[i])
    return sorted(unified.values(), key=lambda s: s.key)


def admission(contexts: Sequence[Sequence[int]], seeds: Sequence[int]) -> list[tuple[int, int]]:
    """admission_order on segment ids (contexts as chronological id lists), each admitted
    segment with the index of the context that admitted it."""
    out, seen = [], set()
    for k, (context, seed) in enumerate(zip(contexts, seeds, strict=True)):
        seed_position = context.index(seed)
        for i in sorted(range(len(context)), key=lambda i: seed_proximity(i, seed_position)):
            if context[i] not in seen:
                seen.add(context[i])
                out.append((context[i], k))
    return out


def admission_order_ids(contexts: Sequence[Sequence[int]], seeds: Sequence[int]) -> list[int]:
    """admission_order on segment ids (contexts as chronological id lists)."""
    return [s for s, _ in admission(contexts, seeds)]


def string_from_query_result(contexts: Sequence[Sequence[Segment]], seeds: Sequence[int], max_num_segments: int | None,
                             fmt: tuple[str | None, str | None] = RENDER_FORMAT) -> str:
    """MemMachine's string_from_query_result: every context of the result restricted to the
    segments the limit admits (build_query_result_context), then rendered and joined."""
    if max_num_segments is not None:
        included = {s.id for s in build_query_result_context(contexts, seeds, max_num_segments)}
        contexts = [[s for s in c if s.id in included] for c in contexts]
    return string_from_segment_contexts(contexts, fmt)
