"""Deictic-resolved verbatim segmenter -- v8l.

V8K + date-prepended embed channel. Embed text changes from
    Audrey: <resolved>
    Dates: February 24, 2024; February 2024
to
    [Saturday, February 24, 2024] Audrey: "<resolved>"

bm25_text keeps the "Dates: <aliases>" append (BM25 still benefits
from the explicit alias tokens for date-anchored queries). block.text
(answerer display) is unchanged.

Tests whether the [<date>] <speaker>: "<text>" embed format raises
recall on first-person rewrites, the same way the user observed it
helps on raw text.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import override
from uuid import uuid4

from babel.dates import format_date, format_time, get_datetime_format
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from memmachine_server.episodic_memory.event_memory.data_types import (
    DecoupledRetrievalContext,
    Event,
    ProducerContext,
    Segment,
    SurroundingEvent,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)


_DATE_STYLE = "full"
_TIME_STYLE = "short"
_LOCALE = "en_US"


class DatedSurroundingEvent(SurroundingEvent):
    """SurroundingEvent + optional timestamp, used by the chat-log
    formatter."""

    timestamp: datetime | None = None


PROMPT_DEICTIC_RESOLVED_V8L = """\
You will propose edits to a chat MESSAGE for indexing in an \
information retrieval system. The edited version — not the original \
— is what gets indexed and later retrieved. The reader at retrieval \
time will have no access to the surrounding context.

An indexed message is useful at retrieval time only if it \
accurately preserves the meaning and nuances of what the speaker \
wrote. The edited version preserves the speaker's wording, except \
for edits that resolve references. Each such edit anchors a \
reference ("you", "it", "they", "this", "that", etc.) to its \
concrete antecedent from the surrounding context, making it \
standalone. Edit only when the resolution is obvious; otherwise the \
reference stays as the speaker wrote it.

WHEN TO EDIT — for each pronoun or demonstrative in the MESSAGE, \
first check whether it is eligible for editing:
- Is the pronoun or demonstrative part of an idiom or common \
saying that shouldn't be resolved literally? (e.g., "it's \
raining", "you know", "this and that") If yes, leave it unchanged.
- Would substituting the reference require copying a passage from \
the surrounding context rather than a single noun phrase? (e.g., \
"that said", "in light of this") If yes, leave it unchanged.
- For each entity referenced by multiple pronouns or \
demonstratives in the message, choose ONE occurrence — typically \
the first — as the resolution point. Leave the other occurrences \
of the same entity as the original pronouns or demonstratives.

If the reference is eligible, ask:
- Is the antecedent not just plausible, but obvious, from the \
surrounding context? If not, leave it unchanged.
- Would every careful reader pick the same antecedent? If not, \
leave it unchanged.
- Is there any alternative interpretation a careful reader might \
prefer? If yes, leave it unchanged.

WHAT EDITS TO MAKE — when a reference passes the checks above, \
apply the matching pattern:
- A "you" addressed to a specific person identifiable from the \
surrounding context gains a vocative with that person's name, at \
the start or end of the sentence. For example, if the addressee is \
Alice, "Do you like pizza?" becomes "Do you like pizza, Alice?", \
not "Do you, Alice, like pizza?". The original "you" stays.
- A third-person pronoun or demonstrative ("it", "they", "this", \
"that", "those", etc.) with one unambiguous antecedent in the \
surrounding context is replaced by the established noun phrase.

HOW TO EDIT — apply these principles to every edit you consider:
- Pay attention to subtle or discreet topic changes in the \
surrounding context. When the conversation shifts subject — a \
follow-up question reframes the topic, an aside introduces a new \
subject, a back-channel acknowledgement returns to an earlier one \
— a noun phrase tied to the prior topic may not be the antecedent \
of a reference in the new topic.
- Do not infer ownership unless the surrounding context makes it \
abundantly clear with words like "my", "your", or "their". \
Pronouns or demonstratives like "it", "they", "this", or "that" \
do not establish ownership, and neither does someone simply \
sharing a photo or describing something. When ownership IS \
established, attribute from the speaker's perspective: "my X" \
if the speaker owns the referent, "your X" if the addressee owns \
it, or "Name's X" for a third-party owner. When ownership is not \
established, use a neutral description ("the X", "a X") rather \
than an ownership form.

WHAT STAYS UNCHANGED — everything in the message outside the edits \
stays as written. The speaker and timestamp are part of the \
message's metadata, so references anchored to them remain as \
written:
- First-person voice ("I", "my", "we", "our") — anchored by the \
speaker.
- Temporal references ("yesterday", "last week", "today") — \
anchored by the timestamp.

OUTPUT FORMAT — return a JSON object with one field, "edits": a \
list of edit pairs to apply to the MESSAGE. Each pair has:
- "original": the exact substring from the MESSAGE to be replaced. \
Include enough surrounding text to uniquely identify the occurrence \
to edit (e.g., "Glad you found" rather than just "you" when "you" \
appears multiple times).
- "replacement": the substring that replaces it in the edited \
message.

Do not propose edits that replace first-person voice ("I", "my", \
"me", "we", "our", "us") with a name or third-person form. For \
example, if the speaker is Bob, the pair {{"original": "I baked a \
cake", "replacement": "Bob baked a cake"}} is WRONG — first-person \
voice stays as the speaker wrote it. Edits that contain a first-\
person word in the original must preserve that word verbatim in \
the replacement.

If no edits are needed, return an empty list. Do not include edits \
where "original" is not a literal substring of the MESSAGE.

SURROUNDING CONTEXT:
{neighbors_block}MESSAGE TO EDIT:
{target_line}"""


_EDITS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["original", "replacement"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}


_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def _date_aliases(event_date: datetime, text: str) -> str:
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for m in _ISO_RE.finditer(text):
        y, mo = int(m.group(1)), int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 0
        if 1 <= mo <= 12:
            dates.add((y, mo, d))
    parts: list[str] = []
    seen: set[str] = set()
    for y, mo, d in sorted(dates):
        name = _MONTHS[mo - 1]
        for alias in (f"{name} {y}", f"{name} {d}, {y}" if d else None):
            if alias and alias not in seen:
                seen.add(alias)
                parts.append(alias)
    return "; ".join(parts)


def _format_timestamp(ts: datetime | None) -> str:
    if ts is None:
        return ""
    date_string = format_date(ts, format=_DATE_STYLE, locale=_LOCALE)
    time_string = format_time(ts, format=_TIME_STYLE, locale=_LOCALE)
    if not time_string:
        return date_string
    if not date_string:
        return time_string
    template = str(get_datetime_format(_TIME_STYLE, locale=_LOCALE))
    return template.replace("{1}", date_string).replace("{0}", time_string)


def _format_event_line(producer: str, text: str, ts: datetime | None) -> str:
    formatted = _format_timestamp(ts)
    timestamp_prefix = f"[{formatted}] " if formatted else ""
    return f"{timestamp_prefix}{producer}: {text}"


def _format_neighbors(before: list, after: list) -> str:
    lines: list[str] = []
    for ev in list(before) + list(after):
        ts = getattr(ev, "timestamp", None)
        lines.append(_format_event_line(ev.producer, ev.text, ts))
    return "\n".join(lines) + ("\n" if lines else "")


def _apply_edits(original_text: str, edits: list[dict]) -> str:
    result = original_text
    for edit in edits:
        orig = edit.get("original", "")
        repl = edit.get("replacement", "")
        if not orig or orig == repl:
            continue
        if orig not in result:
            continue
        result = result.replace(orig, repl, 1)
    return result


class DeicticResolvedV8LSegmenter(Segmenter):
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str = "gpt-5.4-nano",
        reasoning_effort: str = "low",
        prompt_template: str = PROMPT_DEICTIC_RESOLVED_V8L,
        chunk_size: int = 1500,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._prompt = prompt_template
        self._chunk_size = chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator="end",
        )

    async def _resolve(
        self,
        chunk: str,
        speaker: str,
        neighbors_block: str,
        target_line: str,
    ) -> str:
        prompt = self._prompt.format(
            neighbors_block=neighbors_block,
            target_line=target_line,
        )
        resp = await self._client.chat.completions.create(
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "deictic_edits",
                    "strict": True,
                    "schema": _EDITS_JSON_SCHEMA,
                },
            },
        )
        raw = resp.choices[0].message.content or ""
        try:
            payload = json.loads(raw)
            edits = payload.get("edits", [])
        except (json.JSONDecodeError, TypeError):
            edits = []
        return _apply_edits(chunk, edits)

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(
                producer=producer, before=before, after=after
            ):
                speaker = producer
                neighbors_block = _format_neighbors(before, after)
            case ProducerContext(producer=producer):
                speaker = producer
                neighbors_block = ""
            case _:
                speaker = "the speaker"
                neighbors_block = ""

        segments: list[Segment] = []
        for block_index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    chunks = (
                        self._splitter.split_text(text)
                        if len(text) > self._chunk_size
                        else [text]
                    )
                    offset = 0
                    for chunk in chunks:
                        chunk_stripped = chunk.strip()
                        if not chunk_stripped:
                            continue
                        target_line = _format_event_line(
                            speaker, chunk_stripped, event.timestamp
                        )
                        resolved = await self._resolve(
                            chunk_stripped,
                            speaker,
                            neighbors_block,
                            target_line,
                        )
                        if not resolved:
                            continue
                        date_aliases = _date_aliases(event.timestamp, resolved)
                        display_text = f"{speaker}: {resolved}"
                        # V8L change: embed channel uses
                        #   [<full date>] <speaker>: "<resolved>"
                        # bm25 keeps the appended alias list for explicit
                        # date tokens; display is unchanged.
                        full_date = format_date(
                            event.timestamp, format="full", locale="en_US"
                        )
                        embed_text = (
                            f'[{full_date}] {speaker}: "{resolved}"'
                        )
                        bm25_text = display_text
                        if date_aliases:
                            bm25_text = f"{bm25_text}\nDates: {date_aliases}"
                        segments.append(
                            Segment(
                                uuid=uuid4(),
                                event_uuid=event.uuid,
                                index=block_index,
                                offset=offset,
                                timestamp=event.timestamp,
                                block=TextBlock(text=display_text),
                                context=DecoupledRetrievalContext(
                                    producer=speaker,
                                    text_to_embed=embed_text,
                                    text_to_score_bm25=bm25_text,
                                ),
                                properties=event.properties,
                            )
                        )
                        offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block: {type(block).__name__}"
                    )
        return segments
