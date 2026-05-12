r"""LLM-driven text segmenter.

The segmenter shows the raw block text to a `LanguageModel` and asks it to
return a list of standalone memory-worthy quotes, contiguous and verbatim,
in source order. The prompt is the validated v33 of `Mode F-natural` (see
`evaluation/event_memory/longmemeval/llm_pipeline_probe/probe_segmenter_F_natural_v33.py`).
v33 promotes the multi-line non-prose rule into its own sentence in
rule 2 with an explicit FAILURE marker on the drop case. Combined with
v32's whitespace-preservation clause in rule 1, the segmenter is
robust to structural content across all three models. On gpt-5.4-nano
@ low and gpt-5-mini, all five structural cases (code, tables, ASCII
art, arrow diagrams, morse) pass 30/30. On gpt-5-nano @ low, table /
code / arrow / morse are all 100%; pure ASCII-art with backslashes
remains ~33% due to a gpt-5-nano JSON-escaping bug (the model emits
"\\" where it should emit "\") which is a model-specific limit not
solvable via prompting. The stitching code catches this case
(see `_stitch_segments_to_source`) and recovers source-sliced text.
Feedback bench unchanged: 13/14 keep+drop on gpt-5.4-nano @ low and
gpt-5-mini @ low.

v24's unified principle: KEEP what is specific to this passage; DROP
what is interchangeable across similar passages. "Specific" means
content that differentiates this passage from others -- names, places,
dates, decisions, opinions, distinctive phrasing. "Interchangeable"
means pure framing that could open or close any passage regardless of
subject -- bare greetings and sign-offs. Short responses meaningful
only with the prior message ("yes", "no", "ok", "got it", "sounds
good", "acknowledged") are KEPT -- they carry the answer or reaction
the reconstructed memory needs to show.

For inputs longer than `window_chars`, the segmenter first runs a
deterministic `RecursiveCharacterTextSplitter` to cut the passage into
windows, then segments each window independently and concatenates the
results. This is the "whole-book" safety hatch; for typical conversational
turns it never triggers.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)

# v33 prompt -- promotes the multi-line non-prose rule into its own
# sentence with an explicit FAILURE marker. See
# feedback_segment_vs_derivative_roles.md for the iteration history.
# Frozen here so the production class is standalone.
PROMPT_F_NATURAL_V33 = """\
Compress this passage into the parts a human would still want during \
memory reconstruction, broken into a list of standalone memory \
segments. These segments are stored verbatim and shown back when the \
memory is retrieved.

Rules:
1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Never paraphrase, swap synonyms, or change wording -- \
"fabulous" stays "fabulous"; preserve whitespace, newlines, and \
special characters within a segment exactly. The only edits allowed \
are starting and ending the quote at sentence or clause boundaries.
2. KEEP what is specific to this passage; DROP what is interchangeable \
across similar passages. Specific content differentiates this passage \
from any other -- names, places, dates, numbers, identifiers, \
decisions, plans, opinions, preferences, relationships, emotional \
states tied to events, constraints, distinctive phrasing. Multi-line \
non-prose blocks (code, tables, ASCII art, diagrams, encoded text) \
are also specific -- keep each as one segment; dropping such a block \
as decoration is a FAILURE. Interchangeable content has none of \
these specifics -- it is pure framing that could open or close any \
passage regardless of subject (e.g., bare greetings, sign-offs).
3. SHORT RESPONSES that are only meaningful with the prior message \
("yes", "no", "ok", "got it", "sounds good", "acknowledged") are \
KEPT -- they carry the answer or reaction the reconstructed memory \
needs to show.
4. An utterance that begins with a greeting or softener but carries \
substantive content is KEPT -- drop only the leading greeting, not \
the content.
5. PRESERVE original order -- segments appear in the same order as \
their source quotes.
6. SEGMENT NATURALLY -- break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment \
per topic. Do not artificially split a coherent unit, and do not \
artificially merge unrelated ones.
7. STANDALONE -- each segment reads on its own. If a quote depends on \
a referent introduced earlier ("the trip" for "the Tokyo trip in \
May"), widen the quote to start where the referent is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


class _SegmenterResponse(BaseModel):
    """Structured response from the segmenter language model."""

    segments: list[str]


class LLMTextSegmenter(Segmenter):
    """Segments TextBlock events via a LanguageModel.

    Args:
        language_model: The LanguageModel used to produce segments.
            Configure the model and any reasoning effort at construction
            of the LanguageModel itself; this segmenter calls
            `generate_parsed_response(output_format=_SegmenterResponse, ...)`.
        prompt_template: A `.format(passage=...)` template producing the
            full prompt. Defaults to the validated v33 prompt.
        window_chars: For passages longer than this, the segmenter
            pre-chunks deterministically before calling the LLM.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_F_NATURAL_V33,
        window_chars: int = 8000,
        max_attempts: int = 3,
    ) -> None:
        """Initialize the segmenter; see class docstring for arguments."""
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._window_chars = window_chars
        self._max_attempts = max_attempts
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=window_chars,
            chunk_overlap=0,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                "",
            ],
            keep_separator="end",
        )

    async def _segment_text(self, text: str) -> list[str]:
        if len(text) <= self._window_chars:
            raw = await self._call_llm(text)
            return self._stitch_segments_to_source(raw, text)

        windows = self._text_splitter.split_text(text)
        all_segments: list[str] = []
        for window in windows:
            raw = await self._call_llm(window)
            all_segments.extend(self._stitch_segments_to_source(raw, window))
        return all_segments

    async def _call_llm(self, passage: str) -> list[str]:
        prompt = self._prompt_template.format(passage=passage)
        response = await self._language_model.generate_parsed_response(
            output_format=_SegmenterResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return list(response.segments)

    @staticmethod
    def _stitch_segments_to_source(segments: list[str], source: str) -> list[str]:
        r"""Anchor LLM segments to source so "".join(result) reconstructs.

        Reconstructs the source's kept content with dropped content
        silently elided.

        For each LLM-returned segment, locate it in `source` via a
        left-to-right cursor advanced past previously located
        segments, then extend the span both directions:

          Forward (trailing whitespace):
            1. eat horizontal whitespace (spaces, tabs, \r);
            2. eat consecutive newlines;
            3. stop at the next non-newline character.

          Backward (leading indent):
            1. walk backward through horizontal whitespace;
            2. claim the run only if it is bounded by a newline (or
               by the start of source); otherwise the run is part of
               preceding dropped content and is not claimed.

        The asymmetry is deliberate. Forward unconditionally takes
        trailing horizontal whitespace, so two same-line clauses like
        "Hi." and "World." stitch back as "Hi. World.". Backward only
        takes leading horizontal whitespace that is the segment's own
        line indent (separated from the previous line by a newline);
        a leading space that abuts dropped content on the same line
        is left with the dropped content. Orphan whitespace runs
        sitting between newline groups inside a gap of dropped
        content (e.g., a "blank" line that was actually just tabs)
        are dropped along with the gap.

        The emitted text is sliced from `source`, not from the LLM
        output, so the source's exact whitespace, newlines, and
        special characters (ASCII art, code indentation, etc.)
        survive verbatim even if the LLM lightly normalized its
        quoting.

        Segments that cannot be located (LLM paraphrase) are emitted
        as-is with a single trailing space padded on if they don't
        already end in whitespace, so a following segment doesn't
        glue onto them.
        """
        stitched: list[str] = []
        cursor = 0
        for seg in segments:
            if not seg:
                continue
            anchored = LLMTextSegmenter._anchor_segment(seg, source, cursor)
            if anchored is None:
                # Paraphrase fallback: pad with a single space if the
                # segment doesn't already end in whitespace, so a
                # following segment doesn't glue onto it.
                pad = "" if seg[-1].isspace() else " "
                stitched.append(seg + pad)
                continue
            idx, matched = anchored
            start, end = LLMTextSegmenter._expand_segment_span(
                idx, idx + len(matched), source
            )
            stitched.append(source[start:end])
            cursor = end
        return stitched

    @staticmethod
    def _anchor_segment(seg: str, source: str, cursor: int) -> tuple[int, str] | None:
        """Locate `seg` in `source` starting at `cursor`.

        If direct match fails, retry once with JSON-escape sequences
        un-escaped (some models, notably gpt-5-nano, re-apply JSON
        escaping inside string content). Returns ``(idx,
        matched_text)`` or ``None`` if neither match succeeds.
        """
        idx = source.find(seg, cursor)
        if idx >= 0:
            return idx, seg
        unescaped = (
            seg.replace("\\\\", "\\")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
        )
        if unescaped == seg:
            return None
        idx = source.find(unescaped, cursor)
        if idx < 0:
            return None
        return idx, unescaped

    @staticmethod
    def _expand_segment_span(idx: int, end: int, source: str) -> tuple[int, int]:
        """Extend an anchored span both directions per the stitching rules.

        Forward eats trailing horizontal ws then newlines; backward
        eats leading horizontal ws only if newline-bounded. Returns
        ``(start, end)`` indices into source.
        """
        # Forward: phase 1 horizontal ws, phase 2 newlines.
        while end < len(source) and source[end] in (" ", "\t", "\r"):
            end += 1
        while end < len(source) and source[end] == "\n":
            end += 1
        # Backward: walk through horizontal ws; claim only if
        # bounded by a newline (or start-of-source).
        scan = idx - 1
        while scan >= 0 and source[scan] in (" ", "\t", "\r"):
            scan -= 1
        start = scan + 1 if scan < 0 or source[scan] == "\n" else idx
        return start, end

    @override
    async def segment(self, event: Event) -> list[Segment]:
        segments: list[Segment] = []
        for index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    chunks = await self._segment_text(text)
                    segments.extend(
                        Segment(
                            uuid=uuid4(),
                            event_uuid=event.uuid,
                            index=index,
                            offset=offset,
                            timestamp=event.timestamp,
                            block=TextBlock(text=chunk),
                            context=event.context,
                            properties=event.properties,
                        )
                        for offset, chunk in enumerate(chunks)
                    )
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
