"""Deictic-resolved verbatim segmenter -- v6.

v5 had three persistent failures driven by ambiguous framings:
  - ADD-vs-REPLACE for "you" (D8 D21): "ADD vocative" interpreted as
    REPLACE.
  - Generic-vs-addressed (D2 D17): self-check test ignored; default
    leaned addressed.
  - Antecedent hallucination (D23): the Alice example let the model
    extrapolate names from group descriptions ("Audrey's dogs" ->
    "Scout, Toby, and Buddy").
  - Paraphrasing (D20): "no paraphrasing" rule violated.

v6 reframes each rule mechanically, eliminating model-bias entry
points:

  - VOCATIVE INSERTION (rule 1): "the original 'you' must still appear
    in your output unchanged" -- the pronoun is a literal token that
    cannot be removed.
  - DECIDE ADDRESSED-vs-GENERIC (rule 2): default GENERIC; flip to
    ADDRESSED only if BOTH (a) the sentence is a question to
    {addressee} (Are/Do/Did/Have/Can/Will you...) OR makes a claim
    about {addressee}'s specific recent action/state/possession, AND
    (b) substituting "a person" for "you" reads awkwardly.
  - WORD-FOR-WORD antecedent check (rules 3, 4): the substituted noun
    phrase must appear word-for-word in the surrounding turns block;
    if not present, KEEP the pronoun. Removes inference.
  - WORDS-IN-ORDER constraint (output format): only vocative
    insertions and pronoun-for-name substitutions allowed; word order
    preserved.

Output: plain-text completion (no JSON, no pydantic). No examples
beyond mechanical pronoun-level patterns.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    DecoupledRetrievalContext,
    Event,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)


PROMPT_DEICTIC_RESOLVED_V6 = """\
SPEAKER: {speaker}
ADDRESSEE: {addressee}

{speaker} sent the MESSAGE below to {addressee}. Rewrite the MESSAGE \
for a reader who has not seen the SURROUNDING TURNS. The MESSAGE \
words must appear in your output in their original order. The only \
edits you may make are vocative insertions (rule 1) and \
pronoun-for-name substitutions (rules 3, 4, 5).

Rule 1 -- VOCATIVE INSERTION for ADDRESSED "you".
When "you" / "your" / "yours" / "yourself" is ADDRESSED to \
{addressee} (per rule 2), insert ", {addressee}, " (with surrounding \
commas) immediately after the pronoun token. The pronoun token \
itself MUST still appear in your output, unchanged.
- "you"      becomes  "you, {addressee},"
- "your"     becomes  "your, {addressee}'s,"
- "yours"    becomes  "yours, {addressee}'s,"
- "yourself" becomes  "yourself, {addressee},"
HARD CONSTRAINT: the original "you" / "your" / "yours" / "yourself" \
token must appear in your output word-for-word. You are inserting \
content next to it, not replacing it.

Rule 2 -- DECIDE addressed vs generic for each "you".
Default: every "you" is GENERIC and stays unchanged. No vocative.
Override to ADDRESSED only if BOTH of the following hold:
(a) Syntactic test: the sentence is a question to {addressee} \
("Are/Do/Did/Have/Can/Will/Would/Could/Should you ..."), OR the \
sentence makes a claim about {addressee}'s SPECIFIC recent action, \
state, experience, or possession that the SURROUNDING TURNS just \
discussed.
(b) Replace test: substituting "a person" for "you" in the sentence \
makes the sentence read awkwardly or false.
If either (a) or (b) is uncertain, the "you" stays generic. No \
vocative.

Rule 3 -- THIRD-PERSON PRONOUN SUBSTITUTION.
For each "he" / "she" / "they" / "him" / "her" / "them" / "his" / \
"hers" / "theirs" in the MESSAGE:
(a) Look in the SURROUNDING TURNS block for the antecedent (the \
specific person or thing the pronoun refers to).
(b) The substituted noun phrase you write must appear WORD-FOR-WORD \
in the SURROUNDING TURNS block. If you cannot copy the exact phrase \
from the surrounding turns, do not substitute.
(c) If a word-for-word match is found, REPLACE the pronoun with that \
phrase.
(d) If no word-for-word match is found, KEEP the pronoun unchanged.
HARD CONSTRAINT: never output a name or noun phrase that does not \
appear word-for-word in the SURROUNDING TURNS block above.

Rule 4 -- DEMONSTRATIVE SUBSTITUTION.
For each "this" / "that" / "these" / "those" / "here" / "there" / \
bare "it" in the MESSAGE:
(a) Look in the SURROUNDING TURNS block for the referent noun phrase \
(the specific object, place, plan, or event the demonstrative refers \
to).
(b) The substituted noun phrase must appear WORD-FOR-WORD in the \
SURROUNDING TURNS block.
(c) If a word-for-word match is found:
    -- if the SURROUNDING TURNS describe {speaker} as the maker / \
owner / builder / restorer / planter / creator of the referent, \
REPLACE with "my [noun]";
    -- otherwise REPLACE with the noun phrase.
(d) If no word-for-word match is found, KEEP the demonstrative \
unchanged.

Rule 5 -- SPEAKER SELF-REFERENCE.
If a "he" / "she" / "they" / "him" / "her" / "them" / "his" / "her" / \
"their" / "himself" / "herself" / "themself" refers to {speaker} \
themselves (per the SURROUNDING TURNS), substitute first-person:
- "he" / "she" / "they"   becomes  "I"
- "him" / "her" / "them"  becomes  "me"
- "his" / "her" / "their" becomes  "my"
- "himself" / "herself" / "themself" becomes "myself"
HARD CONSTRAINT: never write "{speaker}" as a self-reference \
replacement. Never write "{speaker}: " as a prefix anywhere -- the \
display adds it automatically.

Rule 6 -- KEEP UNCHANGED.
- "I" / "my" / "me" / "mine" / "myself" referring to {speaker}.
- "we" / "us" / "our" (joint references).
- TEMPORAL references such as "yesterday", "last week", "tomorrow", \
"next Friday", "the weekend", "today", "tonight", "now", "just", \
"recently", "X years ago", "this morning". The event date is \
rendered alongside the MESSAGE; downstream tooling resolves these \
at answer time.
- Vocatives already present at the start of the MESSAGE \
("Hey {addressee}!", "{addressee}, ..."): they already identify the \
addressee.
- All other words, punctuation, capitalization, emoji, attached-media \
descriptions.

OUTPUT FORMAT: write ONLY the rewritten MESSAGE on a single line. \
The MESSAGE words must appear in their original order. The only \
permitted edits are (i) vocative insertions ", {addressee}, " next \
to addressed "you" pronouns (rule 1), and (ii) pronoun-for-name \
substitutions (rules 3, 4, 5). No JSON, no quotes, no preamble, no \
commentary. For interchangeable filler (a bare greeting, sign-off, \
thanks, acknowledgement, pleasantry) with no informational content, \
output an empty line.

SURROUNDING TURNS (context only):
{neighbors_block}MESSAGE FROM {speaker} TO {addressee}:
{passage}"""


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


def _format_neighbors(before: list, after: list) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (context only):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    if after:
        lines.append("LATER TURNS (context only):")
        for ev in after:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


def _derive_addressee(speaker: str, before: list, after: list) -> str:
    for ev in list(before) + list(after):
        if ev.producer and ev.producer != speaker:
            return ev.producer
    return "the other person"


class DeicticResolvedV6Segmenter(Segmenter):
    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DEICTIC_RESOLVED_V6,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._lm = language_model
        self._prompt = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
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
        addressee: str,
        neighbors_block: str,
    ) -> str:
        prompt = self._prompt.format(
            speaker=speaker,
            addressee=addressee,
            passage=chunk,
            neighbors_block=neighbors_block,
        )
        text, _ = await self._lm.generate_response(
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        return text.strip() if text else ""

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(
                producer=producer, before=before, after=after
            ):
                speaker = producer
                addressee = _derive_addressee(speaker, before, after)
                neighbors_block = _format_neighbors(before, after)
            case ProducerContext(producer=producer):
                speaker = producer
                addressee = "the other person"
                neighbors_block = ""
            case _:
                speaker = "the speaker"
                addressee = "the other person"
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
                        resolved = await self._resolve(
                            chunk_stripped, speaker, addressee, neighbors_block
                        )
                        if not resolved:
                            continue
                        date_aliases = _date_aliases(event.timestamp, resolved)
                        display_text = f"{speaker}: {resolved}"
                        embed_text = display_text
                        bm25_text = display_text
                        if date_aliases:
                            bm25_text = f"{bm25_text}\nDates: {date_aliases}"
                            embed_text = f"{embed_text}\nDates: {date_aliases}"
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
