"""Translate v22 third-person segments into first-person segment text.

Goal: isolate the answerer/date-handling benefit of first-person segments
WHILE KEEPING THE EMBED IDENTICAL to v22's dual-text. Reads a v22 cached
segments JSON, runs an LLM 3p→1p translator on each segment's block.text,
and writes a NEW cache JSON where:

  - ``block_blob`` carries the new 1p segment text (what the answerer sees)
  - ``context_blob`` is preserved verbatim from the input (the embed text
    stays exactly as v22 produced it -- dual-text 3p_rewrite + 1p raw)

When re-ingested via CachedSegmenter, the segment store rows hold 1p text
and the vector store re-embeds the original dual-text. Retrieval ranking
is identical to v22; only QA-time presentation changes.

The translator's rules:
  - Speaker references in 3p resolve to first-person -- "Alice" → "I",
    "her" → "my", "she" → "I", "Alice's X" → "my X".
  - Other named entities stay in 3p (the addressee, third-party
    participants, places, organizations).
  - Apply v22-dates date-handling rules to the 1p output:
      * Drop the message date from the text (framework prefix carries it).
      * Event-mentioned dates DIFFERENT from the message date are emitted
        inline as ``on YYYY-MM-DD`` in natural prose.
      * Single canonical inline format; no parentheticals, no ``as of``.
  - Preserve every concrete particular verbatim. No paraphrasing of
    specifics, no dropping of details, no hallucination.
  - One statement per input segment (don't split or merge).

Run:
    uv run python probe_segment_3p_to_1p.py \\
        --input segments-cache-v22-nb8b-g3.json \\
        --output segments-cache-v22-nb8b-fp-g3.json \\
        --model gpt-5-nano --reasoning low --concurrency 30
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys

import openai
from pydantic import BaseModel

from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)


PROMPT_TRANSLATE_3P_TO_1P = """\
You translate a third-person memory statement about a speaker into the \
speaker's own first-person voice. The translation must preserve every \
concrete particular verbatim and apply explicit date-handling rules.

SPEAKER: {speaker}
MESSAGE_DATE: {message_date}

THIRD-PERSON STATEMENT:
{statement}

TRANSLATION RULES:

1. The SPEAKER appears in the third-person statement under one or more \
forms: the literal name (``{speaker}``), pronouns (he/she/they/him/her/\
them/his/her/their), and possessives (``{speaker}'s``). Replace all \
references to the SPEAKER with first-person forms:
   - subject pronoun → ``I``
   - object pronoun → ``me``
   - possessive determiner / possessive ``{speaker}'s`` → ``my``
   - possessive pronoun → ``mine``
   - reflexive → ``myself``

   AGREEMENT: after substitution, adjust the verb form to agree with \
``I`` in number and person. Third-person singular verbs become \
first-person:
   - ``is`` → ``am``        (present tense)
   - ``was`` → ``was``      (past tense -- ``was`` is correct for ``I``)
   - ``has`` → ``have``     (present perfect / possessive auxiliary)
   - ``does`` → ``do``      (auxiliary)
   - ``goes/wants/likes/loves/says/has been`` and any other ``-s`` \
ending third-person form → drop the ``-s`` (``go/want/like/love/say/\
have been``).
   Leave verbs that don't change form unchanged (``went``, ``would``, \
``could``, ``had``, infinitives, participles).

2. NAMES other than the SPEAKER stay in third-person: addressees, \
third-party participants, places, brands, organizations, named objects.

3. DATE HANDLING:
   - The system automatically prepends the message timestamp to the \
statement at retrieval time. DO NOT include the message date \
``{message_date}`` in your output in any form -- not as a prefix, not \
as a suffix, not parenthesized, not bracketed.
   - Any date that remains in your output refers to an event MENTIONED \
in the statement that occurred on a DIFFERENT date than the message \
date.
   - When the third-person statement contains an explicit date or a \
resolved-relative date and that date EQUALS ``{message_date}``, DROP \
the date from the output.
   - When the date DIFFERS from ``{message_date}``, emit it inline as \
``on YYYY-MM-DD`` in natural prose -- never as a sentence prefix \
``On YYYY-MM-DD,``, never parenthesized ``(Date: ...)`` or ``(Event \
date: ...)``, never bracketed, never ``as of YYYY-MM-DD``.
   - If the third-person statement already uses a canonical or \
non-canonical date form, normalize to ``on YYYY-MM-DD``.

4. PRESERVE EVERY CONCRETE PARTICULAR verbatim -- names, dates other \
than the dropped message date, numbers, identifiers, quoted phrases, \
emotional states, attached-media descriptions, decisions, plans, \
preferences. Do not paraphrase, summarize, or drop content.

5. Preserve polarity and direction. "Didn't" stays negated; "used to" \
preserves no-longer.

6. Output ONE first-person statement. Do not split. Do not merge.

EXAMPLES (neutral names, neutral domains):

SPEAKER: Alice
MESSAGE_DATE: 2026-04-10
THIRD-PERSON: Alice adopted her two cockatiels on 2023-04-10, right \
before she moved to Portland.
FIRST-PERSON: I adopted my two cockatiels on 2023-04-10, right before \
I moved to Portland.

SPEAKER: Bob
MESSAGE_DATE: 2026-05-02
THIRD-PERSON: Bob considered Charlie's wedding on 2025-06-14 the best \
party he attended in 2025.
FIRST-PERSON: I considered Charlie's wedding on 2025-06-14 the best \
party I attended in 2025.

SPEAKER: Dana
MESSAGE_DATE: 2026-05-18
THIRD-PERSON: Dana promised to stop missing the Thursday mandolin \
practice and to attend every week going forward.
FIRST-PERSON: I promised to stop missing the Thursday mandolin \
practice and to attend every week going forward.

SPEAKER: Charlie
MESSAGE_DATE: 2026-05-18
THIRD-PERSON: On 2026-05-18, Charlie attached a photo of a notepad \
with a note and pen on it, representing his success.
FIRST-PERSON: I attached a photo of a notepad with a note and pen on \
it, representing my success.

SPEAKER: Alice
MESSAGE_DATE: 2026-05-18
THIRD-PERSON: Alice baked a chocolate-cardamom cake for Bob's birthday \
on 2024-09-12 and said it was her favorite recipe right now.
FIRST-PERSON: I baked a chocolate-cardamom cake for Bob's birthday on \
2024-09-12 and said it is my favorite recipe right now.

Output as JSON: {{ "first_person": "..." }}
"""


class _TranslationResponse(BaseModel):
    first_person: str


async def translate_segment(
    lm: OpenAIResponsesLanguageModel,
    statement: str,
    speaker: str,
    message_date: str,
    max_attempts: int = 3,
) -> str:
    prompt = PROMPT_TRANSLATE_3P_TO_1P.format(
        speaker=speaker,
        message_date=message_date,
        statement=statement,
    )
    resp = await lm.generate_parsed_response(
        output_format=_TranslationResponse,
        user_prompt=prompt,
        max_attempts=max_attempts,
    )
    if resp is None or not resp.first_person.strip():
        # Fall back to original on parse failure.
        return statement
    return resp.first_person.strip()


def speaker_and_date_from_record(record: dict) -> tuple[str, str]:
    """Pull (speaker, message_date) for a cached-segment record.

    v22 RewriteContext stores text_to_embed = "{rewrite}\\n{speaker}: {raw_chunk}".
    Speaker is parsed off the second-paragraph prefix before the first colon.
    """
    speaker = None
    try:
        ctx = json.loads(base64.b64decode(record["context_blob"]).decode("utf-8"))
        # Try direct producer field first (some contexts have it).
        speaker = ctx.get("producer")
        if not speaker and ctx.get("context_type") == "rewrite":
            embed = ctx.get("text_to_embed", "")
            # Format: "rewrite_text\nSpeakerName: raw_chunk"
            if "\n" in embed:
                second_line = embed.split("\n", 1)[1]
                if ":" in second_line:
                    speaker = second_line.split(":", 1)[0].strip()
    except Exception:
        pass
    if not speaker:
        speaker = "the speaker"

    # Date: prefer record["timestamp"] (ISO).
    ts = record.get("timestamp") or ""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(ts))
    message_date = m.group(1) if m else ts[:10]
    return speaker, message_date


def extract_segment_text_from_block_blob(blob_b64: str) -> str:
    blob = base64.b64decode(blob_b64).decode("utf-8")
    obj = json.loads(blob)
    if isinstance(obj, dict) and "text" in obj:
        return obj["text"]
    return blob


def encode_block_blob(new_text: str, original_blob_b64: str) -> str:
    """Re-encode block_blob with new text, preserving block_type."""
    blob = base64.b64decode(original_blob_b64).decode("utf-8")
    obj = json.loads(blob)
    if isinstance(obj, dict) and "text" in obj:
        obj["text"] = new_text
        new_blob = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        return base64.b64encode(new_blob).decode("ascii")
    # Fallback: assume plain string blob
    return base64.b64encode(new_text.encode("utf-8")).decode("ascii")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="gpt-5-nano")
    p.add_argument("--reasoning", default="low")
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="Translate only the first N (0=all)")
    args = p.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    print(f"== Translating {len(records)} segments from {args.input}")
    print(f"== model={args.model} reasoning={args.reasoning} concurrency={args.concurrency}")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lm = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=client, model=args.model, reasoning_effort=args.reasoning,
        )
    )

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    lock = asyncio.Lock()

    async def process(idx: int, r: dict) -> None:
        nonlocal done
        async with sem:
            speaker, message_date = speaker_and_date_from_record(r)
            old_text = extract_segment_text_from_block_blob(r["block_blob"])
            new_text = await translate_segment(lm, old_text, speaker, message_date)
            r["block_blob"] = encode_block_blob(new_text, r["block_blob"])
            async with lock:
                done += 1
                if done % 100 == 0 or done == len(records):
                    print(f"   translated {done}/{len(records)}", flush=True)

    await asyncio.gather(*[process(i, r) for i, r in enumerate(records)])

    with open(args.output, "w") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"== wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
