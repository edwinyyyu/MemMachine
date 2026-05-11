"""v33 prompt + pre-escaped passage variant.

Hypothesis: gpt-5-nano's backslash-doubling bug happens because the
model is making JSON-escape decisions on the fly (to output a literal
backslash in a JSON string, it needs to emit \\\\). If we instead
hand the model an already-JSON-escaped passage and tell nothing about
escaping, the model just *copies* the characters into its output. We
then JSON-decode the segments once on receipt to recover the original.

Pre-escape uses `json.dumps(s, ensure_ascii=False)[1:-1]` so non-ASCII
content stays as raw Unicode while only the structural chars (\\, \\n,
\\t, \\r, ") get escaped.
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F_natural import WINDOW_CHARS, call
from probe_segmenter_F_natural_v33 import PROMPT_F_NATURAL_V33

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


def _pre_escape(text: str) -> str:
    """Return the JSON string form of `text` without the surrounding
    quotes (so it can be interpolated into the prompt template)."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _post_unescape(seg: str) -> str:
    """Apply one JSON-decode pass to recover the original characters
    the model 'copied' from the pre-escaped passage."""
    try:
        return json.loads(f'"{seg}"')
    except json.JSONDecodeError:
        return seg


async def segment(client, model, text, reasoning="low", window_chars=WINDOW_CHARS):
    if len(text) <= window_chars:
        encoded = _pre_escape(text)
        prompt = PROMPT_F_NATURAL_V33.format(passage=encoded)
        raw = await call(client, model, prompt, reasoning)
        return [_post_unescape(s) for s in raw]

    splitter = RecursiveCharacterTextSplitter(
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
    windows = splitter.split_text(text)

    async def go(w):
        encoded = _pre_escape(w)
        prompt = PROMPT_F_NATURAL_V33.format(passage=encoded)
        raw = await call(client, model, prompt, reasoning)
        return [_post_unescape(s) for s in raw]

    sub_results = await asyncio.gather(*(go(w) for w in windows))
    flat: list[str] = []
    for r in sub_results:
        flat.extend(r)
    return flat
