"""Tokenizers. Everything is counted with Qwen3's tokenizer (the user, 2026-11-10): reader
tokens (the rendered context a reader LLM gets: budgets, token tables, fill depths, chart
axes), segment sizes, reranker pairs (all Qwen3-Reranker sizes share it) and Qwen3-Embedding
inputs (the same vocabulary: identical counts, checked). It is public, so nothing needs a
Hugging Face login. embeddinggemma's tokenizer (gated) is loaded only for a gemma embedder's
own input limit and billing (the paid first stage since 2026-11-11, so a paid run needs the
login). Tokenizers load from the local Hugging Face cache, else download.

as_message renders a text as EventMemory renders a turn for embedding, less the timestamp."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

GEMMA = "google/embeddinggemma-300m"
QWEN = "Qwen/Qwen3-Reranker-0.6B"
QWEN_EMBED = "Qwen/Qwen3-Embedding-0.6B"  # the Qwen3-Embedding tokenizer (the same for 0.6B, 4B and 8B)
READER = QWEN  # reader tokens; Qwen3-Reranker and Qwen3-Embedding count identically (151,669 tokens)
READER_NAME = "Qwen3 tokenizer"  # how reports name the unit


@cache
def tokenizer(repo: str) -> Tokenizer:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, "tokenizer.json", local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet
        path = hf_hub_download(repo, "tokenizer.json")
    t = Tokenizer.from_file(str(Path(path)))
    t.no_truncation()
    t.no_padding()
    return t


def as_message(producer: str, text: str) -> str:
    """`<producer>: "<text, JSON-escaped>"`: EventMemory's WholeTextDeriver rendering of a
    message (the producer outside the quotes, ensure_ascii off), without the timestamp."""
    return f"{producer}: {json.dumps(text, ensure_ascii=False)}"


def count(texts: list[str], repo: str = READER) -> np.ndarray:
    """Tokens per text, without special tokens."""
    return np.asarray([len(e.ids) for e in tokenizer(repo).encode_batch(list(texts), add_special_tokens=False)], dtype=np.int64)


def token_starts(texts: list[str], repo: str) -> list[np.ndarray]:
    """Character offset where each token starts, per text."""
    out = []
    for start in range(0, len(texts), 256):
        for e in tokenizer(repo).encode_batch(list(texts[start : start + 256]), add_special_tokens=False):
            out.append(np.asarray([s for s, _ in e.offsets], dtype=np.int64))
    return out
