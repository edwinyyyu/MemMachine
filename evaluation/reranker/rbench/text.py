"""Tokenizers. Reader and chunk sizes are counted in embeddinggemma tokens; reranker costs
in Qwen3 tokens (all Qwen3-Reranker sizes share one tokenizer). Both load from the local
Hugging Face cache."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

GEMMA = "google/embeddinggemma-300m"
QWEN = "Qwen/Qwen3-Reranker-0.6B"
QWEN_EMBED = "Qwen/Qwen3-Embedding-0.6B"  # the Qwen3-Embedding tokenizer (the same for 0.6B, 4B and 8B)


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


def count(texts: list[str], repo: str = GEMMA) -> np.ndarray:
    """Tokens per text, without special tokens."""
    return np.asarray([len(e.ids) for e in tokenizer(repo).encode_batch(list(texts), add_special_tokens=False)], dtype=np.int64)


def token_starts(texts: list[str], repo: str) -> list[np.ndarray]:
    """Character offset where each token starts, per text."""
    out = []
    for start in range(0, len(texts), 256):
        for e in tokenizer(repo).encode_batch(list(texts[start : start + 256]), add_special_tokens=False):
            out.append(np.asarray([s for s, _ in e.offsets], dtype=np.int64))
    return out
