"""Run llama.cpp's convert_hf_to_gguf.py on a snapshot that has tokenizer.json but no tokenizer.model.

Gemma3Model.set_vocab builds its SentencePiece ("llama") vocab from tokenizer.model, which
sentence-transformers does not download and Google's gated repository is needed to fetch.
This rebuilds the same vocab from tokenizer.json, following llama.cpp's
ModelBase._create_vocab_sentencepiece (MIT License, Copyright (c) 2023-2026 The ggml
authors) step for step:

- piece text and ids come from the BPE vocab, and byte pieces (<0xXX>) are typed BYTE;
- scores come from the merge list, which the Hugging Face conversion sorted by the merged
  piece's SentencePiece score, highest first, so the order in which a piece first appears
  as a merge result is its score rank (llama.cpp's SPM tokenizer only compares the scores
  of merge results);
- added tokens get the same CONTROL / USER_DEFINED overrides from added_tokens.json and
  tokenizer_config.json as the original routine applies.

Checked on 2026-10-19 against ggml-org's official embeddinggemma GGUF: all tokenizer and
model metadata identical, and the same score order for all 236,249 mergeable pieces.
The converter's own tokenizer.json fallbacks do not work for Gemma: one fails outright,
and _set_vocab_llama_hf sets every score to -1000.

Usage: convert_local.py <llama.cpp source dir> <convert_hf_to_gguf.py args...>
"""

import importlib
import json
import re
import runpy
import sys
from typing import Any

BYTE_PIECE = re.compile(r"<0x[0-9A-F]{2}>")


def vocab_from_tokenizer_json(
    model: Any, token_types: Any
) -> tuple[list[bytes], list[float], list[int]]:
    """Return the SentencePiece-style tokens, scores and types for ``model``'s snapshot."""
    with (model.dir_model / "tokenizer.json").open(encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    vocab = tokenizer_json["model"]["vocab"]
    pieces = sorted(vocab, key=vocab.get)
    if [vocab[piece] for piece in pieces] != list(range(len(pieces))):
        raise ValueError("tokenizer.json vocab ids are not contiguous")
    if model.find_hparam(["vocab_size"]) != len(pieces):
        raise ValueError("tokenizer.json vocab size differs from config.json")

    rank: dict[str, int] = {}
    for merge in tokenizer_json["model"]["merges"]:
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        rank.setdefault(left + right, len(rank))

    tokens: list[bytes] = []
    scores: list[float] = []
    toktypes: list[int] = []
    for piece in pieces:
        if piece == tokenizer_json["model"]["unk_token"]:
            toktype, score = token_types.UNKNOWN, 0.0
        elif BYTE_PIECE.fullmatch(piece):
            toktype, score = token_types.BYTE, 0.0
        else:
            # A piece that never results from a merge cannot be produced by merging,
            # so its score is never compared.
            toktype, score = token_types.NORMAL, -float(rank.get(piece, len(rank)))
        tokens.append(piece.encode("utf-8"))
        scores.append(score)
        toktypes.append(toktype)
    apply_added_tokens(model, token_types, tokens, scores, toktypes)
    return tokens, scores, toktypes


def apply_added_tokens(
    model: Any,
    token_types: Any,
    tokens: list[bytes],
    scores: list[float],
    toktypes: list[int],
) -> None:
    """Apply the added-token overrides _create_vocab_sentencepiece applies, in place."""
    added_tokens_file = model.dir_model / "added_tokens.json"
    if added_tokens_file.is_file():
        with added_tokens_file.open(encoding="utf-8") as f:
            for key, token_id in json.load(f).items():
                if token_id >= len(tokens):
                    continue
                tokens[token_id] = key.encode("utf-8")
                scores[token_id] = -1000.0
                toktypes[token_id] = token_types.USER_DEFINED

    with (model.dir_model / "tokenizer_config.json").open(encoding="utf-8") as f:
        added_tokens_decoder = json.load(f).get("added_tokens_decoder", {})
    for raw_id, token_data in added_tokens_decoder.items():
        token_id, token = int(raw_id), token_data["content"]
        if token_id >= len(tokens):
            continue
        if token_data.get("special") or model.does_token_look_special(token):
            toktypes[token_id] = token_types.CONTROL
        else:
            token = token.replace("▁", " ")  # pre-normalize user-defined spaces
            toktypes[token_id] = token_types.USER_DEFINED
        scores[token_id] = -1000.0
        tokens[token_id] = token.encode("utf-8")


def main() -> None:
    """Patch the Gemma3 converter's vocab step, then run convert_hf_to_gguf.py."""
    src = sys.argv.pop(1)
    sys.path[:0] = [src, f"{src}/gguf-py"]
    gguf = importlib.import_module("gguf")
    token_types = importlib.import_module("conversion.base").SentencePieceTokenTypes
    gemma3_model = importlib.import_module("conversion.gemma").Gemma3Model

    def set_vocab(self: Any) -> None:
        tokens, scores, toktypes = vocab_from_tokenizer_json(self, token_types)
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens)).add_to_gguf(
            self.gguf_writer
        )
        self.gguf_writer.add_add_space_prefix(False)

    gemma3_model.set_vocab = set_vocab
    sys.argv[0] = f"{src}/convert_hf_to_gguf.py"
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
