"""Qwen3-Reranker on this machine's GPU (PyTorch MPS, fp16), for prototyping.

Rows (token ids from rerankers.pair_ids) are sorted by length and packed into batches of
at most BATCH_TOKENS tokens, padding included, left-padded so every row ends at the last
position, where the yes/no logits are read.

Measured 2026-11-05 on an M3 Pro (18 GB): about 2,300-2,800 pair tokens/s with the GPU to
itself. Encoding the shared prompt and query once per query (a prefix cache) gave the same
scores but no speedup on MPS, so it is not used.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import torch

BATCH_TOKENS = 16384
DEVICE = "mps"


@cache
def _load(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(DEVICE).eval()
    return model, tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")


@torch.inference_mode()
def score_rows(rows: list[list[int]], model_name: str) -> np.ndarray:
    """logit(yes) - logit(no) at each row's last position."""
    model, pad, yes, no = _load(model_name)
    out = np.empty(len(rows), dtype=np.float32)
    batch: list[int] = []

    def flush() -> None:
        width = max(len(rows[i]) for i in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, i in enumerate(batch):
            ids[j, width - len(rows[i]) :] = torch.tensor(rows[i])
            mask[j, width - len(rows[i]) :] = 1
        logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE), logits_to_keep=1).logits[:, -1, :]
        out[batch] = (logits[:, yes].float() - logits[:, no].float()).cpu().numpy()

    for i in sorted(range(len(rows)), key=lambda i: len(rows[i])):
        if batch and (len(batch) + 1) * len(rows[i]) > BATCH_TOKENS:  # sorted: row i is the widest
            flush()
            batch = []
        batch.append(i)
    if batch:
        flush()
    return out
