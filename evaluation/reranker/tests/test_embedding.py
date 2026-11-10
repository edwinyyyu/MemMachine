"""MemMachine's averaging: long texts are cut into pieces, embedded, length-weighted, normalized."""

import numpy as np
import pytest

from rbench import embedding


def test_long_texts_are_averaged_by_piece_length(monkeypatch):
    spec = embedding.Embedder("t", "http://x", "q: ", "d: ", max_tokens=2048, max_input_chars=4)
    seen = []

    async def fake(spec_, texts):
        seen.extend(texts)
        return np.asarray([[1.0, 0.0] if t.endswith("aaaa") else [0.0, 1.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(embedding, "_embed_all", fake)
    out = embedding.embed(spec, ["aaaabb", "xy"], "d: ")
    assert seen == ["d: aaaa", "d: bb", "d: xy"]  # pieces of 4 characters, prompt on each
    assert np.allclose(out[0], np.array([4, 2]) / np.linalg.norm([4, 2]))  # weights 4 and 2
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def _gemma_available() -> bool:
    """embeddinggemma's tokenizer is gated: absent without a Hugging Face login (as on EC2)."""
    from rbench.text import GEMMA, tokenizer

    try:
        tokenizer(GEMMA)
    except Exception:  # noqa: BLE001 - gated, offline or not cached
        return False
    return True


def test_over_long_text_is_cut_at_a_token_boundary():
    if not _gemma_available():
        pytest.skip("needs embeddinggemma's (gated) tokenizer")
    spec = embedding.EMBEDDERS["gemma300m"]
    text = "token " * 5000
    cut = embedding._fit(spec, text)
    from rbench.text import GEMMA, count

    assert count([cut], GEMMA)[0] <= spec.max_tokens - 2 < count([text], GEMMA)[0]
    assert text.startswith(cut)


def test_a_text_that_cannot_reach_the_limit_loads_no_tokenizer(monkeypatch):
    """Segments are far below embeddinggemma's 2,048 tokens: its gated tokenizer is never
    loaded for them (a token covers at least one UTF-8 byte, plus one leading-space token)."""
    spec = embedding.EMBEDDERS["gemma300m-di-raw"]

    def refuse(texts, repo):
        raise AssertionError(f"loaded {repo}")

    monkeypatch.setattr(embedding, "token_starts", refuse)
    budget = spec.max_tokens - spec.special_tokens
    for text in ("x" * (budget - 1), "日" * ((budget - 1) // 3), "😀" * ((budget - 1) // 4)):
        assert embedding._fit(spec, text) == text
    with pytest.raises(AssertionError, match="loaded"):
        embedding._fit(spec, "x" * budget)  # could reach it: measured


def test_prompt_setups_share_document_vectors_only_with_the_same_document_prompt():
    from rbench.embedding import EMBEDDERS

    assert EMBEDDERS["gemma300m-q"].documents == EMBEDDERS["gemma300m-raw"].documents != EMBEDDERS["gemma300m"].documents
    assert EMBEDDERS["gemma300m-d"].documents == EMBEDDERS["gemma300m"].documents
    assert EMBEDDERS["gemma300m-q"].doc_prompt == "" and EMBEDDERS["gemma300m-q"].query_prompt.startswith("task:")
    assert EMBEDDERS["gemma300m-raw"].query_prompt == EMBEDDERS["gemma300m-raw"].doc_prompt == ""
    assert EMBEDDERS["gemma300m-di-q"].url.startswith("https://api.deepinfra.com")
