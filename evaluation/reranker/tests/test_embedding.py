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


TRICKY = 'He said "hi"\nthen left \\ ✓ — 日本'


def test_a_user_query_reads_as_event_memory_renders_the_users_turn():
    """`User: "<query>"` is byte for byte what EventMemory's WholeTextDeriver embeds for a User
    turn, less the timestamp: the producer outside the quotes, the text JSON-dumped with
    ensure_ascii off."""
    import json

    from memmachine_server.episodic_memory.event_memory.data_types import ProducerContext
    from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import _format_with_context

    spec = embedding.EMBEDDERS["gemma300m-di-raw-user"]
    assert spec.render_query(TRICKY) == _format_with_context(ProducerContext(producer="User"), json.dumps(TRICKY, ensure_ascii=False))
    assert spec.query_text(TRICKY) == spec.render_query(TRICKY)  # no prompt
    assert embedding.EMBEDDERS["gemma300m-di-raw"].render_query(TRICKY) == TRICKY  # unformatted: as it is


def test_the_instruction_comes_before_the_rendered_query():
    e = embedding.EMBEDDERS
    assert e["qwen3-8b-di-mem-user"].query_text("x") == embedding.QWEN3_MEMORY_QUERY + 'User: "x"'
    assert e["qwen3-8b-di-mem"].query_text("x") == embedding.QWEN3_MEMORY_QUERY + "x"
    assert e["qwen3-8b-di"].query_text("x") == embedding.QWEN3_QUERY + "x"
    assert e["qwen3-8b-di-raw-user"].query_text("x") == 'User: "x"'
    assert embedding.QWEN3_MEMORY_QUERY == ("Instruct: Given a user's query, retrieve relevant passages that provide the facts, "
                                            "preferences, or events needed to answer it.\nQuery:")


def test_the_qwen3_grid_shares_documents_per_size_and_keeps_the_first_setups():
    e = embedding.EMBEDDERS
    names = [f"qwen3-{s}-di{i}{f}" for s in ("8b", "4b", "0.6b") for i in ("", "-mem", "-raw") for f in ("", "-user")]
    assert all(n in e for n in names)
    for size, model, usd in (("8b", "Qwen/Qwen3-Embedding-8B", 0.010), ("4b", "Qwen/Qwen3-Embedding-4B", 0.020),
                             ("0.6b", "Qwen/Qwen3-Embedding-0.6B", 0.010)):
        group = [e[n] for n in names if n.startswith(f"qwen3-{size}-")]
        assert {g.documents for g in group} == {f"qwen3-{size}-di"}  # one document store per size
        assert {(g.model, g.usd_per_mtok, g.dimensions, g.doc_prompt, g.tokenizer) for g in group} == \
            {(model, usd, 1024, "", embedding.QWEN_EMBED)}
    assert not e["qwen3-0.6b-di"].send_dimensions and e["qwen3-4b-di"].send_dimensions  # 0.6B is natively 1,024
    # the setups compared first are what they were: same request, same caches
    assert e["qwen3-8b-di"].query_prompt == embedding.QWEN3_QUERY and e["qwen3-8b-di"].query_producer is None
    assert e["qwen3-8b-di-raw"].query_prompt == "" and e["qwen3-8b-di-raw"].query_producer is None
    assert e["qwen3-8b-di"].documents == e["qwen3-8b-di-raw"].documents == "qwen3-8b-di"
    # every 8B setup has its full-width twin for --widths
    assert all(e[f"{n}-{embedding.FULL}"].dimensions == embedding.FULL for n in names if n.startswith("qwen3-8b-"))


def test_query_vectors_embed_the_rendered_queries(tmp_path, monkeypatch):
    rows = [{"variants": {"question": TRICKY}}, {"variants": {"question": "b"}}]
    sent = {}

    def fake_run(spec, texts, prompt, target):
        sent[spec.name] = (list(texts), prompt)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.zeros((len(texts), 2), dtype=np.float16))

    monkeypatch.setattr(embedding, "VECTORS", tmp_path)
    monkeypatch.setattr(embedding, "queries", lambda dataset: rows)
    monkeypatch.setattr(embedding, "_run", fake_run)
    embedding.query_vectors("qwen3-8b-di-mem-user", "d", "question")
    embedding.query_vectors("qwen3-8b-di", "d", "question")
    assert sent["qwen3-8b-di-mem-user"] == ([embedding.EMBEDDERS["gemma300m-raw-user"].render_query(TRICKY), 'User: "b"'],
                                            embedding.QWEN3_MEMORY_QUERY)
    assert sent["qwen3-8b-di"] == ([TRICKY, "b"], embedding.QWEN3_QUERY)


def test_a_dated_query_embeds_as_the_deriver_heads_a_segment():
    """`[<date>] User: "<query>"` with the date as WholeTextDeriver dates the documents (the date
    alone, MemMachine's default embedding options); a dated embedder needs the date and is not
    run inside EventMemory (which passes the bare question)."""
    import json

    from rbench import memory, partitions as pt

    row = {"variants": {"question": "Where?"}, "features": {"question_date": "2023/05/30 (Tue) 23:31"}}
    date = pt.query_date("longmemeval_m", row, for_embedding=True)
    assert date == "Tuesday, May 30, 2023"
    spec = embedding.EMBEDDERS["gemma300m-di-raw-date-user"]
    assert spec.render_query("Where?", date) == f"[{date}] User: {json.dumps('Where?')}"
    assert spec.documents == embedding.EMBEDDERS["gemma300m-di-raw"].documents  # the same document vectors
    with pytest.raises(ValueError, match="date"):
        spec.render_query("Where?")
    with pytest.raises(NotImplementedError):
        memory.QueryCachingEmbedder("gemma300m-di-raw-date-user")
