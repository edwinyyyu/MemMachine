"""The pair format is the model card's, and cache keys and score scales are stable."""

import numpy as np
import pytest

from dataclasses import replace

from rbench import rerankers
from rbench.rerankers import PREFIX, QWEN3_CARD, SUFFIX, ScoreCache, pair_ids, pair_overhead, to_logit
from rbench.text import QWEN, tokenizer


def test_pair_decodes_to_the_model_card_format():
    ids = pair_ids("what is x", ["x is y"], replace(QWEN3_CARD, instruction="Find it"))[0]
    text = tokenizer(QWEN).decode(ids, skip_special_tokens=False)
    assert text == PREFIX + "<Instruct>: Find it\n<Query>: what is x\n<Document>: x is y" + SUFFIX


def test_long_documents_are_cut_to_the_pair_limit():
    from rbench.rerankers import _fixed_ids

    ids = pair_ids("q", ["word " * 20000], replace(QWEN3_CARD, instruction="i"))[0]
    assert len(ids) == 8192 and ids[-len(_fixed_ids()[1]):] == list(_fixed_ids()[1])  # suffix kept at the end


def test_overhead_excludes_the_document_and_suffix():
    full = len(pair_ids("query", ["doc"], replace(QWEN3_CARD, instruction="inst"))[0])
    assert pair_overhead("query", replace(QWEN3_CARD, instruction="inst")) < full


def test_cache_key_is_content_addressed():
    a = ScoreCache.key("p", "plain", "q", "d")
    assert a == ScoreCache.key("p", "plain", "q", "d") and a != ScoreCache.key("p", "em", "q", "d")
    assert a.startswith("p|plain|") and len(a.split("|")[2]) == 16


def test_probabilities_map_to_logits():
    p = np.array([0.5, 0.9])
    assert np.allclose(to_logit("q8", p), [0.0, np.log(9)])
    assert np.allclose(to_logit("q06-local", np.array([1.5])), [1.5])


def test_pair_is_the_model_card_string_tokenized_in_one_piece():
    from rbench.rerankers import FORMAT, _fixed_ids

    query, doc, inst = "capital of France?", "Paris is the capital.", "Find it"
    body = tokenizer(QWEN).encode(FORMAT.format(instruction=inst, query=query, document=doc), add_special_tokens=False).ids
    prefix, suffix = _fixed_ids()
    assert pair_ids(query, [doc], replace(QWEN3_CARD, instruction=inst))[0] == list(prefix) + body + list(suffix)


def test_documents_are_cut_client_side_to_fit():
    from rbench.rerankers import fit_document

    MAX_PAIR_TOKENS = QWEN3_CARD.max_pair_tokens

    doc = "word " * 20000
    cut = fit_document("q", doc, replace(QWEN3_CARD, instruction="i"))
    assert doc.startswith(cut) and len(cut) < len(doc)
    assert len(pair_ids("q", [cut], replace(QWEN3_CARD, instruction="i"))[0]) <= MAX_PAIR_TOKENS
    assert fit_document("q", "short", replace(QWEN3_CARD, instruction="i")) == "short"


def test_cache_key_changes_with_instruction_text():
    from rbench.rerankers import prompt_key

    assert prompt_key(replace(QWEN3_CARD, instruction="Find it")) != prompt_key(replace(QWEN3_CARD, instruction="Find it."))


def test_the_4b_query_arms_send_what_they_say():
    """`-user` sends the query as the user's message (the embedders' rendering); `-mem` sends the
    user's instruction in the card's fixed format; the bare 4B is unchanged."""
    from rbench import embedding

    q = 'What did I say about "basil"?\nAnything else?'
    assert rerankers.query_for("q4", q) == q and rerankers.query_for("q4-mem", q) == q
    assert rerankers.query_for("q4-user", q) == rerankers.query_for("q4-mem-user", q) == \
        embedding.EMBEDDERS["gemma300m-di-raw-user"].render_query(q)
    card, mem = rerankers.RERANKERS["q4"].template, rerankers.RERANKERS["q4-mem"].template
    assert (mem.prefix, mem.format, mem.suffix) == (card.prefix, card.format, card.suffix)  # DeepInfra's fixed part
    assert mem.instruction == rerankers.MEMORY_INSTRUCTION != card.instruction
    assert rerankers.prompt_key(mem) != rerankers.prompt_key(card)  # its scores never mix with the card's
    for m in ("q4-user", "q4-mem", "q4-mem-user"):
        spec = rerankers.RERANKERS[m]
        assert spec.hf_model == "Qwen/Qwen3-Reranker-4B" and spec.backend == "deepinfra" and spec.usd_per_mtok == 0.025


def test_a_rendered_query_is_what_record_mode_queues_and_keys(tmp_path, monkeypatch):
    """EventMemory passes the bare question; the reranker queues and keys the rendered one."""
    import asyncio

    from rbench import memory

    monkeypatch.setattr(rerankers, "SCORES", tmp_path)
    r = memory.CachedReranker("q4-user", mode="record")
    asyncio.run(r.score("hi?", ["[2023-05-23] User: \"doc\""]))
    assert list(r.work) == list(r.asked) == ['User: "hi?"']
    bare = memory.CachedReranker("q4", mode="record")
    asyncio.run(bare.score("hi?", ["[2023-05-23] User: \"doc\""]))
    assert list(bare.work) == ["hi?"] and r.work['User: "hi?"'][0][0] != bare.work["hi?"][0][0]  # different keys


def test_deepinfra_refuses_only_a_different_format_not_a_different_instruction(monkeypatch):
    sent = []
    monkeypatch.setattr(rerankers, "_deepinfra", lambda *a, **k: sent.append(a[2]) or _done())
    rerankers.score("q4-mem", [], "t", progress=False)
    assert sent and sent[0].instruction == rerankers.MEMORY_INSTRUCTION
    monkeypatch.setitem(rerankers.RERANKERS, "odd", replace(rerankers.RERANKERS["q4"], name="odd",
                                                            template=rerankers.RAW_PAIRS))
    with pytest.raises(SystemExit, match="only the instruction is sent"):
        rerankers.score("odd", [], "t", progress=False)


async def _done() -> None:
    return None


def test_the_tuning_candidates_are_the_reviewed_texts_in_the_card_format():
    """The six candidates the user reviewed (c6 and c7 dropped), each only an instruction change."""
    assert list(rerankers.TUNING) == ["c1", "c2", "c3", "c4", "c5", "c8"]
    assert rerankers.TUNING["c1"][1] == "Given a user's query, retrieve passages relevant to it."
    keys = {rerankers.prompt_key(rerankers.RERANKERS[m].template) for m in ("q4", "q4-mem", *(f"q4-{c}" for c in rerankers.TUNING))}
    assert len(keys) == 8  # every arm's scores are its own
    for c, (_, text) in rerankers.TUNING.items():
        t = rerankers.RERANKERS[f"q4-{c}"].template
        assert t.instruction == text and (t.prefix, t.format, t.suffix) == (QWEN3_CARD.prefix, QWEN3_CARD.format, QWEN3_CARD.suffix)
        assert rerankers.RERANKERS[f"q4-{c}"].query_producer is None and rerankers.RERANKERS[f"q4-{c}"].hf_model.endswith("4B")


def test_a_dated_query_is_headed_as_eventmemory_heads_a_segment(tmp_path, monkeypatch):
    """`[<date>] User: "<query>"`, the date by EventMemory's format_timestamp with the reranker's
    format (as every document the reranker reads is headed); a dated reranker needs the date."""
    import asyncio
    import json

    from rbench import memory, partitions as pt
    from rbench.datasets.longmemeval import session_datetime
    from memmachine_server.episodic_memory.event_memory.formatting import format_timestamp

    row = {"variants": {"question": 'Is "it" here?'}, "features": {"question_date": "2023/05/30 (Tue) 23:31"}}
    date = pt.query_date("longmemeval_m", row)
    assert date == format_timestamp(session_datetime("2023/05/30 (Tue) 23:31"), pt.formats("longmemeval_m").rerank)
    assert date == "Tuesday, May 30, 2023, 11:31 PM"
    q = row["variants"]["question"]
    assert rerankers.query_for("q4-mem-date-user", q, date) == f"[{date}] User: {json.dumps(q, ensure_ascii=False)}"
    assert rerankers.query_for("q4-mem-user", q, date) == f"User: {json.dumps(q, ensure_ascii=False)}"  # no date asked
    with pytest.raises(ValueError, match="date"):
        rerankers.query_for("q4-mem-date-user", q)
    assert pt.query_date("x", {"features": {}}) is None
    monkeypatch.setattr(rerankers, "SCORES", tmp_path)
    monkeypatch.setattr("rbench.schema.queries", lambda dataset: [row])
    r = memory.CachedReranker("q4-mem-date-user", mode="record", dataset="longmemeval_m")
    asyncio.run(r.score(q, ["[Saturday, May 20, 2023, 1:01 PM] User: \"doc\""]))
    assert list(r.work) == [f"[{date}] User: {json.dumps(q, ensure_ascii=False)}"]
    with pytest.raises(ValueError, match="dataset"):
        memory.CachedReranker("q4-mem-date-user", mode="record")
