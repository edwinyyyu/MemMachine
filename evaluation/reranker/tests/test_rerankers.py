"""The pair format is the model card's, and cache keys and score scales are stable."""

import numpy as np

from dataclasses import replace

from rbench.rerankers import PREFIX, QWEN3_CARD, SUFFIX, ScoreCache, depth, pair_ids, pair_overhead, to_logit
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


def test_depth_policy():
    head = np.array([100] * 50)
    assert depth(head, 1000, 5) == 10  # reaches 1,000 tokens with the 10th chunk
    assert depth(head, 1050, 5) == 11  # the chunk that crosses the budget is included
    assert depth(head, 100, 16) == 16  # at least min_depth
    assert depth(np.array([100] * 3), 1000, 16) == 3  # capped by the head


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
