"""compare-embedders' report helpers: segment producers, question types, the report's name."""

import numpy as np
import pytest

from rbench import compare_embedders as ce
from rbench.paths import PREPARED


def test_a_long_list_of_embedders_gets_a_short_stable_name():
    few = ce.summary_name("longmemeval_m", 500, 100, ["qwen3-8b-di", "gemma300m-di-raw"])
    assert few == "embedders__longmemeval_m__seg500__100q__qwen3-8b-di_gemma300m-di-raw"
    many = [f"qwen3-{s}-di{i}{f}" for s in ("8b", "4b", "0.6b") for i in ("", "-mem", "-raw") for f in ("", "-user")]
    name = ce.summary_name("longmemeval_m", 500, 200, many)
    assert name == ce.summary_name("longmemeval_m", 500, 200, list(many)) and len(name) < 100
    assert name.startswith("embedders__longmemeval_m__seg500__200q__18setups_")
    assert name != ce.summary_name("longmemeval_m", 500, 200, many[::-1])  # the order (the reference first) matters


def test_question_types_partition_the_questions():
    rows = [{"features": {"question_type": t}} for t in ("b", "a", "b", "c")]
    masks = ce.by_type(rows, [0, 1, 2, 3])
    assert list(masks) == ["a", "b", "c"]
    assert np.array_equal(np.sum(list(masks.values()), axis=0), np.ones(4))


@pytest.mark.skipif(not (PREPARED / "longmemeval_m" / "queries.jsonl").exists(), reason="needs LongMemEval-M prepared")
def test_each_segments_producer_is_the_one_its_text_names():
    """The producer read from the segment's turn is the one EventMemory rendered into its text."""
    from rbench.samples import sample_order

    rows = ce.queries("longmemeval_m")
    for qi in sample_order("longmemeval_m")[:2]:
        texts, gold, producers = ce.haystack_producers("longmemeval_m", rows[qi], 500)
        assert len(texts) == len(producers) and set(producers) <= {"User", "Assistant"}
        assert all(t.split("] ", 1)[1].startswith(f'{p}: "') for t, p in zip(texts, producers, strict=True))
        assert (texts, gold) == ce.haystack("longmemeval_m", rows[qi], 500)
