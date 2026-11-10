"""The prepared-data format catches bad gold, and samples nest."""

import pytest

from rbench.paths import PREPARED
from rbench.schema import validate


def test_validate_flags_out_of_bounds_and_unknown_docs():
    docs = [{"doc_id": "a", "text": "hello world"}]
    good = {"qid": "1", "split": "test", "variants": {"q": "x"}, "gold": [[["a", 0, 5]]]}
    assert validate(docs, [good]) == []
    bad = {**good, "qid": "2", "gold": [[["a", 3, 99]]]}
    unknown = {**good, "qid": "3", "gold": [[["b", 0, 1]]]}
    problems = validate(docs, [bad, unknown])
    assert len(problems) == 2


@pytest.mark.skipif(not all((PREPARED / d / "queries.jsonl").exists() for d in ("longmemeval_s", "longmemeval_m")),
                    reason="needs LongMemEval-S and -M prepared")
def test_samples_nest_in_one_hash_order():
    from rbench.samples import ORDER, sample

    for dataset in ("longmemeval_s",):
        picked = [set(sample(dataset, name)) for name in ORDER]
        assert all(a <= b for a, b in zip(picked, picked[1:], strict=False))
        assert len(picked[0]) == 8
