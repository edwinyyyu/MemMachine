"""The prepared-data format catches bad gold, and samples nest."""

from rbench.schema import hash_split, validate


def test_validate_flags_out_of_bounds_and_unknown_docs():
    docs = [{"doc_id": "a", "text": "hello world"}]
    good = {"qid": "1", "split": "test", "variants": {"q": "x"}, "gold": [[["a", 0, 5]]]}
    assert validate(docs, [good]) == []
    bad = {**good, "qid": "2", "gold": [[["a", 3, 99]]]}
    unknown = {**good, "qid": "3", "gold": [[["b", 0, 1]]]}
    problems = validate(docs, [bad, unknown])
    assert len(problems) == 2


def test_hash_split_is_stable():
    assert hash_split("doc-1") == hash_split("doc-1")
    shares = [hash_split(str(i)) == "dev" for i in range(2000)]
    assert 0.25 < sum(shares) / len(shares) < 0.35
