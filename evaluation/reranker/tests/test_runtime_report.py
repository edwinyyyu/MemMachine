"""The runtime report's naming of the results set, and its chart text."""

from rbench import runtime_report as rr


def test_the_names_follow_the_results_set():
    assert rr.rid("q4-mem", 5) == "rerank:q4-mem@n5"
    assert rr.rid("q4-mem", 5, "200") == "rerank:q4-mem~200@n5"
    assert rr.rid("q4-mem", 20, "4k", 2) == "rerank:q4-mem@n20+x2"
    assert rr.fid(20, 1) == "first_stage@n20+x1"


def test_dollar_signs_never_open_matplotlib_math():
    assert rr._money("$0.025/M and $0.05/M") == r"\$0.025/M and \$0.05/M"
