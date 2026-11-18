"""The runtime report's naming of the results set, and its chart text."""

from rbench import runtime_report as rr


def test_the_names_follow_the_results_set():
    assert rr.rid("q4-mem", 5) == "rerank:q4-mem@n5"
    assert rr.rid("q4-mem", 5, "200") == "rerank:q4-mem~200@n5"
    assert rr.rid("q4-mem", 20, "4k", 2) == "rerank:q4-mem@n20+x2"
    assert rr.fid(20, 1) == "first_stage@n20+x1"


def test_dollar_signs_never_open_matplotlib_math():
    assert rr._money("$0.025/M and $0.05/M") == r"\$0.025/M and \$0.05/M"


def test_the_same_money_baseline_reads_as_far_as_its_money_goes():
    import pandas as pd

    from rbench.report.data import Results

    df = pd.DataFrame({"ranker": ["first_stage", "first_stage@n5", "rerank:q4-mem@n5", "first_stage+x1",
                                  "first_stage@n5+x1"], "expand": [0, 0, 0, 1, 1], "qid": ["a"] * 5,
                       "depth": [None, "n5", "n5", None, "n5"], "every": [100.0, 900.0, 50.0, 200.0, 800.0]})
    u = rr.uncapped(Results("e", "d", "s", "v", "x", df), (5,))
    assert u.df.set_index("ranker").loc["first_stage@n5", "every"] == 100.0  # the uncapped first stage's rows
    assert u.df.set_index("ranker").loc["first_stage@n5+x1", "every"] == 200.0
    assert u.df.set_index("ranker").loc["rerank:q4-mem@n5", "every"] == 50.0  # the reranked side unchanged
