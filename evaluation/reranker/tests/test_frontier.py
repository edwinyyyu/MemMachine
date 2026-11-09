"""Break-even ratios and relative cost on constructed result sets, chosen and measured on the
same questions (there is no dev split)."""

import numpy as np
import pandas as pd

from rbench.report import frontier
from rbench.report.data import Results


def results(reranked_cost: float = 500.0) -> Results:
    """First stage: every query's gold costs 2,000 reader tokens. Reranked: `reranked_cost`, for
    10,000 reranker tokens."""
    rows = []
    for q in range(100):
        common = dict(qid=f"q{q}", split="test", cluster=f"q{q}", chunking="rec256", expand=0,
                      instruction="default", render="plain")
        rows.append(dict(common, ranker="first_stage", kind="first_stage", model=None, depth=None, first=2000.0,
                         rr_tokens=0.0, rr_prefix_tokens=0.0))
        rows.append(dict(common, ranker="rerank:q8@t8192", kind="rerank", model="q8", depth="t8192",
                         first=reranked_cost, rr_tokens=10000.0, rr_prefix_tokens=4000.0))
    return _results(rows)


def _results(rows: list[dict]) -> Results:
    d = pd.DataFrame(rows)
    if "every" not in d:
        d["every"] = d["first"]  # one gold span per query: first and every coincide
    return Results("e", "d", "corpus", "v", "s", d)


def grid_at_least(x: float) -> float:
    return x  # break-even budgets are exact order statistics, not grid points


def test_break_even_is_reranker_tokens_over_reader_tokens_saved():
    point, lo, hi, base, chosen = frontier.break_even(results(), "q8")
    i = int(np.argmin(np.abs(frontier.TARGETS - 0.9)))
    assert np.isclose(point[i], 10000 / (grid_at_least(2000) - grid_at_least(500)))
    assert chosen[i] == ("rec256", "rerank:q8@t8192") and base[i] == ("rec256", "first_stage")
    assert lo[i] <= point[i] <= hi[i]


def test_no_saving_means_never_cheaper():
    point, *_ = frontier.break_even(results(reranked_cost=2000.0), "q8", with_interval=False)
    i = int(np.argmin(np.abs(frontier.TARGETS - 0.9)))
    assert np.isinf(point[i])


def test_reranking_pays_only_above_the_break_even_price():
    q = 0.05  # reranker $/M
    point, *_ = frontier.break_even(results(), "q8", with_interval=False)
    i = int(np.argmin(np.abs(frontier.TARGETS - 0.9)))
    prices = [point[i] * q * 0.5, point[i] * q * 2, point[i] * q * 100]  # below, above, far above break-even
    curves = frontier.cost_curves(results(), "q8", prices, q)
    no, yes = zip(*(curves[p] for p in prices), strict=True)
    assert yes[0][i] > no[0][i]  # below break-even: reranking costs more
    assert yes[1][i] < no[1][i]  # above: it saves
    assert yes[2][i] / no[2][i] < yes[1][i] / no[1][i]  # and saves a larger share as the reader gets dearer


def test_only_the_reranker_reaching_a_target_is_cheaper_at_any_price():
    # the first stage reaches 90% of questions; the reranker reaches all of them at 500 tokens
    rows = []
    for q in range(100):
        common = dict(qid=f"q{q}", split="test", cluster=f"q{q}", chunking="rec256", expand=0,
                      instruction="default", render="plain")
        rows.append(dict(common, ranker="first_stage", kind="first_stage", model=None, depth=None,
                         first=2000.0 if q < 90 else np.inf, rr_tokens=0.0, rr_prefix_tokens=0.0))
        rows.append(dict(common, ranker="rerank:q8@t8192", kind="rerank", model="q8", depth="t8192", first=500.0,
                         rr_tokens=10000.0, rr_prefix_tokens=4000.0))
    point, *_ = frontier.break_even(_results(rows), "q8", with_interval=False)
    at = lambda t: point[int(np.argmin(np.abs(frontier.TARGETS - t)))]  # noqa: E731
    assert np.isclose(at(0.8), 10000 / (grid_at_least(2000) - grid_at_least(500)))
    assert at(0.95) == 0.0  # only the reranker reaches 0.95


def test_common_depth_ignores_models_not_scored_at_a_chunking():
    rows = []
    for chunking, models in (("rec256", ("q06", "q8")), ("fix64", ("q06",))):
        for m in models:
            rows.append(dict(qid="1", split="test", cluster="1", chunking=chunking, expand=0, instruction="default",
                             render="plain", ranker=f"rerank:{m}@t8192", kind="rerank", model=m, depth="t8192"))
    r = Results("e", "d", "corpus", "v", "s", pd.DataFrame(rows))
    assert r.common_depth("fix64") == "t8192" and r.models_at("fix64") == ["q06"]


def history_results() -> Results:
    """No dev split. First stage 2,000 tokens; first stage expanded 1,500; reranked seeds 1,200
    for 10,000 reranker tokens; reranked windows (expand, then rerank) 500 for 20,000. Every
    gold costs more than first gold, and only every-gold separates the configurations."""
    rows = []
    for q in range(100):
        common = dict(qid=f"q{q}", split="test", cluster=f"q{q}", chunking="seg500", instruction="default", render="em")
        for ranker, kind, model, expand, every, tokens in (
                ("first_stage", "first_stage", None, 0, 2000.0, 0.0),
                ("first_stage+x3", "first_stage", None, 3, 1500.0, 0.0),
                ("rerank:q8@t4096", "rerank", "q8", 0, 1200.0, 10000.0),
                ("xrerank:q8@t4096+x3", "xrerank", "q8", 3, 500.0, 20000.0)):
            rows.append(dict(common, ranker=ranker, kind=kind, model=model, depth=None if model is None else "t4096",
                             expand=expand, first=300.0, every=every, rr_tokens=tokens, rr_prefix_tokens=tokens / 2))
    return _results(rows)


def test_expanded_configurations_compete_on_every_gold():
    point, lo, hi, base, chosen = frontier.break_even(history_results(), "q8")
    i = int(np.argmin(np.abs(frontier.TARGETS - 0.9)))
    assert base[i] == ("seg500", "first_stage+x3")  # the cheaper no-reranker configuration is the expanded one
    assert chosen[i] == ("seg500", "xrerank:q8@t4096+x3")  # 20,000 / 1,000 beats 10,000 / 300
    assert np.isclose(point[i], 20000 / (grid_at_least(1500) - grid_at_least(500)))
    assert lo[i] <= point[i] <= hi[i]


def test_first_gold_is_still_available():
    point, *_ = frontier.break_even(history_results(), "q8", with_interval=False, event="first")
    i = int(np.argmin(np.abs(frontier.TARGETS - 0.9)))
    assert np.isinf(point[i])  # every configuration reads the first gold at the same cost
