"""Reorderings return only the reranked pool: nothing after the depth is read."""

import numpy as np

from rbench.rankers import Depth, Ranker

SIMS = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
LOGITS = np.array([-1.0, 3.0, 0.5, 2.0, 9.0])
GOLD = np.array([False, False, True, False, False])


def test_first_stage_is_identity():
    assert Ranker("first_stage").order(0, SIMS, None, GOLD).tolist() == [0, 1, 2, 3, 4]


def test_rerank_sorts_the_first_n_and_reads_nothing_after():
    order = Ranker("rerank", "m", Depth(4)).order(4, SIMS, LOGITS, GOLD)
    assert order.tolist() == [1, 3, 2, 0]  # position 4 (logit 9) is beyond the depth: not retrieved


def test_oracle_puts_gold_first_within_depth():
    assert Ranker("oracle", depth=Depth(3)).order(3, SIMS, None, GOLD).tolist() == [2, 0, 1]


def test_depth_is_capped_by_the_head():
    assert Depth(10).chunks(np.array([100, 100, 100])) == 3


def test_ids_are_unique():
    from rbench.rankers import for_models

    ids = [r.id for r in for_models(["a", "b"])]
    assert len(ids) == len(set(ids))


def test_first_stage_with_a_vector_search_limit_reads_only_its_top_seeds():
    r = Ranker("first_stage", depth=Depth(3))
    assert r.order(3, SIMS, None, GOLD).tolist() == [0, 1, 2] and r.id == "first_stage@n3"
    assert Ranker("first_stage", depth=Depth(3), expand=3).id == "first_stage@n3+x3"


def test_history_rankers_use_count_depths_and_add_limited_first_stages():
    from rbench.rankers import DEPTHS, for_models

    rs = for_models(["m"], (3,), DEPTHS, limited_first_stage=True)
    assert all(r.depth is None or r.depth.label.startswith("n") for r in rs)
    assert {"first_stage@n20", "first_stage@n20+x3", "xrerank:m@n20+x3"} <= {r.id for r in rs}
    assert len({r.id for r in rs}) == len(rs)
