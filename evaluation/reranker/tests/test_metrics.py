"""Coverage costs, ranks, recall and precision on hand-checkable cases."""

import math

import numpy as np

from rbench.metrics import CURVE_BUDGETS, CURVE_KS, FIELDS, Gold, evaluate

TOKENS = np.array([100, 100, 100, 100])
CHARS = np.array([400, 400, 400, 400])


def _eval(order, tokens, chars, g):
    return evaluate(order, np.cumsum(tokens[order]), chars, g)


def fields(values):
    return dict(zip(FIELDS, values))


def test_single_span_in_one_chunk():
    g = Gold(((((2, 50),),),))  # one alternative, one span: 50 gold characters in chunk 2
    f = fields(_eval(np.array([0, 2, 1, 3]), TOKENS, CHARS, g))
    assert f["first"] == 200 and f["every"] == 200 and f["rank"] == 2 and f["first_k"] == 2
    assert f["recall_k1"] == 0 and f["recall_t250"] == 1.0  # gold is the second chunk read (200 tokens)
    assert math.isclose(f["precision_k2"], 50 / 800)


def test_span_across_two_chunks_needs_both():
    g = Gold(((((1, 30), (3, 30)),),))
    f = fields(_eval(np.array([1, 0, 3, 2]), TOKENS, CHARS, g))
    assert f["first"] == 300  # chunk 3 is read third
    assert f["recall_k1"] == 0.5 and f["recall_k3"] == 1.0


def test_every_needs_all_spans_of_one_alternative_and_first_any():
    g = Gold(((((0, 10),), ((3, 10),)), (((1, 10),),)))  # A = {span in 0, span in 3}; B = {span in 1}
    f = fields(_eval(np.array([0, 2, 3, 1]), TOKENS, CHARS, g))
    assert f["first"] == 100  # A's first span
    assert f["every"] == 300  # A complete at chunk 3 (third); B would need the fourth
    assert f["every_k"] == 3


def test_gold_outside_the_head_is_inf():
    g = Gold(((((5, 10),),),))
    f = fields(_eval(np.array([0, 1]), TOKENS, CHARS, g))
    assert math.isinf(f["first"]) and math.isinf(f["rank"]) and f["recall_k1"] == 0


def test_field_lists_match():
    assert len(FIELDS) == 9 + 2 * len(CURVE_BUDGETS) + 2 * len(CURVE_KS)
    from rbench import config

    # every budget the report reads curves at is on the stored grid
    assert set(config.EVAL.budgets) | set(config.EVAL.columns) <= set(CURVE_BUDGETS)


def test_precision_counts_overlapping_gold_once_and_all_spans_in_a_chunk():
    # chunk 0 holds two gold spans of 30 and 20 characters that do not overlap: 50 gold characters
    g = Gold(((((0, 30),), ((0, 20),)),), chunk_union={0: 50})
    f = fields(_eval(np.array([0, 1]), TOKENS, CHARS, g))
    assert math.isclose(f["precision_k1"], 50 / 400)


def test_trim_drops_edge_whitespace_only():
    from rbench.metrics import trim

    text = "ab \n  code here\n\n xy"
    assert trim(text, 3, 17) == (6, 15)
    assert trim(text, 2, 4) == (2, 4)  # all whitespace: unchanged


def test_merged_union_of_overlapping_spans():
    from rbench.metrics import merged_length

    assert merged_length([(0, 10), (5, 15), (20, 25)]) == 20


def test_touched_needs_any_chunk_of_a_span_complete_needs_all():
    g = Gold(((((1, 30), (3, 30)),),))  # one span across chunks 1 and 3
    f = fields(_eval(np.array([1, 0, 2, 3]), TOKENS, CHARS, g))
    assert f["first_touch"] == 100 and f["first_touch_k"] == 1  # chunk 1 read first
    assert f["first"] == 400 and f["first_k"] == 4  # complete only once chunk 3 is read


def test_every_touch_needs_every_span_of_one_answer_touched():
    g = Gold(((((0, 10), (1, 10)), ((3, 10),)),))  # spans {0,1} and {3}
    f = fields(_eval(np.array([0, 3, 1, 2]), TOKENS, CHARS, g))
    assert f["every_touch"] == 200  # chunk 0 touches span 1, chunk 3 completes span 2
    assert f["every"] == 300  # span 1 is complete only after chunk 1


def test_a_dip_in_rendered_cost_does_not_admit_more_than_the_first_overflow():
    # rendered cost dips at the third chunk (two windows merge): budget 250 stops after the first
    g = Gold(((((2, 50),),),))
    f = fields(evaluate(np.array([0, 1, 2, 3]), np.array([200.0, 260.0, 240.0, 340.0]), CHARS, g))
    assert f["first"] == 260  # chunk 2 is read only once 260 tokens fit
    assert f["recall_t250"] == 0  # the 240 after the dip is not reachable within 250


def test_recall_matches_the_set_definition_on_random_cases():
    """recall_k is max over answers of (gold characters of chunks read, per span) / answer total."""
    from rbench.metrics import CURVE_KS

    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 12
        alternatives = tuple(tuple(tuple((int(c), int(rng.integers(1, 40))) for c in rng.choice(n, rng.integers(1, 3), replace=False))
                                   for _ in range(rng.integers(1, 4))) for _ in range(rng.integers(1, 3)))
        union: dict[int, int] = {}
        for alt in alternatives:
            for span in alt:
                for c, k in span:
                    union[c] = max(union.get(c, 0), k)
        g = Gold(alternatives, chunk_union=union)
        order = rng.permutation(n)
        f = fields(_eval(order, np.full(n, 10.0), np.full(n, 100), g))
        for k in CURVE_KS:
            seen = {int(c) for c in order[:k]}
            want = max(sum(m for span in alt for c, m in span if c in seen) / sum(m for span in alt for _, m in span)
                       for alt in alternatives)
            assert math.isclose(f[f"recall_k{k}"], want)
