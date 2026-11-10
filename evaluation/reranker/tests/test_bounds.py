"""Every bound is valid: no simulated ranker over the same pool reads a question's evidence with
fewer tokens (events), or reaches a higher share of it at any budget (shares). Over lme-test's
simulation; plus the additive seed cost the exact bounds rest on."""

import numpy as np
import pandas as pd
import pytest

from rbench.paths import SIM
from rbench.report import readings as rd

SIMULATED = sorted((SIM / "gemma300m-raw/longmemeval_s/history/question/tiny").glob("*.parquet"))


@pytest.mark.skipif(not SIMULATED, reason="needs the lme-test stage's simulation")
@pytest.mark.parametrize("path", SIMULATED, ids=lambda p: p.stem)
def test_no_ranker_beats_the_bound(path):
    d = pd.read_parquet(path)
    bounds = d[d.kind == "bound"]
    assert not bounds.empty
    checked = 0
    for (depth, expand), b in bounds.groupby(["depth", "expand"]):
        b = b.sort_values("qid")
        others = d[(d.depth == depth) & (d.expand == expand) & (d.kind != "bound")]
        for ranker, rows in others.groupby("ranker"):
            rows = rows.sort_values("qid")
            assert (rows.qid.to_numpy() == b.qid.to_numpy()).all()
            for reading in rd.EVENTS:
                bad = rows[reading].to_numpy() < b[reading].to_numpy() - 1e-9
                assert not bad.any(), (ranker, reading, rows.qid.to_numpy()[bad][:3])
            for reading in rd.SHARES:
                cb, cr = rd.costs(b, reading), rd.costs(rows, reading)
                for budget in np.unique(np.concatenate([rd.steps(cr), rd.steps(cb), [np.inf]])):
                    assert np.all(rd.hit(cr, budget) <= rd.hit(cb, budget) + 1e-9), (ranker, reading, budget)
            checked += 1
    assert checked
    seeds = bounds[bounds.expand == 0]
    assert seeds.bound_exact.astype(bool).all()  # exact with seeds alone on LongMemEval (turns share no segments)


def test_seeds_alone_cost_their_units_plus_separators():
    """The additive cost the seed bounds use equals the reader model for seeds read alone."""
    from rbench.reading import Reader, constants

    rng = np.random.default_rng(0)
    n = 12
    inner, run = rng.integers(5, 60, n), rng.integers(8, 20, n)
    block = np.arange(n)  # a different event each, or the same: seeds are their own runs either way
    seeds = [3, 7, 1, 10]
    reader = Reader([[s] for s in seeds], block, inner, run)
    cost = 0
    for s in seeds:
        cost = reader.admit(s)
    _, newline, separator = constants()
    assert cost == sum(int(inner[s]) + int(run[s]) - newline for s in seeds) + separator * (len(seeds) - 1)
