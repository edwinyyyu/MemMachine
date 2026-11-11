"""The tuning sample: questions 101-200 of the hash order, disjoint from the pilot's."""

import pytest

from rbench import samples
from rbench.paths import PREPARED


def test_a_slice_sample_is_not_among_the_nested_ones():
    assert "tune" not in samples.ORDER and samples.ORDER == ["tiny", "proto", "panel", "large", "all"]


@pytest.mark.skipif(not (PREPARED / "longmemeval_m" / "queries.jsonl").exists(), reason="needs LongMemEval-M prepared")
def test_tune_is_the_second_hundred_and_held_out_from_the_pilot():
    order = samples.sample_order("longmemeval_m")
    tune, proto = samples.sample("longmemeval_m", "tune"), samples.sample("longmemeval_m", "proto")
    assert tune == sorted(order[100:200]) and not set(tune) & set(proto)
    assert set(tune) <= set(samples.sample("longmemeval_m", "panel"))


def test_depth_shares_for_a_slice_look_only_at_its_own_sample(tmp_path, monkeypatch):
    from dataclasses import replace

    from rbench import depth, grid

    monkeypatch.setattr(depth, "DEPTH", tmp_path)
    run = grid.STAGES["lme-tune"][0]
    assert run.sample == "tune" and run.pool_seeds == 50
    assert depth.shares_for(run, "seg500", "gemma300m-di-raw") is None  # nothing measured: no nested fallback, no error
    assert depth.shares_for(replace(run, sample="proto"), "seg500", "gemma300m-di-raw") is None
