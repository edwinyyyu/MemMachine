"""End to end on the lme-test stage's stored partitions (skipped when they are not encoded):
the unified simulation replays the first stage and the perfect reranker from EventMemory."""

import numpy as np
import pytest

from rbench import simulation
from rbench.memory import store_root

READY = (store_root("gemma300m-raw", "longmemeval_s", "seg500") / "done").exists()


@pytest.mark.skipif(not READY, reason="needs the lme-test stage's encoded histories")
def test_simulation_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation, "SIM", tmp_path)
    df = simulation.run("gemma300m-raw", "longmemeval_s", "seg500", "tiny", [], (3,))
    assert {"first_stage", "oracle"} <= set(df.kind) and set(df.expand) == {0, 3}
    one = df[df.ranker == "first_stage"]
    assert len(one) == 8 and one.cluster.notna().all() and (one.seg_tokens > 0).all()
    # with seeds alone the perfect reranker is a bound for every gold read in full: no first
    # stage over the same top L reads all the evidence with fewer tokens
    for limit in ("n10", "n20", "n30"):
        ref = df[df.ranker == f"oracle@{limit}"].sort_values("qid")["every"].to_numpy()
        base = df[df.ranker == f"first_stage@{limit}"].sort_values("qid")["every"].to_numpy()
        assert np.all(ref <= base)
    assert np.isfinite(df[df.ranker == "first_stage+x3"]["recall_t1000"]).all()


@pytest.mark.skipif(not READY, reason="needs the lme-test stage's encoded histories")
def test_an_unchanged_simulation_is_reused(tmp_path, monkeypatch, capsys):
    """run() reuses saved results whose inputs are unchanged (the fingerprint), and simulates
    again when an input changes (here: the expansions replayed)."""
    monkeypatch.setattr(simulation, "SIM", tmp_path)
    first = simulation.run("gemma300m-raw", "longmemeval_s", "seg500", "tiny", [], (3,))
    again = simulation.run("gemma300m-raw", "longmemeval_s", "seg500", "tiny", [], (3,))
    assert "simulation current" in capsys.readouterr().out
    assert first.shape == again.shape and set(first.ranker) == set(again.ranker)
    other = simulation.run("gemma300m-raw", "longmemeval_s", "seg500", "tiny", [], (1,))
    assert set(other.expand) == {0, 1}
