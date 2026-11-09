"""End to end on the lme-test stage's stored partitions (skipped when they are not encoded):
the unified simulation replays the first stage and the perfect reranker from EventMemory."""

import numpy as np
import pytest

from rbench import simulation
from rbench.memory import MEMORY

READY = (MEMORY / "gemma300m-raw" / "longmemeval_s" / "seg500" / "done").exists()


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
