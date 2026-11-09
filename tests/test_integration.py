"""End to end on real prepared data (skipped when the data is not present): simulate and report."""

import numpy as np
import pytest

from rbench import retrieval, simulate
from rbench.paths import CHUNKS, PREPARED

READY = (PREPARED / "conditionalqa" / "queries.jsonl").exists() and (CHUNKS / "conditionalqa" / "rec64.npz").exists() \
    and retrieval.path("gemma300m", "conditionalqa", "corpus", "rec64", "scenario_question").exists()


@pytest.mark.skipif(not READY, reason="needs prepared ConditionalQA with rec64 pools")
def test_simulate_first_stage_and_references_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(simulate, "SIM", tmp_path)
    df = simulate.run("gemma300m", "conditionalqa", "corpus", "scenario_question", "tiny", "rec64", [])
    kinds = set(df.kind)
    assert {"first_stage", "oracle"} <= kinds and set(df.expand) == {0, 3, 15}
    one = df[df.ranker == "first_stage"]
    assert len(one) == 8 and one.cluster.notna().all()
    # within its pool (8,192 head tokens) the gold-first reference never does worse than the first
    # stage; it reads nothing beyond the pool, so compare where the first stage finds gold inside it
    ref = df[df.ranker == "oracle@t8192"].sort_values("qid")["first"].to_numpy()
    base = one.sort_values("qid")["first"].to_numpy()
    inside = base <= 8192
    assert np.all(ref[inside] <= base[inside])
    # expanded reading yields defined recall for every query
    assert np.isfinite(df[df.ranker == "first_stage+x3"]["recall_t1000"]).all()
