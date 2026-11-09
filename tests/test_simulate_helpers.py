"""Bootstrap clusters come from each dataset's registry entry."""

from rbench.simulate import cluster_of, expansions_for


def test_clusters_follow_the_registry():
    assert cluster_of("qasper", {"doc_id": "p1", "features": {}}) == "p1"
    assert cluster_of("multidoc2dial", {"features": {"dialogue_id": "d7"}}) == "d7"
    assert cluster_of("swebench_lite", {"features": {"instance_id": "x-1"}}) == "x-1"


def test_expansions_are_replayed_only_up_to_the_window_cap():
    assert expansions_for("rec64") == (3,)
    assert expansions_for("rec256") == (3,)
    assert expansions_for("rec1024") == ()
