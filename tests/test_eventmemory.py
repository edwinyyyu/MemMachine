"""rbench.eventmemory's copies give exactly MemMachine's outputs (fixtures from scripts/eventmemory_fixtures.py)."""

import datetime
import json
from pathlib import Path

from rbench import eventmemory as em

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "eventmemory.json").read_text())


def segs(case):
    return {s["id"]: em.Segment(s["id"], s["event"], s["index"], s["offset"], datetime.datetime.fromisoformat(s["timestamp"]),
                                s["producer"], s["text"]) for s in case["segments"]}


def test_rendering_matches_memmachine():
    for case in FIXTURES["cases"]:
        by_id = segs(case)
        contexts = [[by_id[i] for i in c["ids"]] for c in case["contexts"]]
        assert [em.string_from_segment_context(c) for c in contexts] == case["rendered_one"]
        assert em.string_from_segment_contexts(contexts) == case["rendered_all"]


def test_admission_matches_memmachine_and_the_prefix_model():
    for case in FIXTURES["cases"]:
        by_id = segs(case)
        contexts = [[by_id[i] for i in c["ids"]] for c in case["contexts"]]
        seeds = [c["seed"] for c in case["contexts"]]
        admitted = sorted(s.id for s in em.build_query_result_context(contexts, seeds, case["limit"]))
        assert admitted == case["admitted"]
        assert sorted(s.id for s in em.admission_order(contexts, seeds)[: case["limit"]]) == admitted


def test_embedding_text_matches_memmachine():
    for case in FIXTURES["cases"]:
        by_id = segs(case)
        assert [em.derivative_text(by_id[s["id"]]) for s in case["segments"][:5]] == case["derivative"]


def test_segmenter_matches_memmachine():
    for case in FIXTURES["segmenter"]:
        assert em.segment_text(case["text"], case["max"]) == case["chunks"]


def test_id_admission_order_matches_segment_version():
    for case in FIXTURES["cases"]:
        by_id = segs(case)
        contexts = [c["ids"] for c in case["contexts"]]
        seeds = [c["seed"] for c in case["contexts"]]
        assert em.admission_order_ids(contexts, seeds) == [s.id for s in em.admission_order(
            [[by_id[i] for i in c] for c in contexts], seeds)]
