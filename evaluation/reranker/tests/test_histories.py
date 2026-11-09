"""LongMemEval-S histories as MemMachine segments them (skipped when the data is not prepared)."""

import numpy as np
import pytest

from rbench import histories as hs
from rbench.paths import PREPARED, SIM
from rbench.schema import queries

READY = (PREPARED / "longmemeval_s" / "queries.jsonl").exists()


def _tiny() -> list[dict]:
    from rbench.samples import sample

    rows = queries("longmemeval_s")
    return [rows[i] for i in sample("longmemeval_s", "tiny")]


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_segments_are_in_eventmemory_order_and_gold_maps_to_segments():
    from rbench.datasets.longmemeval import history, session_datetime

    from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory  # noqa: I001

    for q in _tiny():
        seg, segments, texts = hs.offline("longmemeval_s", q["qid"], "seg500")
        keys = [(s.timestamp, s.event_uuid, s.index, s.offset) for s in segments]
        assert keys == sorted(keys)  # ids are EventMemory's (timestamp, event, index, offset) order
        assert np.all(np.diff(seg.event) >= 0) and all(t.strip() for t in seg.texts)
        firsts = np.flatnonzero(np.r_[True, np.diff(seg.event) > 0])
        assert len({segments[i].timestamp for i in firsts}) == len(firsts)  # every turn has its own timestamp
        turns = hs.gold_segments("longmemeval_s", q["qid"], seg)
        assert len(turns) == len(q["gold_turns"][0]) and all(turns)
        # the increments never show: rendered and embedded timestamps are the session's own
        h = history("longmemeval_s", q["qid"])["sessions"]
        for i in firsts[:: max(1, len(firsts) // 50)]:
            x = segments[int(i)]
            date = session_datetime(h[int(seg.session[i])]["date"])
            plain = x.model_copy(update={"timestamp": date})
            assert EventMemory.string_from_segment_context([x]) == EventMemory.string_from_segment_context([plain])
            assert texts[int(i)].split("]")[0] == f"[{date:%A, %B} {date.day}, {date.year}"


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_sessions_out_of_file_order_are_sorted_by_date():
    from rbench.datasets.longmemeval import history

    found = 0
    for q in queries("longmemeval_s"):  # 199 of the 470 S histories are out of date order (from the 216th on)
        sessions = history("longmemeval_s", q["qid"])["sessions"]
        order = hs.chronological(sessions)
        if order != list(range(len(sessions))) and found < 3:
            found += 1
            seg = hs.offline("longmemeval_s", q["qid"], "seg500")[0]
            seen = list(dict.fromkeys(seg.session.tolist()))
            assert seen == [s for s in order if any(t["content"].strip() for t in sessions[s]["turns"])]
    assert found  # the files do have histories out of date order


SIMULATED = SIM / "gemma300m-raw/longmemeval_s/history/question/tiny/seg500.parquet"


@pytest.mark.skipif(not SIMULATED.exists(), reason="needs the lme-test stage's simulation")
def test_simulated_rows_are_consistent():
    import pandas as pd

    d = pd.read_parquet(SIMULATED)
    assert (d.first_touch <= d["first"]).all() and (d.every_touch <= d.every).all()  # touched never later than complete
    assert (d["first"] <= d.every).all()
    assert {"admitted", "question_type"} <= set(d.columns) and not any(c.startswith("sess_") for c in d.columns)
    assert set(d.expand) == {0, 1, 2, 3, 6}
    # a reordered configuration reads only its pool: at most its seeds' windows (1 + expand
    # segments each), and with no expansion exactly its seeds unless reading hit the 16k cap
    pooled = d[d.kind != "first_stage"]
    pool = np.where(pooled.kind == "oracle", pooled.oracle_pairs, pooled.pairs)
    assert (pooled.admitted <= pool * (1 + pooled.expand)).all()
    plain = (pooled.expand == 0).to_numpy()
    assert (pooled.admitted.to_numpy()[plain] <= pool[plain]).all()
    assert {"rerank", "xrerank", "oracle"} <= set(pooled.kind)
    # the first stage is not limited to a pool: it reads past a shallow one
    assert (d[d.ranker == "first_stage"].admitted > 5).all() and (d[d.ranker == "rerank:q06-local@n5"].admitted <= 5).all()
    # count normalizations: contexts read when events happen, and segments / contexts within budgets
    limited = d[d.ranker == "first_stage@n10+x3"]
    assert (limited.admitted <= 40).all() and (limited.ctx_t16000 <= 10).all()
    for e in ("first", "every"):
        found = np.isfinite(d[f"{e}_k"])
        assert (d.loc[found, f"{e}_ctx"] <= d.loc[found, f"{e}_k"]).all()  # a seed admits at least one segment
        assert np.isinf(d.loc[~found, f"{e}_ctx"]).all()
    plain = d[d.ranker == "first_stage"]
    assert (plain.segs_t1000 == plain.ctx_t1000).all()  # no expansion: one segment per seed
    assert (np.diff(d[[f"segs_t{b}" for b in (250, 500, 1000, 2000)]].to_numpy(), axis=1) >= 0).all()


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_the_prompt_comparison_haystack_keeps_every_gold_turn_in_eventmemory_form():
    from rbench import compare_embedders as ce

    for q in _tiny()[:3]:
        texts, gold = ce.haystack("longmemeval_s", q, 500)
        full = set(hs.offline("longmemeval_s", q["qid"], "seg500")[2])
        assert len(gold) == len(q["gold_turns"][0]) and all(gold)
        assert set(texts) <= full  # the same embedding texts as the whole history
        assert all(texts[i].split("] ", 1)[1].startswith(("User:", "Assistant:")) for turn in gold for i in turn)
