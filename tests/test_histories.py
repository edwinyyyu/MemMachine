"""The history pipeline on real LongMemEval-S data (skipped when it is not prepared)."""

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
    from rbench import eventmemory as em

    for q in _tiny():
        seg, producers = hs.build_segments("longmemeval_s", q["qid"], 500)
        keys = [seg.segment(i, producers).key for i in range(len(seg))]
        assert keys == sorted(keys)  # ids are EventMemory's (timestamp, event, index, offset) order
        assert np.all(np.diff(seg.event) >= 0) and all(t.strip() for t in seg.texts)
        firsts = np.flatnonzero(np.r_[True, np.diff(seg.event) > 0])
        assert len({seg.timestamps[i] for i in firsts}) == len(firsts)  # every turn has its own timestamp
        turns, sessions = hs.gold_segments("longmemeval_s", q["qid"], seg)
        assert len(turns) == len(q["gold_turns"][0]) and all(turns) and all(sessions)
        assert set(producers) <= {"User", "Assistant"}
        # the increments never show: rendered and embedded timestamps are the session's own
        from rbench.datasets.longmemeval import history, session_datetime

        h = history("longmemeval_s", q["qid"])["sessions"]
        for i in firsts[:: max(1, len(firsts) // 50)]:
            date = session_datetime(h[int(seg.session[i])]["date"])
            x = seg.segment(int(i), producers)
            assert em._header(x, em.RENDER_FORMAT) == em._header(em.Segment(0, 0, 0, 0, date, x.producer, ""), em.RENDER_FORMAT)
            assert em.derivative_text(x).split("]")[0] == em.derivative_text(
                em.Segment(0, 0, 0, 0, date, x.producer, x.text)).split("]")[0]


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_sessions_out_of_file_order_are_sorted_by_date():
    from rbench.datasets.longmemeval import history

    found = 0
    for q in queries("longmemeval_s"):  # 199 of the 470 S histories are out of date order (from the 216th on)
        sessions = history("longmemeval_s", q["qid"])["sessions"]
        order = hs.chronological(sessions)
        if order != list(range(len(sessions))) and found < 3:
            found += 1
            seg, _ = hs.build_segments("longmemeval_s", q["qid"], 500)
            seen = list(dict.fromkeys(seg.session.tolist()))
            assert seen == [s for s in order if sessions[s]["turns"]]
    assert found  # the files do have histories out of date order


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_windows_cross_into_the_next_session_and_render_with_dates():
    q = queries("longmemeval_s")[0]
    seg, producers = hs.build_segments("longmemeval_s", q["qid"], 500)
    first = int(seg.session[0])
    last_of_first_session = int(np.flatnonzero(seg.session == first).max())
    w = hs.window(last_of_first_session, len(seg), 3)
    assert len(set(seg.session[w])) == 2 and seg.session[w[-1]] == seg.session[last_of_first_session + 1]
    text = hs.unit_text(seg, producers, w)
    assert text.startswith("[") and text.count("] User: ") + text.count("] Assistant: ") >= 2


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_cached_vectors_and_pools_from_other_segments_are_refused(tmp_path, monkeypatch):
    q = queries("longmemeval_s")[0]
    monkeypatch.setattr(hs, "VECTORS", tmp_path)
    seg, _ = hs.build_segments("longmemeval_s", q["qid"], 500)
    path = hs.vector_path("gemma300m", "longmemeval_s", 500, q["qid"])
    path.parent.mkdir(parents=True)
    np.savez(path, vectors=np.zeros((len(seg), 4), np.float16), fingerprint=np.str_(hs._vector_fingerprint(seg, "gemma300m")))
    assert hs.embed_history("gemma300m", "longmemeval_s", q["qid"], 500).shape == (len(seg), 4)
    np.savez(path, vectors=np.zeros((len(seg), 4), np.float16), fingerprint=np.str_("other"))
    with pytest.raises(RuntimeError, match="other segments"):
        hs.embed_history("gemma300m", "longmemeval_s", q["qid"], 500)
    np.savez(path, vectors=np.zeros((len(seg), 4), np.float16))  # no fingerprint at all
    with pytest.raises(RuntimeError, match="other segments"):
        hs.embed_history("gemma300m", "longmemeval_s", q["qid"], 500)
    from rbench.retrieval import Pools

    pools = hs.HistoryPools(Pools(np.asarray([q["qid"]]), np.asarray([0, 1]), np.asarray([0]), np.asarray([1.0])),
                            np.asarray(["other"]), tmp_path / "pools.npz")
    with pytest.raises(RuntimeError, match="other segments"):
        pools.head(0, seg)


@pytest.mark.skipif(not (SIM / "gemma300m/longmemeval_s/history/question/tiny/seg500__default__em.parquet").exists(),
                    reason="needs the lme-test stage's simulation")
def test_simulated_rows_are_consistent():
    import pandas as pd

    d = pd.read_parquet(SIM / "gemma300m/longmemeval_s/history/question/tiny/seg500__default__em.parquet")
    assert (d.first_touch <= d["first"]).all() and (d.every_touch <= d.every).all()  # touched never later than complete
    assert (d["first"] <= d.every).all()
    assert {"sess_any_k", "sess_all_k", "admitted", "question_type"} <= set(d.columns)
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
    assert (limited.admitted <= 40).all() and (limited.ctx_t16384 <= 10).all()
    for e in ("first", "every"):
        found = np.isfinite(d[f"{e}_k"])
        assert (d.loc[found, f"{e}_ctx"] <= d.loc[found, f"{e}_k"]).all()  # a seed admits at least one segment
        assert np.isinf(d.loc[~found, f"{e}_ctx"]).all()
    plain = d[d.ranker == "first_stage"]
    assert (plain.segs_t1024 == plain.ctx_t1024).all()  # no expansion: one segment per seed
    assert (np.diff(d[[f"segs_t{b}" for b in (128, 256, 512, 1024)]].to_numpy(), axis=1) >= 0).all()


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_an_interrupted_embedding_resumes_from_its_saved_shards(tmp_path, monkeypatch):
    from rbench import embedding

    q = queries("longmemeval_s")[0]
    monkeypatch.setattr(hs, "VECTORS", tmp_path)
    monkeypatch.setattr(hs, "SHARD", 300)
    seg, _ = hs.build_segments("longmemeval_s", q["qid"], 500)
    calls, failed = [], []

    def fake(spec, texts, prompt):
        calls.append(len(texts))
        if len(calls) == 3 and not failed:
            failed.append(True)
            raise RuntimeError("network partition")
        return np.ones((len(texts), 4), np.float32)

    monkeypatch.setattr(embedding, "embed", fake)
    with pytest.raises(RuntimeError, match="partition"):
        hs.embed_history("gemma300m-q", "longmemeval_s", q["qid"], 500)
    done = len(calls) - 1  # shards saved before the failure
    calls.clear()
    vectors = hs.embed_history("gemma300m-q", "longmemeval_s", q["qid"], 500)
    assert vectors.shape == (len(seg), 4)
    assert len(calls) == -(-len(seg) // 300) - done  # only the shards not yet saved were embedded again
    # the "-raw" setup shares the document vectors (same server, same empty document prompt)
    assert hs.embed_history("gemma300m-raw", "longmemeval_s", q["qid"], 500).shape == (len(seg), 4)


@pytest.mark.skipif(not READY, reason="needs prepared LongMemEval-S")
def test_the_prompt_comparison_haystack_keeps_every_gold_turn_in_eventmemory_form():
    from rbench import compare_embedders as ce

    for q in _tiny()[:3]:
        texts, gold = ce.haystack("longmemeval_s", q, 500)
        seg, producers = hs.build_segments("longmemeval_s", q["qid"], 500)
        full = set(hs.derivative_texts(seg, producers))
        assert len(gold) == len(q["gold_turns"][0]) and all(gold)
        assert set(texts) <= full  # the same embedding texts as the whole history (dates are unrendered-equal)
        assert all(texts[i].split("] ", 1)[1].startswith(("User:", "Assistant:")) for turn in gold for i in turn)
