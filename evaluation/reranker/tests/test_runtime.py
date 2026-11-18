"""The runtime replay copies MemMachine: LongTermMemory's episode dedup, EventMemory's stable sort."""

from rbench import runtime


def test_the_first_k_distinct_episodes_in_score_order():
    # contexts 0..5 belong to episodes 7, 7, 3, 9, 3, 5; ranked 1, 0, 2, 3, 4, 5
    episode_of = [7, 7, 3, 9, 3, 5]
    assert runtime.first_k_episodes([1, 0, 2, 3, 4, 5], episode_of, 3) == [7, 3, 9]  # the second 7 and 3 skipped
    assert runtime.first_k_episodes([1, 0, 2, 3, 4, 5], episode_of, 10) == [7, 3, 9, 5]  # fewer than k: all distinct


def test_ties_keep_vector_search_order_as_eventmemory_sorts():
    assert runtime.by_score([1.0, 2.0, 2.0, 0.5, 2.0]) == [1, 2, 4, 0, 3]


def test_the_overfetch_is_memmachines_and_the_configurations_cover_both_rrf_hybrids():
    from memmachine_server.episodic_memory.long_term_memory import long_term_memory

    assert runtime.OVERFETCH == long_term_memory._EVENT_BACKEND_DEDUP_OVERFETCH == 4
    assert runtime.OVERFETCH * max(runtime.KS) <= runtime.POOL
    r = runtime.rankers_for(["q4-mem"])
    assert r == ["first_stage", "bm25", "rrf:identity+bm25", "q4-mem", "rrf:identity+bm25+q4-mem",
                 "rrf:identity+q4-mem", "rrf:bm25+q4-mem"]  # every subset of {identity, bm25} with the cross-encoder
    assert runtime.label("rrf:identity+bm25+q4-mem") == "RRF [identity, BM25, q4-mem]"
    assert runtime.label("rrf:bm25+q4-mem") == "RRF [BM25, q4-mem]"
    assert runtime.label("rrf:identity+bm25") == "RRF [identity, BM25]"


def test_k_segments_counts_a_turn_only_when_all_its_segments_are_kept():
    seeds = [10, 11, 12, 13]  # candidate i is segment seeds[i]
    spans = [(11,), (12, 13)]  # turn A is one segment; turn B is split into two
    assert runtime.turns_in_full([1, 2, 0, 3], seeds, spans, 2) == [True, False]  # 13 not kept: B not in full
    assert runtime.turns_in_full([1, 2, 0, 3], seeds, spans, 4) == [True, True]


def test_the_pools_and_the_input_cache_key(tmp_path):
    from types import SimpleNamespace

    assert [runtime.limit_of(p, 5) for p in runtime.POOLS] == [20, 20, 200]  # k = 5: 4 x k is EventMemory's 20
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / "h1").write_text("stamp-1")
    m = SimpleNamespace(root=tmp_path, embedder="e", width=None, done=lambda key: tmp_path / "done" / key)
    a = runtime._inputs_path(m, "h1", "q1", "When?", "fmt")
    assert a == runtime._inputs_path(m, "h1", "q1", "When?", "fmt") and a.parent == tmp_path / "runtime_inputs" / "h1"
    assert a != runtime._inputs_path(m, "h1", "q2", "When?", "fmt")
    (tmp_path / "done" / "h1").write_text("stamp-2")  # the partition re-encoded: the cache no longer applies
    assert a != runtime._inputs_path(m, "h1", "q1", "When?", "fmt")


def test_the_outputs_stay_clear_of_the_core_pipelines_clean_ups():
    """tables.clear removes <run>__*.md in summary/<embedder>/, and figures.write empties
    figures/<embedder>/<run>/ before drawing: the replay's outputs live in folders of their own."""
    from rbench.paths import FIGURES, SUMMARY

    run = "longmemeval_m__history__question__all"
    assert runtime.summary_dir("e") != SUMMARY / "e"
    assert not (runtime.summary_dir("e") / f"{run}__runtime_topk.md").match(str(SUMMARY / "e" / f"{run}__*.md"))
    assert runtime.figures_dir("e", "longmemeval_m", "all") != FIGURES / "e" / run
