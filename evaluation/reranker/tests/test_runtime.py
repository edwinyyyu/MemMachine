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
    assert r == ["first_stage", "bm25", "rrf:identity+bm25", "q4-mem", "rrf:identity+bm25+q4-mem"]
    assert runtime.label("rrf:identity+bm25+q4-mem") == "RRF [identity, BM25, q4-mem]"
