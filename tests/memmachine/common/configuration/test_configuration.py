import pytest

from memmachine.common.configuration import (
    EpisodicMemoryConf,
    EpisodicMemoryConfPartial,
    LongTermMemoryConf,
    LongTermMemoryConfPartial,
    SessionMemoryConf,
    SessionMemoryConfPartial,
)


@pytest.fixture
def long_term_memory_conf() -> LongTermMemoryConf:
    return LongTermMemoryConf(
        embedder="embedder_v1",
        reranker="reranker_v1",
        vector_graph_store="store_v1",
        enabled=True,
    )


def test_update_long_term_memory_conf(long_term_memory_conf: LongTermMemoryConf):
    update = LongTermMemoryConfPartial(
        embedder="embedder_v2",
    )

    updated = long_term_memory_conf.update(update)
    assert updated.embedder == "embedder_v2"
    assert updated.reranker == "reranker_v1"
    assert updated.vector_graph_store == "store_v1"
    assert updated.enabled is True


@pytest.fixture
def session_memory_conf() -> SessionMemoryConf:
    return SessionMemoryConf(
        model_name="model_v1",
        max_message_length=12345,
        enabled=True,
    )


def test_update_session_memory_conf(session_memory_conf: SessionMemoryConf):
    update = SessionMemoryConfPartial(
        max_token_num=3000,
        max_message_length=54321,
    )

    updated = session_memory_conf.update(update)
    assert updated.model_name == "model_v1"
    assert updated.max_token_num == 3000
    assert updated.max_message_length == 54321
    assert updated.message_capacity == 500
    assert updated.enabled is True


def test_update_episodic_memory_conf(
    long_term_memory_conf: LongTermMemoryConf, session_memory_conf: SessionMemoryConf
):
    base = EpisodicMemoryConf(
        sessionMemory=session_memory_conf,
        long_term_memory=long_term_memory_conf,
    )
    update = EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(embedder="embedder_v2")
    )

    updated = base.update(update)
    assert updated.long_term_memory.embedder == "embedder_v2"
    assert updated.long_term_memory.reranker == "reranker_v1"
    assert updated.sessionMemory.max_message_length == 12345
    assert updated.sessionMemory.message_capacity == 500
