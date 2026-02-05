from datetime import timedelta

from memmachine.common.configuration import SemanticMemoryConf


def test_semantic_config_with_ingestion_triggers():
    conf = SemanticMemoryConf(
        database="database",
        llm_model="llm",
        embedding_model="embedding",
        ingestion_trigger_messages=24,
        ingestion_trigger_age="PT2M",  # type: ignore[arg-type]  # Testing ISO 8601 duration parsing
        config_database="database",
    )
    assert conf.ingestion_trigger_messages == 24
    assert conf.ingestion_trigger_age == timedelta(minutes=2)


def test_semantic_config_timedelta_float():
    conf = SemanticMemoryConf(
        database="database",
        llm_model="llm",
        embedding_model="embedding",
        ingestion_trigger_messages=24,
        ingestion_trigger_age=120.5,  # type: ignore[arg-type]  # Testing float to timedelta conversion
        config_database="database",
    )
    assert conf.ingestion_trigger_messages == 24
    assert conf.ingestion_trigger_age == timedelta(minutes=2, milliseconds=500)
