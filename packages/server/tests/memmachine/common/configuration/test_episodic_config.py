from typing import Any

import pytest
import yaml

from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
)


@pytest.fixture
def episodic_memory_conf() -> dict[str, Any]:
    return {
        "long_term_memory": {
            "embedder": "my_embedder",
            "reranker": "my_reranker",
            "vector_graph_store": "my_neo4j",
        },
        "short_term_memory": {
            "llm_model": "my_model",
            "message_capacity": 500,
        },
    }


def test_episodic_config_to_yaml(episodic_memory_conf):
    conf = EpisodicMemoryConfPartial(
        long_term_memory=episodic_memory_conf["long_term_memory"],
        short_term_memory=episodic_memory_conf["short_term_memory"],
    )
    yaml_str = conf.to_yaml()
    loaded_conf = yaml.safe_load(yaml_str)
    conf_cp = EpisodicMemoryConfPartial(
        long_term_memory=loaded_conf.get("long_term_memory"),
        short_term_memory=loaded_conf.get("short_term_memory"),
    )
    assert conf_cp == conf
    assert conf_cp.long_term_memory == conf.long_term_memory
    assert conf_cp.short_term_memory is not None
    assert conf_cp.short_term_memory.llm_model == "my_model"
    assert conf_cp.short_term_memory == conf.short_term_memory
