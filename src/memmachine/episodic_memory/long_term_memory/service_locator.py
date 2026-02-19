"""Helpers for building long-term memory from configuration."""

from pydantic import InstanceOf

from memmachine.common.configuration.episodic_config import LongTermMemoryConf
from memmachine.common.resource_manager import CommonResourceManager

from .long_term_memory import LongTermMemoryParams


async def long_term_memory_params_from_config(
    config: LongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> LongTermMemoryParams:
    """Build LongTermMemory parameters from configuration and resources."""
    collection = await resource_manager.get_vector_store_collection(
        config.vector_store,
    )
    segment_store = await resource_manager.get_segment_store(
        config.segment_store,
    )
    embedder = await resource_manager.get_embedder(config.embedder, validate=True)
    reranker = await resource_manager.get_reranker(config.reranker, validate=True)
    return LongTermMemoryParams(
        session_key=config.session_key,
        collection=collection,
        segment_store=segment_store,
        embedder=embedder,
        reranker=reranker,
        message_sentence_chunking=config.message_sentence_chunking,
    )
