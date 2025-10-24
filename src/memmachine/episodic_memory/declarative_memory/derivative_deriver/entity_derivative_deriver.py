"""
A derivative deriver that splits episode content into sentences
and creates derivatives for each sentence.
"""

import asyncio
from uuid import uuid4

import spacy
from nltk.corpus import stopwords
from pydantic import BaseModel, Field
from wordfreq import zipf_frequency

from ..data_types import ContentType, Derivative, EpisodeCluster
from .derivative_deriver import DerivativeDeriver

nlp = spacy.load("en_core_web_trf")


class EntityDerivativeDeriverParams(BaseModel):
    """
    Parameters for EntityDerivativeDeriver.

    Attributes:
        derivative_type (str):
            The type to assign to the derived derivatives
            (default: "entity").
        zipf_threshold (float):
            The Zipf frequency threshold for filtering entities
            (default: 5.6).
    """

    derivative_type: str = Field(
        "entity",
        description="The type to assign to the derived derivatives",
    )
    zipf_threshold: float = Field(
        5.6,
        description=("The Zipf frequency threshold for filtering entities"),
    )


class EntityDerivativeDeriver(DerivativeDeriver):
    """
    Derivative deriver that extracts entitites from episode content
    and creates derivatives for each sentence.
    """

    def __init__(self, params: EntityDerivativeDeriverParams):
        """
        Initialize a SentenceDerivativeDeriver
        with the provided parameters.

        Args:
            params (SentenceDerivativeDeriverParams):
                Parameters for the SentenceDerivativeDeriver.
        """
        super().__init__()

        self._derivative_type = params.derivative_type
        self._zipf_threshold = params.zipf_threshold

    async def derive(self, episode_cluster: EpisodeCluster) -> list[Derivative]:
        return [
            Derivative(
                uuid=uuid4(),
                derivative_type=self._derivative_type,
                content_type=ContentType.STRING,
                content=entity,
                timestamp=episode.timestamp,
                filterable_properties=episode.filterable_properties,
                user_metadata=episode.user_metadata,
            )
            for episode in episode_cluster.episodes
            for line in episode.content.splitlines()
            for entity in await EntityDerivativeDeriver._extract_entities(line)
            if zipf_frequency(entity, "en") < 5.3#self._zipf_threshold
        ]

    @staticmethod
    async def _extract_entities(text: str) -> list[str]:
        if not text.strip():
            return []

        doc = await asyncio.to_thread(nlp, text)

        entities = []
        for chunk in doc.noun_chunks:
            chunk_text = "".join(
                [token.text + token.whitespace_ for token in chunk.subtree]
            ).strip()
            if chunk_text.lower() not in stopwords.words("english"):
                entities.append(chunk_text)

        return entities
