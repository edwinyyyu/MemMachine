from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..data_types import Episode as DeclarativeEpisode

class DeclarativeEpisodeStore(ABC):
    @abstractmethod
    async def add_episodes(session_id: str, episodes: Iterable[DeclarativeEpisode]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_episodes(episode_uids: Iterable[str]) -> list[DeclarativeEpisode]:
        raise NotImplementedError
