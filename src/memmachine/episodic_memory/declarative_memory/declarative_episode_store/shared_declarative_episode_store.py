from collections.abc import Iterable

from memmachine.common.episode_store.episode_storage import EpisodeStorage

from .declarative_episode_store import DeclarativeEpisodeStore

from ..data_types import Episode as DeclarativeEpisode

class SharedDeclarativeEpisodeStore(DeclarativeEpisodeStore):
    def __init__(self, episode_storage: EpisodeStorage):
        self._episode_store = episode_storage

    async def add_episodes(session_id: str, episodes: Iterable[DeclarativeEpisode]) -> None:
        pass

    async def get_episodes(episode_uids: Iterable[str]) -> list[DeclarativeEpisode]:
        pass
