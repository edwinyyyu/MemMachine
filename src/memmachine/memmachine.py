from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from .common.resource_initializer import ResourceInitializer

from .episodic_memory import EpisodicMemoryManager
from .profile_memory import ProfileMemory


class ResourceDefinition(BaseModel):
    type: str
    name: str
    config: dict[str, Any]


class MemMachineConfig(BaseModel):
    episodic_memory_manager: dict[str, Any]
    profile_memory: dict[str, Any]
    resources: dict[str, ResourceDefinition]


class MemMachine:
    def __init__(self, config: MemMachineConfig):
        resource_definitions = config.model_dump()["resources"]
        self._resources = ResourceInitializer.initialize(resource_definitions)

        self._episodic_memory_manager = EpisodicMemoryManager(
            config.episodic_memory_manager | {"resources": self._resources}
        )
        self._profile_memory = ProfileMemory(
            config.profile_memory
        )

    @property
    def episodic_memory_manager(self) -> EpisodicMemoryManager:
        return self._episodic_memory_manager

    @property
    def profile_memory(self) -> ProfileMemory:
        return self._profile_memory

    @property
    def resources(self) -> dict[str, Any]:
        return self._resources

    @staticmethod
    def load_config(config_file_path: str) -> MemMachineConfig:
        with open(config_file_path) as config_file:
            try:
                config = MemMachineConfig(**yaml.safe_load(config_file))
            except yaml.YAMLError as e:
                raise ValueError(
                    "Configuration file must contain valid YAML"
                ) from e
            except ValidationError as e:
                raise ValueError(
                    "Configuration file does not conform to the expected schema"
                ) from e

        return config

    @staticmethod
    def from_config_file_path(config_file_path: str):
        config = MemMachine.load_config(config_file_path)
        return MemMachine(config)
