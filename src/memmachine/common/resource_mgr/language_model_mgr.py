"""
Builder for LanguageModel instances.
"""

from typing import Dict, Self

from memmachine.common.configuration.model_conf import LanguageModelConf
from memmachine.common.language_model.amazon_bedrock_language_model import (
    AmazonBedrockLanguageModel,
)
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.language_model.openai_compatible_language_model import (
    OpenAICompatibleLanguageModel,
)
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel


class LanguageModelMgr:
    def __init__(self, conf: LanguageModelConf):
        self.conf = conf
        self.openai_model: Dict[str, OpenAILanguageModel] = {}
        self.aws_bedrock_model: Dict[str, AmazonBedrockLanguageModel] = {}
        self.openai_compatible_model: Dict[str, OpenAICompatibleLanguageModel] = {}

    def build_all(self) -> Dict[str, LanguageModel]:
        self._build_openai_model()
        self._build_aws_bedrock_model()
        self._build_openai_compatible_model()
        return self.language_models

    def get_language_model(self, name: str) -> LanguageModel:
        if name not in self.language_models:
            raise ValueError(f"Language model with name {name} not found.")
        return self.language_models[name]

    @property
    def language_models(self) -> Dict[str, LanguageModel]:
        all_models: Dict[str, LanguageModel] = {}
        all_models.update(self.openai_model)
        all_models.update(self.aws_bedrock_model)
        all_models.update(self.openai_compatible_model)
        return all_models

    def get_model(self, name: str) -> LanguageModel:
        if name in self.openai_model:
            return self.openai_model[name]
        if name in self.aws_bedrock_model:
            return self.aws_bedrock_model[name]
        if name in self.openai_compatible_model:
            return self.openai_compatible_model[name]
        raise ValueError(f"Language model with name {name} not found.")

    def _build_openai_model(self):
        for name, conf in self.conf.openaiConfs.items():
            self.openai_model[name] = OpenAILanguageModel(conf)

    def _build_aws_bedrock_model(self):
        for name, conf in self.conf.awsBedrockConfs.items():
            self.aws_bedrock_model[name] = AmazonBedrockLanguageModel(conf)

    def _build_openai_compatible_model(self):
        for name, conf in self.conf.openaiCompatibleConfs.items():
            self.openai_compatible_model[name] = OpenAICompatibleLanguageModel(conf)
