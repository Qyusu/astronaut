from enum import Enum
from typing import ClassVar, Literal

from pydantic import BaseModel


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMConfig(BaseModel):
    provider: LLMProvider
    api_key: str
    default_model_version: str

    EXCLUDE_FROM_EXPORT: ClassVar[set[str]] = {"provider"}

    def to_dict(self) -> dict:
        return self.model_dump(exclude=self.EXCLUDE_FROM_EXPORT)


class OpenAIConfig(LLMConfig):
    provider: LLMProvider = LLMProvider.OPENAI
    reasoning_effort: Literal["low", "medium", "high"] = "high"


class AnthropicConfig(LLMConfig):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    thinking_model_max_tokens: int = 64000
    basic_model_max_tokens: int = 8192
    max_thinking_budget_tokens: int = 20000


class GoogleConfig(LLMConfig):
    provider: LLMProvider = LLMProvider.GOOGLE
    project_id: str | None = None
