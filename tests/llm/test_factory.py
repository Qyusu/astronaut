import pytest

from astronaut.llm.config import LLMProvider, OpenAIConfig
from astronaut.llm.factory import LLMClientFactory
from astronaut.llm.providers import (
    AnthropicChatClient,
    GoogleChatClient,
    OpenAIChatClient,
)


def test_create_openai_client(openai_config: OpenAIConfig) -> None:
    """Test creating an OpenAI client."""
    client = LLMClientFactory.create(openai_config)
    assert isinstance(client, OpenAIChatClient)


def test_create_anthropic_client(anthropic_config: OpenAIConfig) -> None:
    """Test creating an Anthropic client."""
    client = LLMClientFactory.create(anthropic_config)
    assert isinstance(client, AnthropicChatClient)


def test_create_google_client(google_config: OpenAIConfig) -> None:
    """Test creating a Google client."""
    client = LLMClientFactory.create(google_config)
    assert isinstance(client, GoogleChatClient)


def test_create_unsupported_provider() -> None:
    """Test creating a client with an unsupported provider."""

    class UnsupportedConfig(OpenAIConfig):
        provider: LLMProvider = "unsupported"  # type: ignore

    config = UnsupportedConfig(
        api_key="test_api_key",
        default_model_version="test-model",
    )

    with pytest.raises(ValueError) as exc_info:
        LLMClientFactory.create(config)
    assert "Unsupported LLM provider" in str(exc_info.value)
