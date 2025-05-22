import os

import pytest
from pytest_mock import MockFixture

from astronaut.llm.chat import ChatClient
from astronaut.llm.config import AnthropicConfig, GoogleConfig, OpenAIConfig

# Test execution not tracked by LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture
def openai_config() -> OpenAIConfig:
    """Fixture for OpenAI configuration."""
    return OpenAIConfig(
        api_key="test_api_key",
        default_model_version="gpt-4o-2024-11-20",
        reasoning_effort="high",
    )


@pytest.fixture
def google_config() -> GoogleConfig:
    """Fixture for Google configuration."""
    return GoogleConfig(
        api_key="test_api_key",
        default_model_version="gemini-2.0-flash-001",
        project_id=None,
    )


@pytest.fixture
def anthropic_config() -> AnthropicConfig:
    """Fixture for Anthropic configuration."""
    return AnthropicConfig(
        api_key="test_api_key",
        default_model_version="claude-3-opus-20240229",
        thinking_model_max_tokens=64000,
        basic_model_max_tokens=8192,
        max_thinking_budget_tokens=20000,
    )


@pytest.fixture
def openai_client(openai_config: OpenAIConfig) -> ChatClient:
    """Fixture for OpenAI chat client."""
    return ChatClient(openai_config)


@pytest.fixture
def google_client(google_config: GoogleConfig) -> ChatClient:
    """Fixture for Google chat client."""
    return ChatClient(google_config)


@pytest.fixture
def anthropic_client(anthropic_config: AnthropicConfig) -> ChatClient:
    """Fixture for Anthropic chat client."""
    return ChatClient(anthropic_config)


@pytest.fixture
def mock_sleep(mocker: MockFixture) -> None:
    """Fixture to mock time.sleep to prevent actual waiting during tests."""
    mocker.patch("time.sleep")
