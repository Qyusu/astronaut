import os
from typing import Any

import pytest

from astronaut.llm.chat import ChatClient
from astronaut.llm.config import AnthropicConfig, GoogleConfig, OpenAIConfig


@pytest.fixture
def openai_client() -> ChatClient:
    config = OpenAIConfig(
        api_key="test_api_key",
        default_model_version="gpt-4o-2024-11-20",
        reasoning_effort="high",
    )
    return ChatClient(config)


@pytest.fixture
def google_client() -> ChatClient:
    config = GoogleConfig(
        api_key="test_api_key",
        default_model_version="gemini-2.0-flash-001",
        project_id=None,
    )
    return ChatClient(config)


@pytest.fixture
def anthropic_client() -> ChatClient:
    config = AnthropicConfig(
        api_key="test_api_key",
        default_model_version="claude-3-5-haiku-20241022",
        thinking_model_max_tokens=64000,
        basic_model_max_tokens=8192,
        max_thinking_budget_tokens=20000,
    )
    return ChatClient(config)


def test_initialize_openai_client(openai_client: ChatClient) -> None:
    """Test OpenAI client initialization."""
    assert openai_client.client is not None


def test_initialize_google_client(google_client: ChatClient) -> None:
    """Test Google client initialization."""
    assert google_client.client is not None


def test_initialize_anthropic_client(anthropic_client: ChatClient) -> None:
    """Test Anthropic client initialization."""
    assert anthropic_client.client is not None
