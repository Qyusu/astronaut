import os
from typing import Any

import pytest
from pytest_mock import MockFixture

from astronaut.llm.chat import ChatClient

# Test execution not tracked by LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture
def openai_client(mocker: MockFixture) -> ChatClient:
    """Fixture for initializing the ChatClient with a mocked OpenAI client."""
    mock_openai = mocker.patch("astronaut.llm.chat.ChatClient")

    def mock_parse_call(model: str, *args: Any, **kwargs: Any) -> Any:
        return mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="Test response"))],
            usage=mocker.MagicMock(prompt_tokens=100, completion_tokens=50),
        )

    mock_openai.return_value.beta.chat.completions.parse.side_effect = mock_parse_call

    return ChatClient(
        platform="openai",
        api_key="test_api_key",
        default_model_version="gpt-4o-2024-11-20",
    )


@pytest.fixture
def google_client(mocker: MockFixture) -> ChatClient:
    """Fixture for initializing the ChatClient with a mocked Google client."""
    mock_google = mocker.patch("astronaut.llm.chat.ChatClient")

    def mock_parse_call(model: str, *args: Any, **kwargs: Any) -> Any:
        return mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="Test response"))],
            usage=mocker.MagicMock(prompt_tokens=100, completion_tokens=50),
        )

    mock_google.return_value.beta.chat.completions.parse.side_effect = mock_parse_call

    return ChatClient(
        platform="google",
        api_key="test_api_key",
        default_model_version="gemini-2.0-flash-001",
    )


@pytest.fixture
def anthropic_client(mocker: MockFixture) -> ChatClient:
    """Fixture for initializing the ChatClient with a mocked Anthropic client."""
    mock_anthropic = mocker.patch("astronaut.llm.chat.ChatClient")

    def mock_parse_call(model: str, *args: Any, **kwargs: Any) -> Any:
        return mocker.MagicMock(
            choices=[mocker.MagicMock(message=mocker.MagicMock(content="Test response"))],
            usage=mocker.MagicMock(prompt_tokens=100, completion_tokens=50),
        )

    mock_anthropic.return_value.beta.chat.completions.parse.side_effect = mock_parse_call

    return ChatClient(
        platform="anthropic",
        api_key="test_api_key",
        default_model_version="claude-3-5-haiku-20241022",
    )


def test_initialize_openai_client(openai_client: ChatClient) -> None:
    """Test OpenAI client initialization."""
    assert openai_client.client is not None
    assert openai_client.platform == "openai"


def test_initialize_google_client(google_client: ChatClient) -> None:
    """Test Google client initialization."""
    assert google_client.client is not None
    assert google_client.platform == "google"


def test_initialize_anthropic_client(anthropic_client: ChatClient) -> None:
    """Test Anthropic client initialization."""
    assert anthropic_client.client is not None
    assert anthropic_client.platform == "anthropic"


def test_initialize_client_invalid_platform() -> None:
    """Test initialization with invalid platform."""
    with pytest.raises(ValueError, match="Invalid platform"):
        ChatClient(
            platform="invalid",
            api_key="test_api_key",
            default_model_version="gpt-4o-2024-11-20",
        )
