from typing import Any, List, cast

import pytest
import pytest_mock
from anthropic.types import Message, TextBlock, TextBlockParam, Usage
from pytest_mock import MockFixture

from astronaut.llm.anthropic import AnthropicChatClient


@pytest.fixture
def mock_anthropic_client(mocker: MockFixture) -> Any:
    """Fixture for mocking Anthropic client."""
    mock_client = mocker.patch("anthropic.Anthropic")
    yield mock_client


@pytest.fixture
def anthropic_chat_client(mock_anthropic_client: Any) -> AnthropicChatClient:
    """Fixture for initializing the AnthropicChatClient with a mocked Anthropic client."""
    return AnthropicChatClient(api_key="test_api_key", default_model_version="claude-3-opus-20240229")


@pytest.fixture
def mock_sleep(mocker: MockFixture) -> None:
    """Fixture to mock time.sleep to prevent actual waiting during tests."""
    mocker.patch("time.sleep")


@pytest.fixture
def mock_parse_client(mocker: MockFixture) -> Any:
    """Fixture for mocking parse_client."""
    mock_client = mocker.patch("astronaut.llm.anthropic.parse_client")
    yield mock_client


def test_construct_message(anthropic_chat_client: AnthropicChatClient) -> None:
    """Test message construction for Anthropic models."""
    user_prompt = {"content": "Hello"}
    message_history = []

    messages = anthropic_chat_client._construct_message(user_prompt, message_history)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert len(messages[0]["content"]) == 1
    content = cast(List[TextBlockParam], messages[0]["content"])
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Hello"


def test_update_history(anthropic_chat_client: AnthropicChatClient) -> None:
    """Test updating message history."""
    user_prompt = {"content": "Hello"}
    message_history = []
    content = "Hi there!"

    updated_history = anthropic_chat_client._update_history(user_prompt, message_history, content)
    assert len(updated_history) == 2
    assert updated_history[0]["role"] == "user"
    assert updated_history[1]["role"] == "assistant"
    assert isinstance(updated_history[0]["content"], str)
    assert isinstance(updated_history[1]["content"], str)
    assert updated_history[0]["content"] == "Hello"
    assert updated_history[1]["content"] == "Hi there!"


def test_get_model_name_from_version(anthropic_chat_client: AnthropicChatClient) -> None:
    """Test extracting base model name from version string."""
    model_name = anthropic_chat_client._get_model_name_from_version("claude-3-opus-20240229")
    assert model_name == "claude-3-opus"


def test_get_token_count(anthropic_chat_client: AnthropicChatClient) -> None:
    """Test token count calculation for different usage scenarios."""
    mock_message = Message(
        id="test-id",
        model="claude-3-opus",
        role="assistant",
        type="message",
        content=[TextBlock(type="text", text="Test response")],
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    input_tokens, cached_tokens, output_tokens = anthropic_chat_client._get_token_count(mock_message)
    assert input_tokens == 10
    assert cached_tokens == 0  # Anthropic doesn't provide cached_tokens
    assert output_tokens == 20


def test_parse_response(anthropic_chat_client: AnthropicChatClient, mock_sleep: None, mock_parse_client: Any) -> None:
    """Test parsing response from Anthropic completion."""
    mock_message = Message(
        id="test-id",
        model="claude-3-opus",
        role="assistant",
        type="message",
        content=[TextBlock(type="text", text="Test response")],
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    mock_parse_client.parse_chat.return_value = ("Test response", [], 0.0)
    response = anthropic_chat_client._parse_response(mock_message, response_format=None)
    assert response == "Test response"
    mock_parse_client.parse_chat.assert_called_once()


@pytest.mark.parametrize("mocker", [pytest_mock.mocker], indirect=True)
def test_parse_chat(
    mocker: MockFixture, anthropic_chat_client: AnthropicChatClient, mock_sleep: None, mock_parse_client: Any
) -> None:
    """Test parsing chat completion for Anthropic model."""
    mock_chat = mocker.patch.object(anthropic_chat_client.client.messages, "create")
    mock_message = Message(
        id="test-id",
        model="claude-3-opus",
        role="assistant",
        type="message",
        content=[TextBlock(type="text", text="Test response")],
        usage=Usage(input_tokens=10, output_tokens=20),
    )
    mock_chat.return_value = mock_message

    system_prompt = {"content": "You are a helpful assistant"}
    user_prompt = {"content": "Hello"}
    mock_parse_client.parse_chat.return_value = ("Test response", [], 0.0)
    response, history, cost = anthropic_chat_client.parse_chat(
        system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7, max_tokens=1000
    )

    assert response == "Test response"
    assert len(history) == 2
    assert cost > 0
    mock_chat.assert_called_once()


@pytest.mark.parametrize("mocker", [pytest_mock.mocker], indirect=True)
def test_parse_chat_thinking_model(
    mocker: MockFixture, anthropic_chat_client: AnthropicChatClient, mock_sleep: None, mock_parse_client: Any
) -> None:
    """Test parsing chat completion for thinking model."""
    mock_chat_thinking = mocker.patch.object(anthropic_chat_client.client.messages, "create")
    mock_message = Message(
        id="test-id",
        model="claude-3-opus",
        role="assistant",
        type="message",
        content=[TextBlock(type="text", text="Test response")],
        usage=Usage(input_tokens=10, output_tokens=20),
    )
    mock_chat_thinking.return_value = mock_message

    system_prompt = {"content": "You are a helpful assistant"}
    user_prompt = {"content": "Hello"}
    mock_parse_client.parse_chat.return_value = ("Test response", [], 0.0)
    response, history, cost = anthropic_chat_client.parse_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_version="claude-3-opus-20240229",
        max_thinking_tokens=20000,
    )

    assert response == "Test response"
    assert len(history) == 2
    assert cost > 0
    mock_chat_thinking.assert_called_once()
