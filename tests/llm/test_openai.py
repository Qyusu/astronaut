from typing import Any, Dict, cast

import pytest
import pytest_mock
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from pydantic import BaseModel
from pytest_mock import MockFixture

from astronaut.llm.openai import OpenAIChatClient


class TestResponse(BaseModel):
    """Test response model for testing."""

    content: str


@pytest.fixture
def mock_openai_client(mocker: MockFixture) -> Any:
    """Fixture for mocking OpenAI client."""
    mock_client = mocker.patch("openai.OpenAI")
    yield mock_client


@pytest.fixture
def openai_chat_client(mock_openai_client: Any) -> OpenAIChatClient:
    """Fixture for initializing the OpenAIChatClient with a mocked OpenAI client."""
    return OpenAIChatClient(api_key="test_api_key", default_model_version="gpt-4o-2024-11-20")


@pytest.fixture
def mock_sleep(mocker: MockFixture) -> None:
    """Fixture to mock time.sleep to prevent actual waiting during tests."""
    mocker.patch("time.sleep")


def test_construct_message(openai_chat_client: OpenAIChatClient) -> None:
    """Test message construction for different model types."""
    system_prompt = {"content": "You are a helpful assistant"}
    user_prompt = {"content": "Hello"}
    message_history = []

    # GPT model case
    messages = openai_chat_client._construct_message("gpt", system_prompt, user_prompt, message_history)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Reasoning model case
    messages = openai_chat_client._construct_message("reasoning", system_prompt, user_prompt, message_history)
    assert len(messages) == 2
    assert messages[0]["role"] == "developer"
    assert messages[1]["role"] == "user"


def test_update_history(openai_chat_client: OpenAIChatClient) -> None:
    """Test updating message history."""
    user_prompt = {"content": "Hello"}
    message_history = []
    content = "Hi there!"

    updated_history = openai_chat_client._update_history(user_prompt, message_history, content)
    updated_history = cast(list[Dict[str, str]], updated_history)

    assert len(updated_history) == 2
    assert updated_history[0]["role"] == "user"
    assert updated_history[1]["role"] == "assistant"
    assert updated_history[0]["content"] == "Hello"
    assert updated_history[1]["content"] == "Hi there!"


def test_parse_response(openai_chat_client: OpenAIChatClient) -> None:
    """Test parsing response from OpenAI completion."""
    mock_message = ChatCompletionMessage(role="assistant", content="Test response")
    mock_choice = Choice(message=mock_message, finish_reason="stop", index=0)
    mock_completion = ChatCompletion(
        id="test-id",
        model="gpt-4o",
        choices=[mock_choice],
        created=1234567890,
        object="chat.completion",
    )

    response = openai_chat_client._parse_response(mock_completion)
    assert response == "Test response"


def test_get_token_count(openai_chat_client: OpenAIChatClient) -> None:
    """Test token count calculation for different usage scenarios."""
    # when usage is None
    mock_completion = ChatCompletion(
        id="test-id", model="gpt-4o", choices=[], created=1234567890, object="chat.completion", usage=None
    )

    input_tokens, cached_tokens, output_tokens = openai_chat_client._get_token_count(mock_completion)
    assert input_tokens == 0
    assert cached_tokens == 0
    assert output_tokens == 0

    # when prompt_tokens_details is None
    mock_usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    mock_completion = ChatCompletion(
        id="test-id", model="gpt-4o", choices=[], created=1234567890, object="chat.completion", usage=mock_usage
    )

    input_tokens, cached_tokens, output_tokens = openai_chat_client._get_token_count(mock_completion)
    assert input_tokens == 10
    assert cached_tokens == 0
    assert output_tokens == 20

    # when prompt_tokens_details is not None
    mock_usage = CompletionUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=5),
    )
    mock_completion = ChatCompletion(
        id="test-id", model="gpt-4o", choices=[], created=1234567890, object="chat.completion", usage=mock_usage
    )

    input_tokens, cached_tokens, output_tokens = openai_chat_client._get_token_count(mock_completion)
    assert input_tokens == 10
    assert cached_tokens == 5
    assert output_tokens == 20


@pytest.mark.parametrize("mocker", [pytest_mock.mocker], indirect=True)
def test_parse_chat_gpt(mocker: MockFixture, openai_chat_client: OpenAIChatClient, mock_sleep: None) -> None:
    """Test parsing chat completion for GPT model."""
    mock_chat = mocker.patch("astronaut.llm.openai.OpenAIChatClient._chat")
    mock_message = ChatCompletionMessage(role="assistant", content="Test response")
    mock_choice = Choice(message=mock_message, finish_reason="stop", index=0)
    mock_completion = ChatCompletion(
        id="test-id",
        model="gpt-4o",
        choices=[mock_choice],
        created=1234567890,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    mock_chat.return_value = mock_completion

    system_prompt = {"content": "You are a helpful assistant"}
    user_prompt = {"content": "Hello"}
    response, history, cost = openai_chat_client.parse_chat(
        system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7, n=1
    )

    assert response == "Test response"
    assert len(history) == 2
    assert cost > 0
    mock_chat.assert_called_once()


@pytest.mark.parametrize("mocker", [pytest_mock.mocker], indirect=True)
def test_parse_chat_reasoning(mocker: MockFixture, openai_chat_client: OpenAIChatClient, mock_sleep: None) -> None:
    """Test parsing chat completion for reasoning model."""
    mock_chat_reasoning = mocker.patch("astronaut.llm.openai.OpenAIChatClient._chat_reasoning_model")
    mock_message = ChatCompletionMessage(role="assistant", content="Test response")
    mock_choice = Choice(message=mock_message, finish_reason="stop", index=0)
    mock_completion = ChatCompletion(
        id="test-id",
        model="o1-mini",
        choices=[mock_choice],
        created=1234567890,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    mock_chat_reasoning.return_value = mock_completion

    system_prompt = {"content": "You are a helpful assistant"}
    user_prompt = {"content": "Hello"}
    response, history, cost = openai_chat_client.parse_chat(
        system_prompt=system_prompt, user_prompt=user_prompt, model_version="o1-mini", reasoning_effort="high"
    )

    assert response == "Test response"
    assert len(history) == 2
    assert cost > 0
    mock_chat_reasoning.assert_called_once()
