from typing import Any

import pytest
import pytest_mock
from google.genai.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)
from pytest_mock import MockFixture

from astronaut.llm.google import GoogleChatClient


@pytest.fixture
def mock_google_client(mocker: MockFixture) -> Any:
    """Fixture for mocking Google client."""
    mock_client = mocker.patch("google.genai.Client")
    yield mock_client


@pytest.fixture
def google_chat_client(mock_google_client: Any) -> GoogleChatClient:
    """Fixture for initializing the GoogleChatClient with a mocked Google client."""
    return GoogleChatClient(api_key="test_api_key", default_model_version="gemini-2.0-flash-001")


@pytest.fixture
def mock_sleep(mocker: MockFixture) -> None:
    """Fixture to mock time.sleep to prevent actual waiting during tests."""
    mocker.patch("time.sleep")


def test_construct_message(google_chat_client: GoogleChatClient) -> None:
    """Test message construction for Google models."""
    user_prompt = {"role": "user", "content": "Hello"}
    message_history = []

    messages = google_chat_client._construct_message(user_prompt, message_history)
    assert isinstance(messages, str)
    assert "user: Hello" in messages


def test_update_history(google_chat_client: GoogleChatClient) -> None:
    """Test updating message history."""
    user_prompt = {"role": "user", "content": "Hello"}
    message_history = []
    content = "Hi there!"

    updated_history = google_chat_client._update_history(user_prompt, message_history, content)
    assert len(updated_history) == 2
    assert "user: Hello" in updated_history[0]
    assert "model: Hi there!" in updated_history[1]


def test_get_model_name_from_version(google_chat_client: GoogleChatClient) -> None:
    """Test extracting base model name from version string."""
    model_name = google_chat_client._get_model_name_from_version("gemini-2.0-flash-001")
    assert model_name == "gemini-2.0-flash"


def test_get_token_count(google_chat_client: GoogleChatClient) -> None:
    """Test token count calculation for different usage scenarios."""
    # when usage is None
    mock_response = GenerateContentResponse(candidates=[])

    input_tokens, cached_tokens, output_tokens = google_chat_client._get_token_count(mock_response)
    assert input_tokens == 0
    assert cached_tokens == 0
    assert output_tokens == 0

    # when usage exists
    mock_usage = GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        cached_content_token_count=5,
        candidates_token_count=20,
    )
    mock_response = GenerateContentResponse(
        candidates=[],
        usage_metadata=mock_usage,
    )

    input_tokens, cached_tokens, output_tokens = google_chat_client._get_token_count(mock_response)
    assert input_tokens == 10
    assert cached_tokens == 5
    assert output_tokens == 20


def test_parse_response(google_chat_client: GoogleChatClient) -> None:
    """Test parsing response from Google completion."""
    mock_part = Part(text="Test response")
    mock_content = Content(parts=[mock_part])
    mock_candidate = Candidate(content=mock_content)
    mock_response = GenerateContentResponse(candidates=[mock_candidate])

    response = google_chat_client._parse_response(mock_response)
    assert response == "Test response"


@pytest.mark.parametrize("mocker", [pytest_mock.mocker], indirect=True)
def test_parse_chat(mocker: MockFixture, google_chat_client: GoogleChatClient, mock_sleep: None) -> None:
    """Test parsing chat completion for Google model."""
    mock_part = Part(text="Test response")
    mock_content = Content(parts=[mock_part])
    mock_candidate = Candidate(content=mock_content)
    mock_usage = GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        cached_content_token_count=5,
        candidates_token_count=20,
    )
    mock_response = GenerateContentResponse(
        candidates=[mock_candidate],
        usage_metadata=mock_usage,
    )
    mock_chat = mocker.patch.object(google_chat_client.client.models, "generate_content")
    mock_chat.return_value = mock_response

    system_prompt = {"content": "You are a helpful assistant"}
    user_prompt = {"role": "user", "content": "Hello"}
    response, history, cost = google_chat_client.parse_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        n=1,
    )

    assert response == "Test response"
    assert len(history) == 2
    assert cost > 0
    mock_chat.assert_called_once()
