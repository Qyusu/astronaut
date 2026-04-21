from typing import Dict, cast

import pytest
import pytest_mock
from openai.types.responses.parsed_response import ParsedResponse
from pydantic import BaseModel
from pytest_mock import MockFixture

from astronaut.llm.config import OpenAIConfig
from astronaut.llm.providers.openai import OpenAIChatClient
from astronaut.schema import MessageHistory


class TestResponse(BaseModel):
    """Test response model for testing."""

    content: str


@pytest.fixture
def openai_chat_client() -> OpenAIChatClient:
    """Fixture for initializing the OpenAIChatClient with a config."""
    config = OpenAIConfig(
        api_key="test_api_key",
        default_model_version="gpt-5-2025-08-07",
        reasoning_effort="high",
    )
    return OpenAIChatClient(**config.to_dict())


@pytest.fixture
def mock_sleep(mocker: MockFixture) -> None:
    """Fixture to mock time.sleep to prevent actual waiting during tests."""
    mocker.patch("time.sleep")


def test_update_history(openai_chat_client: OpenAIChatClient) -> None:
    """Test updating message history."""
    message_history: MessageHistory = []
    previous_response_id = "resp_6896fd"

    updated_history = openai_chat_client._update_history(
        message_history, previous_response_id
    )
    updated_history = cast(list[Dict[str, str]], updated_history)

    assert len(updated_history) == 1
    assert updated_history[-1] == previous_response_id


def test_parse_response(
    openai_chat_client: OpenAIChatClient, mocker: MockFixture
) -> None:
    """Test parsing response from OpenAI ParsedResponse."""
    # Create a mock ParsedResponse with the expected structure
    mock_content = mocker.Mock()
    mock_content.text = "Test response"

    mock_output = mocker.Mock()
    mock_output.content = [mock_content]

    mock_response = mocker.Mock(spec=ParsedResponse)
    mock_response.output = [mock_output]

    response = openai_chat_client._parse_response(mock_response)
    assert response == "Test response"


def test_get_token_count(
    openai_chat_client: OpenAIChatClient, mocker: MockFixture
) -> None:
    """Test token count calculation for different usage scenarios."""
    # when usage is None
    mock_response = mocker.Mock(spec=ParsedResponse)
    mock_response.usage = None

    input_tokens, cached_tokens, output_tokens = (
        openai_chat_client._get_token_count(mock_response)
    )
    assert input_tokens == 0
    assert cached_tokens == 0
    assert output_tokens == 0

    # when usage is provided with basic tokens
    mock_usage = mocker.Mock()
    mock_usage.model_dump.return_value = {
        "input_tokens": 10,
        "output_tokens": 20,
        "input_tokens_details": None,
    }

    mock_response = mocker.Mock(spec=ParsedResponse)
    mock_response.usage = mock_usage

    input_tokens, cached_tokens, output_tokens = (
        openai_chat_client._get_token_count(mock_response)
    )
    assert input_tokens == 10
    assert cached_tokens == 0
    assert output_tokens == 20

    # when usage includes cached tokens
    mock_usage = mocker.Mock()
    mock_usage.model_dump.return_value = {
        "input_tokens": 10,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 5},
    }

    mock_response = mocker.Mock(spec=ParsedResponse)
    mock_response.usage = mock_usage

    input_tokens, cached_tokens, output_tokens = (
        openai_chat_client._get_token_count(mock_response)
    )
    assert input_tokens == 10
    assert cached_tokens == 5
    assert output_tokens == 20


@pytest.mark.parametrize("mocker", [pytest_mock.mocker], indirect=True)
def test_parse_chat(
    mocker: MockFixture, openai_chat_client: OpenAIChatClient
) -> None:
    """Test parsing chat completion for GPT model."""
    mock_chat = mocker.patch(
        "astronaut.llm.providers.openai.OpenAIChatClient._chat_by_responses_api"
    )

    # Create a mock ParsedResponse
    mock_content = mocker.Mock()
    mock_content.text = "Test response"

    mock_output = mocker.Mock()
    mock_output.content = [mock_content]

    mock_usage = mocker.Mock()
    mock_usage.model_dump.return_value = {
        "input_tokens": 10,
        "output_tokens": 20,
        "input_tokens_details": None,
    }

    mock_response = mocker.Mock(spec=ParsedResponse)
    mock_response.output = [mock_output]
    mock_response.id = "resp_6896fd"
    mock_response.usage = mock_usage

    mock_chat.return_value = mock_response

    developer_prompt = {"content": "You are a helpful assistant"}
    user_prompt = {"content": "Hello"}
    response, history, cost = openai_chat_client.parse_chat(
        developer_prompt=developer_prompt,
        user_prompt=user_prompt,
    )

    assert response == "Test response"
    assert len(history) == 1
    assert cost > 0
    mock_chat.assert_called_once()
