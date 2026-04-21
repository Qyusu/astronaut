import pytest
from pytest_mock import MockerFixture

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


def test_parse_chat_openai(
    openai_client: ChatClient, mocker: MockerFixture
) -> None:
    # parameters
    dev_prompt = {"role": "system", "content": "sys"}
    user_prompt = {"role": "user", "content": "user"}

    # mock inner client
    mock_inner = mocker.Mock()
    mock_inner.parse_chat.return_value = ("response", ["hist"], 0.5)
    openai_client.client = mock_inner

    response = openai_client.parse_chat(
        developer_prompt=dev_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        reasoning_effort="high",  # openai specific
    )

    assert response.content == "response"
    assert response.message_history == ["hist"]
    assert response.cost == 0.5

    # Validate arguments passed to inner client
    # OpenAIChatRequest arguments should be flattened
    call_kwargs = mock_inner.parse_chat.call_args.kwargs
    assert call_kwargs["developer_prompt"] == dev_prompt
    assert call_kwargs["user_prompt"] == user_prompt
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["reasoning_effort"] == "high"


def test_parse_chat_other(
    google_client: ChatClient, mocker: MockerFixture
) -> None:
    # parameters
    dev_prompt = {"role": "system", "content": "sys"}
    user_prompt = {"role": "user", "content": "user"}

    # mock inner client
    mock_inner = mocker.Mock()
    mock_inner.parse_chat.return_value = ("gemini_res", ["h"], 0.1)
    google_client.client = mock_inner

    response = google_client.parse_chat(
        developer_prompt=dev_prompt, user_prompt=user_prompt, temperature=0.5
    )

    assert response.content == "gemini_res"

    # Validate arguments
    # ChatRequest arguments flattened
    call_kwargs = mock_inner.parse_chat.call_args.kwargs
    assert call_kwargs["developer_prompt"] == dev_prompt
    assert call_kwargs["temperature"] == 0.5
    # Should not have openai specific params if we didn't pass them,
    # but more importantly, it uses ChatRequest model, not OpenAIChatRequest.
    # ChatRequest doesn't have reasoning_effort field.
    assert "reasoning_effort" not in call_kwargs
