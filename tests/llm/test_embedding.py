import os

import pytest
from pytest_mock import MockFixture

from astronaut.llm.embedding import EmbeddingClient

# Test execution not tracked by LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture
def openai_client(mocker: MockFixture) -> EmbeddingClient:
    """Fixture for initializing the ChatClient with a mocked OpenAI client."""
    mock_openai = mocker.patch("astronaut.llm.embedding.OpenAI")
    mock_openai.return_value.embeddings.create.return_value = mocker.MagicMock(
        data=[mocker.MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(3)], usage=mocker.MagicMock(total_tokens=1000)
    )
    return EmbeddingClient(
        platform="openai",
        api_key="test_api_key",
        embeddings_model_version="text-embedding-3-small",
    )


@pytest.fixture
def mock_sleep(mocker: MockFixture) -> None:
    """Fixture to mock time.sleep to prevent actual waiting during tests."""
    mocker.patch("time.sleep")


def test_embeddings(openai_client: EmbeddingClient) -> None:
    """Test successful embedding generation."""
    embeddings, cost = openai_client.embeddings(["text1", "text2", "text3"])
    assert len(embeddings) == 3
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert cost > 0


def test_embeddings_with_retry(mocker: MockFixture, openai_client: EmbeddingClient, mock_sleep: None) -> None:
    """Test embedding generation with retry mechanism."""
    # First call raises exception, second call succeeds
    mock_embeddings = mocker.patch.object(openai_client, "_embeddings")
    mock_embeddings.side_effect = [
        Exception("API Error"),
        mocker.MagicMock(data=[mocker.MagicMock(embedding=[0.1, 0.2, 0.3])], usage=mocker.MagicMock(total_tokens=1000)),
    ]

    embeddings, cost = openai_client.embeddings(["text1"])
    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert cost > 0
    assert mock_embeddings.call_count == 2


def test_embeddings_max_retries_exceeded(mocker: MockFixture, openai_client: EmbeddingClient, mock_sleep: None) -> None:
    """Test that exception is raised when max retries are exceeded."""
    mock_embeddings = mocker.patch.object(openai_client, "_embeddings")
    mock_embeddings.side_effect = Exception("API Error")

    with pytest.raises(ValueError, match="Failed to get embeddings from OpenAI after 3 attempts"):
        openai_client.embeddings(["text1"], max_retries=3)

    assert mock_embeddings.call_count == 3


def test_initialize_client_invalid_platform() -> None:
    """Test initialization with invalid platform."""
    with pytest.raises(ValueError, match="Embedding model only supports OpenAI"):
        EmbeddingClient(
            platform="invalid",
            api_key="test_api_key",
            embeddings_model_version="text-embedding-3-small",
        )
