from typing import Any

import pytest
from pinecone import QueryResponse
from pytest_mock import MockerFixture

from astronaut.db.client import PineconeClient
from astronaut.llm import EmbeddingClient


@pytest.fixture
def mock_embedding_client(mocker: MockerFixture) -> EmbeddingClient:
    return mocker.Mock(spec=EmbeddingClient)


@pytest.fixture
def mock_pinecone_instance(mocker: MockerFixture) -> Any:
    return mocker.Mock()


@pytest.fixture
def pinecone_client(
    mocker: MockerFixture,
    mock_embedding_client: EmbeddingClient,
    mock_pinecone_instance: Any,
) -> PineconeClient:
    mocker.patch(
        "astronaut.db.client.Pinecone", return_value=mock_pinecone_instance
    )
    mock_pinecone_instance.list_indexes.return_value.names.return_value = [
        "existing-index"
    ]
    mock_pinecone_instance.Index.return_value = mocker.Mock()
    return PineconeClient(
        api_key="test-key",
        index_name="existing-index",
        embed_client=mock_embedding_client,
    )


def test_init_existing_index(
    mocker: MockerFixture,
    mock_embedding_client: EmbeddingClient,
    mock_pinecone_instance: Any,
) -> None:
    mocker.patch(
        "astronaut.db.client.Pinecone", return_value=mock_pinecone_instance
    )
    mock_pinecone_instance.list_indexes.return_value.names.return_value = [
        "existing-index"
    ]

    client = PineconeClient(
        api_key="test-key",
        index_name="existing-index",
        embed_client=mock_embedding_client,
    )

    assert client.index is not None
    mock_pinecone_instance.Index.assert_called_once_with("existing-index")


def test_init_new_index(
    mocker: MockerFixture,
    mock_embedding_client: EmbeddingClient,
    mock_pinecone_instance: Any,
) -> None:
    mocker.patch(
        "astronaut.db.client.Pinecone", return_value=mock_pinecone_instance
    )
    mock_pinecone_instance.list_indexes.return_value.names.return_value = []

    client = PineconeClient(
        api_key="test-key",
        index_name="new-index",
        embed_client=mock_embedding_client,
    )

    assert client.index is None
    mock_pinecone_instance.Index.assert_not_called()


def test_create_index(
    mocker: MockerFixture,
    pinecone_client: PineconeClient,
    mock_pinecone_instance: Any,
) -> None:
    pinecone_client.index_name = "new-index"
    mock_pinecone_instance.list_indexes.return_value.names.return_value = []

    # Mock describe_index status ready
    mock_pinecone_instance.describe_index.return_value.status = {"ready": True}

    pinecone_client.create_index(dimension=1536)

    mock_pinecone_instance.create_index.assert_called_once_with(
        name="new-index",
        dimension=1536,
        metric="cosine",
        spec=mocker.ANY,
    )
    assert pinecone_client.index is not None


def test_create_index_already_exists(
    pinecone_client: PineconeClient,
    mock_pinecone_instance: Any,
) -> None:
    pinecone_client.index_name = "existing-index"
    mock_pinecone_instance.list_indexes.return_value.names.return_value = [
        "existing-index"
    ]

    pinecone_client.create_index(dimension=1536)

    mock_pinecone_instance.create_index.assert_not_called()
    assert pinecone_client.index is not None


def test_chunk_by_size(
    mocker: MockerFixture, pinecone_client: PineconeClient
) -> None:
    mock_encoding = mocker.Mock()
    mock_encoding.encode.return_value = list(range(100))
    mock_encoding.decode.return_value = "chunk"
    mocker.patch("tiktoken.get_encoding", return_value=mock_encoding)

    text = "Hello world " * 10
    chunks = pinecone_client._chunk_by_size(text, chunk_size=5)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_upsert(
    mocker: MockerFixture,
    pinecone_client: PineconeClient,
    mock_embedding_client: EmbeddingClient,
) -> None:
    pinecone_client.index = mocker.Mock()

    # Mock fetch to return empty (document doesn't exist)
    pinecone_client.index.fetch.return_value = mocker.Mock(vectors={})

    # Mock embeddings
    mock_embedding_client.embeddings.return_value = ([[0.1] * 1536], 0.001)

    # Mock chunk methods which use tiktoken
    mocker.patch.object(
        pinecone_client, "_chunk_by_size", return_value=["chunk1"]
    )

    pinecone_client.upsert(
        document_id="doc1",
        text="some text",
        chunk_size=100,
        metadata={"key": "val"},
        chunk_method="size",
    )

    pinecone_client.index.upsert.assert_called()
    assert pinecone_client.total_cost == 0.001


def test_upsert_existing_no_update(
    mocker: MockerFixture,
    pinecone_client: PineconeClient,
) -> None:
    pinecone_client.index = mocker.Mock()
    pinecone_client.index.fetch.return_value = mocker.Mock(vectors={"doc1": {}})

    pinecone_client.upsert(
        document_id="doc1", text="some text", chunk_size=100, allow_update=False
    )

    pinecone_client.index.upsert.assert_not_called()


def test_query(
    mocker: MockerFixture,
    pinecone_client: PineconeClient,
    mock_embedding_client: EmbeddingClient,
) -> None:
    pinecone_client.index = mocker.Mock()

    # Mock embeddings
    mock_embedding_client.embeddings.return_value = ([[0.1] * 1536], 0.001)

    # Mock query response
    mock_response = mocker.Mock(spec=QueryResponse)
    pinecone_client.index.query.return_value = mock_response

    response = pinecone_client.query("query text")

    assert response == mock_response
    pinecone_client.index.query.assert_called_once()
    assert pinecone_client.total_cost == 0.001


def test_delete_document(
    mocker: MockerFixture,
    pinecone_client: PineconeClient,
) -> None:
    pinecone_client.index = mocker.Mock()
    pinecone_client.index.list.return_value = [["id1", "id2"]]

    pinecone_client.delete_document("doc_prefix")

    pinecone_client.index.delete.assert_called_with(ids=["id1", "id2"])


def test_delete_index(
    pinecone_client: PineconeClient,
    mock_pinecone_instance: Any,
) -> None:
    mock_pinecone_instance.list_indexes.return_value = [
        {"name": "existing-index"}
    ]

    pinecone_client.delete_index("existing-index")

    mock_pinecone_instance.delete_index.assert_called_once_with(
        "existing-index"
    )
    assert pinecone_client.index is None


def test_check_connection(
    mocker: MockerFixture,
    pinecone_client: PineconeClient,
) -> None:
    pinecone_client.index = None
    assert pinecone_client.check_connection() is False

    pinecone_client.index = mocker.Mock()
    pinecone_client.index.describe_index_stats.return_value = "stats"
    assert pinecone_client.check_connection() is True
