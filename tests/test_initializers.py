from typing import Any

import pytest
from pytest_mock import MockerFixture

from astronaut.configs import ExperimentSettings, Settings
from astronaut.initializers import (
    initialize_algorithm_context,
    initialize_embedding_client,
    initialize_llm_client,
    initialize_pinecone_clients,
    load_dataset_configs,
)
from astronaut.llm.config import LLMProvider
from astronaut.logics import StrategyType
from astronaut.schema import ReflectionType, ResearchMode, TaskType


@pytest.fixture
def mock_settings(mocker: MockerFixture) -> Settings:
    return mocker.patch("astronaut.initializers.settings")


def test_initialize_llm_client_openai(
    mock_settings: Settings, mocker: MockerFixture
) -> None:
    mock_settings.CHAT_PLATFORM = LLMProvider.OPENAI
    mock_settings.OPENAI_API_KEY = "test_key"
    mock_settings.DEFAULT_MODEL_VERSION = "gpt-5"

    MockChatClient = mocker.patch("astronaut.initializers.ChatClient")
    client = initialize_llm_client()
    assert client is MockChatClient.return_value
    MockChatClient.assert_called_once()


def test_initialize_llm_client_invalid(mock_settings: Settings) -> None:
    mock_settings.CHAT_PLATFORM = "invalid"  # type: ignore
    with pytest.raises(ValueError, match="Invalid platform"):
        initialize_llm_client()


def test_initialize_embedding_client_none(mock_settings: Settings) -> None:
    mock_settings.EMBEDDING_MODEL_VERSION = None
    assert initialize_embedding_client() is None


def test_initialize_embedding_client_openai(
    mock_settings: Settings, mocker: MockerFixture
) -> None:
    mock_settings.EMBEDDING_MODEL_VERSION = "text-embedding-3"
    mock_settings.EMBEDDING_PLATFORM = LLMProvider.OPENAI
    mock_settings.OPENAI_API_KEY = "test_key"

    MockEmbedClient = mocker.patch("astronaut.initializers.EmbeddingClient")
    client = initialize_embedding_client()
    assert client is MockEmbedClient.return_value


def test_initialize_pinecone_clients_missing_embed(
    mock_settings: Settings,
) -> None:
    mock_settings.PINECONE_API_KEY = "key"
    mock_settings.PENNLYLANE_INDEX_NAME = "index"

    with pytest.raises(ValueError, match="Embedding client is not provided"):
        initialize_pinecone_clients(None)


def test_initialize_pinecone_clients_success(
    mock_settings: Settings, mocker: MockerFixture
) -> None:
    mock_settings.PINECONE_API_KEY = "key"
    mock_settings.PENNLYLANE_INDEX_NAME = "index_qml"
    mock_settings.ARXIV_INDEX_NAME = "index_arxiv"

    MockPineconeClient = mocker.patch("astronaut.initializers.PineconeClient")
    mock_embed = mocker.Mock()

    qml, arxiv = initialize_pinecone_clients(mock_embed)

    assert qml is not None
    assert arxiv is not None
    assert MockPineconeClient.call_count == 2


def test_load_dataset_configs_file_found(mocker: MockerFixture) -> None:
    mock_exp_settings = mocker.Mock(spec=ExperimentSettings)
    mock_exp_settings.dataset_config_path = "datasets.yaml"

    mocker.patch("os.path.isabs", return_value=True)
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("builtins.open", mocker.mock_open(read_data="key: value"))
    mocker.patch("yaml.safe_load", return_value={"key": "value"})

    configs = load_dataset_configs("dummy_path", mock_exp_settings)
    assert configs == {"key": "value"}


def test_load_dataset_configs_file_not_found(mocker: MockerFixture) -> None:
    mock_exp_settings = mocker.Mock(spec=ExperimentSettings)
    mock_exp_settings.dataset_config_path = "datasets.yaml"

    mocker.patch("os.path.isabs", return_value=True)
    mocker.patch("os.path.exists", return_value=False)

    with pytest.raises(FileNotFoundError):
        load_dataset_configs("dummy_path", mock_exp_settings)


def test_initialize_algorithm_context_quantum_kernel(
    mocker: MockerFixture,
) -> None:
    mock_exp_settings = mocker.Mock(spec=ExperimentSettings)
    mock_exp_settings.validation_qubits = [0, 1]
    mock_exp_settings.max_trial_num = 1
    mock_exp_settings.max_idea_num = 1
    mock_exp_settings.max_suggestion_num = 1
    mock_exp_settings.expanding_steps = [1]
    mock_exp_settings.research_mode = ResearchMode.NONE
    mock_exp_settings.reflection_type = ReflectionType.NONE
    mock_exp_settings.max_reflection_round = 1
    mock_exp_settings.max_scoring_round = 1
    mock_exp_settings.review_metric = "NONE"
    mock_exp_settings.existing_research_report_path = None

    mock_chat_client = mocker.Mock()
    mock_arxiv_db = mocker.Mock()
    mock_qml_db = mocker.Mock()
    mock_experiment = mocker.Mock()
    dataset_configs: dict[str, Any] = {}

    mocker.patch(
        "astronaut.initializers.load_code", return_value="def code(): pass"
    )
    MockContext = mocker.patch("astronaut.initializers.QuantumAlgorithmContext")

    context = initialize_algorithm_context(
        task_type=TaskType.QUANTUM_KERNEL,
        experiment_settings=mock_exp_settings,
        chat_client=mock_chat_client,
        arxiv_db_client=mock_arxiv_db,
        qml_db_client=mock_qml_db,
        experiment=mock_experiment,
        dataset_configs=dataset_configs,
    )

    assert context == MockContext.return_value

    MockContext.assert_called_once()
    call_kwargs = MockContext.call_args.kwargs
    assert call_kwargs["strategy_type"] == StrategyType.QUANTUM_KERNEL
    assert call_kwargs["seed_code"] == "def code(): pass"
    assert call_kwargs["llm_client"] == mock_chat_client
