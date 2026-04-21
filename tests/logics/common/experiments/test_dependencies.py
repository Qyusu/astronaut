from astronaut.logics.common.experiments.dependencies import ExperimentDependencies
from pytest_mock import MockerFixture

def test_experiment_dependencies_init(mocker: MockerFixture) -> None:
    mock_llm_client = mocker.Mock()
    mock_arxiv_db_client = mocker.Mock()
    mock_qml_db_client = mocker.Mock()
    mock_experiment = mocker.Mock()

    dependencies = ExperimentDependencies(
        llm_client=mock_llm_client,
        arxiv_db_client=mock_arxiv_db_client,
        qml_db_client=mock_qml_db_client,
        experiment=mock_experiment,
    )

    assert dependencies.llm_client == mock_llm_client
    assert dependencies.arxiv_db_client == mock_arxiv_db_client
    assert dependencies.qml_db_client == mock_qml_db_client
    assert dependencies.experiment == mock_experiment

def test_experiment_dependencies_init_optional_none(mocker: MockerFixture) -> None:
    mock_llm_client = mocker.Mock()
    mock_experiment = mocker.Mock()

    dependencies = ExperimentDependencies(
        llm_client=mock_llm_client,
        arxiv_db_client=None,
        qml_db_client=None,
        experiment=mock_experiment,
    )

    assert dependencies.llm_client == mock_llm_client
    assert dependencies.arxiv_db_client is None
    assert dependencies.qml_db_client is None
    assert dependencies.experiment == mock_experiment
