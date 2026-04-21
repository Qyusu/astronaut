from typing import Any

import pytest
from pytest_mock import MockFixture

from astronaut.logics.vqe.strategy import VQEStrategy
from astronaut.schema import ReflectionType, ResearchMode


@pytest.fixture
def mock_clients(mocker: MockFixture) -> dict[str, Any]:
    return {
        "llm": mocker.Mock(),
        "arxiv": mocker.Mock(),
        "qml": mocker.Mock(),
        "experiment": mocker.Mock(),
    }


def test_vqe_strategy_execute_baseline(
    mocker: MockFixture, mock_clients: dict[str, Any]
) -> None:
    mock_run_baseline = mocker.patch(
        "astronaut.logics.vqe.strategy.run_baseline"
    )

    mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
    strategy = VQEStrategy(
        llm_client=mock_clients["llm"],
        arxiv_db_client=mock_clients["arxiv"],
        qml_db_client=mock_clients["qml"],
        seed_code="seed",
        experiment=mock_clients["experiment"],
        dataset_configs={"H2": {}},
        validation_qubits=[4, 6],
        max_trial_num=3,
        max_idea_num=1,
        max_suggestion_num=5,
        expanding_steps=[4, 6],
        research_mode=ResearchMode.NONE,
        reflection_type=ReflectionType.NONE,
        max_reflection_round=3,
        max_scoring_round=1,
    )

    strategy.execute_baseline()
    assert mock_run_baseline.called


def test_vqe_strategy_execute_experiment(
    mocker: MockFixture, mock_clients: dict[str, MockFixture]
) -> None:
    mock_run = mocker.patch("astronaut.logics.vqe.strategy.run")

    mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
    strategy = VQEStrategy(
        llm_client=mock_clients["llm"],
        arxiv_db_client=mock_clients["arxiv"],
        qml_db_client=mock_clients["qml"],
        seed_code="seed",
        experiment=mock_clients["experiment"],
        dataset_configs={"H2": {}},
        validation_qubits=[4, 6],
        max_trial_num=3,
        max_idea_num=1,
        max_suggestion_num=5,
        expanding_steps=[4, 6],
        research_mode=ResearchMode.NONE,
        reflection_type=ReflectionType.NONE,
        max_reflection_round=3,
        max_scoring_round=1,
    )

    strategy.execute_experiment()
    assert mock_run.called


def test_vqe_strategy_validate(
    mocker: MockFixture, mock_clients: dict[str, MockFixture]
) -> None:
    strategy = VQEStrategy(
        llm_client=mock_clients["llm"],
        arxiv_db_client=mock_clients["arxiv"],
        qml_db_client=mock_clients["qml"],
        seed_code="seed",
        experiment=mock_clients["experiment"],
        dataset_configs={"H2": {}},
        validation_qubits=[4, 6],
        max_trial_num=3,
        max_idea_num=1,
        max_suggestion_num=5,
        expanding_steps=[4, 6],
        research_mode=ResearchMode.NONE,
        reflection_type=ReflectionType.NONE,
        max_reflection_round=3,
        max_scoring_round=1,
    )

    # All connected and initialized
    mock_clients["arxiv"].check_connection.return_value = True
    mock_clients["qml"].check_connection.return_value = True
    mock_clients["experiment"].runs_to_dataframe.return_value = mocker.Mock(
        empty=True
    )

    assert strategy.validate() is True

    # Arxiv disconnected
    mock_clients["arxiv"].check_connection.return_value = False
    assert strategy.validate() is False
