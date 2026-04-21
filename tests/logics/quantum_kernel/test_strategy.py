from typing import Any

from astronaut.logics.common import ReviewMetric
from astronaut.logics.quantum_kernel.strategy import QuantumKernelStrategy
from astronaut.schema import ReflectionType, ResearchMode


def test_validation_success(
    mock_llm_client: Any,
    mock_experiment: Any,
    dataset_configs: dict[str, Any],
) -> None:
    mock_experiment.runs_to_dataframe.return_value.empty = True

    strategy = QuantumKernelStrategy(
        llm_client=mock_llm_client,
        arxiv_db_client=None,
        qml_db_client=None,
        seed_code="print('hello')",
        experiment=mock_experiment,
        dataset_configs=dataset_configs,
        validation_qubits=[1, 2],
        max_trial_num=5,
        max_idea_num=2,
        max_suggestion_num=10,
        expanding_steps=[10, 20],
        research_mode=ResearchMode.LIGHT,
        reflection_type=ReflectionType.SELF_REFLECTION,
        max_reflection_round=2,
        max_scoring_round=2,
    )

    assert strategy.validate() is True


def test_validation_fail_db_connection(
    mock_llm_client: Any,
    mock_experiment: Any,
    dataset_configs: dict[str, Any],
    mocker: Any,
) -> None:
    mock_db = mocker.MagicMock()
    mock_db.check_connection.return_value = False
    mock_experiment.runs_to_dataframe.return_value.empty = True

    strategy = QuantumKernelStrategy(
        llm_client=mock_llm_client,
        arxiv_db_client=mock_db,
        qml_db_client=None,
        seed_code="print('hello')",
        experiment=mock_experiment,
        dataset_configs=dataset_configs,
        validation_qubits=[1, 2],
        max_trial_num=5,
        max_idea_num=2,
        max_suggestion_num=10,
        expanding_steps=[10, 20],
        research_mode=ResearchMode.LIGHT,
        reflection_type=ReflectionType.SELF_REFLECTION,
        max_reflection_round=2,
        max_scoring_round=2,
    )

    assert strategy.validate() is False


def test_execute_baseline(
    mock_llm_client: Any,
    mock_experiment: Any,
    dataset_configs: dict[str, Any],
    mocker: Any,
) -> None:
    mock_run_baseline = mocker.patch(
        "astronaut.logics.quantum_kernel.strategy.run_baseline"
    )

    strategy = QuantumKernelStrategy(
        llm_client=mock_llm_client,
        arxiv_db_client=None,
        qml_db_client=None,
        seed_code="print('hello')",
        experiment=mock_experiment,
        dataset_configs=dataset_configs,
        validation_qubits=[1, 2],
        max_trial_num=5,
        max_idea_num=2,
        max_suggestion_num=10,
        expanding_steps=[10, 20],
        research_mode=ResearchMode.LIGHT,
        reflection_type=ReflectionType.SELF_REFLECTION,
        max_reflection_round=2,
        max_scoring_round=2,
    )

    strategy.execute_baseline()
    mock_run_baseline.assert_called_once_with(mock_experiment, dataset_configs)


def test_execute_experiment(
    mock_llm_client: Any,
    mock_experiment: Any,
    dataset_configs: dict[str, Any],
    mocker: Any,
) -> None:
    mock_run = mocker.patch("astronaut.logics.quantum_kernel.strategy.run")

    strategy = QuantumKernelStrategy(
        llm_client=mock_llm_client,
        arxiv_db_client=None,
        qml_db_client=None,
        seed_code="print('hello')",
        experiment=mock_experiment,
        dataset_configs=dataset_configs,
        validation_qubits=[1, 2],
        max_trial_num=5,
        max_idea_num=2,
        max_suggestion_num=10,
        expanding_steps=[10, 20],
        research_mode=ResearchMode.LIGHT,
        reflection_type=ReflectionType.SELF_REFLECTION,
        max_reflection_round=2,
        max_scoring_round=2,
        review_metric=ReviewMetric.AGGREGATED_ACCURACY,
        existing_research_report_path="report.md",
    )

    strategy.execute_experiment()
    mock_run.assert_called_once()
    # verify kwargs
    _, kwargs = mock_run.call_args
    assert kwargs["llm_client"] == mock_llm_client
    assert kwargs["arxiv_db_client"] is None
    assert kwargs["experiment"] == mock_experiment
    assert kwargs["dataset_configs"] == dataset_configs
    assert kwargs["review_metric"] == ReviewMetric.AGGREGATED_ACCURACY
    assert kwargs["existing_research_report_path"] == "report.md"
