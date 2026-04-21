from typing import Any

import pandas as pd
from pytest_mock import MockFixture

import astronaut.schema as base_schema
from astronaut.logics.vqe.run_experiment import (
    IdeaProcessor,
    SingleIdeaExperimentRunner,
    get_best_result_id_vqe,
    initialize_run_context,
)


def test_initialize_run_context(mock_experiment: MockFixture) -> None:
    context = initialize_run_context(
        experiment=mock_experiment,
        max_trial_num=3,
        max_idea_num=1,
        max_suggestion_num=5,
        max_scoring_round=1,
        reflection_type=base_schema.ReflectionType.SELF_REFLECTION,
        max_reflection_round=3,
        seed_code="seed",
    )
    assert isinstance(context, base_schema.RunContext)
    assert context.max_trial_num == 3


def test_get_best_result_id_vqe() -> None:
    df = pd.DataFrame(
        [
            {"id": "1_1", "aggregated_cost": 0.5},
            {"id": "1_2", "aggregated_cost": 0.1},
            {"id": "1_3", "aggregated_cost": 0.3},
        ]
    )
    best_id = get_best_result_id_vqe(df, "id")
    assert best_id == "1_2"


class TestSingleIdeaExperimentRunner:
    def test_run_experiments_by_multiple_molecules(
        self,
        mocker: MockFixture,
        mock_experiment: MockFixture,
        run_context: base_schema.RunContext,
        valid_dataset_configs: dict[str, Any],
    ) -> None:
        runner = SingleIdeaExperimentRunner(
            experiment=mock_experiment,
            context=run_context,
            dataset_configs=valid_dataset_configs,
        )

        mock_run_bond = mocker.patch.object(
            runner, "_run_experiments_by_bond_length"
        )
        mock_run_bond.return_value = pd.DataFrame()

        result = base_schema.GeneratedResult(
            idea=base_schema.GeneratedIdea(
                name="Idea",
                summary="S",
                explanation="E",
                formula="F",
                key_sentences=["K"],
            ),
            score=None,
            implement=base_schema.GeneratedImpl(
                class_name="Ansatz1", code="code1"
            ),
        )

        runner.run_experiments_by_multiple_molecules(result, "1_1", "base.yaml")
        assert mock_run_bond.called


class TestIdeaProcessor:
    def test_run_experiments(
        self,
        mocker: Any,
        run_context: base_schema.RunContext,
        valid_dataset_configs: dict[str, Any],
    ) -> None:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        deps = mocker.Mock()
        deps.experiment.name = "test_exp"
        mock_runner = mocker.patch(
            "astronaut.logics.vqe.run_experiment.SingleIdeaExperimentRunner",
            autospec=True,
        )
        instance = mock_runner.return_value
        instance.run_experiments_by_multiple_molecules.return_value = (
            pd.DataFrame()
        )

        processor = IdeaProcessor(
            dependencies=deps,
            context=run_context,
            dataset_configs=valid_dataset_configs,
            validation_qubits=[4],
        )
        instance.run_experiments_by_multiple_molecules.return_value = (
            pd.DataFrame()
        )

        result = base_schema.GeneratedResult(
            idea=base_schema.GeneratedIdea(
                name="Idea",
                summary="S",
                explanation="E",
                formula="F",
                key_sentences=["K"],
            ),
            score=None,
            implement=base_schema.GeneratedImpl(
                class_name="Ansatz1", code="code1"
            ),
        )

        processor._run_experiments(result, "1_1")
        assert instance.run_experiments_by_multiple_molecules.called

    def test_aggregate_results(
        self,
        mocker: MockFixture,
        run_context: base_schema.RunContext,
        valid_dataset_configs: dict[str, Any],
    ) -> None:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        deps = mocker.Mock()
        deps.experiment.name = "test_exp"
        processor = IdeaProcessor(
            dependencies=deps,
            context=run_context,
            dataset_configs=valid_dataset_configs,
            validation_qubits=[4],
        )

        mocker.patch.object(
            processor,
            "_detect_ansatz_scalability",
            return_value=("O(1)", "O(1)"),
        )
        mocker.patch.object(
            processor, "_aggregate_single_idea_result", return_value=pd.Series()
        )

        result = base_schema.GeneratedResult(
            idea=base_schema.GeneratedIdea(
                name="Idea",
                summary="S",
                explanation="E",
                formula="F",
                key_sentences=["K"],
            ),
            score=None,
            implement=base_schema.GeneratedImpl(
                class_name="Ansatz1", code="code1"
            ),
        )

        processor._aggregate_results("1_1", pd.DataFrame(), result)
        assert processor._detect_ansatz_scalability.called
