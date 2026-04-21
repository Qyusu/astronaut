from typing import Any

import pandas as pd
import pytest

import astronaut.logics.quantum_kernel.constants as qk_const
import astronaut.schema as base_schema
from astronaut.logics.common import ReviewMetric
from astronaut.logics.quantum_kernel.run_experiment import (
    ExperimentReviewer,
    IdeaProcessor,
    QuantumKernelExperimentRunner,
    SingleIdeaExperimentRunner,
    get_best_result_id_qkernel,
    initialize_run_context,
    run,
    run_baseline,
)


def test_initialize_run_context(mock_experiment: Any) -> None:
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


def test_get_best_result_id_qkernel(mocker: Any) -> None:
    mocker.patch(
        "astronaut.logics.quantum_kernel.utils.get_primary_metric_info",
        return_value=("metric", True),
    )
    df = pd.DataFrame(
        [
            {
                qk_const.IDEA_ID_COL: "1_1",
                f"{qk_const.AGGREGATED_COL_PREFIX}_metric_{qk_const.VALIDATION_SUFFIX}": 0.5,
            },
            {
                qk_const.IDEA_ID_COL: "1_2",
                f"{qk_const.AGGREGATED_COL_PREFIX}_metric_{qk_const.VALIDATION_SUFFIX}": 0.1,
            },
            {
                qk_const.IDEA_ID_COL: "1_3",
                f"{qk_const.AGGREGATED_COL_PREFIX}_metric_{qk_const.VALIDATION_SUFFIX}": 0.3,
            },
        ]
    )
    best_id = get_best_result_id_qkernel(df, qk_const.IDEA_ID_COL)
    assert best_id == "1_2"


class TestSingleIdeaExperimentRunner:
    def test_run_experiment_by_multiple_datasets(
        self,
        mocker: Any,
        mock_experiment: Any,
        run_context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
    ) -> None:
        runner = SingleIdeaExperimentRunner(
            experiment=mock_experiment,
            context=run_context,
            dataset_configs=dataset_configs,
        )

        mock_run_single = mocker.patch.object(
            runner, "_run_single_dataset_experiment"
        )
        mock_run_single.return_value = pd.Series({"metrics": 0.1})

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
                class_name="FeatureMap1", code="code1"
            ),
        )

        dataset_names = ["iris", "digits"]
        df = runner.run_experiment_by_multiple_datasets(
            result, "1_1", dataset_names
        )

        assert mock_run_single.call_count == 2
        assert len(df) == 2

    def test_search_hyperparameters(
        self,
        mocker: Any,
        mock_experiment: Any,
        run_context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
    ) -> None:
        runner = SingleIdeaExperimentRunner(
            experiment=mock_experiment,
            context=run_context,
            dataset_configs=dataset_configs,
        )

        mock_experiment.run.return_value = (
            mocker.MagicMock(
                model=mocker.MagicMock(), dataset=mocker.MagicMock()
            ),
            "res",
        )
        mock_experiment.run.return_value[
            0
        ].model.hyperparameter_search.return_value = {"C": 1.0}

        mocker.patch("builtins.open", mocker.mock_open(read_data="key: value"))
        mocker.patch(
            "yaml.safe_load",
            return_value={"dataset": {"raw_preprocess_logic": {"params": {}}}},
        )
        mocker.patch("yaml.dump")
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("os.remove")
        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.qk_utils.get_primary_metric_info",
            return_value=("metric", True),
        )

        params = runner._search_hyperparameters("config.yaml", 10)
        assert params == {"C": 1.0}
        assert mock_experiment.run.called


class TestIdeaProcessor:
    def test_run_experiments(
        self,
        mocker: Any,
        run_context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
    ) -> None:
        deps = mocker.MagicMock()
        deps.experiment.name = "test_exp"

        mock_runner_cls = mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.SingleIdeaExperimentRunner",
            autospec=True,
        )
        instance = mock_runner_cls.return_value
        instance.run_experiment_by_multiple_datasets.return_value = (
            pd.DataFrame()
        )

        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.common_logics.CodeValidationOrchestrator"
        )

        processor = IdeaProcessor(
            dependencies=deps,
            context=run_context,
            dataset_configs=dataset_configs,
            validation_qubits=[4],
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
                class_name="FeatureMap1", code="code1"
            ),
        )

        processor._run_experiments(result, "1_1")
        assert instance.run_experiment_by_multiple_datasets.called

    def test_aggregate_results(
        self,
        mocker: Any,
        run_context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
    ) -> None:
        deps = mocker.MagicMock()
        deps.experiment.name = "test_exp"

        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.SingleIdeaExperimentRunner"
        )
        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.common_logics.CodeValidationOrchestrator"
        )

        processor = IdeaProcessor(
            dependencies=deps,
            context=run_context,
            dataset_configs=dataset_configs,
            validation_qubits=[4],
        )

        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.qk_utils.get_primary_metric_info",
            return_value=("metric", True),
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
                class_name="FeatureMap1", code="code1"
            ),
        )

        raw_df_data = [
            {
                qk_const.DATASET_NAME_COL: "iris",
                qk_const.RUN_TIME_COL: 10.0,
                f"metric_{qk_const.VALIDATION_SUFFIX}": 0.1,
                "metric": 0.2,
            },
            {
                qk_const.DATASET_NAME_COL: "digits",
                qk_const.RUN_TIME_COL: 20.0,
                f"metric_{qk_const.VALIDATION_SUFFIX}": 0.05,
                "metric": 0.1,
            },
        ]
        raw_df = pd.DataFrame(raw_df_data)

        series = processor._aggregate_results("1_1", raw_df, result)

        assert series[qk_const.IDEA_ID_COL] == "1_1"
        assert series[qk_const.AGGREGATED_RUN_TIME_COL] == 15.0
        assert series[
            f"{qk_const.AGGREGATED_COL_PREFIX}_metric_{qk_const.VALIDATION_SUFFIX}"
        ] == pytest.approx(0.075)
        assert series[
            f"{qk_const.AGGREGATED_COL_PREFIX}_metric_{qk_const.TEST_SUFFIX}"
        ] == pytest.approx(0.15)

    def test_build_dry_run_config_list(
        self,
        mocker: Any,
        run_context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
    ) -> None:
        deps = mocker.MagicMock()
        deps.experiment.name = "test_exp"

        # Mocking open for reading yaml
        mocker.patch("builtins.open", mocker.mock_open(read_data="key: value"))
        mocker.patch(
            "yaml.safe_load",
            return_value={
                "feature_map": {},
                "device": {"n_qubits": 4},
            },
        )
        # Mock SingleIdeaExperimentRunner inside __init__
        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.SingleIdeaExperimentRunner"
        )
        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.common_logics.CodeValidationOrchestrator"
        )

        processor = IdeaProcessor(
            dependencies=deps,
            context=run_context,
            dataset_configs=dataset_configs,
            validation_qubits=[1, 2],
        )

        configs = processor._build_dry_run_config_list("Impl")
        assert len(configs) == 2
        assert configs[0]["device"]["n_qubits"] == 1
        assert configs[1]["device"]["n_qubits"] == 2
        assert configs[0]["feature_map"]["implement_name"] == "Impl"


class TestExperimentReviewer:
    def test_review_prompt_builder_fn(
        self, mocker: Any, run_context: base_schema.RunContext
    ) -> None:
        deps = mocker.MagicMock()
        reviewer = ExperimentReviewer(
            dependencies=deps,
            context=run_context,
            review_metric=ReviewMetric.AGGREGATED_ACCURACY,
        )

        mock_prompt_cls = mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.qk_prompts.ReviewIdeaPrompt"
        )
        mock_prompt_instance = mock_prompt_cls.return_value
        mock_prompt_instance.build.return_value = (
            {"developer": "p"},
            {"user": "p"},
        )

        dev, user = reviewer._review_prompt_builder_fn(
            max_suggestion_num=5,
            last_trial_num=1,
            last_trial_results="results",
            performance_review="review",
        )

        assert dev == {"developer": "p"}
        assert user == {"user": "p"}
        mock_prompt_cls.assert_called_with(max_suggestion_num=5)
        mock_prompt_instance.build.assert_called_with(
            last_trial_num=1,
            last_trial_results="results",
            performance_review="review",
        )


class TestQuantumKernelExperimentRunner:
    @pytest.fixture
    def runner(
        self,
        mocker: Any,
        mock_experiment: Any,
        dataset_configs: dict[str, Any],
    ) -> QuantumKernelExperimentRunner:
        deps = mocker.MagicMock()
        deps.experiment = mock_experiment
        return QuantumKernelExperimentRunner(
            dependencies=deps,
            dataset_configs=dataset_configs,
            validation_qubits=[1],
            review_metric=ReviewMetric.AGGREGATED_ACCURACY,
        )

    def test_initialize(
        self, mocker: Any, runner: QuantumKernelExperimentRunner
    ) -> None:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        runner.initialize(
            max_trial_num=3,
            max_idea_num=1,
            max_suggestion_num=5,
            max_scoring_round=1,
            reflection_type=base_schema.ReflectionType.SELF_REFLECTION,
            max_reflection_round=3,
            seed_code="seed",
        )
        assert runner.context is not None
        assert runner.reviewer is not None
        assert runner.idea_processor is not None
        assert runner.trial_executor is not None

    def test_expanding_idea(
        self, mocker: Any, runner: QuantumKernelExperimentRunner
    ) -> None:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        runner.initialize(
            max_trial_num=3,
            max_idea_num=1,
            max_suggestion_num=5,
            max_scoring_round=1,
            reflection_type=base_schema.ReflectionType.SELF_REFLECTION,
            max_reflection_round=3,
            seed_code="seed",
        )

        mock_researcher = mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.common_logics.Researcher",
            autospec=True,
        )
        mock_researcher.return_value.research.return_value = ("report", 0.1)

        mock_expanding_executor = mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.exp_common.ExpandingExecutor",
            autospec=True,
        )
        mock_expanding_executor.return_value.execute_idea_expanding.return_value = (
            ["idea"],
            "report",
            0.2,
        )

        runner.expanding_idea(
            research_mode=base_schema.ResearchMode.LIGHT, expanding_steps=[5]
        )

        assert mock_researcher.return_value.research.called
        assert (
            mock_expanding_executor.return_value.execute_idea_expanding.called
        )
        assert runner.context.total_cost == pytest.approx(0.3)

    def test_run_trials(
        self, mocker: Any, runner: QuantumKernelExperimentRunner
    ) -> None:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        runner.initialize(
            max_trial_num=1,
            max_idea_num=1,
            max_suggestion_num=5,
            max_scoring_round=1,
            reflection_type=base_schema.ReflectionType.SELF_REFLECTION,
            max_reflection_round=3,
            seed_code="seed",
        )

        mocker.patch.object(
            runner,
            "_review_trial_if_needed",
            return_value=(runner.context, 0.1, False),
        )
        mocker.patch.object(
            runner,
            "_execute_trial",
            return_value=(
                runner.context,
                0.2,
                base_schema.GeneratedResults(results=[]),
                pd.DataFrame(),
                pd.DataFrame(),
            ),
        )
        mocker.patch.object(
            runner,
            "_update_context_with_trial_results",
            return_value=pd.DataFrame(),
        )
        mocker.patch.object(runner, "_log_trial_completion")
        mocker.patch.object(runner, "_log_experiment_completion")

        runner.run_trials()

        assert runner._execute_trial.called
        assert runner._update_context_with_trial_results.called

    def test_execute_trial(
        self, mocker: Any, runner: QuantumKernelExperimentRunner
    ) -> None:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        runner.initialize(
            max_trial_num=1,
            max_idea_num=1,
            max_suggestion_num=5,
            max_scoring_round=1,
            reflection_type=base_schema.ReflectionType.SELF_REFLECTION,
            max_reflection_round=3,
            seed_code="seed",
        )

        runner.trial_executor = mocker.MagicMock()
        runner.trial_executor.execute_trial.return_value = (
            base_schema.GeneratedResults(results=[]),
            pd.DataFrame(),
            pd.DataFrame(),
            0.5,
            runner.context,
        )

        runner._execute_trial(1)

        assert runner.trial_executor.update_context.called
        assert runner.trial_executor.execute_trial.called

    def test_update_context_with_trial_results(
        self, mocker: Any, runner: QuantumKernelExperimentRunner
    ) -> None:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        runner.initialize(
            max_trial_num=1,
            max_idea_num=1,
            max_suggestion_num=5,
            max_scoring_round=1,
            reflection_type=base_schema.ReflectionType.SELF_REFLECTION,
            max_reflection_round=3,
            seed_code="seed",
        )

        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.get_best_result_id_qkernel",
            return_value="1_1",
        )
        mocker.patch(
            "astronaut.logics.quantum_kernel.run_experiment.FeedbackFormatter"
        )

        validated_results = base_schema.GeneratedResults(results=[])
        trial_result_df = pd.DataFrame(
            [{qk_const.IDEA_ID_COL: "1_1", "metric": 0.1}]
        )
        all_idea_result_df = pd.DataFrame()
        best_result_df = pd.DataFrame()

        runner._update_context_with_trial_results(
            validated_results,
            trial_result_df,
            all_idea_result_df,
            best_result_df,
            0.5,
        )

        assert runner.context.total_cost == pytest.approx(0.5)


def test_run_baseline(
    mocker: Any, mock_experiment: Any, dataset_configs: dict[str, Any]
) -> None:
    mock_executor = mocker.patch(
        "astronaut.logics.quantum_kernel.run_experiment.exp_common.BaselineExecutor"
    )

    run_baseline(mock_experiment, dataset_configs)

    assert mock_executor.return_value.execute.called


def test_run_orchestrator(
    mocker: Any,
    mock_llm_client: Any,
    mock_experiment: Any,
    dataset_configs: dict[str, Any],
) -> None:
    mock_runner_cls = mocker.patch(
        "astronaut.logics.quantum_kernel.run_experiment.QuantumKernelExperimentRunner"
    )
    mock_runner = mock_runner_cls.return_value

    run(
        llm_client=mock_llm_client,
        arxiv_db_client=None,
        qml_db_client=None,
        seed_code="seed",
        experiment=mock_experiment,
        dataset_configs=dataset_configs,
        validation_qubits=[1],
        max_trial_num=1,
        max_idea_num=1,
        max_suggestion_num=5,
        expanding_steps=[5],
        research_mode=base_schema.ResearchMode.LIGHT,
        reflection_type=base_schema.ReflectionType.SELF_REFLECTION,
        max_reflection_round=3,
        max_scoring_round=1,
        review_metric=ReviewMetric.AGGREGATED_ACCURACY,
    )

    assert mock_runner.initialize.called
    assert mock_runner.expanding_idea.called
    assert mock_runner.run_trials.called
