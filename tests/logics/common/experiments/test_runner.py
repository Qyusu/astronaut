import pytest
from typing import Any
from pytest_mock import MockerFixture
from astronaut.logics.common.experiments.runner import BaseExperimentRunner, BaseIdeaProcessor
import astronaut.schema as base_schema
from astronaut.logics.common.experiments.dependencies import ExperimentDependencies

@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> Any:
    return mocker.Mock(spec=ExperimentDependencies)

@pytest.fixture
def mock_context(mocker: MockerFixture) -> Any:
    ctx = mocker.Mock(spec=base_schema.RunContext)
    ctx.gen_code_dirc = "/tmp"
    return ctx

class TestBaseExperimentRunner:
    class ConcreteRunner(BaseExperimentRunner):
        def initialize(self, *args: Any, **kwargs: Any) -> None: pass
        def run_trials(self) -> None: pass
        def _execute_trial(self, trial_num: int) -> None: pass
        def _update_context_with_trial_results(self, *args: Any, **kwargs: Any) -> None: pass

    def test_sync_context_to_components_success(self, mock_dependencies: Any, mock_context: Any, mocker: MockerFixture) -> None:
        runner = self.ConcreteRunner(mock_dependencies)
        runner.context = mock_context
        runner.reviewer = mocker.Mock()
        runner.trial_executor = mocker.Mock()

        runner._sync_context_to_components()

        assert runner.reviewer.context == mock_context
        runner.trial_executor.update_context.assert_called_with(mock_context)

    def test_sync_context_to_components_raises(self, mock_dependencies: Any) -> None:
        runner = self.ConcreteRunner(mock_dependencies)
        # Missing context and components
        with pytest.raises(ValueError):
            runner._sync_context_to_components()

    def test_review_trial_if_needed(self, mock_dependencies: Any, mock_context: Any, mocker: MockerFixture) -> None:
        runner = self.ConcreteRunner(mock_dependencies)
        runner.context = mock_context
        runner.reviewer = mocker.Mock()
        runner.reviewer.review_if_needed.return_value = (mock_context, 0.5, False)

        updated_ctx, cost, is_completed = runner._review_trial_if_needed(1)

        assert updated_ctx == mock_context
        assert cost == 0.5
        assert not is_completed
        assert runner.reviewer.context == mock_context
        runner.reviewer.review_if_needed.assert_called_with(1)

    def test_review_trial_if_needed_raises(self, mock_dependencies: Any) -> None:
        runner = self.ConcreteRunner(mock_dependencies)
        with pytest.raises(ValueError):
            runner._review_trial_if_needed(1)

import pandas as pd

class TestBaseIdeaProcessor:
    class ConcreteIdeaProcessor(BaseIdeaProcessor):
        def _run_experiments(self, validated_result: base_schema.GeneratedResult, idea_id: str) -> pd.DataFrame:
            return pd.DataFrame([{"res": 1}])
        def _aggregate_results(self, idea_id: str, raw_result_df: pd.DataFrame, validated_result: base_schema.GeneratedResult) -> pd.Series:
            return pd.Series({"agg": 1})
        def _create_dummy_result(self, idea_id: str) -> tuple[pd.Series, pd.DataFrame]:
            return pd.Series({"agg": 0}), pd.DataFrame([{"res": 0}])
        def _build_dry_run_config_list(self, implement_name: str) -> list[dict]:
            return [{"conf": 1}]

    def test_process_idea_success(self, mock_dependencies: Any, mock_context: Any, mocker: MockerFixture) -> None:
        processor = self.ConcreteIdeaProcessor(
            dependencies=mock_dependencies,
            context=mock_context,
            dry_run_config_path="path",
            code_file_prefix="prefix"
        )
        processor.validation_orchestrator = mocker.Mock()

        mock_result = mocker.Mock(spec=base_schema.GeneratedResult)
        mock_result.implement = mocker.Mock(spec=base_schema.GeneratedImpl)
        mock_result.implement.class_name = "Class"
        mock_result.implement.code = "code"
        mock_result.idea = "idea"
        mock_result.score = 1.0

        processor.validation_orchestrator.validate_and_retry.return_value = (mock_result, 0.1)
        mocker.patch("astronaut.logics.common.experiments.runner.save_generated_code")

        validated, agg, raw, cost = processor.process_idea(mock_result, "1_1")

        assert validated == mock_result
        assert cost == 0.1
        assert agg["agg"] == 1
        assert raw.iloc[0]["res"] == 1

        processor.validation_orchestrator.validate_and_retry.assert_called_once()

    def test_process_idea_exception(self, mock_dependencies: Any, mock_context: Any, mocker: MockerFixture) -> None:
        processor = self.ConcreteIdeaProcessor(
            dependencies=mock_dependencies,
            context=mock_context,
            dry_run_config_path="path",
            code_file_prefix="prefix"
        )

        generated_idea = base_schema.GeneratedIdea(
            name="idea1", summary="sum", explanation="exp", formula="f", key_sentences=["k"]
        )
        mock_result = base_schema.GeneratedResult(
            idea=generated_idea, score=None,
            implement=base_schema.GeneratedImpl(code="c", class_name="Class")
        )

        validated, agg, raw, cost = processor.process_idea(mock_result, "1_1")

        assert cost == 0.0
        assert agg["agg"] == 0
        assert raw.iloc[0]["res"] == 0

    def test_update_context(self, mock_dependencies: Any, mock_context: Any, mocker: MockerFixture) -> None:
        processor = self.ConcreteIdeaProcessor(mock_dependencies, mock_context, "path", "prefix")
        processor.validation_orchestrator = mocker.Mock()

        new_ctx = mocker.Mock(spec=base_schema.RunContext)
        processor.update_context(new_ctx)

        assert processor.context == new_ctx
        assert processor.validation_orchestrator.context == new_ctx
