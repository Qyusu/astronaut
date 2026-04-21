import pytest
from typing import Any
from pytest_mock import MockerFixture
from astronaut.logics.common.experiments.reviewers import ExperimentReviewer
from astronaut.logics.common.experiments.dependencies import ExperimentDependencies
import astronaut.schema as base_schema
import astronaut.constants as exp_const
import astronaut.logics.common as common_logics

@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> Any:
    deps = mocker.Mock(spec=ExperimentDependencies)
    deps.llm_client = mocker.Mock()
    return deps

@pytest.fixture
def mock_context(mocker: MockerFixture) -> Any:
    context = mocker.Mock(spec=base_schema.RunContext)
    context.model_versions = mocker.Mock()
    context.model_versions.review = "v_review"
    context.need_idea_review = True
    context.eval_result_df = mocker.Mock()
    # Mock model_copy to return specific updated contexts if needed, or self
    context.model_copy.return_value = context
    return context

class TestExperimentReviewer:
    def test_review_if_needed_not_needed(self, mock_dependencies: Any, mock_context: Any) -> None:
        mock_context.need_idea_review = False
        reviewer = ExperimentReviewer(
            dependencies=mock_dependencies,
            context=mock_context,
            review_metric=common_logics.ReviewMetric.NONE
        )

        ctx, cost, is_completed = reviewer.review_if_needed(trial_num=2)
        assert ctx == mock_context
        assert cost == 0.0
        assert not is_completed

    def test_review_if_needed_none_metric(self, mock_dependencies: Any, mock_context: Any, mocker: MockerFixture) -> None:
        mock_review_idea_cls = mocker.patch("astronaut.logics.common.experiments.reviewers.common_logics.ReviewIdea")
        mock_review_idea_instance = mock_review_idea_cls.return_value
        mock_review_idea_instance.review.return_value = ("Comment", base_schema.MessageHistory(), 0.5)

        reviewer = ExperimentReviewer(
            dependencies=mock_dependencies,
            context=mock_context,
            review_metric=common_logics.ReviewMetric.NONE
        )

        ctx, cost, is_completed = reviewer.review_if_needed(trial_num=2)

        assert cost == 0.5
        assert not is_completed
        mock_review_idea_instance.review.assert_called_once()
        assert "performance_review" in mock_review_idea_instance.review.call_args.kwargs
        assert mock_review_idea_instance.review.call_args.kwargs["performance_review"] is None

    def test_review_if_needed_with_metric(self, mock_dependencies: Any, mock_context: Any, mocker: MockerFixture) -> None:
        mock_review_idea_cls = mocker.patch("astronaut.logics.common.experiments.reviewers.common_logics.ReviewIdea")
        mock_review_idea_instance = mock_review_idea_cls.return_value
        mock_review_idea_instance.review.return_value = (exp_const.COMPLETED, base_schema.MessageHistory(), 0.5)

        mock_perf_reviewer_cls = mocker.patch("astronaut.logics.common.experiments.reviewers.common_logics.ReviewPerformance")
        mock_perf_reviewer_instance = mock_perf_reviewer_cls.return_value
        mock_perf_reviewer_instance.review.return_value = "Perf Review"

        reviewer = ExperimentReviewer(
            dependencies=mock_dependencies,
            context=mock_context,
            review_metric=common_logics.ReviewMetric.AGGREGATED_ACCURACY
        )

        ctx, cost, is_completed = reviewer.review_if_needed(trial_num=2)

        assert cost == 0.5
        assert is_completed
        mock_perf_reviewer_instance.review.assert_called_once()
        mock_review_idea_instance.review.assert_called_once()
        assert mock_review_idea_instance.review.call_args.kwargs["performance_review"] == "Perf Review"

    def test_review_prompt_builder_fn_not_implemented(self, mock_dependencies: Any, mock_context: Any) -> None:
        reviewer = ExperimentReviewer(
            dependencies=mock_dependencies,
            context=mock_context,
            review_metric=common_logics.ReviewMetric.NONE
        )

        with pytest.raises(NotImplementedError):
            reviewer._review_prompt_builder_fn(1, 1, "", None)
