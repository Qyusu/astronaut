from typing import Any

import pandas as pd
import pytest

from astronaut.logics.common.review import (
    PerformanceStatus,
    ReviewIdea,
    ReviewMetric,
    ReviewPerformance,
)
from astronaut.schema import (
    MessageHistory,
    MessageHistoryNum,
    ModelVersions,
    RunContext,
)


@pytest.fixture
def mock_client(mocker: Any) -> Any:
    return mocker.MagicMock()


@pytest.fixture
def review_idea(mock_client: Any) -> ReviewIdea:
    return ReviewIdea(client=mock_client, model_version="test-model")


class TestReviewIdea:
    def test_review_success(
        self, review_idea: ReviewIdea, mock_client: Any, mocker: Any
    ) -> None:
        # Mock Context
        context = RunContext(
            gen_config_dirc="/tmp",
            gen_code_dirc="/tmp",
            local_paper_dir="/tmp",
            model_versions=ModelVersions(
                default="m",
                expanding="m",
                idea="m",
                scoring="m",
                summary="m",
                expert="m",
                reflection="m",
                reference="m",
                code="m",
                review="m",
            ),
            max_trial_num=10,
            max_idea_num=10,
            max_suggestion_num=3,
            max_scoring_round=3,
            last_code="print(1)",
            total_cost=0.0,
            need_idea_review=True,
            review_comment="",
            last_trial_results="",
            score_histories="",
            message_history=MessageHistory(review=[], idea=[], code=[]),
            n_message_history=MessageHistoryNum(review=1, idea=1, code=1),
            eval_result_df=pd.DataFrame(),
        )

        # Mock LLM response
        mock_response = mocker.MagicMock()
        mock_response.content = (
            '{"keep_points": ["Good point"], "suggestions": ["Try X"]}'
        )
        mock_response.message_history = ["rev_msg"]
        mock_response.cost = 0.03
        mock_client.parse_chat.return_value = mock_response

        prompt_builder = mocker.MagicMock()
        prompt_builder.return_value = (
            {"role": "system", "content": "dev"},
            {"role": "user", "content": "user"},
        )

        comment, history, cost = review_idea.review(
            prompt_builder_fn=prompt_builder, last_trial_num=1, context=context
        )

        assert "- Good point" in comment
        assert "- Try X" in comment
        assert history.review == ["rev_msg"]
        assert cost == 0.03


class TestReviewPerformance:
    def test_review_too_few_trials(self) -> None:
        reviewer = ReviewPerformance()
        df = pd.DataFrame({"aggregated_accuracy_validation": [0.5]})
        assert reviewer.review(df, ReviewMetric.AGGREGATED_ACCURACY) is None

    def test_review_improvement(self) -> None:
        reviewer = ReviewPerformance()
        df = pd.DataFrame({"aggregated_accuracy_validation": [0.5, 0.8]})
        comment = reviewer.review(df, ReviewMetric.AGGREGATED_ACCURACY)
        assert "Significantly improved" in comment

    def test_review_drop(self) -> None:
        reviewer = ReviewPerformance()
        df = pd.DataFrame({"aggregated_accuracy_validation": [0.8, 0.4]})
        comment = reviewer.review(df, ReviewMetric.AGGREGATED_ACCURACY)
        assert "Dropped significantly" in comment

    def test_review_unchanged(self) -> None:
        reviewer = ReviewPerformance()
        df = pd.DataFrame({"aggregated_accuracy_validation": [0.5, 0.5]})
        comment = reviewer.review(df, ReviewMetric.AGGREGATED_ACCURACY)
        assert "Unchanged" in comment

    def test_review_cost_improvement(self) -> None:
        reviewer = ReviewPerformance()
        df = pd.DataFrame({"aggregated_cost": [10.0, 5.0]})
        comment = reviewer.review(df, ReviewMetric.AGGREGATED_COST)
        assert "Significantly improved" in comment

    def test_compute_metric_value_invalid(self) -> None:
        reviewer = ReviewPerformance()
        df = pd.DataFrame({"col": [1, 2]})
        with pytest.raises(
            ValueError, match="invalid for metric value computation"
        ):
            reviewer._compute_metric_value(df, ReviewMetric.NONE)

    def test_get_performance_status_out_of_range(self) -> None:
        reviewer = ReviewPerformance()
        status = reviewer._get_performance_status(
            -2.0, ReviewMetric.AGGREGATED_ACCURACY
        )
        assert status == PerformanceStatus.OUT_OF_RANGE
