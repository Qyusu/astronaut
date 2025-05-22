import pandas as pd

from astronaut.logics.common.review import (
    PerformanceStatus,
    ReviewMetric,
    ReviewPerformance,
)
from astronaut.schema import IdeaScore, Score


class TestReviewPerformance:
    def setup_method(self) -> None:
        self.significantly_improved_first_sentence = (
            "Please review the changes or factors that likely led to this improvement "
        )
        self.improved_first_sentence = (
            "Please examine the elements that contributed to this progress by referencing all past trials, "
        )
        self.marginally_improved_first_sentence = (
            "Please examine the elements that contributed to this progress by referencing all past trials, "
        )

        self.unchanged_first_sentence = (
            "Please investigate the potential reasons for this stagnation by comparing all past trials, "
        )

        self.dropped_slightly_first_sentence = (
            "Please review the factors or changes that may have negatively impacted the results "
        )

        self.dropped_significantly_first_sentence = (
            "Please thoroughly analyze the root causes of this drop by referencing all past trials, "
        )

        self.out_of_range_first_sentence = "The performance metric is out of the expected range."

    def test_init(self) -> None:
        review_performance = ReviewPerformance(review_comment_template="This is test template.")
        assert review_performance.review_comment_template == "This is test template."

        default_review_performance = ReviewPerformance()
        assert default_review_performance.review_comment_template is not None

    def test_get_performance_status_accuracy(self) -> None:
        review_performance = ReviewPerformance()
        metric = ReviewMetric.ACCURACY

        # Test significantly improved range
        assert review_performance._get_performance_status(0.21, metric) == PerformanceStatus.SIGNIFICANTLY_IMPROVED
        assert review_performance._get_performance_status(0.5, metric) == PerformanceStatus.SIGNIFICANTLY_IMPROVED
        assert review_performance._get_performance_status(1.0, metric) == PerformanceStatus.SIGNIFICANTLY_IMPROVED

        # Test improved range
        assert review_performance._get_performance_status(0.051, metric) == PerformanceStatus.IMPROVED
        assert review_performance._get_performance_status(0.1, metric) == PerformanceStatus.IMPROVED
        assert review_performance._get_performance_status(0.2, metric) == PerformanceStatus.IMPROVED

        # Test marginally improved range
        assert review_performance._get_performance_status(0.001, metric) == PerformanceStatus.MARGINALLY_IMPROVED
        assert review_performance._get_performance_status(0.025, metric) == PerformanceStatus.MARGINALLY_IMPROVED
        assert review_performance._get_performance_status(0.05, metric) == PerformanceStatus.MARGINALLY_IMPROVED

        # Test unchanged
        assert review_performance._get_performance_status(0.0, metric) == PerformanceStatus.UNCHANGED

        # Test dropped slightly range
        assert review_performance._get_performance_status(-0.1, metric) == PerformanceStatus.DROPPED_SLIGHTLY
        assert review_performance._get_performance_status(-0.15, metric) == PerformanceStatus.DROPPED_SLIGHTLY
        assert review_performance._get_performance_status(-0.2, metric) == PerformanceStatus.DROPPED_SLIGHTLY

        # Test dropped significantly range
        assert review_performance._get_performance_status(-0.21, metric) == PerformanceStatus.DROPPED_SIGNIFICANTLY
        assert review_performance._get_performance_status(-0.5, metric) == PerformanceStatus.DROPPED_SIGNIFICANTLY
        assert review_performance._get_performance_status(-1.0, metric) == PerformanceStatus.DROPPED_SIGNIFICANTLY

        # Test out of range
        assert review_performance._get_performance_status(1.1, metric) == PerformanceStatus.OUT_OF_RANGE
        assert review_performance._get_performance_status(-1.1, metric) == PerformanceStatus.OUT_OF_RANGE

    def test_get_performance_status_cost(self) -> None:
        review_performance = ReviewPerformance()
        metric = ReviewMetric.COST

        # Test significantly improved range
        assert review_performance._get_performance_status(-0.1, metric) == PerformanceStatus.SIGNIFICANTLY_IMPROVED
        assert review_performance._get_performance_status(-1.0, metric) == PerformanceStatus.SIGNIFICANTLY_IMPROVED
        assert review_performance._get_performance_status(-99.9, metric) == PerformanceStatus.SIGNIFICANTLY_IMPROVED

        # Test improved range
        assert review_performance._get_performance_status(-0.01, metric) == PerformanceStatus.IMPROVED
        assert review_performance._get_performance_status(-0.05, metric) == PerformanceStatus.IMPROVED
        assert review_performance._get_performance_status(-0.09, metric) == PerformanceStatus.IMPROVED

        # Test marginally improved range
        assert review_performance._get_performance_status(-1e-10, metric) == PerformanceStatus.MARGINALLY_IMPROVED
        assert review_performance._get_performance_status(-0.005, metric) == PerformanceStatus.MARGINALLY_IMPROVED
        assert review_performance._get_performance_status(-0.009, metric) == PerformanceStatus.MARGINALLY_IMPROVED

        # Test unchanged
        assert review_performance._get_performance_status(0.0, metric) == PerformanceStatus.UNCHANGED

        # Test dropped slightly range
        assert review_performance._get_performance_status(1e-10, metric) == PerformanceStatus.DROPPED_SLIGHTLY
        assert review_performance._get_performance_status(0.05, metric) == PerformanceStatus.DROPPED_SLIGHTLY
        assert review_performance._get_performance_status(0.09, metric) == PerformanceStatus.DROPPED_SLIGHTLY

        # Test dropped significantly range
        assert review_performance._get_performance_status(0.1, metric) == PerformanceStatus.DROPPED_SIGNIFICANTLY
        assert review_performance._get_performance_status(10.0, metric) == PerformanceStatus.DROPPED_SIGNIFICANTLY
        assert review_performance._get_performance_status(99.9, metric) == PerformanceStatus.DROPPED_SIGNIFICANTLY

        # Test out of range
        assert review_performance._get_performance_status(-100.1, metric) == PerformanceStatus.OUT_OF_RANGE
        assert review_performance._get_performance_status(100.0, metric) == PerformanceStatus.OUT_OF_RANGE

    def test_review(self) -> None:
        review_performance = ReviewPerformance()

        # case1: data length is not enough to review
        evaluation_df = pd.DataFrame({"accuracy": [0.1]})
        score_list = [
            IdeaScore(
                originality=Score(score=5.0, reason=""),
                feasibility=Score(score=6.2, reason=""),
                versatility=Score(score=2.0, reason=""),
            )
        ]
        assert review_performance.review(evaluation_df, score_list, ReviewMetric.ACCURACY) is None

        common_score_list = [
            IdeaScore(
                originality=Score(score=5.0, reason=""),
                feasibility=Score(score=6.2, reason=""),
                versatility=Score(score=2.0, reason=""),
            ),
            IdeaScore(
                originality=Score(score=6.0, reason=""),
                feasibility=Score(score=7.2, reason=""),
                versatility=Score(score=3.0, reason=""),
            ),
        ]

        # case2: data length is enough to review and metrics are "Significantly improved"
        evaluation_df = pd.DataFrame({"accuracy": [0.1, 0.5]})
        review_comment = review_performance.review(evaluation_df, common_score_list, ReviewMetric.ACCURACY)
        assert review_comment is not None and (self.significantly_improved_first_sentence in review_comment)

        # case3: data length is enough to review and metrics are "Improved"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.26]})
        review_comment = review_performance.review(evaluation_df, common_score_list, ReviewMetric.ACCURACY)
        assert review_comment is not None and (self.improved_first_sentence in review_comment)

        # case4: data length is enough to review and metrics are "Marginally improved"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.21]})
        review_comment = review_performance.review(evaluation_df, common_score_list, ReviewMetric.ACCURACY)
        assert review_comment is not None and (self.marginally_improved_first_sentence in review_comment)

        # case5: data length is enough to review and metrics are "Unchanged"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.2]})
        review_comment = review_performance.review(evaluation_df, common_score_list, ReviewMetric.ACCURACY)
        assert review_comment is not None and (self.unchanged_first_sentence in review_comment)

        # case6: data length is enough to review and metrics are "Dropped slightly"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.15]})
        review_comment = review_performance.review(evaluation_df, common_score_list, ReviewMetric.ACCURACY)
        assert review_comment is not None and (self.dropped_slightly_first_sentence in review_comment)

        # case7: data length is enough to review and metrics are "Dropped significantly"
        evaluation_df = pd.DataFrame({"accuracy": [0.5, 0.1]})
        review_comment = review_performance.review(evaluation_df, common_score_list, ReviewMetric.ACCURACY)
        assert review_comment is not None and (self.dropped_significantly_first_sentence in review_comment)

        # case8: data length is enough to review and metrics are "Out of Range"
        evaluation_df = pd.DataFrame({"accuracy": [0.5, 100.0]})
        review_comment = review_performance.review(evaluation_df, common_score_list, ReviewMetric.ACCURACY)
        assert review_comment is not None and (self.out_of_range_first_sentence in review_comment)

    def test_review_with_empty_df_and_score_list(self) -> None:
        review_performance = ReviewPerformance()
        assert review_performance.review(pd.DataFrame(), [], ReviewMetric.ACCURACY) is None
