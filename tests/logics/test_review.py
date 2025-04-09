import pandas as pd

from astronaut.logics.review import ReviewPerformance
from astronaut.schema import IdeaScore, Score


class TestReviewPerformance:
    def setup_method(self) -> None:
        self.significantly_improved_review = (
            "Please review the changes or factors that likely led to this improvement "
            "by referring to all past trials, analyze their impact, "
            "and propose how we can enhance these aspects further to sustain or amplify the positive trend."
        )
        self.improved_review = (
            "Please examine the elements that contributed to this progress by referencing all past trials, "
            "assess their effectiveness, and suggest additional refinements or "
            "strategies to achieve more significant advancements."
        )
        self.marginally_improved_review = (
            "Please examine the elements that contributed to this progress by referencing all past trials, "
            "assess their effectiveness, and suggest additional refinements or "
            "strategies to achieve more significant advancements."
        )
        self.unchanged_review = (
            "Please investigate the potential reasons for this stagnation by comparing all past trials, "
            "identify any bottlenecks or limitations, and propose actionable strategies "
            "to introduce meaningful progress."
        )
        self.dropped_slightly_review = (
            "Please review the factors or changes that may have negatively impacted the results "
            "by analyzing all past trials, evaluate their significance, and propose targeted solutions "
            "to recover or improve performance in subsequent trials."
        )
        self.dropped_significantly_review = (
            "Please thoroughly analyze the root causes of this drop by referencing all past trials, "
            "including any critical changes or issues in the process, and recommend urgent actions "
            "or adjustments to address these challenges effectively and recover performance."
        )
        self.out_of_range_review = ""

    def test_init(self) -> None:
        review_performance = ReviewPerformance(review_comment_template="This is test template.")
        assert review_performance.review_comment_template == "This is test template."

        default_review_performance = ReviewPerformance()
        assert default_review_performance.review_comment_template is not None

    def test_discreatize_score(self) -> None:
        review_performance = ReviewPerformance()
        assert review_performance._discreatize_score(31.0) == "Out of Range"
        assert review_performance._discreatize_score(30.0) == "Significantly improved"
        assert review_performance._discreatize_score(3.1) == "Significantly improved"
        assert review_performance._discreatize_score(3.0) == "Improved"
        assert review_performance._discreatize_score(0.6) == "Improved"
        assert review_performance._discreatize_score(0.5) == "Marginally improved"
        assert review_performance._discreatize_score(0.1) == "Marginally improved"
        assert review_performance._discreatize_score(0.0) == "Unchanged"
        assert review_performance._discreatize_score(-0.1) == "Dropped slightly"
        assert review_performance._discreatize_score(-1.5) == "Dropped slightly"
        assert review_performance._discreatize_score(-1.6) == "Dropped significantly"
        assert review_performance._discreatize_score(-30.0) == "Dropped significantly"
        assert review_performance._discreatize_score(-31.0) == "Out of Range"

    def test_discreatize_metric(self) -> None:
        review_performance = ReviewPerformance()
        assert review_performance._discreatize_metric(1.1) == "Out of Range"
        assert review_performance._discreatize_metric(1.0) == "Significantly improved"
        assert review_performance._discreatize_metric(0.21) == "Significantly improved"
        assert review_performance._discreatize_metric(0.2) == "Improved"
        assert review_performance._discreatize_metric(0.06) == "Improved"
        assert review_performance._discreatize_metric(0.05) == "Marginally improved"
        assert review_performance._discreatize_metric(0.01) == "Marginally improved"
        assert review_performance._discreatize_metric(0.0) == "Unchanged"
        assert review_performance._discreatize_metric(-0.01) == "Dropped slightly"
        assert review_performance._discreatize_metric(-0.2) == "Dropped slightly"
        assert review_performance._discreatize_metric(-0.21) == "Dropped significantly"
        assert review_performance._discreatize_metric(-1.0) == "Dropped significantly"
        assert review_performance._discreatize_metric(-1.1) == "Out of Range"

    def test_review(self) -> None:
        # [NOTE]: Currently, the review method is not use score_list.
        review_performance = ReviewPerformance()

        # case1: data lenght is not enough to review
        evaluation_df = pd.DataFrame({"accuracy": [0.1]})
        score_list = [
            IdeaScore(
                originality=Score(score=5.0, reason=""),
                feasibility=Score(score=6.2, reason=""),
                versatility=Score(score=2.0, reason=""),
            )
        ]
        assert review_performance.review(evaluation_df, score_list) is None

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
        # case2: data lenght is enough to review and metrics are ""Significantly improved"
        evaluation_df = pd.DataFrame({"accuracy": [0.1, 0.5]})
        review_comment = review_performance.review(evaluation_df, common_score_list)
        assert review_comment is not None and (self.significantly_improved_review in review_comment)

        # case3: data lenght is enough to review and metrics are "Improved"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.26]})
        review_comment = review_performance.review(evaluation_df, common_score_list)
        assert review_comment is not None and (self.improved_review in review_comment)

        # case4: data lenght is enough to review and metrics are "Marginally improved"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.21]})
        review_comment = review_performance.review(evaluation_df, common_score_list)
        assert review_comment is not None and (self.marginally_improved_review in review_comment)

        # case5: data lenght is enough to review and metrics are "Unchanged"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.2]})
        review_comment = review_performance.review(evaluation_df, common_score_list)
        assert review_comment is not None and (self.unchanged_review in review_comment)

        # case6: data lenght is enough to review and metrics are "Dropped slightly"
        evaluation_df = pd.DataFrame({"accuracy": [0.2, 0.15]})
        review_comment = review_performance.review(evaluation_df, common_score_list)
        assert review_comment is not None and (self.dropped_slightly_review in review_comment)

        # case7: data lenght is enough to review and metrics are "Dropped significantly"
        evaluation_df = pd.DataFrame({"accuracy": [0.5, 0.1]})
        review_comment = review_performance.review(evaluation_df, common_score_list)
        assert review_comment is not None and (self.dropped_significantly_review in review_comment)

        # case8: data lenght is enough to review and metrics are "Out of Range"
        evaluation_df = pd.DataFrame({"accuracy": [0.5, 100.0]})
        review_comment = review_performance.review(evaluation_df, common_score_list)
        assert review_comment is not None and (self.out_of_range_review in review_comment)
