import json
from enum import Enum
from typing import Callable, Literal

import pandas as pd
from langsmith import traceable
from loguru import logger
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel, Field

from astronaut.llm import ChatClient
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    MessageHistory,
    ReviewIdeaResult,
    RunContext,
)

LANGFUSE_TRACKING_TAGS = ["review", "idea"]


class ReviewIdea:
    """A class for reviewing and analyzing feature map ideas.

    This class provides functionality to review and analyze feature map ideas using
    language models. It supports both standard and reasoning-based model versions,
    with appropriate parsing and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use

    Methods:
        review: Reviews and analyzes feature map ideas
    """

    def __init__(
        self,
        client: ChatClient,
        model_version: str,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        self.client = client
        self.model_version = model_version
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    @traceable(tags=LANGFUSE_TRACKING_TAGS)
    def review(
        self,
        prompt_builder_fn: Callable[..., tuple[dict[str, str], dict[str, str]]],
        last_trial_num: int,
        context: RunContext,
        performance_review: str | None = None,
    ) -> tuple[str, MessageHistory, float]:
        """Review idea using a provided prompt builder function.

        Args:
            prompt_builder_fn: Function that builds developer and user prompts.
                Should accept (max_suggestion_num, last_trial_num,
                last_trial_results, performance_review) and return (developer_prompt, user_prompt)
            last_trial_num: Number of the last trial
            context: Current run context
            performance_review: Optional performance review string

        Returns:
            tuple: (review_comment, updated_message_history, cost)
        """
        developer_prompt, user_prompt = prompt_builder_fn(
            max_suggestion_num=context.max_suggestion_num,
            last_trial_num=last_trial_num,
            last_trial_results=context.last_trial_results,
            performance_review=performance_review,
        )

        result, history, cost = self._review_by_llm(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=context.message_history.review,
            n_history=context.n_message_history.review,
        )

        updated_message_history = MessageHistory(
            review=history,
            idea=context.message_history.idea,
            code=context.message_history.code,
        )

        return result.review_comment(), updated_message_history, cost

    def _review_by_llm(
        self,
        developer_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: int | None,
    ) -> tuple[ReviewIdeaResult, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Reviewing idea...")
        logger.debug(
            f"developer_prompt: {developer_prompt}, user_prompt: {user_prompt}"
        )

        response = self.client.parse_chat(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=message_history,
            n_history=n_history,
            response_format=ReviewIdeaResult,
            model_version=self.model_version,
            reasoning=self.reasoning_effort,
            verbosity=self.verbosity,
        )

        result = json.loads(response.content)
        logger.debug(f"Idea review result: {result}")
        logger.info("Idea review is done and result is parsed as JSON.")

        return (
            ReviewIdeaResult(**result),
            response.message_history,
            response.cost,
        )


class PerformanceStatus(Enum):
    SIGNIFICANTLY_IMPROVED = "Significantly improved"
    IMPROVED = "Improved"
    MARGINALLY_IMPROVED = "Marginally improved"
    UNCHANGED = "Unchanged"
    DROPPED_SLIGHTLY = "Dropped slightly"
    DROPPED_SIGNIFICANTLY = "Dropped significantly"
    OUT_OF_RANGE = "Out of Range"


class MetricThresholds(BaseModel):
    significant_improve: tuple[float, float] = Field(
        ...,
        description="Threshold range for significant improvement (lower_bound, upper_bound)",
    )
    improve: tuple[float, float] = Field(
        ...,
        description="Threshold range for improvement (lower_bound, upper_bound)",
    )
    marginal_improve: tuple[float, float] = Field(
        ...,
        description="Threshold range for marginal improvement (lower_bound, upper_bound)",
    )
    slight_drop: tuple[float, float] = Field(
        ...,
        description="Threshold range for slight drop (lower_bound, upper_bound)",
    )
    significant_drop: tuple[float, float] = Field(
        ...,
        description="Threshold range for significant drop (lower_bound, upper_bound)",
    )


class ReviewMetric(Enum):
    ACCURACY = (
        "accuracy",
        MetricThresholds(
            significant_improve=(0.2, 1.0),
            improve=(0.05, 0.2),
            marginal_improve=(0.0, 0.05),
            slight_drop=(-0.2, 0.0),
            significant_drop=(-1.0, -0.2),
        ),
    )
    COST = (
        "cost",
        MetricThresholds(
            significant_improve=(-100.0, -0.1),
            improve=(-0.1, -0.01),
            marginal_improve=(-0.01, -1e-10),
            slight_drop=(1e-10, 0.1),
            significant_drop=(0.1, 100.0),
        ),
    )
    AGGREGATED_COST = (
        "aggregated_cost",
        MetricThresholds(
            significant_improve=(-100.0, -0.1),
            improve=(-0.1, -0.01),
            marginal_improve=(-0.01, -1e-10),
            slight_drop=(1e-10, 0.1),
            significant_drop=(0.1, 100.0),
        ),
    )

    def __init__(self, value: str, thresholds: MetricThresholds):
        self._value_ = value
        self.thresholds = thresholds


class ReviewDirection(BaseModel):
    status: PerformanceStatus
    message: str = Field(
        ...,
        description="Review direction message for the given performance status",
    )


class ReviewPerformance:
    """A class for reviewing and analyzing performance metrics.

    This class provides functionality to review and analyze performance metrics
    of feature map implementations. It evaluates changes in various metrics to
    provide targeted feedback and improvement suggestions based on performance status.

    Args:
        review_comment_template (str | None): Template for review comments.
            Defaults to a standard template if not provided.

    Attributes:
        review_directions (dict[PerformanceStatus, ReviewDirection]): Mapping of
            performance statuses to their corresponding review directions.
        review_comment_template (str): Template string for formatting review comments.

    Methods:
        _get_performance_status: Determines the performance status based on metric changes.
        review: Reviews performance metrics and generates feedback based on changes.
            Returns formatted review comments including performance status and
            targeted improvement suggestions.
    """

    def __init__(self, review_comment_template: str | None = None) -> None:
        self.review_directions = {
            PerformanceStatus.SIGNIFICANTLY_IMPROVED: ReviewDirection(
                status=PerformanceStatus.SIGNIFICANTLY_IMPROVED,
                message=(
                    "Please review the changes or factors that likely led to this improvement "
                    "by referring to all past trials, analyze their impact, "
                    "and propose how we can enhance these aspects further to sustain or amplify the positive trend."
                ),
            ),
            PerformanceStatus.IMPROVED: ReviewDirection(
                status=PerformanceStatus.IMPROVED,
                message=(
                    "Please examine the elements that contributed to this progress by referencing all past trials, "
                    "assess their effectiveness, and suggest additional refinements or "
                    "strategies to achieve more significant advancements."
                ),
            ),
            PerformanceStatus.MARGINALLY_IMPROVED: ReviewDirection(
                status=PerformanceStatus.MARGINALLY_IMPROVED,
                message=(
                    "Please examine the elements that contributed to this progress by referencing all past trials, "
                    "assess their effectiveness, and suggest additional refinements or "
                    "strategies to achieve more significant advancements."
                ),
            ),
            PerformanceStatus.UNCHANGED: ReviewDirection(
                status=PerformanceStatus.UNCHANGED,
                message=(
                    "Please investigate the potential reasons for this stagnation by comparing all past trials, "
                    "identify any bottlenecks or limitations, and propose actionable strategies "
                    "to introduce meaningful progress."
                ),
            ),
            PerformanceStatus.DROPPED_SLIGHTLY: ReviewDirection(
                status=PerformanceStatus.DROPPED_SLIGHTLY,
                message=(
                    "Please review the factors or changes that may have negatively impacted the results "
                    "by analyzing all past trials, evaluate their significance, and propose targeted solutions "
                    "to recover or improve performance in subsequent trials."
                ),
            ),
            PerformanceStatus.DROPPED_SIGNIFICANTLY: ReviewDirection(
                status=PerformanceStatus.DROPPED_SIGNIFICANTLY,
                message=(
                    "Please thoroughly analyze the root causes of this drop by referencing all past trials, "
                    "including any critical changes or issues in the process, and recommend urgent actions "
                    "to address these challenges effectively and recover performance."
                ),
            ),
            PerformanceStatus.OUT_OF_RANGE: ReviewDirection(
                status=PerformanceStatus.OUT_OF_RANGE,
                message="The performance metric is out of the expected range.",
            ),
        }

        if review_comment_template is None:
            self.review_comment_template = (
                'In the previous trial, the model\'s performance "{discreatize_performance}". '
                "{review_direction}"
            )
        else:
            self.review_comment_template = review_comment_template

    @traceable(tags=LANGFUSE_TRACKING_TAGS)
    def review(
        self,
        evaluation_df: pd.DataFrame,
        metric: ReviewMetric,
    ) -> str | None:
        if len(evaluation_df) < 2:
            return None

        diff_metric = (
            evaluation_df[metric.value].values[-1]
            - evaluation_df[metric.value].values[-2]
        )
        status = self._get_performance_status(diff_metric, metric)
        review_direction = self.review_directions.get(
            status, ReviewDirection(status=status, message="")
        )

        return self.review_comment_template.format(
            discreatize_performance=status.value,
            review_direction=review_direction.message,
        )

    def _get_performance_status(
        self, diff_metric: float, metric: ReviewMetric
    ) -> PerformanceStatus:
        thresholds = metric.thresholds

        if (
            thresholds.significant_improve[0]
            < diff_metric
            <= thresholds.significant_improve[1]
        ):
            return PerformanceStatus.SIGNIFICANTLY_IMPROVED
        elif thresholds.improve[0] < diff_metric <= thresholds.improve[1]:
            return PerformanceStatus.IMPROVED
        elif (
            thresholds.marginal_improve[0]
            < diff_metric
            <= thresholds.marginal_improve[1]
        ):
            return PerformanceStatus.MARGINALLY_IMPROVED
        elif diff_metric == 0.0:
            return PerformanceStatus.UNCHANGED
        elif (
            thresholds.slight_drop[0] <= diff_metric < thresholds.slight_drop[1]
        ):
            return PerformanceStatus.DROPPED_SLIGHTLY
        elif (
            thresholds.significant_drop[0]
            <= diff_metric
            < thresholds.significant_drop[1]
        ):
            return PerformanceStatus.DROPPED_SIGNIFICANTLY
        else:
            return PerformanceStatus.OUT_OF_RANGE
