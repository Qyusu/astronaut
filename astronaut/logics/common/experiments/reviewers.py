from loguru import logger

from astronaut.constants import COMPLETED
from astronaut.logics.common import (
    ReviewIdea,
    ReviewMetric,
    ReviewPerformance)
from astronaut.logics.common.experiments.dependencies import ExperimentDependencies
from astronaut.logics.common.experiments.runner import BaseExperimentReviewer
from astronaut.logics.common.experiments.utils import update_run_context
from astronaut.schema import RunContext
from astronaut.params import ReviewParams


class ExperimentReviewer(BaseExperimentReviewer):
    """Handles idea review functionality."""

    def __init__(
        self,
        dependencies: ExperimentDependencies,
        context: RunContext,
        review_metric: ReviewMetric,
    ) -> None:
        self.deps = dependencies
        self.context = context
        self.review_metric = review_metric
        self.idea_reviewer = ReviewIdea(
            client=dependencies.llm_client,
            model_version=context.model_versions.review,
            reasoning_effort=ReviewParams().reasoning_effort,
            verbosity=ReviewParams().verbosity,
        )
        self.performance_reviewer = ReviewPerformance()

    def review_if_needed(
        self, trial_num: int
    ) -> tuple[RunContext, float, bool]:
        """Review previous trial results if needed."""
        if not self.context.need_idea_review:
            return self.context, 0.0, False

        # [NOTE]: The ablation study shows that the performance review is not
        # affected by the result. Then we temporarily disable it.
        # performance_review = self.performance_reviewer.review(
        #     self.context.eval_result_df,
        #     self.review_metric,
        # )
        performance_review = None

        updated_review_comment, updated_message_history, cost = (
            self.idea_reviewer.review(
                prompt_builder_fn=self._review_prompt_builder_fn,
                last_trial_num=trial_num - 1,
                context=self.context,
                performance_review=performance_review,
            )
        )

        updated_context = update_run_context(
            self.context,
            message_history=updated_message_history,
            review_comment=updated_review_comment,
        )

        self.context = updated_context  # Update internal context

        is_completed = updated_review_comment == COMPLETED
        if is_completed:
            logger.info("Work is done. The idea generation is completed.")

        return updated_context, cost, is_completed

    def _review_prompt_builder_fn(
        self,
        max_suggestion_num: int,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Create domain-specific review prompts.

        This method should be overridden by subclasses to provide
        domain-specific prompt building logic.
        """
        raise NotImplementedError(
            "Subclasses must implement _review_prompt_builder_fn"
        )
