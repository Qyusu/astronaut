from __future__ import annotations

from abc import ABC, abstractmethod
import traceback
from typing import Any, TYPE_CHECKING

import pandas as pd
from loguru import logger

from astronaut.logics.common import save_generated_code
import astronaut.schema as base_schema


if TYPE_CHECKING:
    from astronaut.logics.common.experiments import ExperimentDependencies
    from astronaut.logics.common.validation import CodeValidationOrchestrator


class BaseExperimentRunner(ABC):
    """Abstract base class for all experiment runners."""

    def __init__(self, dependencies: "ExperimentDependencies") -> None:
        self.deps: "ExperimentDependencies" = dependencies
        self.context: base_schema.RunContext | None = None
        self.reviewer: Any = None
        self.trial_executor: Any = None

    @abstractmethod
    def initialize(
        self,
        max_trial_num: int,
        max_idea_num: int,
        max_suggestion_num: int,
        max_reflection_round: int,
        seed_code: str,
    ) -> None:
        """Initialize the experiment context and components."""
        pass

    @abstractmethod
    def run_trials(self) -> None:
        """Execute all trials of the experiment."""
        pass

    def _sync_context_to_components(self) -> None:
        """Synchronize the current context to all components."""
        if (
            not self.context
            or not hasattr(self, "reviewer")
            or not hasattr(self, "trial_executor")
        ):
            raise ValueError("Components not initialized")

        if self.reviewer:
            self.reviewer.context = self.context
        if self.trial_executor:
            self.trial_executor.update_context(self.context)

    def _review_trial_if_needed(
        self, trial_num: int
    ) -> tuple[base_schema.RunContext, float, bool]:
        """Review previous trial results if needed."""
        if not self.context or not self.reviewer:
            raise ValueError("Components not initialized")

        # Ensure reviewer has the latest context
        self.reviewer.context = self.context

        updated_context, review_cost, is_completed = (
            self.reviewer.review_if_needed(trial_num)
        )
        return updated_context, review_cost, is_completed

    @abstractmethod
    def _execute_trial(self, trial_num: int) -> None:
        """Execute a single trial.

        This method should be implemented by concrete classes to handle
        their specific trial execution patterns and return formats.
        """
        pass

    @abstractmethod
    def _update_context_with_trial_results(
        self, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        """Update context with results from completed trial."""
        pass

    def _log_trial_completion(
        self, trial_num: int, cost: float, raw_result_df: pd.DataFrame, aggregated_result_df: pd.DataFrame
    ) -> None:
        """Log completion of a trial."""
        from loguru import logger

        logger.info(f"Trial {trial_num} is done. [COST]: ${cost:.2f}")
        logger.debug(f"Raw Result:\n{raw_result_df}")
        logger.info(f"Evaluation Result:\n{aggregated_result_df}")
        logger.info("######################################################")

    def _log_experiment_completion(self, raw_result_df: pd.DataFrame, aggregated_result_df: pd.DataFrame) -> None:
        """Log completion of the entire experiment."""
        from loguru import logger

        if not self.context:
            logger.error("Context not available for final logging")
            return

        logger.info(
            f"All trials are done. Total Cost: ${self.context.total_cost:.2f}"
        )
        logger.debug(f"Raw Result:\n{raw_result_df}")
        logger.info(f"All trials Result:\n{aggregated_result_df}")


class BaseSingleIdeaExperimentRunner(ABC):
    """Abstract base class for single idea experiment execution."""

    def __init__(self, experiment: Any, context: base_schema.RunContext) -> None:
        self.experiment = experiment
        self.context = context

    @abstractmethod
    def run_experiment(self, *args: Any, **kwargs: Any) -> pd.Series:
        """Run experiment for a validated result.

        This method should be implemented by concrete classes to handle
        the specific experiment execution logic for their domain.
        """
        pass


class BaseIdeaProcessor(ABC):
    """Abstract base class for idea processors."""

    def __init__(
        self,
        dependencies: ExperimentDependencies,
        context: base_schema.RunContext,
        dry_run_config_path: str,
        code_file_prefix: str,
    ) -> None:
        self.deps = dependencies
        self.context = context
        self.dry_run_config_path = dry_run_config_path
        self.code_file_prefix = code_file_prefix

        self.validation_orchestrator: "CodeValidationOrchestrator"

    def process_idea(
        self, result: base_schema.GeneratedResult, idea_id: str
    ) -> tuple[base_schema.GeneratedResult, pd.Series, pd.DataFrame, float]:
        """Process a single idea through validation and experiment execution."""

        try:
            config_list = self._build_dry_run_config_list(
                implement_name=result.implement.class_name
            )

            validated_result, validation_cost = (
                self._common_validation_and_save(
                    result,
                    config_list,
                    save_file_name=f"{self.code_file_prefix}_{idea_id}.py",
                )
            )

            raw_result_df = self._run_experiments(validated_result, idea_id)
            aggregated_series = self._aggregate_results(idea_id, raw_result_df, validated_result)

            return (
                validated_result,
                aggregated_series,
                raw_result_df,
                validation_cost,
            )

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Error occurred in processing idea {idea_id}: {e}")

            validated_result = base_schema.GeneratedResult(
                idea=result.idea, score=result.score, implement=result.implement
            )
            aggregated_series, raw_result_df = self._create_dummy_result(idea_id)
            return validated_result, aggregated_series, raw_result_df, 0.0

    @abstractmethod
    def _run_experiments(
        self, validated_result: base_schema.GeneratedResult, idea_id: str
    ) -> pd.DataFrame:
        """Run experiments for the validated result."""
        pass

    @abstractmethod
    def _aggregate_results(
        self, idea_id: str, raw_result_df: pd.DataFrame, validated_result: base_schema.GeneratedResult
    ) -> pd.Series:
        """Aggregate experiment results."""
        pass

    @abstractmethod
    def _create_dummy_result(
        self, idea_id: str
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Create dummy results in case of failure."""
        pass

    @abstractmethod
    def _build_dry_run_config_list(self, implement_name: str) -> list[dict]:
        """Build dry run config list."""
        pass

    def update_context(self, context: base_schema.RunContext) -> None:
        """Update the context for this idea processor and its components."""
        self.context = context
        if self.validation_orchestrator:
            self.validation_orchestrator.context = context

    def _common_validation_and_save(
        self, result: base_schema.GeneratedResult, config_list: list[dict], save_file_name: str
    ) -> tuple[base_schema.GeneratedResult, float]:
        """Common validation and code saving logic.

        Args:
            result: Generated result.
            config_list: List of validation configs.
            save_file_name: File name to save the generated code.

        Returns:
            Tuple of validated result and validation cost.
        """
        if not self.validation_orchestrator:
            raise ValueError("Validation orchestrator not initialized")

        try:
            validated_result, validation_cost = (
                self.validation_orchestrator.validate_and_retry(result, config_list)
            )

            save_generated_code(
                f"{self.context.gen_code_dirc}/{save_file_name}",
                validated_result.implement.code,
            )

            return validated_result, validation_cost

        except Exception as e:
            import traceback

            from loguru import logger

            logger.debug(traceback.format_exc())
            raise e


class BaseTrialExecutor(ABC):
    """Abstract base class for trial executors."""

    @abstractmethod
    def execute_trial(
        self, trial_num: int
    ) -> tuple[base_schema.GeneratedResults, pd.DataFrame, float, base_schema.RunContext]:
        """Execute a single trial including idea generation and processing."""
        pass

    @abstractmethod
    def update_context(self, context: base_schema.RunContext) -> None:
        """Update the context for this trial executor."""
        pass


class BaseExperimentReviewer(ABC):
    """Abstract base class for experiment reviewers."""

    @abstractmethod
    def review_if_needed(
        self, trial_num: int
    ) -> tuple[base_schema.RunContext, float, bool]:
        """Review previous trial results if needed."""
        pass

    @abstractmethod
    def _review_prompt_builder_fn(
        self,
        max_suggestion_num: int,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Create domain-specific review prompts."""
        pass
