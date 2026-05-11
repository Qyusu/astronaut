from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from astronaut.logics.common.experiments import ExperimentDependencies
from astronaut.logics.common.validation import CodeValidationOrchestrator
from astronaut.schema import GeneratedResult, GeneratedResults, RunContext


class BaseExperimentRunner(ABC):
    """Abstract base class for all experiment runners."""

    def __init__(self, dependencies: ExperimentDependencies) -> None:
        self.deps: ExperimentDependencies = dependencies
        self.context: RunContext | None = None
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
    ) -> tuple[RunContext, float, bool]:
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
    def _execute_trial(self, trial_num: int):
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
        self, trial_num: int, cost: float, result_df: pd.DataFrame
    ) -> None:
        """Log completion of a trial."""
        from loguru import logger

        logger.info(f"Trial {trial_num} is done. [COST]: ${cost:.2f}")
        logger.info(f"Evaluation Result:\n{result_df}")
        logger.info("######################################################")

    def _log_experiment_completion(self, result_df: pd.DataFrame) -> None:
        """Log completion of the entire experiment."""
        from loguru import logger

        if not self.context:
            logger.error("Context not available for final logging")
            return

        logger.info(
            f"All trials are done. Total Cost: ${self.context.total_cost:.2f}"
        )
        logger.info(f"All trials Result:\n{result_df}")


class BaseSingleIdeaExperimentRunner(ABC):
    """Abstract base class for single idea experiment execution."""

    def __init__(self, experiment: Any, context: RunContext) -> None:
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
        context: RunContext,
        dry_run_config_path: str,
        code_file_prefix: str,
    ) -> None:
        self.deps = dependencies
        self.context = context
        self.dry_run_config_path = dry_run_config_path
        self.code_file_prefix = code_file_prefix

        # Initialize validation orchestrator for reuse across idea processing
        self.validation_orchestrator: CodeValidationOrchestrator

    @abstractmethod
    def process_idea(
        self, result: GeneratedResult, trial_num: int, idea_num: int
    ):
        """Process a single idea through validation and experiment execution.

        Each concrete class should implement this with their specific return pattern.
        """
        pass

    def update_context(self, context: RunContext) -> None:
        """Update the context for this idea processor and its components."""
        self.context = context
        # Update validation orchestrator context if needed
        if self.validation_orchestrator:
            self.validation_orchestrator.context = context

    def _common_validation_and_save(
        self, result: GeneratedResult, idea_id: str
    ) -> tuple[GeneratedResult, float]:
        """Common validation and code saving logic."""
        if not self.validation_orchestrator:
            raise ValueError("Validation orchestrator not initialized")

        try:
            # Validate generated code using orchestrator
            validated_result, validation_cost = (
                self.validation_orchestrator.validate_and_retry(result)
            )

            # Save generated code
            from astronaut.logics.common import save_generated_code

            save_generated_code(
                f"{self.context.gen_code_dirc}/{self.code_file_prefix}_{idea_id}.py",
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
    ) -> tuple[GeneratedResults, pd.DataFrame, float, RunContext]:
        """Execute a single trial including idea generation and processing."""
        pass

    @abstractmethod
    def update_context(self, context: RunContext) -> None:
        """Update the context for this trial executor."""
        pass


class BaseExperimentReviewer(ABC):
    """Abstract base class for experiment reviewers."""

    @abstractmethod
    def review_if_needed(
        self, trial_num: int
    ) -> tuple[RunContext, float, bool]:
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
