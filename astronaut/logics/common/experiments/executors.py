import copy
from typing import Any, Callable

import pandas as pd
import qxmt
import yaml
from langsmith import traceable
from loguru import logger
from qxmt.configs import ExperimentConfig

import astronaut.constants as exp_const
import astronaut.logics.common as common_logics
from astronaut.logics.common.experiments.dependencies import ExperimentDependencies
from astronaut.logics.common.experiments.runner import BaseIdeaProcessor
from astronaut.logics.common.experiments.utils import (
    format_batch_results,
    update_run_context,
)
from astronaut.params import GenerateIdeaParams
import astronaut.schema as base_schema



class TrialExecutor:
    """Handles execution of a single trial including idea generation and processing.

    This class orchestrates the flow of determining a new idea, processing it (validation & experiment),
    and returning the results.
    """

    def __init__(
        self,
        dependencies: ExperimentDependencies,
        context: base_schema.RunContext,
        idea_processor: BaseIdeaProcessor,
        idea_generator_fn: Callable[..., tuple[base_schema.GeneratedResults, base_schema.MessageHistory, float]],
    ) -> None:
        self.deps = dependencies
        self.context = context
        self.idea_processor = idea_processor
        self.idea_generator_fn = idea_generator_fn

    def execute_trial(
        self, trial_num: int
    ) -> tuple[base_schema.GeneratedResults, pd.DataFrame, pd.DataFrame, float, base_schema.RunContext]:
        """Execute a single trial including idea generation and processing."""
        total_cost = 0.0

        # Generate idea and code using the injected generator function
        generated_idea_code, updated_message_history, cost = self.idea_generator_fn(
            llm_client=self.deps.llm_client,
            arxiv_db_client=self.deps.arxiv_db_client,
            trial_num=trial_num,
            context=self.context,
        )

        # Update context with message history from idea generation
        # coding history is reset to empty list at each trial
        self.context = update_run_context(
            self.context,
            message_history=base_schema.MessageHistory(
                review=updated_message_history.review,
                idea=updated_message_history.idea,
                code=[],
            ),
        )
        total_cost += cost

        trial_result_df = pd.DataFrame()
        all_idea_result_df = pd.DataFrame()
        validated_result_list: list[base_schema.GeneratedResult] = []

        # Process each generated idea
        for i, result in enumerate(generated_idea_code.results, start=1):
            idea_id = f"{trial_num}_{i}"
            (
                validated_result,
                single_idea_aggregated_result_series,
                single_idea_result_df,
                validation_cost,
            ) = self.idea_processor.process_idea(result, idea_id)
            total_cost += validation_cost

            validated_result_list.append(validated_result)
            trial_result_df = pd.concat(
                [
                    trial_result_df,
                    single_idea_aggregated_result_series.to_frame().T,
                ],
                axis=0,
            ).reset_index(drop=True)
            all_idea_result_df = pd.concat(
                [all_idea_result_df, single_idea_result_df], axis=0
            ).reset_index(drop=True)

        validated_results = base_schema.GeneratedResults(results=validated_result_list)
        return (
            validated_results,
            trial_result_df,
            all_idea_result_df,
            total_cost,
            self.context,
        )

    def update_context(self, context: base_schema.RunContext) -> None:
        """Update the context for this trial executor."""
        self.context = context
        self.idea_processor.update_context(context)


class ExpandingExecutor:
    """Handles execution of idea expanding phase."""

    def __init__(
        self,
        dependencies: ExperimentDependencies,
        context: base_schema.RunContext,
        expanding_steps: list[int],
        idea_processor: BaseIdeaProcessor,
        research_based_prompt_factory: Callable[[int, str], tuple[str, str]],
        expanding_prompt_factory: Callable[[int, str], tuple[str, str]],
        code_generator_fn: Callable[..., tuple[base_schema.GeneratedResults, base_schema.MessageHistory, float]],
        feedback_formatter_fn: Callable[
            [base_schema.GeneratedResults, pd.DataFrame, pd.DataFrame], str
        ],
        seed_phase_id_prefix: str,
        expanding_phase_id_prefix: str,
    ) -> None:
        self.deps = dependencies
        self.context = context
        self.expanding_steps = expanding_steps
        if len(self.expanding_steps) < 2:
            raise ValueError(
                "Expanding steps must be a list of at least two integers."
            )

        self.idea_processor = idea_processor
        self.research_based_prompt_factory = research_based_prompt_factory
        self.expanding_prompt_factory = expanding_prompt_factory
        self.code_generator_fn = code_generator_fn
        self.feedback_formatter_fn = feedback_formatter_fn
        self.seed_phase_id_prefix = seed_phase_id_prefix
        self.expanding_phase_id_prefix = expanding_phase_id_prefix

        self.idea_generator = common_logics.GenerateIdea(
            client=self.deps.llm_client,
            model_version=self.context.model_versions.expanding,
            reasoning_effort=GenerateIdeaParams().reasoning_effort,
            verbosity=GenerateIdeaParams().verbosity,
        )

    @traceable(tags=exp_const.LANGSMITH_EXPANDING_TAGS)
    def execute_idea_expanding(
        self, research_report: str
    ) -> tuple[base_schema.GeneratedIdeaResult, str, float]:
        total_cost = 0.0

        ideas, cost = self._generate_seed_ideas_from_research_report(
            research_report, seed_idea_num=self.expanding_steps[0]
        )
        total_cost += cost

        experiment_report, cost = self._make_experiment_report(
            ideas.results, id_prefix=self.seed_phase_id_prefix
        )
        total_cost += cost

        for expanding_num in self.expanding_steps[1:]:
            ideas, cost = self._expanding_ideas(
                experiment_report, expanding_num
            )
            total_cost += cost

            single_idea_report, cost = self._make_experiment_report(
                ideas.results,
                id_prefix=f"{self.expanding_phase_id_prefix}_{expanding_num}",
            )
            experiment_report += single_idea_report
            total_cost += cost

        return ideas, experiment_report, total_cost

    def _generate_seed_ideas_from_research_report(
        self, research_report: str, seed_idea_num: int
    ) -> tuple[base_schema.GeneratedIdeaResult, float]:
        developer_prompt, user_prompt = self.research_based_prompt_factory(
            seed_idea_num, research_report
        )

        logger.info(f"Generating {seed_idea_num} seed ideas from research report...")
        logger.debug(
            f"Developer prompt: {developer_prompt}, User prompt: {user_prompt}"
        )

        generated_seed_ideas, _, cost = self.idea_generator.generate(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=None,
        )

        generated_idea_num = len(generated_seed_ideas.results)
        logger.info(
            f"Generating {generated_idea_num} seed ideas is done and result is parsed as JSON."
        )

        return generated_seed_ideas, cost

    def _expanding_ideas(
        self, experiment_report: str, expanding_num: int
    ) -> tuple[base_schema.GeneratedIdeaResult, float]:
        developer_prompt, user_prompt = self.expanding_prompt_factory(
            expanding_num, experiment_report
        )

        logger.info("Expanding ideas...")
        logger.debug(
            f"Developer prompt: {developer_prompt}, User prompt: {user_prompt}"
        )

        expanded_ideas, _, cost = self.idea_generator.generate(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=None,
        )

        expanded_idea_num = len(expanded_ideas.results)
        logger.info(f"Expanding {expanded_idea_num} ideas is done and result is parsed as JSON.")

        return expanded_ideas, cost

    def _make_experiment_report(
        self, ideas: list[base_schema.GeneratedIdea], id_prefix: str
    ) -> tuple[str, float]:
        total_cost = 0.0

        generated_codes, _, cost = self.code_generator_fn(
            llm_client=self.deps.llm_client,
            code_model_version=self.context.model_versions.code,
            reference_model_version=self.context.model_versions.reference,
            last_code=self.context.last_code,
            finalized_ideas=ideas,
            message_history=[],
            n_code_history=None,
        )
        total_cost += cost

        # During the expanding phase, the idea score is not always computed.
        formatted_results = format_batch_results(
            ideas=ideas,
            scores=[None for _ in ideas],
            implements=generated_codes.results,
        )

        aggregated_result_df = pd.DataFrame()
        raw_result_df = pd.DataFrame()
        validated_result_list: list[base_schema.GeneratedResult] = []
        for i, result in enumerate(formatted_results.results, start=1):
            idea_id = f"{id_prefix}_{i}"
            (
                validated_result,
                single_idea_aggregated_result_series,
                single_idea_result_df,
                cost,
            ) = self.idea_processor.process_idea(result, idea_id)

            total_cost += cost

            validated_result_list.append(validated_result)
            aggregated_result_df = pd.concat(
                [
                    aggregated_result_df,
                    single_idea_aggregated_result_series.to_frame().T,
                ],
                axis=0,
            ).reset_index(drop=True)
            raw_result_df = pd.concat(
                [raw_result_df, single_idea_result_df], axis=0
            ).reset_index(drop=True)

        logger.debug(f"Raw Expanding Result: {raw_result_df}")
        logger.info(f"Expanding Result: {aggregated_result_df}")

        experiment_report = (
            f"------- {id_prefix} -------\n"
            + self.feedback_formatter_fn(
                base_schema.GeneratedResults(results=validated_result_list),
                aggregated_result_df,
                raw_result_df,
            )
        )

        return experiment_report, total_cost


class BaselineExecutor:
    """Handles execution of baseline experiments."""

    def __init__(
        self,
        experiment: qxmt.Experiment,
        dataset_configs: dict[str, Any],
        config_updater_fn: Callable[[dict[str, Any], Any], None],
        log_result_fn: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.experiment = experiment
        self.dataset_configs = dataset_configs
        self.config_updater_fn = config_updater_fn
        self.log_result_fn = log_result_fn

    def execute(self, config_source: str) -> None:
        """Execute baseline experiment."""
        logger.info("Run Experiment by QXMT with Seed Model...")

        with open(config_source, "r") as f:
            base_config = yaml.safe_load(f)

        for target_name, config_dict in self.dataset_configs.items():
            logger.info(f"Run Baseline Experiment for: {target_name}...")
            current_config = copy.deepcopy(base_config)
            self.config_updater_fn(current_config, config_dict)

            _, baseline_result = self.experiment.run(
                config_source=ExperimentConfig(**current_config),
                add_results=False,
            )

            if self.log_result_fn:
                self.log_result_fn(target_name, baseline_result)
