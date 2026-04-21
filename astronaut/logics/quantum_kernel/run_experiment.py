from __future__ import annotations

import copy
import os
import traceback
from typing import TYPE_CHECKING, Any, cast

import optuna
import pandas as pd
import qxmt
import yaml
from langsmith import traceable
from loguru import logger

import astronaut.constants as exp_const
import astronaut.logics.common as common_logics
import astronaut.logics.common.experiments as exp_common
import astronaut.logics.quantum_kernel.constants as qk_const
import astronaut.logics.quantum_kernel.schema as qk_schema
import astronaut.logics.quantum_kernel.utils as qk_utils
import astronaut.prompts.quantum_kernel as qk_prompts
import astronaut.schema as base_schema
from astronaut.logics.quantum_kernel.feedback import FeedbackFormatter
from astronaut.logics.quantum_kernel.generate_feature_map import (
    generate_feature_map,
    generate_feature_map_codes,
)

if TYPE_CHECKING:
    from qxmt.experiment.schema import Dataset, Evaluations, RunTime
    from qxmt.models.qkernels import QSVC

    from astronaut.db import PineconeClient
    from astronaut.llm import ChatClient


optuna.logging.disable_default_handler()

QKERNEL_BASE_CONFIG_PATH = exp_const.BASE_CONFIG_PATH.replace(
    exp_const.TASK_TYPE_PLACEHOLDER, exp_const.QKERNEL_MODEL_TYPE
)
QKERNEL_DRYRUN_CONFIG_PATH = exp_const.DRYRUN_CONFIG_PATH.replace(
    exp_const.TASK_TYPE_PLACEHOLDER, exp_const.QKERNEL_MODEL_TYPE
)


def initialize_run_context(
    experiment: qxmt.Experiment,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    max_scoring_round: int,
    reflection_type: base_schema.ReflectionType,
    max_reflection_round: int,
    seed_code: str,
) -> base_schema.RunContext:
    """Initialize run context specific to quantum kernel experiments."""
    base_context = exp_common.initialize_run_context_base(
        experiment,
        max_trial_num,
        max_idea_num,
        max_suggestion_num,
        max_scoring_round,
        reflection_type,
        max_reflection_round,
        seed_code,
    )

    return base_schema.RunContext(**base_context)


def get_best_result_id_qkernel(result_df: pd.DataFrame, id_col: str) -> str:
    """Get best result ID for quantum kernel experiments (dynamic metric)."""
    primary_metric, is_minimize = qk_utils.get_primary_metric_info()
    target_col = f"{qk_const.AGGREGATED_COL_PREFIX}_{primary_metric}_{qk_const.VALIDATION_SUFFIX}"
    return exp_common.get_best_result_id(
        result_df, target_col, id_col, minimize=is_minimize
    )


def gen_dummy_result_series(
    idea_id: str, dataset_names: list[str]
) -> pd.Series:
    primary_metric, _ = qk_utils.get_primary_metric_info()
    result_dict = {
        qk_const.IDEA_ID_COL: idea_id,
        qk_const.AGGREGATED_RUN_TIME_COL: 0.0,
        f"{qk_const.AGGREGATED_COL_PREFIX}_{primary_metric}_{qk_const.VALIDATION_SUFFIX}": 0.0,
        f"{qk_const.AGGREGATED_COL_PREFIX}_{primary_metric}_{qk_const.TEST_SUFFIX}": 0.0,
    }

    for dataset_name in dataset_names:
        result_dict[
            f"{dataset_name}_{primary_metric}_{qk_const.VALIDATION_SUFFIX}"
        ] = 0.0
        result_dict[
            f"{dataset_name}_{primary_metric}_{qk_const.TEST_SUFFIX}"
        ] = 0.0

    return pd.Series(
        qk_schema.QuantumKernelAggregatedResult(**result_dict).model_dump()
    )


class SingleIdeaExperimentRunner:
    """Handles experiment execution for a single idea."""

    def __init__(
        self,
        experiment: qxmt.Experiment,
        context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
    ) -> None:
        self.experiment = experiment
        self.context = context
        self.dataset_configs = dataset_configs

    def run_experiment_by_multiple_datasets(
        self,
        validated_result: base_schema.GeneratedResult,
        idea_id: str,
        dataset_names: list[str],
    ) -> pd.DataFrame:
        datasets_result_df = pd.DataFrame()
        for dataset_name in dataset_names:
            logger.info(
                f"[idea_id={idea_id}, dataset={dataset_name}] Run Experiment by QXMT..."
            )
            single_dataset_result_series = self._run_single_dataset_experiment(
                validated_result, idea_id, dataset_name
            )
            datasets_result_df = pd.concat(
                [datasets_result_df, single_dataset_result_series.to_frame().T],
                axis=0,
            ).reset_index(drop=True)

        return datasets_result_df

    def _run_single_dataset_experiment(
        self,
        validated_result: base_schema.GeneratedResult,
        idea_id: str,
        dataset_name: str,
    ) -> pd.Series:
        config_path = self._update_config(
            base_config_path=QKERNEL_BASE_CONFIG_PATH,
            config_dirc=self.context.gen_config_dirc,
            experiment_name=str(self.experiment.name),
            module_suffix=idea_id,
            feature_map_name=validated_result.implement.class_name,
            summary=validated_result.idea.summary,
            dataset_name=dataset_name,
        )

        best_params = self._search_hyperparameters(
            config_path, n_samples=qk_const.HP_SEARCH_N_SAMPLES
        )
        self._update_config_with_best_params(config_path, best_params)

        self.experiment.run(config_source=config_path, n_jobs=1)
        eval_result_df = self.experiment.runs_to_dataframe(
            include_validation=True
        )
        run_time = cast("RunTime", self.experiment.exp_db.runs[-1].runtime)  # type: ignore

        single_dataset_result_series = pd.Series(
            qk_schema.QuantumKernelExperimentResult(
                **eval_result_df.iloc[-1].to_dict(),
                idea_id=idea_id,
                dataset_name=dataset_name,
                run_time=run_time.train_seconds,
            ).model_dump()
        )

        return single_dataset_result_series

    def _update_config(
        self,
        base_config_path: str,
        config_dirc: str,
        experiment_name: str,
        module_suffix: str,
        feature_map_name: str,
        summary: str,
        dataset_name: str,
    ) -> str:
        config = yaml.safe_load(open(base_config_path))
        config[qk_const.QXMT_CONFIG_KEY_DESCRIPTION] = summary
        config[qk_const.QXMT_CONFIG_KEY_DATASET] = qk_utils.get_dataset_config(
            dataset_name, self.dataset_configs
        )
        config[qk_const.QXMT_CONFIG_KEY_FEATURE_MAP][
            qk_const.QXMT_CONFIG_KEY_MODULE_NAME
        ] = f"{exp_const.GENERATED_MODULE_ROOT}.{experiment_name}.{qk_const.CODE_FILE_PREFIX}_{module_suffix}"
        config[qk_const.QXMT_CONFIG_KEY_FEATURE_MAP][
            qk_const.QXMT_CONFIG_KEY_IMPLEMENT_NAME
        ] = feature_map_name

        config_path = (
            f"{config_dirc}/{qk_const.CONFIG_FILE_PREFIX}_{module_suffix}.yaml"
        )
        with open(config_path, "w") as file:
            yaml.dump(config, file)
        return config_path

    def _update_config_with_best_params(
        self, config_path: str, best_params: dict
    ) -> None:
        """Update config with best parameters."""
        config = yaml.safe_load(open(config_path))
        for key, value in best_params.items():
            config[qk_const.QXMT_CONFIG_KEY_MODEL][
                qk_const.QXMT_CONFIG_KEY_PARAMS
            ][key] = value
        with open(config_path, "w") as file:
            yaml.dump(config, file)

    def _make_search_config(self, config_path: str, n_samples: int) -> str:
        """Make a search config for a given config path."""
        config_dict = yaml.safe_load(open(config_path))
        config_dict[qk_const.QXMT_CONFIG_KEY_DATASET][
            qk_const.QXMT_CONFIG_KEY_RAW_PREPROCESS_LOGIC
        ][qk_const.QXMT_CONFIG_KEY_PARAMS][
            qk_const.QXMT_CONFIG_KEY_N_SAMPLES
        ] = n_samples
        search_config_path = config_path.replace(
            ".yaml", f"_{qk_const.HP_CONFIG_SUFFIX}.yaml"
        )
        with open(search_config_path, "w") as file:
            yaml.dump(config_dict, file)

        return search_config_path

    def _search_hyperparameters(self, config_path: str, n_samples: int) -> dict:
        """Search hyperparameters for a generated feature map."""
        search_config_path = self._make_search_config(config_path, n_samples)

        artifact, _ = self.experiment.run(
            config_source=search_config_path,
            n_jobs=1,
            show_progress=False,
            add_results=False,
        )
        model = cast("QSVC", artifact.model)
        dataset = cast("Dataset", artifact.dataset)

        _, is_minimize = qk_utils.get_primary_metric_info()
        search_args = qk_schema.HyperparameterSearchArguments(
            cv=qk_const.HP_SEARCH_CV,
            direction="minimize" if is_minimize else "maximize",
            n_jobs=qk_const.HP_SEARCH_N_JOBS,
            n_trials=qk_const.HP_SEARCH_N_TRIALS,
        ).model_dump()

        logger.info("Start hyperparameter search...")
        best_params = model.hyperparameter_search(
            X=dataset.X_train,
            y=dataset.y_train,
            sampler_type=qk_const.HP_SAMPLER_TYPE,
            search_space=qk_const.HP_SEARCH_SPACE,
            search_args=search_args,
            objective=None,
            refit=False,
        )
        logger.info(f"Best parameters: {best_params}")

        if os.path.exists(search_config_path):
            os.remove(search_config_path)

        return best_params


class IdeaProcessor(exp_common.BaseIdeaProcessor):
    """Handles processing of individual ideas including validation and experiment execution."""

    def __init__(
        self,
        dependencies: exp_common.ExperimentDependencies,
        context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
        validation_qubits: list[int],
    ) -> None:
        super().__init__(
            dependencies=dependencies,
            context=context,
            dry_run_config_path=QKERNEL_DRYRUN_CONFIG_PATH,
            code_file_prefix=qk_const.CODE_FILE_PREFIX,
        )
        self.target_dataset_names = list(dataset_configs.keys())
        self.validation_qubits = validation_qubits
        self.single_idea_experiment_runner = SingleIdeaExperimentRunner(
            dependencies.experiment, context, dataset_configs
        )

        self.validation_orchestrator = common_logics.CodeValidationOrchestrator(
            llm_client=dependencies.llm_client,
            db_client=dependencies.qml_db_client,
            context=context,
            experiment=dependencies.experiment,
            dry_run_config_path=QKERNEL_DRYRUN_CONFIG_PATH,
        )

    def _build_dry_run_config_list(self, implement_name: str) -> list[dict]:
        """Build dry run config list for multiple qubit settings."""
        with open(self.dry_run_config_path, "r") as f:
            base_config = yaml.safe_load(f)

        base_config[qk_const.QXMT_CONFIG_KEY_FEATURE_MAP][
            qk_const.QXMT_CONFIG_KEY_MODULE_NAME
        ] = f"{exp_const.GENERATED_MODULE_ROOT}.{self.deps.experiment.name}.{qk_const.DRYRUN_FILE_PREFIX}"
        base_config[qk_const.QXMT_CONFIG_KEY_FEATURE_MAP][
            qk_const.QXMT_CONFIG_KEY_IMPLEMENT_NAME
        ] = implement_name

        qubit_list = self.validation_qubits

        if not qubit_list:
            return [base_config]

        config_list = []
        for qubit in qubit_list:
            cfg = copy.deepcopy(base_config)
            cfg[qk_const.QXMT_CONFIG_KEY_DEVICE][
                qk_const.QXMT_CONFIG_KEY_N_QUBITS
            ] = qubit
            config_list.append(cfg)

        return config_list

    def _run_experiments(
        self, validated_result: base_schema.GeneratedResult, idea_id: str
    ) -> pd.DataFrame:
        """Run experiments for the validated result."""
        # Quantum Kernel-specific: Run single experiment
        return self.single_idea_experiment_runner.run_experiment_by_multiple_datasets(
            validated_result,
            idea_id,
            dataset_names=self.target_dataset_names,
        )

    def _aggregate_results(
        self,
        idea_id: str,
        raw_result_df: pd.DataFrame,
        validated_result: base_schema.GeneratedResult,
    ) -> pd.Series:
        """Aggregate experiment results."""
        return self._aggregate_single_idea_result(
            idea_id,
            raw_result_df,
        )

    def _create_dummy_result(
        self, idea_id: str
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Create dummy results in case of failure."""
        aggregated_series = gen_dummy_result_series(
            idea_id, self.target_dataset_names
        )
        return aggregated_series, pd.DataFrame()

    def _aggregate_single_idea_result(
        self,
        idea_id: str,
        single_idea_result_df: pd.DataFrame,
    ) -> pd.Series:
        primary_metric, _ = qk_utils.get_primary_metric_info()

        # Aggregated metrics
        result_dict = {
            qk_const.IDEA_ID_COL: idea_id,
            qk_const.AGGREGATED_RUN_TIME_COL: float(
                single_idea_result_df[qk_const.RUN_TIME_COL].mean()
            ),
            f"{qk_const.AGGREGATED_COL_PREFIX}_{primary_metric}_{qk_const.VALIDATION_SUFFIX}": float(
                single_idea_result_df[
                    f"{primary_metric}_{qk_const.VALIDATION_SUFFIX}"
                ].mean()
            ),
            f"{qk_const.AGGREGATED_COL_PREFIX}_{primary_metric}_{qk_const.TEST_SUFFIX}": float(
                single_idea_result_df[primary_metric].mean()
            ),
        }

        # Individual dataset metrics
        for _, row in single_idea_result_df.iterrows():
            dataset_name = row[qk_const.DATASET_NAME_COL]
            result_dict[
                f"{dataset_name}_{primary_metric}_{qk_const.VALIDATION_SUFFIX}"
            ] = float(row[f"{primary_metric}_{qk_const.VALIDATION_SUFFIX}"])
            result_dict[
                f"{dataset_name}_{primary_metric}_{qk_const.TEST_SUFFIX}"
            ] = float(row[primary_metric])

        return pd.Series(
            qk_schema.QuantumKernelAggregatedResult(**result_dict).model_dump()
        )


class ExperimentReviewer(exp_common.ExperimentReviewer):
    """Handles idea review functionality."""

    def __init__(
        self,
        dependencies: exp_common.ExperimentDependencies,
        context: base_schema.RunContext,
        review_metric: common_logics.ReviewMetric,
    ) -> None:
        super().__init__(dependencies, context, review_metric)

    def _review_prompt_builder_fn(
        self,
        max_suggestion_num: int,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Create quantum kernel specific review prompts."""
        return qk_prompts.ReviewIdeaPrompt(
            max_suggestion_num=max_suggestion_num
        ).build(
            last_trial_num=last_trial_num,
            last_trial_results=last_trial_results,
            performance_review=performance_review,
        )


class QuantumKernelExperimentRunner(exp_common.BaseExperimentRunner):
    """Main orchestrator for Quantum Kernel experiments."""

    def __init__(
        self,
        dependencies: exp_common.ExperimentDependencies,
        dataset_configs: dict[str, Any],
        validation_qubits: list[int],
        review_metric: common_logics.ReviewMetric,
    ) -> None:
        super().__init__(dependencies)
        self.researcher: common_logics.Researcher | None = None
        self.reviewer: ExperimentReviewer | None = None
        self.trial_executor: exp_common.TrialExecutor | None = None
        self.dataset_configs = dataset_configs
        self.validation_qubits = validation_qubits
        self.review_metric = review_metric
        # Initialize idea processor placeholder, actual initialization in initialize()
        self.idea_processor: IdeaProcessor | None = None

    def initialize(
        self,
        max_trial_num: int,
        max_idea_num: int,
        max_suggestion_num: int,
        max_scoring_round: int,
        reflection_type: base_schema.ReflectionType,
        max_reflection_round: int,
        seed_code: str,
    ) -> None:
        """Initialize the experiment context and components."""
        self.context = initialize_run_context(
            self.deps.experiment,
            max_trial_num,
            max_idea_num,
            max_suggestion_num,
            max_scoring_round,
            reflection_type,
            max_reflection_round,
            seed_code,
        )
        self.reviewer = ExperimentReviewer(
            self.deps,
            self.context,
            review_metric=self.review_metric,
        )
        self.idea_processor = IdeaProcessor(
            self.deps,
            self.context,
            self.dataset_configs,
            self.validation_qubits,
        )
        self.trial_executor = exp_common.TrialExecutor(
            dependencies=self.deps,
            context=self.context,
            idea_processor=self.idea_processor,
            idea_generator_fn=generate_feature_map,
        )

    def expanding_idea(
        self,
        research_mode: base_schema.ResearchMode,
        expanding_steps: list[int],
        existing_research_report_path: str | None = None,
    ) -> None:
        if not self.context:
            raise ValueError(
                "Context not initialized. Call initialize() first."
            )

        # generate research report
        if existing_research_report_path is None:
            research_report, research_cost = common_logics.Researcher(
                client=self.deps.llm_client, research_mode=research_mode
            ).research(
                user_prompt=qk_prompts.RESEARCHER_USER_PROMPT,
                save_report_path=f"{self.context.gen_code_dirc}/{exp_const.RESEARCH_REPORT_FILE_NAME}",
            )
        else:
            research_report, research_cost = (
                common_logics.Researcher.load_existing_report(
                    existing_research_report_path
                )
            )

        # generate multiple ideas from research report
        expanding_executor = exp_common.ExpandingExecutor(
            dependencies=self.deps,
            context=self.context,
            expanding_steps=expanding_steps,
            idea_processor=self.idea_processor,
            research_based_prompt_factory=lambda n,
            r: qk_prompts.ResearchBasedFeatureMapIdeaPrompt().build(n, r),
            expanding_prompt_factory=lambda n,
            r: qk_prompts.ExpandingFeatureMapIdeaPrompt().build(n, r),
            code_generator_fn=generate_feature_map_codes,
            feedback_formatter_fn=lambda r, agg, raw: FeedbackFormatter(
                r, agg, raw
            ).format(),
            seed_phase_id_prefix=qk_const.SEED_PHASE_ID_PREFIX,
            expanding_phase_id_prefix=qk_const.EXPANDING_PHASE_ID_PREFIX,
        )
        _, experiment_report, expanding_cost = (
            expanding_executor.execute_idea_expanding(research_report)
        )

        self.context = exp_common.update_run_context(
            self.context,
            total_cost=self.context.total_cost + research_cost + expanding_cost,
            last_trial_results=experiment_report,
            need_idea_review=True,
        )

    def run_trials(self) -> None:
        """Execute all trials of the experiment."""
        if not self.context or not self.reviewer or not self.trial_executor:
            raise ValueError(
                "Experiment not initialized. Call initialize() first."
            )

        best_result_df = pd.DataFrame()
        try:
            for i in range(self.context.max_trial_num):
                trial_num = i + 1
                logger.info(
                    f"Starting trial ({trial_num}/{self.context.max_trial_num})..."
                )

                # Review previous trial if needed
                self.context, review_cost, is_completed = (
                    self._review_trial_if_needed(trial_num)
                )
                if is_completed:
                    break

                # Execute current trial
                (
                    self.context,
                    trial_cost,
                    validated_results,
                    trial_result_df,
                    all_idea_result_df,
                ) = self._execute_trial(trial_num)

                total_iter_cost = review_cost + trial_cost

                # Update context with trial results
                best_result_df = self._update_context_with_trial_results(
                    validated_results,
                    trial_result_df,
                    all_idea_result_df,
                    best_result_df,
                    total_iter_cost,
                )

                self._log_trial_completion(
                    trial_num,
                    total_iter_cost,
                    all_idea_result_df,
                    best_result_df,
                )

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Error occurred: {e}")
        finally:
            self._log_experiment_completion(all_idea_result_df, best_result_df)

    def _execute_trial(
        self, trial_num: int
    ) -> tuple[
        base_schema.RunContext,
        float,
        base_schema.GeneratedResults,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Execute a single trial with Quantum Kernel-specific pattern."""
        if not self.context or not self.trial_executor:
            raise ValueError("Components not initialized")

        # Sync context to trial executor before execution
        self.trial_executor.update_context(self.context)

        (
            validated_results,
            trial_result_df,
            all_idea_result_df,
            trial_cost,
            updated_context,
        ) = self.trial_executor.execute_trial(trial_num)

        return (
            updated_context,
            trial_cost,
            validated_results,
            trial_result_df,
            all_idea_result_df,
        )

    def _update_context_with_trial_results(
        self,
        validated_results: base_schema.GeneratedResults,
        trial_result_df: pd.DataFrame,
        all_idea_result_df: pd.DataFrame,
        best_result_df: pd.DataFrame,
        iter_cost: float,
    ) -> pd.DataFrame:
        """Update context with results from completed trial."""
        if not self.context:
            raise ValueError("Context not initialized")

        # Get best result from trial
        best_result_id = get_best_result_id_qkernel(
            trial_result_df, qk_const.IDEA_ID_COL
        )
        updated_best_result_df = cast(
            pd.DataFrame,
            pd.concat(
                [
                    best_result_df,
                    trial_result_df[
                        trial_result_df[qk_const.IDEA_ID_COL] == best_result_id
                    ],
                ],
                ignore_index=True,
            ),
        )

        self.context = exp_common.update_run_context(
            self.context,
            last_trial_results=FeedbackFormatter(
                validated_results,
                trial_result_df,
                all_idea_result_df,
            ).format(),
            score_histories=exp_common.update_score_histories(
                self.context.score_histories, validated_results
            ),
            need_idea_review=True,
            total_cost=self.context.total_cost + iter_cost,
            eval_result_df=updated_best_result_df,
        )

        self._sync_context_to_components()

        return updated_best_result_df


def run_baseline(
    experiment: qxmt.Experiment, dataset_configs: dict[str, Any]
) -> None:
    """Run baseline experiment."""

    def config_updater_fn(config: dict[str, Any], dataset_config: Any) -> None:
        config["dataset"] = dataset_config

    def log_result_fn(target_name: str, baseline_result: Any) -> None:
        baseline_evaluations = cast("Evaluations", baseline_result.evaluations)
        logger.info(
            f"""
                Base Score [{target_name}]:
                    - Validataion: {baseline_evaluations.validation}
                    - Test: {baseline_evaluations.test}
                """
        )

    exp_common.BaselineExecutor(
        experiment=experiment,
        dataset_configs=dataset_configs,
        config_updater_fn=config_updater_fn,
        log_result_fn=log_result_fn,
    ).execute(config_source=QKERNEL_BASE_CONFIG_PATH)


@traceable(tags=exp_const.LANGSMITH_EXPERIMENT_TAGS)
def run(
    llm_client: ChatClient,
    arxiv_db_client: PineconeClient | None,
    qml_db_client: PineconeClient | None,
    seed_code: str,
    experiment: qxmt.Experiment,
    dataset_configs: dict[str, Any],
    validation_qubits: list[int],
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    expanding_steps: list[int],
    research_mode: base_schema.ResearchMode,
    reflection_type: base_schema.ReflectionType,
    max_reflection_round: int,
    max_scoring_round: int,
    review_metric: common_logics.ReviewMetric,
    existing_research_report_path: str | None = None,
) -> None:
    """Runs the quantum feature map generation and evaluation experiment.

    This function orchestrates the entire experiment process, including:
    1. Feature map idea research and expanding
    2. Feature map idea generation and review
    3. Code generation and validation
    4. Experiment execution and evaluation
    4. Performance tracking and improvement

    The process iterates for max_trial_num times, with each trial:
    - Generating multiple feature map ideas
    - Validating and executing the generated code
    - Reviewing performance and suggesting improvements
    - Tracking costs and results

    Args:
        llm_client (ChatClient): Client for interacting with the language model
        arxiv_db_client (PineconeClient | None):
            Client for accessing academic paper database. If None, the arxiv DB is not used.
        qml_db_client (PineconeClient | None):
            Client for accessing Pennylane documentation database. If None, the qml DB is not used.
        seed_code (str): Initial code to start the experiment
        experiment (qxmt.Experiment): QXMT experiment object for execution
        target_dataset_names (list[str]): List of target dataset names for evaluation
        validation_qubits (list[int]): List of qubit numbers for validation
        max_trial_num (int): Maximum number of trials to run
        max_idea_num (int): Maximum number of ideas to generate per trial
        max_suggestion_num (int): Maximum number of improvement suggestions per review
        expanding_steps (list[int]): Number of ideas for each steps of expanding
        research_mode (ResearchMode): Mode for initial idea research
        reflection_type (ReflectionType): Type of reflection to use
        max_reflection_round (int): Maximum number of reflection rounds per idea
        max_scoring_round (int): Maximum number of scoring rounds per idea
        review_metric (ReviewMetric): Metric to use for review
        existing_research_report_path (str | None): Path to existing research report markdown file

    Returns:
        None: Results are logged and saved during execution
    """
    dependencies = exp_common.ExperimentDependencies(
        llm_client=llm_client,
        arxiv_db_client=arxiv_db_client,
        qml_db_client=qml_db_client,
        experiment=experiment,
    )

    runner = QuantumKernelExperimentRunner(
        dependencies,
        dataset_configs=dataset_configs,
        validation_qubits=validation_qubits,
        review_metric=review_metric,
    )
    runner.initialize(
        max_trial_num=max_trial_num,
        max_idea_num=max_idea_num,
        max_suggestion_num=max_suggestion_num,
        max_scoring_round=max_scoring_round,
        reflection_type=reflection_type,
        max_reflection_round=max_reflection_round,
        seed_code=seed_code,
    )

    if (
        research_mode != base_schema.ResearchMode.NONE
        or existing_research_report_path is not None
    ):
        runner.expanding_idea(
            research_mode, expanding_steps, existing_research_report_path
        )

    runner.run_trials()
