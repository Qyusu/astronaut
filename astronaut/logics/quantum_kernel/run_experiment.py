import os
import traceback
from textwrap import dedent
from typing import cast

import optuna
import pandas as pd
import qxmt
import yaml
from langsmith import traceable
from loguru import logger
from qxmt.experiment.schema import Dataset, RunTime
from qxmt.models.qkernels import QSVC

from astronaut.constants import (
    BASE_CONFIG_PATH,
    DRYRUN_CONFIG_PATH,
    GENERATED_MODULE_ROOT,
    MODEL_TYPE_PLACEHOLDER,
    QKERNEL_MODEL_TYPE,
    RESEARCH_REPORT_FILE_NAME,
)
from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.logics.common import (
    CodeValidationOrchestrator,
    GenerateIdea,
    Researcher,
    ReviewMetric,
)
from astronaut.logics.common.experiments import (
    BaseExperimentRunner,
    BaseIdeaProcessor,
    ExperimentDependencies,
    format_batch_results,
    get_best_idea_abstract,
    get_best_result_id,
    initialize_run_context_base,
    update_run_context,
    update_score_histories,
)
from astronaut.logics.common.experiments import (
    ExperimentReviewer as BaseExperimentReviewer,
)
from astronaut.logics.quantum_kernel import generate_feature_map
from astronaut.logics.quantum_kernel.generate_feature_map import (
    generate_feature_map_codes,
)
from astronaut.logics.quantum_kernel.utils import (
    get_n_qubits_from_config,
)
from astronaut.params import GenerateIdeaParams
from astronaut.prompts.quantum_kernel import (
    RESEARCHER_USER_PROMPT,
    FilteringFeatureMapIdeaPrompt,
    ResearchBasedFeatureMapIdeaPrompt,
    ReviewIdeaPrompt,
)
from astronaut.schema import (
    GeneratedIdea,
    GeneratedIdeaResult,
    GeneratedResult,
    GeneratedResults,
    MessageHistory,
    ReflectionType,
    ResearchMode,
    RunContext,
)

optuna.logging.disable_default_handler()

QKERNEL_BASE_CONFIG_PATH = BASE_CONFIG_PATH.replace(
    MODEL_TYPE_PLACEHOLDER, QKERNEL_MODEL_TYPE
)
QKERNEL_DRYRUN_CONFIG_PATH = DRYRUN_CONFIG_PATH.replace(
    MODEL_TYPE_PLACEHOLDER, QKERNEL_MODEL_TYPE
)

HYPERPARAMETER_SEARCH_N_SAMPLES = 10000


def format_trial_results(
    generated_results: GeneratedResults, trial_result_df: pd.DataFrame
) -> str:
    template = dedent(
        """
        ## Idea Number: {idea_num}
        ### Idea Explanation:
        {idea_explanation}

        ### Idea Mathematical Expression:
        {idea_formula}

        ### Idea Implementation:
        ```python
        {idea_implementation}
        ```

        ### Trial Results:
        #### Run Time: {run_time:.2f} [seconds]

        #### Result:
            - Accuracy: {accuracy}
            - Precision: {precision}
            - Recall: {recall}
            - F1 Score: {f1_score}
        """
    )

    idea_order = []
    past_traials = []
    generated_result_list = generated_results.results
    sorted_result_df = trial_result_df.sort_values(
        "accuracy_validation", ascending=False
    )
    for row in sorted_result_df.iterrows():
        idea_id = str(row[1]["idea_id"])
        idea_num = int(idea_id.split("_")[-1])
        idea_order.append(f'"Idea Number: {idea_num}"')

        current_result = generated_result_list[idea_num - 1]
        past_traial = template.format(
            idea_num=idea_num,
            idea_explanation=current_result.idea.explanation,
            idea_formula=current_result.idea.formula,
            idea_implementation=current_result.implement.code,
            run_time=row[1]["run_time"],
            accuracy=row[1]["accuracy_validation"],
            precision=row[1]["precision_validation"],
            recall=row[1]["recall_validation"],
            f1_score=row[1]["f1_score_validation"],
        )
        past_traials.append(past_traial)

    idea_order_str = (
        "Current trial ideas sorted by accuracy (scores from highest to lowest): "
        + " > ".join(idea_order)
    )
    past_traials_str = "\n".join(past_traials)

    return f"{idea_order_str}\n{past_traials_str}"


def initialize_run_context(
    experiment: qxmt.Experiment,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    max_scoring_round: int,
    reflection_type: ReflectionType,
    max_reflection_round: int,
    seed_code: str,
) -> RunContext:
    """Initialize run context specific to quantum kernel experiments."""
    base_context = initialize_run_context_base(
        experiment,
        max_trial_num,
        max_idea_num,
        max_suggestion_num,
        max_scoring_round,
        reflection_type,
        max_reflection_round,
        seed_code,
    )

    return RunContext(
        model_type=QKERNEL_MODEL_TYPE,
        n_qubits=get_n_qubits_from_config(QKERNEL_BASE_CONFIG_PATH),
        **base_context,
    )


def get_best_result_id_qkernel(
    result_df: pd.DataFrame, id_col: str = "idea_id"
) -> str:
    """Get best result ID for quantum kernel experiments (maximize accuracy)."""
    return get_best_result_id(
        result_df, "accuracy_validation", id_col, minimize=False
    )


def gen_dummy_result_series(idea_id: str) -> pd.Series:
    return pd.Series(
        {
            "run_id": -1,
            "idea_id": idea_id,
            "run_time": 0.0,
            "accuracy_validation": 0.0,
            "precision_validation": 0.0,
            "recall_validation": 0.0,
            "f1_score_validation": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
    )


class SingleIdeaExperimentRunner:
    """Handles experiment execution for a single idea."""

    def __init__(
        self, experiment: qxmt.Experiment, context: RunContext
    ) -> None:
        self.experiment = experiment
        self.context = context

    def run_experiment(
        self, validated_result: GeneratedResult, idea_id: str
    ) -> pd.Series:
        """Run experiment for a validated result."""
        logger.info(f"[idea_id={idea_id}] Run Experiment by QXMT...")

        config_path = self._update_config(
            base_config_path=QKERNEL_BASE_CONFIG_PATH,
            config_dirc=self.context.gen_config_dirc,
            experiment_name=str(self.experiment.name),
            module_suffix=idea_id,
            feature_map_name=validated_result.implement.class_name,
            summary=validated_result.idea.summary,
        )

        best_params = self._search_hyperparameters(
            config_path, n_samples=HYPERPARAMETER_SEARCH_N_SAMPLES
        )
        self._update_config_with_best_params(config_path, best_params)

        self.experiment.run(config_source=config_path, n_jobs=1)
        eval_result_df = self.experiment.runs_to_dataframe(
            include_validation=True
        )
        run_time = cast(RunTime, self.experiment.exp_db.runs[-1].runtime)  # type: ignore
        single_idea_result_series = eval_result_df.iloc[-1].copy()
        single_idea_result_series["idea_id"] = idea_id
        single_idea_result_series["run_time"] = run_time.train_seconds

        return single_idea_result_series

    def _update_config(
        self,
        base_config_path: str,
        config_dirc: str,
        experiment_name: str,
        module_suffix: str,
        feature_map_name: str,
        summary: str,
    ) -> str:
        config = yaml.safe_load(open(base_config_path))
        config["description"] = summary
        config["feature_map"]["module_name"] = (
            f"{GENERATED_MODULE_ROOT}.{experiment_name}.feature_map_{module_suffix}"
        )
        config["feature_map"]["implement_name"] = feature_map_name

        config_path = f"{config_dirc}/config_{module_suffix}.yaml"
        with open(config_path, "w") as file:
            yaml.dump(config, file)
        return config_path

    def _update_config_with_best_params(
        self, config_path: str, best_params: dict
    ) -> None:
        """Update config with best parameters."""
        config = yaml.safe_load(open(config_path))
        for key, value in best_params.items():
            config["model"]["params"][key] = value
        with open(config_path, "w") as file:
            yaml.dump(config, file)

    def _make_search_config(self, config_path: str, n_samples: int) -> str:
        """Make a search config for a given config path."""
        config_dict = yaml.safe_load(open(config_path))
        config_dict["dataset"]["raw_preprocess_logic"]["params"][
            "n_samples"
        ] = n_samples
        search_config_path = config_path.replace(".yaml", "_search.yaml")
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
        model = cast(QSVC, artifact.model)
        dataset = cast(Dataset, artifact.dataset)

        search_space = {"C": [0.001, 100.0]}
        search_args = {
            "cv": 5,
            "direction": "maximize",
            "n_jobs": 3,
            "n_trials": 500,
        }

        logger.info(f"Start hyperparameter search for {search_space}...")
        best_params = model.hyperparameter_search(
            X=dataset.X_train,
            y=dataset.y_train,
            sampler_type="tpe",
            search_space=search_space,
            search_args=search_args,
            objective=None,
            refit=False,
        )
        logger.info(f"Best parameters: {best_params}")

        if os.path.exists(search_config_path):
            os.remove(search_config_path)

        return best_params


class IdeaProcessor(BaseIdeaProcessor):
    """Handles processing of individual ideas including validation and experiment execution."""

    def __init__(
        self, dependencies: ExperimentDependencies, context: RunContext
    ) -> None:
        super().__init__(
            dependencies=dependencies,
            context=context,
            dry_run_config_path=QKERNEL_DRYRUN_CONFIG_PATH,
            code_file_prefix="feature_map",
        )
        self.single_idea_experiment_runner = SingleIdeaExperimentRunner(
            dependencies.experiment, context
        )

        # Initialize validation orchestrator for reuse across idea processing
        self.validation_orchestrator = CodeValidationOrchestrator(
            llm_client=dependencies.llm_client,
            db_client=dependencies.qml_db_client,
            context=context,
            experiment=dependencies.experiment,
            dry_run_config_path=QKERNEL_DRYRUN_CONFIG_PATH,
        )

    def process_idea(
        self, result: GeneratedResult, idea_id: str
    ) -> tuple[GeneratedResult, pd.Series, float]:
        """Process a single idea through validation and experiment execution with Quantum Kernel-specific pattern."""

        try:
            # Use common validation and save logic
            validated_result, validation_cost = (
                self._common_validation_and_save(result, idea_id)
            )

            # Quantum Kernel-specific: Run single experiment
            single_idea_result_series = (
                self.single_idea_experiment_runner.run_experiment(
                    validated_result, idea_id
                )
            )

            return validated_result, single_idea_result_series, validation_cost

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Error occurred in processing idea {idea_id}: {e}")
            validated_result = GeneratedResult(
                idea=result.idea, score=result.score, implement=result.implement
            )
            single_idea_result_series = gen_dummy_result_series(idea_id)
            return validated_result, single_idea_result_series, 0.0


class FilteringExecutor:
    def __init__(
        self,
        dependencies: ExperimentDependencies,
        context: RunContext,
        filtering_steps: list[int],
    ) -> None:
        self.deps = dependencies
        self.context = context
        self.filtering_steps = filtering_steps
        self.idea_processor = IdeaProcessor(dependencies, context)

        self.idea_generator = GenerateIdea(
            client=self.deps.llm_client,
            model_version=self.context.model_versions.filtering,
            reasoning_effort=GenerateIdeaParams().reasoning_effort,
            verbosity=GenerateIdeaParams().verbosity,
        )

    @traceable(tags=["filtering"])
    def execute_idea_filtering(
        self, research_report: str
    ) -> tuple[GeneratedIdeaResult, str, float]:
        total_cost = 0.0

        ideas, cost = self._generate_seed_ideas_from_research_report(
            research_report, seed_idea_num=self.filtering_steps[0]
        )
        total_cost += cost

        experiment_report, cost = self._make_experiment_report(
            ideas.results, id_prefix="seed"
        )
        total_cost += cost

        for filtering_num in self.filtering_steps[1:]:
            ideas, cost = self._filtering_ideas(
                experiment_report, filtering_num
            )
            total_cost += cost

            current_report, cost = self._make_experiment_report(
                ideas.results, id_prefix=f"filtering_step_{filtering_num}"
            )
            experiment_report += current_report
            total_cost += cost

        return ideas, experiment_report, total_cost

    def _generate_seed_ideas_from_research_report(
        self, research_report: str, seed_idea_num: int
    ) -> tuple[GeneratedIdeaResult, float]:
        developer_prompt, user_prompt = (
            ResearchBasedFeatureMapIdeaPrompt().build(
                idea_num=seed_idea_num,
                research_report=research_report,
            )
        )

        logger.info("Generating seed ideas from research report...")
        logger.debug(
            f"Developer prompt: {developer_prompt}, User prompt: {user_prompt}"
        )

        generated_seed_ideas, _, cost = self.idea_generator.generate(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=None,
        )

        logger.info(
            "Generating seed ideas is done and result is parsed as JSON."
        )

        return generated_seed_ideas, cost

    def _filtering_ideas(
        self, experiment_report: str, filtering_num: int
    ) -> tuple[GeneratedIdeaResult, float]:
        developer_prompt, user_prompt = FilteringFeatureMapIdeaPrompt().build(
            idea_num=filtering_num,
            experiment_report=experiment_report,
        )

        logger.info("Filtering ideas...")
        logger.debug(
            f"Developer prompt: {developer_prompt}, User prompt: {user_prompt}"
        )

        filtered_ideas, _, cost = self.idea_generator.generate(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=None,
        )

        logger.info("Filtering ideas is done and result is parsed as JSON.")

        return filtered_ideas, cost

    def _make_experiment_report(
        self, ideas: list[GeneratedIdea], id_prefix: str
    ) -> tuple[str, float]:
        total_cost = 0.0

        generated_codes, _, cost = generate_feature_map_codes(
            llm_client=self.deps.llm_client,
            code_model_version=self.context.model_versions.code,
            reference_model_version=self.context.model_versions.reference,
            last_code=self.context.last_code,
            finalized_ideas=ideas,
            message_history=[],
            n_code_history=None,
        )
        total_cost += cost

        formatted_results = format_batch_results(
            ideas,
            [None for _ in ideas],
            generated_codes.results,
        )

        filetring_result_df = pd.DataFrame()
        validated_result_list: list[GeneratedResult] = []
        for i, result in enumerate(formatted_results.results, start=1):
            idea_id = f"{id_prefix}_{i}"
            (
                validated_result,
                single_idea_result_series,
                cost,
            ) = self.idea_processor.process_idea(result, idea_id)

            total_cost += cost

            validated_result_list.append(validated_result)
            filetring_result_df = pd.concat(
                [filetring_result_df, single_idea_result_series.to_frame().T],
                axis=0,
            ).reset_index(drop=True)

        logger.info(f"Fitering Result: {filetring_result_df}")
        experiment_report = (
            f"------- {id_prefix} -------\n"
            + format_trial_results(
                GeneratedResults(results=validated_result_list),
                filetring_result_df,
            )
        )

        return experiment_report, total_cost


class TrialExecutor:
    """Handles execution of a single trial including idea generation and processing."""

    def __init__(
        self, dependencies: ExperimentDependencies, context: RunContext
    ) -> None:
        self.deps = dependencies
        self.context = context
        self.idea_processor = IdeaProcessor(dependencies, context)

    def execute_trial(
        self, trial_num: int
    ) -> tuple[GeneratedResults, pd.DataFrame, float, RunContext]:
        """Execute a single trial including idea generation and processing."""
        iter_cost = 0.0

        # Generate feature map idea and code
        generate_first_results, updated_message_history, cost = (
            generate_feature_map(
                llm_client=self.deps.llm_client,
                arxiv_db_client=self.deps.arxiv_db_client,
                trial_num=trial_num,
                context=self.context,
            )
        )

        # Update context with message history from feature map generation
        # coding history is reset to empty list at each trial
        self.context = update_run_context(
            self.context,
            message_history=MessageHistory(
                review=updated_message_history.review,
                idea=updated_message_history.idea,
                code=[],
            ),
        )
        iter_cost += cost

        # Process each generated idea
        trial_result_df = pd.DataFrame()
        validated_result_list: list[GeneratedResult] = []

        for i, result in enumerate(generate_first_results.results, start=1):
            idea_id = f"{trial_num}_{i}"
            validated_result, single_idea_result_series, validation_cost = (
                self.idea_processor.process_idea(result, idea_id)
            )
            iter_cost += validation_cost

            validated_result_list.append(validated_result)
            trial_result_df = pd.concat(
                [trial_result_df, single_idea_result_series.to_frame().T],
                axis=0,
            ).reset_index(drop=True)

        validated_results = GeneratedResults(results=validated_result_list)
        return validated_results, trial_result_df, iter_cost, self.context

    def update_context(self, context: RunContext) -> None:
        """Update the context for this trial executor."""
        self.context = context
        self.idea_processor.update_context(context)


class ExperimentReviewer(BaseExperimentReviewer):
    """Handles idea review functionality."""

    def __init__(
        self,
        dependencies: ExperimentDependencies,
        context: RunContext,
        review_metric: ReviewMetric,
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
        return ReviewIdeaPrompt(max_suggestion_num=max_suggestion_num).build(
            last_trial_num=last_trial_num,
            last_trial_results=last_trial_results,
            performance_review=performance_review,
        )


class QuantumKernelExperimentRunner(BaseExperimentRunner):
    """Main orchestrator for Quantum Kernel experiments."""

    def __init__(
        self,
        dependencies: ExperimentDependencies,
        review_metric: ReviewMetric = ReviewMetric.ACCURACY,
    ) -> None:
        super().__init__(dependencies)
        self.researcher: Researcher | None = None
        self.reviewer: ExperimentReviewer | None = None
        self.trial_executor: TrialExecutor | None = None
        self.review_metric = review_metric

    def initialize(
        self,
        max_trial_num: int,
        max_idea_num: int,
        max_suggestion_num: int,
        max_scoring_round: int,
        reflection_type: ReflectionType,
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
            self.deps, self.context, review_metric=self.review_metric
        )
        self.trial_executor = TrialExecutor(self.deps, self.context)

    def filtering_idea(
        self,
        research_mode: ResearchMode,
        filtering_steps: list[int],
        existing_research_report_path: str | None = None,
    ) -> None:
        if not self.context:
            raise ValueError(
                "Context not initialized. Call initialize() first."
            )

        if research_mode == ResearchMode.NONE:
            raise ValueError(
                'Idea filtering needs to specify research mode from "deep" or "light".'
            )

        # generate research report
        if existing_research_report_path is None:
            research_report, research_cost = Researcher(
                client=self.deps.llm_client, research_mode=research_mode
            ).research(
                user_prompt=RESEARCHER_USER_PROMPT,
                save_report_path=f"{self.context.gen_code_dirc}/{RESEARCH_REPORT_FILE_NAME}",
            )
        else:
            research_report, research_cost = Researcher.load_existing_report(
                existing_research_report_path
            )

        # generate multiple ideas from research report
        filtering_executor = FilteringExecutor(
            self.deps, self.context, filtering_steps
        )
        _, experiment_report, filtering_cost = (
            filtering_executor.execute_idea_filtering(research_report)
        )

        self.context = update_run_context(
            self.context,
            total_cost=self.context.total_cost + research_cost + filtering_cost,
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
                self.context, trial_cost, validated_results, trial_result_df = (
                    self._execute_trial(trial_num)
                )

                total_iter_cost = review_cost + trial_cost

                # Update context with trial results
                best_result_df = self._update_context_with_trial_results(
                    validated_results,
                    trial_result_df,
                    best_result_df,
                    total_iter_cost,
                )

                self._log_trial_completion(
                    trial_num, total_iter_cost, best_result_df
                )

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Error occurred: {e}")
        finally:
            self._log_experiment_completion(best_result_df)

    def _execute_trial(
        self, trial_num: int
    ) -> tuple[RunContext, float, GeneratedResults, pd.DataFrame]:
        """Execute a single trial with Quantum Kernel-specific pattern."""
        if not self.context or not self.trial_executor:
            raise ValueError("Components not initialized")

        # Sync context to trial executor before execution
        self.trial_executor.update_context(self.context)

        validated_results, trial_result_df, trial_cost, updated_context = (
            self.trial_executor.execute_trial(trial_num)
        )

        return updated_context, trial_cost, validated_results, trial_result_df

    def _update_context_with_trial_results(
        self,
        validated_results: GeneratedResults,
        trial_result_df: pd.DataFrame,
        best_result_df: pd.DataFrame,
        iter_cost: float,
    ) -> pd.DataFrame:
        """Update context with results from completed trial."""
        if not self.context:
            raise ValueError("Context not initialized")

        # Get best result from trial
        best_result_id = get_best_result_id_qkernel(trial_result_df)
        updated_best_result_df = cast(
            pd.DataFrame,
            pd.concat(
                [
                    best_result_df,
                    trial_result_df[
                        trial_result_df["idea_id"] == best_result_id
                    ],
                ],
                ignore_index=True,
            ),
        )

        best_result = validated_results.results[
            int(best_result_id.split("_")[-1]) - 1
        ]

        # Update context
        self.context = update_run_context(
            self.context,
            last_trial_results=format_trial_results(
                validated_results, trial_result_df
            ),
            score_histories=update_score_histories(
                self.context.score_histories, validated_results
            ),
            need_idea_review=True,
            best_idea_abstract=get_best_idea_abstract(best_result),
            total_cost=self.context.total_cost + iter_cost,
            eval_result_df=updated_best_result_df,
        )

        # Sync updated context to all components
        self._sync_context_to_components()

        return updated_best_result_df


@traceable(tags=["experiment"])
def run(
    llm_client: ChatClient,
    arxiv_db_client: PineconeClient | None,
    qml_db_client: PineconeClient | None,
    seed_code: str,
    experiment: qxmt.Experiment,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    filtering_steps: list[int],
    research_mode: ResearchMode,
    reflection_type: ReflectionType,
    max_reflection_round: int,
    max_scoring_round: int,
    review_metric: ReviewMetric = ReviewMetric.ACCURACY,
    existing_research_report_path: str | None = None,
) -> None:
    """Runs the quantum feature map generation and evaluation experiment.

    This function orchestrates the entire experiment process, including:
    1. Feature map idea research and filtering
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
        max_trial_num (int): Maximum number of trials to run
        max_idea_num (int): Maximum number of ideas to generate per trial
        max_suggestion_num (int): Maximum number of improvement suggestions per review
        filtering_steps (list[int]): Number of ideas for each steps of filtering
        research_mode (ResearchMode): Mode for initial idea research
        reflection_type (ReflectionType): Type of reflection to use
        max_reflection_round (int): Maximum number of reflection rounds per idea
        max_scoring_round (int): Maximum number of scoring rounds per idea
        review_metric (ReviewMetric): Metric to use for review
        existing_research_report_path (str | None): Path to existing research report markdown file

    Returns:
        None: Results are logged and saved during execution
    """
    dependencies = ExperimentDependencies(
        llm_client=llm_client,
        arxiv_db_client=arxiv_db_client,
        qml_db_client=qml_db_client,
        experiment=experiment,
    )

    runner = QuantumKernelExperimentRunner(
        dependencies, review_metric=review_metric
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
        research_mode != ResearchMode.NONE
        or existing_research_report_path is not None
    ):
        runner.filtering_idea(
            research_mode, filtering_steps, existing_research_report_path
        )

    runner.run_trials()
