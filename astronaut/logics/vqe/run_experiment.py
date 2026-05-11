import traceback
from typing import cast

import pandas as pd
import qxmt
import yaml
from langsmith import traceable
from loguru import logger
from qxmt.experiment.schema import VQERunTime
from qxmt.models.vqe import BaseVQE

from astronaut.constants import (
    BASE_CONFIG_PATH,
    DRYRUN_CONFIG_PATH,
    GENERATED_MODULE_ROOT,
    MODEL_TYPE_PLACEHOLDER,
    RESEARCH_REPORT_FILE_NAME,
    VQE_MODEL_TYPE,
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
from astronaut.logics.vqe import cost_function, generate_ansatz
from astronaut.logics.vqe.ansatz_scalability import AnsatzScalabilityDetector
from astronaut.logics.vqe.cost_function import MultiCostAggregator
from astronaut.logics.vqe.feedback import FeedbackFormatter, FeedbackFormatType
from astronaut.logics.vqe.generate_ansatz import generate_ansatz_codes
from astronaut.logics.vqe.target_molecules import (
    TARGET_MOLECULES,
    TargetMolecule,
)
from astronaut.logics.vqe.utils import (
    get_bond_lengths,
    get_coordinates,
    get_n_qubits_from_config,
)
from astronaut.params import GenerateIdeaParams
from astronaut.prompts.vqe import (
    RESEARCHER_USER_PROMPT,
    FilteringAnsatzIdeaPrompt,
    ResearchBasedAnsatzIdeaPrompt,
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

VQE_BASE_CONFIG_PATH = BASE_CONFIG_PATH.replace(
    MODEL_TYPE_PLACEHOLDER, VQE_MODEL_TYPE
)
VQE_DRYRUN_CONFIG_PATH = DRYRUN_CONFIG_PATH.replace(
    MODEL_TYPE_PLACEHOLDER, VQE_MODEL_TYPE
)


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
    """Initialize run context specific to VQE experiments."""
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
        model_type=VQE_MODEL_TYPE,
        n_qubits=get_n_qubits_from_config(VQE_BASE_CONFIG_PATH),
        **base_context,
    )


def get_best_result_id_vqe(
    result_df: pd.DataFrame, id_col: str = "idea_id"
) -> str:
    """Get best result ID for VQE experiments (minimize aggregated cost)."""
    return get_best_result_id(
        result_df, "aggregated_cost", id_col, minimize=True
    )


def gen_dummy_aggregated_result_series(idea_id: str) -> pd.Series:
    return pd.Series(
        {
            "idea_id": idea_id,
            "num_gates_order": "unknown",
            "num_params_order": "unknown",
            "aggregated_num_iter": 0,
            "aggregated_run_time": 0.0,
            "aggregated_cost": float("inf"),
        }
    )


class SingleIdeaExperimentRunner:
    """Handles bond length experiment execution for a single idea."""

    def __init__(
        self, experiment: qxmt.Experiment, context: RunContext
    ) -> None:
        self.experiment = experiment
        self.context = context

    def run_experiments_by_multiple_molecules(
        self,
        molecules_settings: dict[str, TargetMolecule],
        validated_result: GeneratedResult,
        idea_id: str,
        base_config_path: str = VQE_BASE_CONFIG_PATH,
    ) -> pd.DataFrame:
        molecules_result_df = pd.DataFrame()
        for molecule_settings in molecules_settings.values():
            logger.info(
                f"[idea_id={idea_id}, molecule={molecule_settings.name}] Run Experiment by QXMT..."
            )
            molecule_result_df = self._run_experiments_by_bond_length(
                validated_result,
                idea_id,
                base_config_path,
                molecule_settings,
            )
            molecules_result_df = pd.concat(
                [molecules_result_df, molecule_result_df], axis=0
            ).reset_index(drop=True)

        return molecules_result_df

    def _run_experiments_by_bond_length(
        self,
        validated_result: GeneratedResult,
        idea_id: str,
        config_path: str,
        molecule_settings: TargetMolecule,
    ) -> pd.DataFrame:
        """Run experiments across multiple bond lengths for a validated result."""
        bond_length_settings = molecule_settings.bond_length_settings
        bond_lengths = get_bond_lengths(
            min_bond_length=bond_length_settings.min_bond_length,
            max_bond_length=bond_length_settings.max_bond_length,
            sample_nums=bond_length_settings.sample_nums,
            equilibrium_bond_length=bond_length_settings.equilibrium_bond_length,
        )

        optimized_params = None
        bond_length_result_df = pd.DataFrame()
        for bond_length in bond_lengths:
            logger.info(f"Run Experiment with bond length: {bond_length}...")
            result_series, optimized_params = (
                self._run_single_bond_length_experiment(
                    validated_result,
                    config_path,
                    idea_id,
                    bond_length,
                    molecule_settings,
                    optimized_params,
                )
            )
            bond_length_result_df = pd.concat(
                [bond_length_result_df, result_series.to_frame().T], axis=0
            ).reset_index(drop=True)

        return bond_length_result_df

    def _run_single_bond_length_experiment(
        self,
        validated_result: GeneratedResult,
        config_path: str,
        idea_id: str,
        bond_length: float,
        molecule_settings: TargetMolecule,
        init_params: list[float] | None = None,
    ) -> tuple[pd.Series, list[float]]:
        """Run a single bond length experiment."""
        config_path = self._update_config(
            base_config_path=config_path,
            config_dirc=self.context.gen_config_dirc,
            experiment_name=str(self.experiment.name),
            module_suffix=idea_id,
            ansatz_name=validated_result.implement.class_name,
            summary=validated_result.idea.summary,
            molecule_settings=molecule_settings,
            bond_length=bond_length,
            init_params=init_params,
        )

        artifact, _ = self.experiment.run(config_source=config_path, n_jobs=1)
        vqe_model = cast(BaseVQE, artifact.model)
        optimized_params = self._get_optimized_params(vqe_model)
        eval_result_df = self.experiment.runs_to_dataframe(
            include_validation=False
        )
        run_time = cast(VQERunTime, self.experiment.exp_db.runs[-1].runtime)  # type: ignore
        num_iter = len(vqe_model.cost_history)

        bond_length_series = eval_result_df.iloc[-1].copy()
        bond_length_series["idea_id"] = idea_id
        bond_length_series["molecule_name"] = molecule_settings.name
        bond_length_series["num_iter"] = num_iter
        bond_length_series["run_time"] = run_time.optimize_seconds
        bond_length_series["bond_length"] = bond_length
        bond_length_series["cost"] = cost_function(
            final_energy=bond_length_series["final_cost"],
            exact_energy=bond_length_series[molecule_settings.exact_energy],
        )

        return bond_length_series, optimized_params

    def _get_optimized_params(self, vqe_model: BaseVQE) -> list[float]:
        return vqe_model.params_history[-1].tolist()

    def _update_config(
        self,
        base_config_path: str,
        config_dirc: str,
        experiment_name: str,
        module_suffix: str,
        ansatz_name: str,
        summary: str,
        molecule_settings: TargetMolecule,
        bond_length: float,
        init_params: list[float] | None = None,
    ) -> str:
        config = yaml.safe_load(open(base_config_path))
        config["description"] = summary

        config["hamiltonian"]["params"]["symbols"] = molecule_settings.symbols
        config["hamiltonian"]["params"]["coordinates"] = get_coordinates(
            molecule_settings.name, bond_length
        )
        config["hamiltonian"]["params"]["charge"] = (
            molecule_settings.structures.charge
        )
        config["hamiltonian"]["params"]["multi"] = (
            molecule_settings.structures.multi
        )
        config["hamiltonian"]["params"]["active_electrons"] = (
            molecule_settings.structures.active_electrons
        )
        config["hamiltonian"]["params"]["active_orbitals"] = (
            molecule_settings.structures.active_orbitals
        )

        energy_cache = molecule_settings.energy_cache
        if energy_cache is not None and bond_length in energy_cache:
            config["hamiltonian"]["params"]["hf_energy"] = energy_cache[
                bond_length
            ].hf
            config["hamiltonian"]["params"]["casci_energy"] = energy_cache[
                bond_length
            ].casci
            config["hamiltonian"]["params"]["casscf_energy"] = energy_cache[
                bond_length
            ].casscf
            config["hamiltonian"]["params"]["fci_energy"] = energy_cache[
                bond_length
            ].fci

        config["device"]["n_qubits"] = molecule_settings.n_qubits

        config["ansatz"]["module_name"] = (
            f"{GENERATED_MODULE_ROOT}.{experiment_name}.ansatz_{module_suffix}"
        )
        config["ansatz"]["implement_name"] = ansatz_name

        if init_params is not None:
            config["model"]["params"]["init_params"] = {
                "type": "custom",
                "values": init_params,
            }

        config_path = f"{config_dirc}/config_{module_suffix}.yaml"
        with open(config_path, "w") as file:
            yaml.dump(config, file)

        return config_path


class IdeaProcessor(BaseIdeaProcessor):
    """Handles processing of individual ideas including validation and experiment execution."""

    def __init__(
        self, dependencies: ExperimentDependencies, context: RunContext
    ) -> None:
        super().__init__(
            dependencies=dependencies,
            context=context,
            dry_run_config_path=VQE_DRYRUN_CONFIG_PATH,
            code_file_prefix="ansatz",
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
            dry_run_config_path=VQE_DRYRUN_CONFIG_PATH,
        )

        self.code_module_dirc = (
            f"{GENERATED_MODULE_ROOT}.{dependencies.experiment.name}"
        )

    def process_idea(
        self, result: GeneratedResult, idea_id: str
    ) -> tuple[GeneratedResult, pd.Series, pd.DataFrame, float]:
        """Process a single idea through validation and experiment execution with VQE-specific pattern."""

        try:
            # Use common validation and save logic
            validated_result, validation_cost = (
                self._common_validation_and_save(result, idea_id)
            )

            # VQE-specific: Run single idea experiment across multiple bond lengths
            single_idea_result_df = self.single_idea_experiment_runner.run_experiments_by_multiple_molecules(
                TARGET_MOLECULES,
                validated_result,
                idea_id,
                VQE_BASE_CONFIG_PATH,
            )

            # store ansatz spec and execution statistics
            num_gates_order, num_params_order = self._detect_ansatz_scalability(
                module_path=f"{self.code_module_dirc}.ansatz_{idea_id}",
                ansatz_name=validated_result.implement.class_name,
            )
            aggregated_series = self._aggregate_single_idea_result(
                idea_id,
                single_idea_result_df,
                num_gates_order,
                num_params_order,
            )

            return (
                validated_result,
                aggregated_series,
                single_idea_result_df,
                validation_cost,
            )

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Error occurred in processing idea {idea_id}: {e}")
            validated_result = GeneratedResult(
                idea=result.idea, score=result.score, implement=result.implement
            )
            aggregated_series = gen_dummy_aggregated_result_series(idea_id)
            return validated_result, aggregated_series, pd.DataFrame(), 0.0

    def _detect_ansatz_scalability(
        self, module_path: str, ansatz_name: str
    ) -> tuple[str, str]:
        logger.info("Detecting ansatz scalability...")
        detector = AnsatzScalabilityDetector(module_path, ansatz_name)
        num_gates_order = detector.get_n_gates_order()
        num_params_order = detector.detect_n_params_order()
        return num_gates_order, num_params_order

    def _aggregate_single_idea_result(
        self,
        idea_id: str,
        single_idea_result_df: pd.DataFrame,
        num_gates_order: str,
        num_params_order: str,
    ) -> pd.Series:
        aggregator = MultiCostAggregator(single_idea_result_df)
        return pd.Series(
            {
                "idea_id": idea_id,
                "num_gates_order": num_gates_order,
                "num_params_order": num_params_order,
                "aggregated_num_iter": aggregator.aggregate(
                    aggregate_method="mean", cost_col="num_iter"
                ),
                "aggregated_run_time": aggregator.aggregate(
                    aggregate_method="mean", cost_col="run_time"
                ),
                "aggregated_cost": aggregator.aggregate(
                    aggregate_method="mean", cost_col="cost"
                ),
            }
        )


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
        if len(self.filtering_steps) < 2:
            raise ValueError(
                "Filtering steps must be a list of at least two integers."
            )

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
        developer_prompt, user_prompt = ResearchBasedAnsatzIdeaPrompt().build(
            idea_num=seed_idea_num,
            research_report=research_report,
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
        developer_prompt, user_prompt = FilteringAnsatzIdeaPrompt().build(
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

        generated_codes, _, cost = generate_ansatz_codes(
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

        aggregated_result_df = pd.DataFrame()
        raw_result_df = pd.DataFrame()
        validated_result_list: list[GeneratedResult] = []
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

        logger.info(f"Fitering Result: {aggregated_result_df}")
        experiment_report = (
            f"------- {id_prefix} -------\n"
            + FeedbackFormatter(
                FeedbackFormatType.RAW_TEXT,
                GeneratedResults(results=validated_result_list),
                aggregated_result_df,
                raw_result_df,
            ).format()
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
    ) -> tuple[GeneratedResults, pd.DataFrame, pd.DataFrame, float, RunContext]:
        """Execute a single trial including idea generation and processing."""
        iter_cost = 0.0

        # Generate ansatz idea and code
        generated_idea_code, updated_message_history, cost = generate_ansatz(
            llm_client=self.deps.llm_client,
            arxiv_db_client=self.deps.arxiv_db_client,
            trial_num=trial_num,
            context=self.context,
        )

        # Update context with message history from ansatz generation
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

        trial_result_df = pd.DataFrame()
        all_idea_result_df = pd.DataFrame()
        validated_result_list: list[GeneratedResult] = []

        # Process each generated idea
        for i, result in enumerate(generated_idea_code.results, start=1):
            idea_id = f"{trial_num}_{i}"
            (
                validated_result,
                single_idea_aggregated_result_series,
                single_idea_result_df,
                validation_cost,
            ) = self.idea_processor.process_idea(result, idea_id)
            iter_cost += validation_cost

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

        validated_results = GeneratedResults(results=validated_result_list)
        return (
            validated_results,
            trial_result_df,
            all_idea_result_df,
            iter_cost,
            self.context,
        )

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
        """Create VQE specific review prompts."""
        target_molecule_names = ", ".join(TARGET_MOLECULES.keys())
        return ReviewIdeaPrompt(
            target_molecule_names=target_molecule_names,
            max_suggestion_num=max_suggestion_num,
        ).build(
            last_trial_num=last_trial_num,
            last_trial_results=last_trial_results,
            performance_review=performance_review,
        )


class VQEExperimentRunner(BaseExperimentRunner):
    """Main orchestrator for VQE experiments."""

    def __init__(
        self,
        dependencies: ExperimentDependencies,
        feedback_format_type: FeedbackFormatType = FeedbackFormatType.RAW_TEXT,
        review_metric: ReviewMetric = ReviewMetric.AGGREGATED_COST,
    ) -> None:
        super().__init__(dependencies)
        self.reviewer: ExperimentReviewer | None = None
        self.trial_executor: TrialExecutor | None = None
        self.feedback_format_type = feedback_format_type
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
                    trial_num, total_iter_cost, best_result_df
                )

        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Error occurred: {e}")
        finally:
            self._log_experiment_completion(best_result_df)

    def _execute_trial(
        self, trial_num: int
    ) -> tuple[RunContext, float, GeneratedResults, pd.DataFrame, pd.DataFrame]:
        """Execute a single trial with VQE-specific pattern."""
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
        validated_results: GeneratedResults,
        trial_result_df: pd.DataFrame,
        all_idea_result_df: pd.DataFrame,
        best_result_df: pd.DataFrame,
        iter_cost: float,
    ) -> pd.DataFrame:
        """Update context with results from completed trial."""
        if not self.context:
            raise ValueError("Context not initialized")

        # Get best result from trial
        best_result_id = get_best_result_id_vqe(trial_result_df)
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
            last_trial_results=FeedbackFormatter(
                self.feedback_format_type,
                validated_results,
                trial_result_df,
                all_idea_result_df,
            ).format(),
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
    feedback_format_type: FeedbackFormatType = FeedbackFormatType.RAW_TEXT,
    review_metric: ReviewMetric = ReviewMetric.AGGREGATED_COST,
    existing_research_report_path: str | None = None,
) -> None:
    """Runs the quantum variational ansatz generation and evaluation experiment.

    This function orchestrates the entire experiment process, including:
    1. Ansatz idea research and filtering
    2. Ansatz idea generation and review
    3. Code generation and validation
    4. Experiment execution and evaluation
    4. Performance tracking and improvement

    The process iterates for max_trial_num times, with each trial:
    - Generating multiple ansatz ideas
    - Validating and executing the generated code
    - Reviewing performance and suggesting improvements
    - Tracking costs and results

    Args:
        llm_client (ChatClient): Client for interacting with the language model
        arxiv_db_client (PineconeClient | None): Client for accessing academic paper database
        qml_db_client (PineconeClient | None): Client for accessing Pennylane documentation database
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
        feedback_format_type (FeedbackFormatType): Type of feedback format to use
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

    runner = VQEExperimentRunner(
        dependencies,
        feedback_format_type=feedback_format_type,
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
        research_mode != ResearchMode.NONE
        or existing_research_report_path is not None
    ):
        runner.filtering_idea(
            research_mode, filtering_steps, existing_research_report_path
        )

    runner.run_trials()
