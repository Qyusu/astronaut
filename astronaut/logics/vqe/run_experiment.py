from __future__ import annotations

import copy
import traceback
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import qxmt
import yaml
from langsmith import traceable
from loguru import logger

import astronaut.constants as exp_const
import astronaut.logics.common as common_logics
import astronaut.logics.common.experiments as exp_common
import astronaut.logics.vqe.constants as vqe_const
import astronaut.logics.vqe.schema as vqe_schema
import astronaut.logics.vqe.utils as vqe_utils
import astronaut.prompts.vqe as vqe_prompts
import astronaut.schema as base_schema
from astronaut.logics.vqe.ansatz_scalability import AnsatzScalabilityDetector
from astronaut.logics.vqe.cost_function import cost_function
from astronaut.logics.vqe.feedback import FeedbackFormatter, FeedbackFormatType
from astronaut.logics.vqe.generate_ansatz import (
    generate_ansatz,
    generate_ansatz_codes,
)
from astronaut.logics.vqe.molecule_configs import (
    TargetMolecule,
    get_molecule_config_from_qubits,
)

if TYPE_CHECKING:
    from qxmt.experiment.schema import VQEEvaluations, VQERunTime
    from qxmt.models.vqe import BaseVQE

    from astronaut.db import PineconeClient
    from astronaut.llm import ChatClient
    from astronaut.schema import GeneratedResult, GeneratedResults

VQE_BASE_CONFIG_PATH = exp_const.BASE_CONFIG_PATH.replace(
    exp_const.TASK_TYPE_PLACEHOLDER, exp_const.VQE_MODEL_TYPE
)
VQE_DRYRUN_CONFIG_PATH = exp_const.DRYRUN_CONFIG_PATH.replace(
    exp_const.TASK_TYPE_PLACEHOLDER, exp_const.VQE_MODEL_TYPE
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
    """Initialize run context specific to VQE experiments."""
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


def get_best_result_id_vqe(result_df: pd.DataFrame, id_col: str) -> str:
    """Get best result ID for VQE experiments (minimize aggregated cost)."""
    return exp_common.get_best_result_id(
        result_df, vqe_const.AGGREGATED_COST_COL, id_col, minimize=True
    )


def gen_dummy_aggregated_result_series(idea_id: str) -> pd.Series:
    return pd.Series(
        vqe_schema.VQEAggregatedResult(
            idea_id=idea_id,
            num_gates_order="unknown",
            num_params_order="unknown",
            aggregated_num_iter=0,
            aggregated_run_time=0.0,
            aggregated_cost=float("inf"),
        ).model_dump()
    )


class SingleIdeaExperimentRunner:
    """Handles bond length experiment execution for a single idea."""

    def __init__(
        self,
        experiment: qxmt.Experiment,
        context: base_schema.RunContext,
        dataset_configs: dict[str, Any],
    ) -> None:
        self.experiment = experiment
        self.context = context
        self.dataset_configs = dataset_configs

    def run_experiments_by_multiple_molecules(
        self,
        validated_result: GeneratedResult,
        idea_id: str,
        base_config_path: str,
    ) -> pd.DataFrame:
        molecules_result_df = pd.DataFrame()
        for mol_name, config_dict in self.dataset_configs.items():
            logger.info(
                f"[idea_id={idea_id}, molecule={mol_name}] Run Experiment by QXMT..."
            )
            molecule_settings = TargetMolecule(**config_dict)
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
        bond_lengths = vqe_utils.get_bond_lengths(
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
        vqe_model = cast("BaseVQE", artifact.model)
        optimized_params = self._get_optimized_params(vqe_model)
        eval_result_df = self.experiment.runs_to_dataframe(
            include_validation=False
        )
        run_time = cast("VQERunTime", self.experiment.exp_db.runs[-1].runtime)  # type: ignore
        num_iter = len(vqe_model.cost_history)

        result = vqe_schema.VQEExperimentResult(
            **eval_result_df.iloc[-1].to_dict(),
            idea_id=idea_id,
            molecule_name=molecule_settings.name,
            num_iter=num_iter,
            run_time=run_time.optimize_seconds,
            bond_length=bond_length,
            cost=cost_function(
                final_energy=eval_result_df.iloc[-1][
                    vqe_const.QXMT_FINAL_COST_COL
                ],
                exact_energy=eval_result_df.iloc[-1][
                    molecule_settings.exact_energy
                ],
            ),
        )
        bond_length_series = pd.Series(result.model_dump())

        return bond_length_series, optimized_params

    def _get_optimized_params(self, vqe_model: "BaseVQE") -> list[float]:
        params = vqe_model.params_history[-1]
        return params.flatten().tolist()

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
        config[vqe_const.QXMT_CONFIG_KEY_DESCRIPTION] = summary

        vqe_utils.update_config_for_molecule(
            config, molecule_settings, bond_length
        )

        config[vqe_const.QXMT_CONFIG_KEY_ANSATZ][
            vqe_const.QXMT_CONFIG_KEY_MODULE_NAME
        ] = f"{exp_const.GENERATED_MODULE_ROOT}.{experiment_name}.{vqe_const.CODE_FILE_PREFIX}_{module_suffix}"
        config[vqe_const.QXMT_CONFIG_KEY_ANSATZ][
            vqe_const.QXMT_CONFIG_KEY_IMPLEMENT_NAME
        ] = ansatz_name

        if init_params is not None:
            config[vqe_const.QXMT_CONFIG_KEY_MODEL][
                vqe_const.QXMT_CONFIG_KEY_PARAMS
            ][vqe_const.QXMT_CONFIG_KEY_INIT_PARAMS] = {
                vqe_const.INIT_PARAMS_TYPE: vqe_const.INIT_PARAMS_TYPE_CUSTOM,
                vqe_const.INIT_PARAMS_VALUES: init_params,
            }

        config_path = (
            f"{config_dirc}/{vqe_const.CONFIG_FILE_PREFIX}_{module_suffix}.yaml"
        )
        with open(config_path, "w") as file:
            yaml.dump(config, file)

        return config_path


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
            dry_run_config_path=VQE_DRYRUN_CONFIG_PATH,
            code_file_prefix=vqe_const.CODE_FILE_PREFIX,
        )
        self.target_dataset_names = list(dataset_configs.keys())
        self.dataset_configs = {
            name: TargetMolecule(**config)
            for name, config in dataset_configs.items()
        }
        self.validation_qubits = validation_qubits
        self.single_idea_experiment_runner = SingleIdeaExperimentRunner(
            dependencies.experiment, context, dataset_configs
        )

        self.validation_orchestrator = common_logics.CodeValidationOrchestrator(
            llm_client=dependencies.llm_client,
            db_client=dependencies.qml_db_client,
            context=context,
            experiment=dependencies.experiment,
            dry_run_config_path=VQE_DRYRUN_CONFIG_PATH,
        )

        self.code_module_dirc = (
            f"{exp_const.GENERATED_MODULE_ROOT}.{dependencies.experiment.name}"
        )

    def _build_dry_run_config_list(self, implement_name: str) -> list[dict]:
        """Build dry run config list for multiple qubit settings."""
        base_config = self._load_dry_run_base_config(implement_name)
        min_mol_config = self._get_min_qubit_molecule()
        qubits_list = self.validation_qubits
        if not qubits_list:
            qubits_list = [min_mol_config.n_qubits]

        return [
            self._create_dry_run_config_for_qubits(
                base_config, qubits, min_mol_config
            )
            for qubits in qubits_list
        ]

    def _load_dry_run_base_config(self, implement_name: str) -> dict[str, Any]:
        """Load and prepare the base configuration for dry runs."""
        with open(self.dry_run_config_path, "r") as f:
            base_config = yaml.safe_load(f)

        base_config[vqe_const.QXMT_CONFIG_KEY_ANSATZ][
            vqe_const.QXMT_CONFIG_KEY_MODULE_NAME
        ] = f"{exp_const.GENERATED_MODULE_ROOT}.{self.deps.experiment.name}.{vqe_const.DRYRUN_FILE_PREFIX}"
        base_config[vqe_const.QXMT_CONFIG_KEY_ANSATZ][
            vqe_const.QXMT_CONFIG_KEY_IMPLEMENT_NAME
        ] = implement_name
        return base_config

    def _get_min_qubit_molecule(self) -> TargetMolecule:
        """Get the molecule configuration with the minimum number of qubits."""
        return min(self.dataset_configs.values(), key=lambda x: x.n_qubits)

    def _create_dry_run_config_for_qubits(
        self,
        base_config: dict[str, Any],
        qubits: int,
        fallback_mol_config: TargetMolecule,
    ) -> dict[str, Any]:
        """Create a dry run configuration for a specific number of qubits."""
        cfg = copy.deepcopy(base_config)
        mol_config = get_molecule_config_from_qubits(
            qubits, self.dataset_configs
        )

        if mol_config is None:
            # Use fallback (min qubit molecule) if no matching config found
            mol_config = fallback_mol_config.model_copy(
                update={"n_qubits": qubits}
            )

        vqe_utils.update_config_for_molecule(
            cfg,
            mol_config,
            mol_config.bond_length_settings.equilibrium_bond_length,
        )
        return cfg

    def _run_experiments(
        self, validated_result: GeneratedResult, idea_id: str
    ) -> pd.DataFrame:
        """Run experiments for the validated result."""
        return self.single_idea_experiment_runner.run_experiments_by_multiple_molecules(
            validated_result,
            idea_id,
            VQE_BASE_CONFIG_PATH,
        )

    def _aggregate_results(
        self,
        idea_id: str,
        raw_result_df: pd.DataFrame,
        validated_result: GeneratedResult,
    ) -> pd.Series:
        """Aggregate experiment results."""
        num_gates_order, num_params_order = self._detect_ansatz_scalability(
            module_path=f"{self.code_module_dirc}.{vqe_const.CODE_FILE_PREFIX}_{idea_id}",
            ansatz_name=validated_result.implement.class_name,
        )
        return self._aggregate_single_idea_result(
            idea_id,
            raw_result_df,
            num_gates_order,
            num_params_order,
        )

    def _create_dummy_result(
        self, idea_id: str
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Create dummy results in case of failure."""
        aggregated_series = gen_dummy_aggregated_result_series(idea_id)
        return aggregated_series, pd.DataFrame()

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
        aggregation_source = vqe_schema.VQEAggregatedResult.aggregation_source()
        aggregated_values = {
            agg_field: float(single_idea_result_df[source_field].mean())
            for agg_field, source_field in aggregation_source.items()
        }

        return pd.Series(
            vqe_schema.VQEAggregatedResult(
                idea_id=idea_id,
                num_gates_order=num_gates_order,
                num_params_order=num_params_order,
                **aggregated_values,
            ).model_dump()
        )


class ExperimentReviewer(exp_common.ExperimentReviewer):
    """Handles idea review functionality."""

    def __init__(
        self,
        dependencies: exp_common.ExperimentDependencies,
        context: base_schema.RunContext,
        review_metric: common_logics.ReviewMetric,
        target_molecule_names_str: str,
    ) -> None:
        super().__init__(dependencies, context, review_metric)
        self.target_molecule_names_str = target_molecule_names_str

    def _review_prompt_builder_fn(
        self,
        max_suggestion_num: int,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Create VQE specific review prompts."""
        return vqe_prompts.ReviewIdeaPrompt(
            target_molecule_names=self.target_molecule_names_str,
            max_suggestion_num=max_suggestion_num,
        ).build(
            last_trial_num=last_trial_num,
            last_trial_results=last_trial_results,
            performance_review=performance_review,
        )


class VQEExperimentRunner(exp_common.BaseExperimentRunner):
    """Main orchestrator for VQE experiments."""

    def __init__(
        self,
        dependencies: exp_common.ExperimentDependencies,
        dataset_configs: dict[str, Any],
        validation_qubits: list[int],
        review_metric: common_logics.ReviewMetric,
        feedback_format_type: FeedbackFormatType,
    ) -> None:
        super().__init__(dependencies)
        self.dataset_configs = dataset_configs
        self.validation_qubits = validation_qubits
        self.reviewer: ExperimentReviewer | None = None
        self.trial_executor: exp_common.TrialExecutor | None = None
        self.review_metric = review_metric
        self.feedback_format_type = feedback_format_type
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
            target_molecule_names_str=", ".join(
                list(self.dataset_configs.keys())
            ),
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
            idea_generator_fn=partial(
                generate_ansatz, dataset_configs=self.dataset_configs
            ),
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
                user_prompt=vqe_prompts.RESEARCHER_USER_PROMPT,
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
            r: vqe_prompts.ResearchBasedAnsatzIdeaPrompt().build(n, r),
            expanding_prompt_factory=lambda n,
            r: vqe_prompts.ExpandingAnsatzIdeaPrompt().build(n, r),
            code_generator_fn=generate_ansatz_codes,
            feedback_formatter_fn=lambda r, agg, raw: FeedbackFormatter(
                FeedbackFormatType.RAW_TEXT, r, agg, raw
            ).format(),
            seed_phase_id_prefix=vqe_const.SEED_PHASE_ID_PREFIX,
            expanding_phase_id_prefix=vqe_const.EXPANDING_PHASE_ID_PREFIX,
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
        GeneratedResults,
        pd.DataFrame,
        pd.DataFrame,
    ]:
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
        best_result_id = get_best_result_id_vqe(
            trial_result_df, vqe_const.IDEA_ID_COL
        )
        updated_best_result_df = cast(
            pd.DataFrame,
            pd.concat(
                [
                    best_result_df,
                    trial_result_df[
                        trial_result_df[vqe_const.IDEA_ID_COL] == best_result_id
                    ],
                ],
                ignore_index=True,
            ),
        )

        self.context = exp_common.update_run_context(
            self.context,
            last_trial_results=FeedbackFormatter(
                self.feedback_format_type,
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

    def config_updater_fn(
        config: dict[str, Any], molecule_settings_dict: Any
    ) -> None:
        if isinstance(molecule_settings_dict, dict):
            molecule_settings = TargetMolecule(**molecule_settings_dict)
        else:
            molecule_settings = molecule_settings_dict

        vqe_utils.update_config_for_molecule(
            config,
            molecule_settings,
            molecule_settings.bond_length_settings.equilibrium_bond_length,
        )

    def log_result_fn(target_name: str, baseline_result: Any) -> None:
        baseline_evaluations = cast(
            "VQEEvaluations", baseline_result.evaluations
        )
        logger.info(
            f"""
                Base Score [{target_name}]:
                    {baseline_evaluations}
                """
        )

    exp_common.BaselineExecutor(
        experiment=experiment,
        dataset_configs=dataset_configs,
        config_updater_fn=config_updater_fn,
        log_result_fn=log_result_fn,
    ).execute(config_source=VQE_BASE_CONFIG_PATH)


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
    feedback_format_type: FeedbackFormatType = FeedbackFormatType.RAW_TEXT,
    existing_research_report_path: str | None = None,
) -> None:
    """Runs the quantum variational ansatz generation and evaluation experiment.

    This function orchestrates the entire experiment process, including:
    1. Ansatz idea research and expanding
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
        target_dataset_names (list[str]): List of target molecule names for evaluation
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
        feedback_format_type (FeedbackFormatType): Type of feedback format to use
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

    runner = VQEExperimentRunner(
        dependencies,
        dataset_configs=dataset_configs,
        validation_qubits=validation_qubits,
        review_metric=review_metric,
        feedback_format_type=feedback_format_type,
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
