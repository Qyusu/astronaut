from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astronaut.logics.common import ReviewMetric
from astronaut.logics.strategies import QuantumStrategy
from astronaut.logics.vqe.run_experiment import run, run_baseline

if TYPE_CHECKING:
    from qxmt.experiment import Experiment

    from astronaut.db import PineconeClient
    from astronaut.llm import ChatClient
    from astronaut.schema import ReflectionType, ResearchMode


class VQEStrategy(QuantumStrategy):
    def __init__(
        self,
        llm_client: "ChatClient",
        arxiv_db_client: "PineconeClient" | None,
        qml_db_client: "PineconeClient" | None,
        seed_code: str,
        experiment: "Experiment",
        dataset_configs: dict[str, Any],
        validation_qubits: list[int],
        max_trial_num: int,
        max_idea_num: int,
        max_suggestion_num: int,
        expanding_steps: list[int],
        research_mode: "ResearchMode",
        reflection_type: "ReflectionType",
        max_reflection_round: int,
        max_scoring_round: int,
        review_metric: ReviewMetric = ReviewMetric.NONE,
        existing_research_report_path: str | None = None,
    ):
        self.llm_client = llm_client
        self.arxiv_db_client = arxiv_db_client
        self.qml_db_client = qml_db_client
        self.seed_code = seed_code
        self.experiment = experiment
        self.dataset_configs = dataset_configs
        self.validation_qubits = validation_qubits
        self.max_trial_num = max_trial_num
        self.max_idea_num = max_idea_num
        self.max_suggestion_num = max_suggestion_num
        self.expanding_steps = expanding_steps
        self.research_mode = research_mode
        self.reflection_type = reflection_type
        self.max_reflection_round = max_reflection_round
        self.max_scoring_round = max_scoring_round
        self.review_metric = review_metric
        self.existing_research_report_path = existing_research_report_path

    def execute_baseline(self, *args: Any, **kwargs: Any) -> None:
        return run_baseline(
            experiment=self.experiment,
            dataset_configs=self.dataset_configs,
        )

    def execute_experiment(self, *args: Any, **kwargs: Any) -> None:
        return run(
            llm_client=self.llm_client,
            arxiv_db_client=self.arxiv_db_client,
            qml_db_client=self.qml_db_client,
            seed_code=self.seed_code,
            experiment=self.experiment,
            dataset_configs=self.dataset_configs,
            validation_qubits=self.validation_qubits,
            max_trial_num=self.max_trial_num,
            max_idea_num=self.max_idea_num,
            max_suggestion_num=self.max_suggestion_num,
            expanding_steps=self.expanding_steps,
            research_mode=self.research_mode,
            reflection_type=self.reflection_type,
            max_reflection_round=self.max_reflection_round,
            max_scoring_round=self.max_scoring_round,
            review_metric=self.review_metric,
            existing_research_report_path=self.existing_research_report_path,
        )

    def validate(self) -> bool:
        # check connection to arxiv DB
        if self.arxiv_db_client is not None:
            arxiv_db_connected = self.arxiv_db_client.check_connection()
        else:
            arxiv_db_connected = True

        # check connection to qml DB
        if self.qml_db_client is not None:
            qml_db_connected = self.qml_db_client.check_connection()
        else:
            qml_db_connected = True

        # check experiment initialization
        is_initialized = self.experiment.runs_to_dataframe().empty

        return arxiv_db_connected and qml_db_connected and is_initialized
