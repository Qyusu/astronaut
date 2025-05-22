from typing import Any, cast

from loguru import logger
from qxmt.experiment import Experiment
from qxmt.experiment.schema import Evaluations

from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.logics.quantum_kernel import run
from astronaut.logics.strategies import QuantumStrategy


class QuantumKernelStrategy(QuantumStrategy):
    def __init__(
        self,
        llm_client: ChatClient,
        arxiv_db_client: PineconeClient | None,
        qml_db_client: PineconeClient | None,
        seed_code: str,
        experiment: Experiment,
        max_trial_num: int,
        max_idea_num: int,
        max_suggestion_num: int,
        max_reflection_round: int,
    ):
        self.llm_client = llm_client
        self.arxiv_db_client = arxiv_db_client
        self.qml_db_client = qml_db_client
        self.seed_code = seed_code
        self.experiment = experiment
        self.max_trial_num = max_trial_num
        self.max_idea_num = max_idea_num
        self.max_suggestion_num = max_suggestion_num
        self.max_reflection_round = max_reflection_round

    def execute_baseline(self, config_source: str) -> None:
        logger.info("Run Experiment by QXMT with Seed Feature Map...")
        _, baseline_result = self.experiment.run(config_source=config_source, add_results=False)
        baseline_evaluations = cast(Evaluations, baseline_result.evaluations)
        logger.info(
            f"""
            Base Score:
                - Validataion: {baseline_evaluations.validation}
                - Test: {baseline_evaluations.test}
            """
        )

    def execute_experiment(self, *args: Any, **kwargs: Any) -> None:
        return run(
            self.llm_client,
            self.arxiv_db_client,
            self.qml_db_client,
            self.seed_code,
            self.experiment,
            self.max_trial_num,
            self.max_idea_num,
            self.max_suggestion_num,
            self.max_reflection_round,
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
