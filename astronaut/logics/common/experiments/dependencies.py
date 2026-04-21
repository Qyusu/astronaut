import qxmt

from astronaut.db import PineconeClient
from astronaut.llm import ChatClient


class ExperimentDependencies:
    """Container for experiment dependencies."""

    def __init__(
        self,
        llm_client: ChatClient,
        arxiv_db_client: PineconeClient | None,
        qml_db_client: PineconeClient | None,
        experiment: qxmt.Experiment,
    ) -> None:
        self.llm_client = llm_client
        self.arxiv_db_client = arxiv_db_client
        self.qml_db_client = qml_db_client
        self.experiment = experiment
