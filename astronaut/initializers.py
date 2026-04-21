import os
import sys
from typing import Any

import qxmt
import yaml
from loguru import logger

import astronaut.constants as exp_const
from astronaut.configs import ExperimentSettings, settings
from astronaut.db import PineconeClient
from astronaut.llm import ChatClient, EmbeddingClient
from astronaut.llm.config import (
    AnthropicConfig,
    GoogleConfig,
    LLMProvider,
    OpenAIConfig,
)
from astronaut.logics import QuantumAlgorithmContext, StrategyType
from astronaut.logics.common import load_code
from astronaut.schema import TaskType


def initialize_llm_client() -> ChatClient:
    """Initialize the LLM client based on the configured platform."""
    if settings.CHAT_PLATFORM == LLMProvider.OPENAI:
        llm_config = OpenAIConfig(
            api_key=settings.OPENAI_API_KEY,
            default_model_version=settings.DEFAULT_MODEL_VERSION,
        )
    elif settings.CHAT_PLATFORM == LLMProvider.GOOGLE:
        llm_config = GoogleConfig(
            api_key=settings.GOOGLE_API_KEY,
            default_model_version=settings.DEFAULT_MODEL_VERSION,
        )
    elif settings.CHAT_PLATFORM == LLMProvider.ANTHROPIC:
        llm_config = AnthropicConfig(
            api_key=settings.ANTHROPIC_API_KEY,
            default_model_version=settings.DEFAULT_MODEL_VERSION,
        )
    else:
        raise ValueError(f"Invalid platform: {settings.CHAT_PLATFORM}")

    return ChatClient(llm_config)


def initialize_embedding_client() -> EmbeddingClient | None:
    """Initialize the embedding client based on the configured platform."""
    if settings.EMBEDDING_MODEL_VERSION is None:
        logger.info(
            "Embedding model version is not provided. Embedding client will not be initialized."
        )
        return None

    if settings.EMBEDDING_PLATFORM == LLMProvider.OPENAI:
        embed_api_key = settings.OPENAI_API_KEY
    else:
        raise ValueError(f"Invalid platform: {settings.EMBEDDING_PLATFORM}")

    return EmbeddingClient(
        platform=settings.EMBEDDING_PLATFORM,
        api_key=embed_api_key,
        embeddings_model_version=settings.EMBEDDING_MODEL_VERSION,
    )


def initialize_pinecone_clients(
    embed_client: EmbeddingClient | None,
) -> tuple[PineconeClient | None, PineconeClient | None]:
    """Initialize Pinecone clients for QML and arXiv databases."""
    if (
        settings.PINECONE_API_KEY is not None
        and settings.PENNLYLANE_INDEX_NAME is not None
    ):
        if embed_client is None:
            raise ValueError(
                """Embedding client is not provided that is required to initialize Pinecone client.
                Please check embedding settings on .env file."""
            )
        qml_db_client = PineconeClient(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PENNLYLANE_INDEX_NAME,
            embed_client=embed_client,
        )
    else:
        logger.info("Database for Pennylane documents is not provided.")
        qml_db_client = None

    if (
        settings.PINECONE_API_KEY is not None
        and settings.ARXIV_INDEX_NAME is not None
    ):
        if embed_client is None:
            raise ValueError(
                """Embedding client is not provided that is required to initialize Pinecone client.
                Please check embedding settings on .env file."""
            )
        arxiv_db_client = PineconeClient(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.ARXIV_INDEX_NAME,
            embed_client=embed_client,
        )
    else:
        logger.info("Database for academic papers is not provided.")
        arxiv_db_client = None

    return qml_db_client, arxiv_db_client


def initialize_clients() -> tuple[
    ChatClient,
    EmbeddingClient | None,
    PineconeClient | None,
    PineconeClient | None,
]:
    """Initialize all required clients for the application."""
    chat_client = initialize_llm_client()
    embed_client = initialize_embedding_client()
    qml_db_client, arxiv_db_client = initialize_pinecone_clients(embed_client)
    return chat_client, embed_client, qml_db_client, arxiv_db_client


def initialize_logger(
    experiment_name: str, console_log_level: str, file_log_level: str
) -> None:
    # reconfigure logger with specified console log level
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                "level": console_log_level.upper(),
                "colorize": True,
                "backtrace": True,
                "diagnose": True,
            }
        ]
    )

    logger_root_dir = f"../logs/{experiment_name}"
    if not os.path.exists(logger_root_dir):
        os.makedirs(logger_root_dir)

    logger.add(
        f"{logger_root_dir}/logfile_{experiment_name}.log",
        rotation="1 MB",
        compression="zip",
        level=file_log_level.upper(),
    )


def load_dataset_configs(
    config_path: str, experiment_settings: ExperimentSettings
) -> dict[str, Any]:
    """Load dataset configurations from the specified path."""

    dataset_configs: dict[str, Any] = {}
    if experiment_settings.dataset_config_path:
        dataset_config_path = experiment_settings.dataset_config_path
        # Handle relative path from experiment config file
        if not os.path.isabs(dataset_config_path):
            base_dir = os.path.dirname(os.path.abspath(config_path))
            dataset_config_path = os.path.join(base_dir, dataset_config_path)

        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, "r") as f:
                dataset_configs = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(
                f"Dataset config file not found: {dataset_config_path}"
            )
    return dataset_configs


def initialize_algorithm_context(
    task_type: TaskType,
    experiment_settings: ExperimentSettings,
    chat_client: ChatClient,
    arxiv_db_client: PineconeClient | None,
    qml_db_client: PineconeClient | None,
    experiment: qxmt.Experiment,
    dataset_configs: dict[str, Any],
) -> QuantumAlgorithmContext:
    """Initialize the quantum algorithm context based on the task type."""
    if task_type == TaskType.QUANTUM_KERNEL:
        seed_code_path = exp_const.QKERNEL_SEED_CODE_PATH
        strategy_type = StrategyType.QUANTUM_KERNEL
    elif task_type == TaskType.VQE:
        seed_code_path = exp_const.VQE_SEED_CODE_PATH
        strategy_type = StrategyType.VQE
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    seed_code = load_code(seed_code_path)

    return QuantumAlgorithmContext(
        strategy_type=strategy_type,
        llm_client=chat_client,
        arxiv_db_client=arxiv_db_client,
        qml_db_client=qml_db_client,
        seed_code=seed_code,
        experiment=experiment,
        dataset_configs=dataset_configs,
        validation_qubits=experiment_settings.validation_qubits,
        max_trial_num=experiment_settings.max_trial_num,
        max_idea_num=experiment_settings.max_idea_num,
        max_suggestion_num=experiment_settings.max_suggestion_num,
        expanding_steps=experiment_settings.expanding_steps,
        research_mode=experiment_settings.research_mode,
        reflection_type=experiment_settings.reflection_type,
        max_reflection_round=experiment_settings.max_reflection_round,
        max_scoring_round=experiment_settings.max_scoring_round,
        review_metric=experiment_settings.review_metric,
        existing_research_report_path=experiment_settings.existing_research_report_path,
    )
