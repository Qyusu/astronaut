import os
import sys

import qxmt
from loguru import logger

from astronaut.configs import ExperimentSettings, settings
from astronaut.constants import (
    BASE_CONFIG_PATH,
    MODEL_TYPE_PLACEHOLDER,
    QKERNEL_SEED_CODE_PATH,
    VQE_SEED_CODE_PATH,
)
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


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m astronaut.main <RUN_CONFIG_PATH.yaml>"
        )

    config_path = sys.argv[1]
    experiment_settings = ExperimentSettings.from_yaml(config_path)

    initialize_logger(
        experiment_settings.experiment_name,
        experiment_settings.console_log_level,
        experiment_settings.log_level,
    )

    # initialize all clients
    chat_client, _, qml_db_client, arxiv_db_client = initialize_clients()

    # initialize QXMT experiment
    experiment = qxmt.Experiment(
        name=experiment_settings.experiment_name,
        desc=experiment_settings.desc,
        auto_gen_mode=False,
    ).init()

    if experiment_settings.model_type == "quantum_kernel":
        seed_code = load_code(QKERNEL_SEED_CODE_PATH)
        context = QuantumAlgorithmContext(
            strategy_type=StrategyType.QUANTUM_KERNEL,
            llm_client=chat_client,
            arxiv_db_client=arxiv_db_client,
            qml_db_client=qml_db_client,
            seed_code=seed_code,
            experiment=experiment,
            max_trial_num=experiment_settings.max_trial_num,
            max_idea_num=experiment_settings.max_idea_num,
            max_suggestion_num=experiment_settings.max_suggestion_num,
            filtering_steps=experiment_settings.filtering_steps,
            research_mode=experiment_settings.research_mode,
            reflection_type=experiment_settings.reflection_type,
            max_reflection_round=experiment_settings.max_reflection_round,
            max_scoring_round=experiment_settings.max_scoring_round,
            existing_research_report_path=experiment_settings.existing_research_report_path,
        )
    elif experiment_settings.model_type == "vqe":
        seed_code = load_code(VQE_SEED_CODE_PATH)
        context = QuantumAlgorithmContext(
            strategy_type=StrategyType.VQE,
            llm_client=chat_client,
            arxiv_db_client=arxiv_db_client,
            qml_db_client=qml_db_client,
            seed_code=seed_code,
            experiment=experiment,
            max_trial_num=experiment_settings.max_trial_num,
            max_idea_num=experiment_settings.max_idea_num,
            max_suggestion_num=experiment_settings.max_suggestion_num,
            filtering_steps=experiment_settings.filtering_steps,
            research_mode=experiment_settings.research_mode,
            reflection_type=experiment_settings.reflection_type,
            max_reflection_round=experiment_settings.max_reflection_round,
            max_scoring_round=experiment_settings.max_scoring_round,
            existing_research_report_path=experiment_settings.existing_research_report_path,
        )
    else:
        raise ValueError(
            f"Invalid model type: {experiment_settings.model_type}"
        )

    # excute baseline and full experiment
    context.execute_baseline(
        BASE_CONFIG_PATH.replace(
            MODEL_TYPE_PLACEHOLDER, experiment_settings.model_type
        )
    )
    context.execute_experiment()


if __name__ == "__main__":
    main()
