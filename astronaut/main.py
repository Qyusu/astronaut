import sys

import click
import qxmt
from loguru import logger

from astronaut.configs import settings
from astronaut.constants import (
    BASE_CONFIG_PATH,
    DEFAULT_MAX_IDEA_NUM,
    DEFAULT_MAX_REFLECTION_ROUND,
    DEFAULT_MAX_SUGGESTION_NUM,
    DEFAULT_MAX_TRIAL_NUM,
    MODEL_TYPE_PLACEHOLDER,
    QKERNEL_SEED_CODE_PATH,
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

logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            "level": "INFO",
            "colorize": True,
            "backtrace": True,
            "diagnose": True,
        }
    ]
)


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
        logger.info("Embedding model version is not provided. Embedding client will not be initialized.")
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
    if settings.PINECONE_API_KEY is not None and settings.PENNLYLANE_INDEX_NAME is not None:
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

    if settings.PINECONE_API_KEY is not None and settings.ARXIV_INDEX_NAME is not None:
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


def initialize_clients() -> tuple[ChatClient, EmbeddingClient | None, PineconeClient | None, PineconeClient | None]:
    """Initialize all required clients for the application."""
    chat_client = initialize_llm_client()
    embed_client = initialize_embedding_client()
    qml_db_client, arxiv_db_client = initialize_pinecone_clients(embed_client)
    return chat_client, embed_client, qml_db_client, arxiv_db_client


@click.command()
@click.option(
    "--model_type",
    type=str,
    required=True,
    help="Specify the model type from 'quantum_kernel'.",
)
@click.option("--experiment_name", type=str, required=True, help="Specify the name of the experiment.")
@click.option(
    "--desc",
    type=str,
    default="Automatically generated code for quantum feature map.",
    required=False,
    help="Specify the description of the experiment.",
)
@click.option(
    "--max_trial_num",
    type=int,
    default=DEFAULT_MAX_TRIAL_NUM,
    required=False,
    help="Specify the maximum number of trials to run.",
)
@click.option(
    "--max_idea_num",
    type=int,
    default=DEFAULT_MAX_IDEA_NUM,
    required=False,
    help="Specify the maximum number of ideas to generate at one trial in the generation component.",
)
@click.option(
    "--max_suggestion_num",
    type=int,
    default=DEFAULT_MAX_SUGGESTION_NUM,
    required=False,
    help="Specify the maximum number of suggestions during idea review.",
)
@click.option(
    "--max_reflection_round",
    type=int,
    default=DEFAULT_MAX_REFLECTION_ROUND,
    required=False,
    help="Specify the maximum number of reflection rounds during idea refinement.",
)
def main(
    model_type: str,
    experiment_name: str,
    desc: str,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    max_reflection_round: int,
) -> None:
    # initialize logger
    logger.add(f"../logs/logfile_{experiment_name}.log", rotation="1 MB", compression="zip")

    # initialize all clients
    chat_client, _, qml_db_client, arxiv_db_client = initialize_clients()

    # initialize QXMT experiment
    experiment = qxmt.Experiment(
        name=experiment_name,
        desc=desc,
        auto_gen_mode=False,
    ).init()

    if model_type == "quantum_kernel":
        seed_code = load_code(QKERNEL_SEED_CODE_PATH)
        context = QuantumAlgorithmContext(
            strategy_type=StrategyType.QUANTUM_KERNEL,
            llm_client=chat_client,
            arxiv_db_client=arxiv_db_client,
            qml_db_client=qml_db_client,
            seed_code=seed_code,
            experiment=experiment,
            max_trial_num=max_trial_num,
            max_idea_num=max_idea_num,
            max_suggestion_num=max_suggestion_num,
            max_reflection_round=max_reflection_round,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # excute baseline and full experiment
    context.execute_baseline(BASE_CONFIG_PATH.replace(MODEL_TYPE_PLACEHOLDER, model_type))
    context.execute_experiment()


if __name__ == "__main__":
    main()
