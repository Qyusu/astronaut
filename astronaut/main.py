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
    SEED_CODE_PATH,
)
from astronaut.db import PineconeClient
from astronaut.experiment_utils import load_code
from astronaut.llm import ChatClient, EmbeddingClient
from astronaut.run_experiment import run

logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])


@click.command()
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
    experiment_name: str,
    desc: str,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    max_reflection_round: int,
) -> None:
    # initialize logger
    logger.add(f"../logs/logfile_{experiment_name}.log", rotation="1 MB", compression="zip")

    # initialize LLM client
    if settings.CHAT_PLATFORM == "openai":
        chat_api_key = settings.OPENAI_API_KEY
    elif settings.CHAT_PLATFORM == "google":
        chat_api_key = settings.GOOGLE_API_KEY
    elif settings.CHAT_PLATFORM == "anthropic":
        chat_api_key = settings.ANTHROPIC_API_KEY
    else:
        raise ValueError(f"Invalid platform: {settings.CHAT_PLATFORM}")

    chat_client = ChatClient(
        platform=settings.CHAT_PLATFORM,
        api_key=chat_api_key,
        default_model_version=settings.DEFAULT_MODEL_VERSION,
    )

    # initialize Embedding client
    if settings.EMBEDDING_PLATFORM == "openai":
        embed_api_key = settings.OPENAI_API_KEY
    else:
        raise ValueError(f"Invalid platform: {settings.EMBEDDING_PLATFORM}")

    embed_client = EmbeddingClient(
        platform=settings.EMBEDDING_PLATFORM,
        api_key=embed_api_key,
        embeddings_model_version=settings.EMBEDDING_MODEL_VERSION,
    )

    # initialize Vectorized DB client for pennylane docs and arXiv papers
    qml_db_client = PineconeClient(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PENNLYLANE_INDEX_NAME,
        embed_client=embed_client,
    )
    arxiv_db_client = PineconeClient(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.ARXIV_INDEX_NAME,
        embed_client=embed_client,
    )

    # load seed feature map
    seed_code = load_code(SEED_CODE_PATH)

    # initialize QXMT experiment
    experiment = qxmt.Experiment(
        name=experiment_name,
        desc=desc,
        auto_gen_mode=False,
    ).init()

    # run by seed feature map
    logger.info("Run Experiment by QXMT with Seed Feature Map...")
    _, baseline_result = experiment.run(config_source=BASE_CONFIG_PATH, add_results=False)
    logger.info(
        f"""
        Base Score:
            - Validataion: {baseline_result.evaluations.validation}
            - Test: {baseline_result.evaluations.test}
        """
    )

    # generate feature map code and run experiment
    run(
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


if __name__ == "__main__":
    main()
