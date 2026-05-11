import os
from textwrap import dedent
from typing import Any

import pandas as pd
import qxmt
from loguru import logger

from astronaut.configs import settings
from astronaut.constants import GENERATED_CODE_DIRC, GENERATED_CONFIG_DIRC, PAPER_PDF_DIRC
from astronaut.schema import (
    GeneratedIdea,
    GeneratedImpl,
    GeneratedResult,
    GeneratedResults,
    IdeaScore,
    MessageHistory,
    MessageHistoryNum,
    ModelVersions,
    RunContext,
    Score,
    ReflectionConfig,
    ReflectionType,
)


def setup_generated_dircs(
    experiment_name: str,
    config_dirc: str = GENERATED_CONFIG_DIRC,
    code_dirc: str = GENERATED_CODE_DIRC,
) -> tuple[str, str]:
    """Set up generated configuration and code directories for the experiment.

    Args:
        experiment_name: Name of the experiment
        config_dirc: Directory for generated configurations
        code_dirc: Directory for generated code

    Returns:
        tuple: (generated_config_directory, generated_code_directory)
    """
    gen_config_dirc = f"{config_dirc}/{experiment_name}"
    if not os.path.exists(gen_config_dirc):
        os.makedirs(gen_config_dirc)

    gen_code_dirc = f"{code_dirc}/{experiment_name}"
    if not os.path.exists(gen_code_dirc):
        os.makedirs(gen_code_dirc)

    return gen_config_dirc, gen_code_dirc


def initialize_run_context_base(
    experiment: qxmt.Experiment,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    max_scoring_round: int,
    reflection_type: ReflectionType,
    max_reflection_round: int,
    seed_code: str,
) -> dict[str, Any]:
    """Initialize the base run context data that is common across all experiment types.

    Args:
        experiment: QXMT experiment instance
        max_trial_num: Maximum number of trials
        max_idea_num: Maximum number of ideas per trial
        max_suggestion_num: Maximum number of suggestions
        max_scoring_round: Maximum number of scoring rounds
        reflection_type: Type of reflection to use
        max_reflection_round: Maximum number of reflection rounds
        seed_code: Initial seed code

    Returns:
        dict: Base context data that can be extended by specific experiment types
    """
    gen_config_dirc, gen_code_dirc = setup_generated_dircs(str(experiment.name))
    local_paper_dir = f"{PAPER_PDF_DIRC}/{settings.ARXIV_INDEX_NAME}"
    model_versions = ModelVersions(
        default=settings.DEFAULT_MODEL_VERSION,
        filtering=settings.FILTERING_MODEL_VERSION,
        idea=settings.IDEA_MODEL_VERSION,
        scoring=settings.SCORING_MODEL_VERSION,
        summary=settings.SUMMARY_MODEL_VERSION,
        expert=settings.EXPERT_MODEL_VERSION,
        reflection=settings.REFLECTION_MODEL_VERSION,
        reference=settings.REFERENCE_EXTRACTION_MODEL_VERSION,
        code=settings.CODE_MODEL_VERSION,
        validation=settings.VALIDATION_MODEL_VERSION,
        review=settings.REVIEW_MODEL_VERSION,
    )

    # Create reflection config based on max_reflection_round
    reflection_config = ReflectionConfig(
        type=reflection_type,
        max_rounds=max_reflection_round,
    )

    return {
        "gen_config_dirc": gen_config_dirc,
        "gen_code_dirc": gen_code_dirc,
        "local_paper_dir": local_paper_dir,
        "model_versions": model_versions,
        "max_trial_num": max_trial_num,
        "max_idea_num": max_idea_num,
        "max_suggestion_num": max_suggestion_num,
        "max_scoring_round": max_scoring_round,
        "reflection_config": reflection_config,
        "best_idea_abstract": "",
        "last_code": seed_code,
        "total_cost": 0.0,
        "need_idea_review": False,  # first iteration does not need idea review
        "review_comment": "",
        "last_trial_results": "",
        "score_histories": "",
        "message_history": MessageHistory(),
        "n_message_history": MessageHistoryNum(),
        "eval_result_df": pd.DataFrame(),
    }


def update_run_context(context: RunContext, **updates: Any) -> RunContext:
    """Update the run context with new values.

    Args:
        context: Current run context
        **updates: Key-value pairs to update in the context

    Returns:
        RunContext: Updated run context

    Raises:
        ValueError: If an invalid key is provided
    """
    updated_key_list = []
    for key in updates:
        if key in list(RunContext.model_fields.keys()):
            updated_key_list.append(key)
        else:
            raise ValueError(f"Invalid key: {key}")

    logger.debug(f"Update Run Context: {updated_key_list}")

    return context.model_copy(update=updates)


def get_best_result_id(
    result_df: pd.DataFrame,
    metric_col: str,
    id_col: str = "idea_id",
    minimize: bool = False,
) -> str:
    """Get the ID of the best result based on the specified metric.

    Args:
        result_df: DataFrame containing results
        metric_col: Column name for the metric to optimize
        id_col: Column name for the ID (default: "idea_id")
        minimize: Whether to minimize the metric (True) or maximize it (False)

    Returns:
        str: ID of the best result
    """
    if minimize:
        best_row_index = result_df[metric_col].idxmin()
    else:
        best_row_index = result_df[metric_col].idxmax()

    best_value = result_df.loc[best_row_index, id_col]
    return str(best_value)


def get_best_idea_abstract(best_idea: GeneratedResult) -> str:
    """Get formatted abstract of the best idea.

    Args:
        best_idea: The best generated result

    Returns:
        str: Formatted abstract string
    """
    template = dedent(
        """
        ### Idea Explanation:
        {idea_explanation}

        ### Idea Mathematical Expression:
        {idea_formula}
        """
    )

    return template.format(
        idea_explanation=best_idea.idea.explanation,
        idea_formula=best_idea.idea.formula,
    )


def update_score_histories(
    score_histories: str, generate_results: GeneratedResults
) -> str:
    """Update score histories with new results.

    Args:
        score_histories: Current score histories string
        generate_results: New generated results to add

    Returns:
        str: Updated score histories string
    """
    template = dedent(
        """
        {score_histories}

        - {algorithm_name}:
            - explanation:
                - {summary}

            - scores:
                {scores}
        """
    )

    for result in generate_results.results:
        score_histories = template.format(
            score_histories=score_histories,
            algorithm_name=result.idea.name,
            summary=result.idea.summary,
            scores=str(result.score).lstrip(),
        )

    return score_histories


def format_batch_results(
    ideas: list[GeneratedIdea],
    scores: list[IdeaScore | None],
    implements: list[GeneratedImpl],
) -> GeneratedResults:
    """Format batch results into a GeneratedResults object.

    Args:
        ideas: List of generated ideas
        scores: List of idea scores
        implements: List of generated implementations

    Returns:
        GeneratedResults: Formatted results
    """

    results = []
    for i in range(len(ideas)):
        result = GeneratedResult(
            idea=ideas[i], score=scores[i], implement=implements[i]
        )
        results.append(result)
    return GeneratedResults(results=results)
