import os
import sys
from textwrap import dedent
from typing import Any, Optional

import pandas as pd
import qxmt
import yaml
from langsmith import traceable
from loguru import logger

from astronaut.configs import settings
from astronaut.constants import (
    BASE_CONFIG_PATH,
    COMPLETED,
    GENERATED_CODE_DIRC,
    GENERATED_CONFIG_DIRC,
    GENERATED_MODULE_ROOT,
)
from astronaut.db import PineconeClient
from astronaut.experiment_utils import save_generated_feature_map_code
from astronaut.llm import ChatClient
from astronaut.logics import generate_feature_map, validate_generated_code
from astronaut.logics.review import ReviewIdea, ReviewPerformance
from astronaut.prompts import ReviewIdeaPrompt
from astronaut.schema import (
    GeneratedResult,
    GeneratedResults,
    IdeaScore,
    MessageHistory,
    MessageHistoryNum,
    ModelVersions,
    RunContext,
    Score,
)

logger.configure(handlers=[{"sink": sys.stderr, "format": "{time} {level} {message}", "level": "INFO"}])


def format_trial_results(generated_results: GeneratedResults, current_result_df: pd.DataFrame) -> str:
    template = dedent(
        """
        ## Idea Number: {idea_num}
        ### Idea Explanation:
        {idea_explanation}

        ### Idea Mathematical Expression:
        {idea_formula}

        ### Trial Results:
        #### Run Time: {run_time:.2f} [seconds]

        #### Result:
            - Accuracy: {accuracy:.2f}
            - Precision: {precision:.2f}
            - Recall: {recall:.2f}
            - F1 Score: {f1_score:.2f}
        """
    )

    idea_order = []
    past_traials = []
    generated_result_list = generated_results.results
    sorted_result_df = current_result_df.sort_values("accuracy_validation", ascending=False)
    for row in sorted_result_df.iterrows():
        idea_id = row[1]["idea_id"]
        idea_num = int(idea_id.split("_")[-1])
        idea_order.append(f'"Idea Number: {idea_num}"')

        current_result = generated_result_list[idea_num - 1]
        past_traial = template.format(
            idea_num=idea_num,
            idea_explanation=current_result.idea.explanation,
            idea_formula=current_result.idea.formula,
            run_time=row[1]["run_time"],
            accuracy=row[1]["accuracy_validation"],
            precision=row[1]["precision_validation"],
            recall=row[1]["recall_validation"],
            f1_score=row[1]["f1_score_validation"],
        )
        past_traials.append(past_traial)

    idea_order_str = "Current trial ideas sorted by accuracy (scores from highest to lowest): " + " > ".join(idea_order)
    past_traials_str = "\n".join(past_traials)

    return f"{idea_order_str}\n{past_traials_str}"


def update_score_histories(score_histories: str, generate_results: GeneratedResults) -> str:
    template = dedent(
        """
        {score_histories}

        - {feature_map_name}:
            - explanation:
                - {summary}

            - scores:
                {scores}
        """
    )

    for result in generate_results.results:
        score_histories = template.format(
            score_histories=score_histories,
            feature_map_name=result.idea.feature_map_name,
            summary=result.idea.summary,
            scores=str(result.score).lstrip(),
        )

    return score_histories


def update_config(
    config_dirc: str,
    experiment_name: str,
    module_suffix: str,
    feature_map_name: str,
    summary: str,
) -> str:
    config = yaml.safe_load(open(BASE_CONFIG_PATH))
    config["description"] = summary
    config["feature_map"]["module_name"] = f"{GENERATED_MODULE_ROOT}.{experiment_name}.feature_map_{module_suffix}"
    config["feature_map"]["implement_name"] = feature_map_name

    config_path = f"{config_dirc}/config_{module_suffix}.yaml"
    with open(config_path, "w") as file:
        yaml.dump(config, file)
    return config_path


def setup_generated_dircs(experiment_name: str) -> tuple[str, str]:
    gen_config_dirc = f"{GENERATED_CONFIG_DIRC}/{experiment_name}"
    if not os.path.exists(gen_config_dirc):
        os.makedirs(gen_config_dirc)

    gen_code_dirc = f"{GENERATED_CODE_DIRC}/{experiment_name}"
    if not os.path.exists(gen_code_dirc):
        os.makedirs(gen_code_dirc)

    return gen_config_dirc, gen_code_dirc


def review_idea(
    reviewer: ReviewIdea,
    last_trial_num: int,
    context: RunContext,
    performance_review: Optional[str],
) -> tuple[str, MessageHistory, float]:
    system_prompt, user_prompt = ReviewIdeaPrompt(
        llm_model_version=reviewer.model_version, max_suggestion_num=context.max_suggestion_num
    ).build(
        last_trial_num=last_trial_num,
        last_trial_results=context.last_trial_results,
        performance_review=performance_review,
    )

    result, history, cost = reviewer.review(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        message_history=context.message_history.review,
        n_history=context.n_message_history.review,
    )
    updated_message_history = MessageHistory(
        review=history, idea=context.message_history.idea, code=context.message_history.code
    )

    return result.review_comment(), updated_message_history, cost


def get_n_qubits_from_config(config_path: str) -> int:
    config = yaml.safe_load(open(config_path))
    n_qubits = config.get("device", {}).get("n_qubits", None)
    if n_qubits is None:
        raise ValueError("n_qubits is not found in the config.")
    return n_qubits


def initialize_run_context(
    experiment: qxmt.Experiment,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    max_reflection_round: int,
    seed_code: str,
) -> RunContext:
    gen_config_dirc, gen_code_dirc = setup_generated_dircs(str(experiment.name))
    model_versions = ModelVersions(
        default=settings.DEFAULT_MODEL_VERSION,
        idea=settings.IDEA_MODEL_VERSION,
        scoring=settings.SCORING_MODEL_VERSION,
        summary=settings.SUMMARY_MODEL_VERSION,
        reflection=settings.REFLECTION_MODEL_VERSION,
        code=settings.CODE_MODEL_VERSION,
        validation=settings.VALIDATION_MODEL_VERSION,
        review=settings.REVIEW_MODEL_VERSION,
        parser=settings.PARSER_MODEL_VERSION,
    )
    score_list = [
        IdeaScore(
            originality=Score(score=0.0, reason=""),
            feasibility=Score(score=0.0, reason=""),
            versatility=Score(score=0.0, reason=""),
        )
    ]

    return RunContext(
        gen_config_dirc=gen_config_dirc,
        gen_code_dirc=gen_code_dirc,
        model_versions=model_versions,
        n_qubits=get_n_qubits_from_config(BASE_CONFIG_PATH),
        max_trial_num=max_trial_num,
        max_idea_num=max_idea_num,
        max_suggestion_num=max_suggestion_num,
        max_reflection_round=max_reflection_round,
        best_idea_abstract="",
        last_code=seed_code,
        score_list=score_list,
        total_cost=0.0,
        need_idea_review=False,  # first iteration does not need idea review
        review_comment="",
        last_trial_results="",
        score_histories="",
        message_history=MessageHistory(),
        n_message_history=MessageHistoryNum(),
        eval_result_df=pd.DataFrame(),
    )


def update_run_context(context: RunContext, **updates: Any) -> RunContext:
    updated_key_list = []
    for key in updates:
        if key in list(RunContext.model_fields.keys()):
            updated_key_list.append(key)
        else:
            raise ValueError(f"Invalid key: {key}")

    logger.debug(f"Update Run Context: {updated_key_list}")

    return context.model_copy(update=updates)


def get_best_result_id(
    result_df: pd.DataFrame, metric_col: str = "accuracy_validation", id_col: str = "idea_id"
) -> str:
    best_row_index = result_df[metric_col].idxmax()
    best_value = result_df.loc[best_row_index, id_col]

    return str(best_value)


def get_best_idea_abstract(best_idea: GeneratedResult) -> str:
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


def gen_dummy_result_series(idea_id: str) -> pd.Series:
    return pd.Series(
        {
            "run_id": -1,
            "idea_id": idea_id,
            "run_time": 0.0,
            "accuracy_validation": 0.0,
            "precision_validation": 0.0,
            "recall_validation": 0.0,
            "f1_score_validation": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
    )


@traceable(tags=["experiment"])
def run(
    llm_client: ChatClient,
    arxiv_db_client: PineconeClient,
    qml_db_client: PineconeClient,
    seed_code: str,
    experiment: qxmt.Experiment,
    max_trial_num: int,
    max_idea_num: int,
    max_suggestion_num: int,
    max_reflection_round: int,
) -> None:
    """Runs the quantum feature map generation and evaluation experiment.

    This function orchestrates the entire experiment process, including:
    1. Feature map idea generation and review
    2. Code generation and validation
    3. Experiment execution and evaluation
    4. Performance tracking and improvement

    The process iterates for max_trial_num times, with each trial:
    - Generating multiple feature map ideas
    - Validating and executing the generated code
    - Reviewing performance and suggesting improvements
    - Tracking costs and results

    Args:
        llm_client (ChatClient): Client for interacting with the language model
        arxiv_db_client (PineconeClient): Client for accessing academic paper database
        qml_db_client (PineconeClient): Client for accessing Pennylane documentation database
        seed_code (str): Initial code to start the experiment
        experiment (qxmt.Experiment): QXMT experiment object for execution
        max_trial_num (int): Maximum number of trials to run
        max_idea_num (int): Maximum number of ideas to generate per trial
        max_suggestion_num (int): Maximum number of improvement suggestions per review
        max_reflection_round (int): Maximum number of reflection rounds per idea

    Returns:
        None: Results are logged and saved during execution
    """
    context = initialize_run_context(
        experiment, max_trial_num, max_idea_num, max_suggestion_num, max_reflection_round, seed_code
    )
    idea_reviewer = ReviewIdea(
        client=llm_client,
        model_version=context.model_versions.review,
        parser_model_version=context.model_versions.parser,
    )
    performance_reviewer = ReviewPerformance()
    best_result_df = pd.DataFrame()

    try:
        for i in range(max_trial_num):
            iter_cost = 0.0
            trial_num = i + 1
            logger.info(f"Start trial {trial_num}...")

            # review last feature map idea
            if context.need_idea_review:
                performance_review = performance_reviewer.review(context.eval_result_df, context.score_list)
                updated_review_comment, updated_message_history, cost = review_idea(
                    reviewer=idea_reviewer,
                    last_trial_num=trial_num - 1,
                    context=context,
                    performance_review=performance_review,
                )
                context = update_run_context(
                    context, message_history=updated_message_history, review_comment=updated_review_comment
                )
                iter_cost += cost

                if updated_review_comment == COMPLETED:
                    logger.info("Work is done. The idea generation is completed.")
                    break

            # generate feature map idea and code
            generate_first_results, updated_message_history, cost = generate_feature_map(
                llm_client=llm_client,
                arxiv_db_client=arxiv_db_client,
                trial_num=trial_num,
                context=context,
            )
            context = update_run_context(context, message_history=updated_message_history)
            iter_cost += cost

            iter_result_df = pd.DataFrame()
            validated_result_list: list[GeneratedResult] = []
            for i, result in enumerate(generate_first_results.results, start=1):
                idea_id = f"{trial_num}_{i}"
                try:
                    # validate generated code
                    validated_result, validation_cost = validate_generated_code(
                        llm_client=llm_client,
                        db_client=qml_db_client,
                        context=context,
                        generate_result=result,
                        experiment=experiment,
                    )
                    iter_cost += validation_cost
                    logger.debug(f"[trial={trial_num}, idea_num={i}] Generated Result: {validated_result}")

                    # save generated code
                    save_generated_feature_map_code(
                        f"{context.gen_code_dirc}/feature_map_{idea_id}.py", validated_result.implement.code
                    )

                    # update experiment config by validated result
                    logger.info(f"[trial]={trial_num}, idea_num={i}] Run Experiment by QXMT...")
                    config_path = update_config(
                        config_dirc=context.gen_config_dirc,
                        experiment_name=str(experiment.name),
                        module_suffix=idea_id,
                        feature_map_name=validated_result.implement.class_name,
                        summary=validated_result.idea.summary,
                    )

                    # run experimet by QXMT
                    experiment.run(config_source=config_path, n_jobs=1)
                    eval_result_df = experiment.runs_to_dataframe(include_validation=True)
                    run_time = experiment.exp_db.runs[-1].runtime  # type: ignore
                    current_series = eval_result_df.iloc[-1].copy()
                    current_series["idea_id"] = idea_id
                    current_series["run_time"] = run_time.train_seconds
                except Exception as e:
                    logger.error(f"Error occurred in Run: {e}")
                    validated_result = GeneratedResult(idea=result.idea, score=result.score, implement=result.implement)
                    current_series = gen_dummy_result_series(idea_id)
                    continue
                finally:
                    validated_result_list.append(validated_result)
                    iter_result_df = pd.concat([iter_result_df, current_series.to_frame().T], axis=0).reset_index(
                        drop=True
                    )

            # get best idea from the generated results
            best_result_id = get_best_result_id(iter_result_df)
            best_result_df = pd.concat([best_result_df, iter_result_df[iter_result_df["idea_id"] == best_result_id]])
            validated_results = GeneratedResults(results=validated_result_list)
            best_result = validated_result_list[int(best_result_id.split("_")[-1]) - 1]

            # update by current iteration results
            context = update_run_context(
                context,
                last_trial_results=format_trial_results(validated_results, iter_result_df),
                score_histories=update_score_histories(context.score_histories, validated_results),
                need_idea_review=True,
                best_idea_abstract=get_best_idea_abstract(best_result),
                score_list=context.score_list + [best_result.score],
                total_cost=context.total_cost + iter_cost,
                eval_result_df=best_result_df,
            )
            logger.info(f"Trial {trial_num} is done. [COST]: ${iter_cost:.2f}")
            logger.info(f"Evaluation Result:\n{best_result_df}")
            logger.info("######################################################")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        logger.info(f"All trials are done. Total Cost: ${context.total_cost:.2f}")
        logger.info(f"All trials Result:\n{best_result_df}")
