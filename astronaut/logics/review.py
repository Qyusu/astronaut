import json
from textwrap import dedent
from typing import Optional

import pandas as pd
from langsmith import traceable
from loguru import logger

from astronaut.constants import (
    GPT_MAX_TOKENS,
    REASONING_MAX_TOKENS,
    REASONING_SERIES,
    STABLE_REASONING_MODEL_VERSIONS,
)
from astronaut.llm import ChatClient
from astronaut.logics.parser import ParseGeneratedResult
from astronaut.schema import MESSAGE_HISTORY_TYPE, IdeaScore, ReviewIdeaResult


class ReviewIdea:
    """A class for reviewing and analyzing feature map ideas.

    This class provides functionality to review and analyze feature map ideas using
    language models. It supports both standard and reasoning-based model versions,
    with appropriate parsing and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use
        parser_model_version (str): Version of the parser model to use

    Methods:
        review: Reviews and analyzes feature map ideas
    """

    def __init__(self, client: ChatClient, model_version: str, parser_model_version: str) -> None:
        self.client = client
        self.model_version = model_version
        self.parser = ParseGeneratedResult(client, parser_model_version)

    @traceable(tags=["review", "idea"])
    def review(
        self,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: Optional[int],
    ) -> tuple[ReviewIdeaResult, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Review Last Idea...")
        logger.debug(f"Input User Prompt: {user_prompt}")
        if self.model_version in REASONING_SERIES:
            if self.model_version in STABLE_REASONING_MODEL_VERSIONS:
                response_format = ReviewIdeaResult
            else:
                response_format = None
                system_prompt = {"role": "system", "content": ""}

            content, updated_message_history, cost = self.client.parse_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                n=1,
                max_tokens=REASONING_MAX_TOKENS,
                response_format=response_format,
                model_version=self.model_version,
            )
            # parse output result by light model
            if response_format is None:
                content, parse_cost = self.parser.parse(content, "review")
                cost += parse_cost
        else:
            content, updated_message_history, cost = self.client.parse_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                temperature=0.2,
                n=1,
                max_tokens=GPT_MAX_TOKENS,
                response_format=ReviewIdeaResult,
                model_version=self.model_version,
            )

        result = json.loads(content)
        logger.info("Review Last Step Feature Map Idea is done. And result is parsed as JSON.")
        logger.debug(f"Reviewed Last Idea: {result}")

        return ReviewIdeaResult(**result), updated_message_history, cost


class ReviewPerformance:
    """A class for reviewing and analyzing performance metrics.

    This class provides functionality to review and analyze performance metrics
    of feature map implementations. It evaluates changes in accuracy and other
    metrics to provide targeted feedback and improvement suggestions.

    Args:
        review_comment_template (Optional[str]): Template for review comments

    Methods:
        _discreatize_score: Discretizes score differences into categories
        _discreatize_metric: Discretizes metric differences into categories
        review: Reviews performance metrics and generates feedback
    """

    def __init__(self, review_comment_template: Optional[str] = None) -> None:
        if review_comment_template is None:
            self.review_comment_template = (
                'In the previous trial, the model\'s classification accuracy "{discreatize_performance}". '
                "{review_direction}"
            )
        else:
            self.review_comment_template = review_comment_template

    def _discreatize_score(self, diff_score: float) -> str:
        if 3.0 < diff_score <= 30.0:
            return "Significantly improved"
        elif 0.5 < diff_score <= 3.0:
            return "Improved"
        elif 0.0 < diff_score <= 0.5:
            return "Marginally improved"
        elif diff_score == 0.0:
            return "Unchanged"
        elif -1.5 <= diff_score < 0.0:
            return "Dropped slightly"
        elif -30.0 <= diff_score < -1.5:
            return "Dropped significantly"
        else:
            return "Out of Range"

    def _discreatize_metric(self, diff_metric: float) -> str:
        if 0.2 < diff_metric <= 1.0:
            return "Significantly improved"
        elif 0.05 < diff_metric <= 0.2:
            return "Improved"
        elif 0.0 < diff_metric <= 0.05:
            return "Marginally improved"
        elif diff_metric == 0.0:
            return "Unchanged"
        elif -0.2 <= diff_metric < 0.0:
            return "Dropped slightly"
        elif -1.0 <= diff_metric < -0.2:
            return "Dropped significantly"
        else:
            return "Out of Range"

    def review(
        self, evaluation_df: pd.DataFrame, score_list: list[IdeaScore], metric: str = "accuracy"
    ) -> Optional[str]:
        if (len(evaluation_df) < 2) or (len(score_list) < 2):
            return None

        diff_score = score_list[-1].diff(score_list[-2])
        discreatize_score = self._discreatize_score(diff_score)
        diff_metric = evaluation_df[metric].values[-1] - evaluation_df[metric].values[-2]
        discreatize_metric = self._discreatize_metric(diff_metric)

        # if discreatize_score in negative_case or discreatize_metric in negative_case:
        if discreatize_metric in ["Significantly improved"]:
            review_direction = (
                "Please review the changes or factors that likely led to this improvement "
                "by referring to all past trials, analyze their impact, "
                "and propose how we can enhance these aspects further to sustain or amplify the positive trend."
            )
        elif discreatize_metric in ["Marginally improved", "Improved"]:
            review_direction = (
                "Please examine the elements that contributed to this progress by referencing all past trials, "
                "assess their effectiveness, and suggest additional refinements or "
                "strategies to achieve more significant advancements."
            )
        elif discreatize_metric in ["Unchanged"]:
            review_direction = (
                "Please investigate the potential reasons for this stagnation by comparing all past trials, "
                "identify any bottlenecks or limitations, and propose actionable strategies "
                "to introduce meaningful progress."
            )
        elif discreatize_metric in ["Dropped slightly"]:
            review_direction = (
                "Please review the factors or changes that may have negatively impacted the results "
                "by analyzing all past trials, evaluate their significance, and propose targeted solutions "
                "to recover or improve performance in subsequent trials."
            )
        elif discreatize_metric in ["Dropped significantly"]:
            review_direction = (
                "Please thoroughly analyze the root causes of this drop by referencing all past trials, "
                "including any critical changes or issues in the process, and recommend urgent actions "
                "or adjustments to address these challenges effectively and recover performance."
            )
        else:
            review_direction = ""

        return self.review_comment_template.format(
            discreatize_performance=discreatize_metric, review_direction=review_direction
        )
