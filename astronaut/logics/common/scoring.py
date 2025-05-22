import json

from langsmith import traceable
from loguru import logger

from astronaut.constants import GPT_MAX_TOKENS, REASONING_SERIES
from astronaut.llm import ChatClient
from astronaut.prompts import ScoringIdeaPrompt
from astronaut.schema import MESSAGE_HISTORY_TYPE, GeneratedIdea, ScoringResult


class ScoringIdea:
    """A class for scoring feature map ideas using language models.

    This class provides functionality to score feature map ideas using language models.
    It supports both standard and reasoning-based model versions, with appropriate parsing
    and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use

    Methods:
        score: Scores feature map ideas
    """

    def __init__(self, client: ChatClient, model_version: str) -> None:
        self.client = client
        self.model_version = model_version
        self.is_o1_series = model_version in REASONING_SERIES

    @traceable(tags=["scoring", "idea"])
    def score(
        self,
        idea: GeneratedIdea,
        related_work: str,
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: int | None,
        round: int,
        max_round: int,
        score_histories: str = "",
    ) -> tuple[ScoringResult, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Scoring Idea...")
        if self.is_o1_series:
            raise NotImplementedError("Scoring Idea is not supported in O1 series.")
        else:
            system_prompt, user_prompt = ScoringIdeaPrompt(score_histories=score_histories).build(
                idea=idea.explanation, related_work=related_work, round=round, max_round=max_round
            )
            response = self.client.parse_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                temperature=0.0,
                n=1,
                max_tokens=GPT_MAX_TOKENS,
                response_format=ScoringResult,
                model_version=self.model_version,
            )

        result = json.loads(response.content)
        score_str = ", ".join([f"{k}={v.get('score', 0.0)}" for k, v in result.get("score", {}).items()])
        logger.info(f"[Round {round}/{max_round}] Scoring Idea is done ({score_str}). And result is parsed as JSON.")
        logger.debug(f"Score: {result}")

        return ScoringResult(**result), response.message_history, response.cost
