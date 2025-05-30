import json

from langsmith import traceable
from loguru import logger

from astronaut.constants import GPT_MAX_TOKENS, REASONING_MAX_TOKENS, REASONING_SERIES
from astronaut.llm import ChatClient
from astronaut.logics.common.parser import ParseGeneratedResult
from astronaut.schema import MESSAGE_HISTORY_TYPE, ReflectIdeaResult


class ReflectIdea:
    """A class for reflecting on and improving generated feature map ideas.

    This class provides functionality to reflect on and improve generated feature map
    ideas using language models. It supports both standard and reasoning-based model
    versions, with appropriate parsing and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use
        parser_model_version (str): Version of the parser model to use

    Methods:
        reflect: Reflects on and improves generated feature map ideas
    """

    def __init__(self, client: ChatClient, model_version: str, parser_model_version: str) -> None:
        self.client = client
        self.model_version = model_version
        self.parser = ParseGeneratedResult(client, parser_model_version)

    @traceable(tags=["reflection", "idea"])
    def reflect(
        self,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: int | None,
    ) -> tuple[ReflectIdeaResult, MESSAGE_HISTORY_TYPE, float]:
        logger.debug(f"Input User Prompt: {user_prompt}")
        if self.model_version in REASONING_SERIES:
            response = self.client.parse_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                n=1,
                max_tokens=REASONING_MAX_TOKENS,
                response_format=ReflectIdeaResult,
                model_version=self.model_version,
            )
        else:
            response = self.client.parse_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                temperature=0.2,
                n=1,
                max_tokens=GPT_MAX_TOKENS,
                response_format=ReflectIdeaResult,
                model_version=self.model_version,
            )

        result = json.loads(response.content)
        logger.info("Reflected Feature Map Idea is done. And result is parsed as JSON.")
        logger.debug(f"Reflected Idea: {result}")

        return ReflectIdeaResult(**result), response.message_history, response.cost
