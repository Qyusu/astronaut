import json
from typing import Optional

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
from astronaut.schema import MESSAGE_HISTORY_TYPE, GeneratedIdeaResult, GeneratedImpl


class GenerateIdea:
    """A class for generating quantum feature map ideas using LLM.

    This class provides functionality to generate ideas for quantum feature maps
    using language models. It supports both standard and reasoning-based model
    versions, with appropriate parsing and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use
        parser_model_version (str): Version of the parser model to use

    Methods:
        generate: Generates feature map ideas using the language model
    """

    def __init__(self, client: ChatClient, model_version: str, parser_model_version: str) -> None:
        self.client = client
        self.model_version = model_version
        self.is_o1_series = model_version in REASONING_SERIES
        self.parser = ParseGeneratedResult(client, parser_model_version)

    @traceable(tags=["generation", "idea"])
    def generate(
        self,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: Optional[int],
    ) -> tuple[GeneratedIdeaResult, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Generate Feature Map Idea...")
        logger.debug(f"Input User Prompt: {user_prompt}")
        if self.is_o1_series:
            if self.model_version in STABLE_REASONING_MODEL_VERSIONS:
                response_format = GeneratedIdeaResult
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
                content, parse_cost = self.parser.parse(content, "idea", reflect_mode=False)
                cost += parse_cost
        else:
            content, updated_message_history, cost = self.client.parse_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                temperature=0.8,
                n=1,
                max_tokens=GPT_MAX_TOKENS,
                response_format=GeneratedIdeaResult,
                model_version=self.model_version,
            )

        result = json.loads(content)
        logger.info("Generated Feature Map Idea is done. And result is parsed as JSON.")
        logger.debug(f"Generated Idea: {result}")

        return GeneratedIdeaResult(**result), updated_message_history, cost


class GenerateCode:
    """A class for generating quantum feature map code using LLM.

    This class provides functionality to generate Python code for quantum feature maps
    using language models. It supports both standard and reasoning-based model
    versions, with appropriate parsing and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use
        parser_model_version (str): Version of the parser model to use

    Methods:
        generate: Generates feature map code using the language model
    """

    def __init__(self, client: ChatClient, model_version: str, parser_model_version: str) -> None:
        self.client = client
        self.model_version = model_version
        self.is_o1_series = model_version in REASONING_SERIES
        self.parser = ParseGeneratedResult(client, parser_model_version)

    @traceable(tags=["generation", "code"])
    def generate(
        self,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: Optional[int] = None,
    ) -> tuple[GeneratedImpl, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Generate Feature Map Code...")
        logger.debug(f"Input User Prompt: {user_prompt}")
        if self.is_o1_series:
            if self.model_version in STABLE_REASONING_MODEL_VERSIONS:
                response_format = GeneratedImpl
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
                content, parse_cost = self.parser.parse(content, "code", reflect_mode=False)
                cost += parse_cost
        else:
            content, updated_message_history, cost = self.client.parse_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                temperature=0.0,
                n=1,
                max_tokens=GPT_MAX_TOKENS,
                response_format=GeneratedImpl,
                model_version=self.model_version,
            )

        result = json.loads(content)
        logger.info("Generated Feature Map Code is done. And result is parsed as JSON.")
        logger.debug(f"Generated Code: {result}")

        return GeneratedImpl(**result), updated_message_history, cost
