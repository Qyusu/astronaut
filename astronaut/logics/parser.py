from langsmith import traceable
from loguru import logger

from astronaut.constants import GPT_MAX_TOKENS
from astronaut.llm import ChatClient
from astronaut.prompts import ParseJsonPrompt
from astronaut.schema import (
    GeneratedIdeaResult,
    GeneratedImplResult,
    ReflectIdeaResult,
    ReviewIdeaResult,
)


class ParseGeneratedResult:
    """A class for parsing and formatting generated results from language models.

    This class provides functionality to parse and format various types of generated
    results, including ideas, code, and reviews. When the language model cannot directly
    output in the required schema format, it uses a lightweight model to parse the raw
    output into the appropriate dataclass schema.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use

    Methods:
        _parse_review_idea: Parses reviewed idea results into ReviewIdeaResult schema
        _parse_gen_idea: Parses generated idea results into GeneratedIdeaResult or ReflectIdeaResult schema
        _parse_gen_code: Parses generated code results into GeneratedImplResult schema
        parse: Main method for parsing different types of results with schema validation
    """

    def __init__(self, client: ChatClient, model_version: str) -> None:
        self.client = client
        self.model_version = model_version

    @traceable(tags=["parse", "idea"])
    def _parse_review_idea(self, raw_content: str) -> tuple[str, float]:
        logger.info("Parse reviewed idea result...")
        system_prompt, user_prompt = ParseJsonPrompt(raw_content).build()
        content, _, cost = self.client.parse_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=0,
            n=1,
            max_tokens=GPT_MAX_TOKENS,
            response_format=ReviewIdeaResult,
            model_version=self.model_version,
        )
        logger.info("Parse reviewed idea result is done.")

        return content, cost

    @traceable(tags=["parse", "idea"])
    def _parse_gen_idea(self, raw_content: str, reflect_mode: bool) -> tuple[str, float]:
        response_format = ReflectIdeaResult if reflect_mode else GeneratedIdeaResult
        logger.info("Parse generated idea result...")
        system_prompt, user_prompt = ParseJsonPrompt(raw_content).build()
        content, _, cost = self.client.parse_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=0,
            n=1,
            max_tokens=GPT_MAX_TOKENS,
            response_format=response_format,
            model_version=self.model_version,
        )
        logger.info("Parse generated idea result is done.")

        return content, cost

    @traceable(tags=["parse", "code"])
    def _parse_gen_code(self, raw_content: str) -> tuple[str, float]:
        logger.info("Parse generated code result...")
        system_prompt, user_prompt = ParseJsonPrompt(raw_content).build()
        content, _, cost = self.client.parse_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=0,
            n=1,
            max_tokens=GPT_MAX_TOKENS,
            response_format=GeneratedImplResult,
            model_version=self.model_version,
        )
        logger.info("Parse generated code result is done.")

        return content, cost

    def parse(self, raw_content: str, target: str, reflect_mode: bool = False) -> tuple[str, float]:
        if target == "review":
            content, cost = self._parse_review_idea(raw_content=raw_content)
        elif target == "idea":
            content, cost = self._parse_gen_idea(raw_content=raw_content, reflect_mode=reflect_mode)
        elif target == "code":
            content, cost = self._parse_gen_code(raw_content=raw_content)
        else:
            raise ValueError(f"Unsupported type: {type}")

        return content, cost
