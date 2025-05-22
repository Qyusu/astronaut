import re
import time
from textwrap import dedent
from typing import Type, cast

import anthropic
from anthropic.types import Message, MessageParam
from langsmith import traceable
from loguru import logger
from pydantic import BaseModel, ValidationError

from astronaut.configs import settings
from astronaut.constants import ANTHOROPIC_THINKING_SERIES, GPT_MAX_TOKENS
from astronaut.llm.base import BaseLLMClient
from astronaut.llm.providers.openai import OpenAIChatClient
from astronaut.prompts import ParseJsonPrompt
from astronaut.schema import (
    ANTHOROPIC_MESSAGE_HISTORY_TYPE,
    MESSAGE_HISTORY_TYPE,
    get_schema_string,
)

# Anthropic API doesn't support "response_format".
# So, we use OpenAI API to parse the response.
if settings.OPENAI_API_KEY is not None:
    parse_client = OpenAIChatClient(
        api_key=settings.OPENAI_API_KEY, default_model_version="gpt-4o-mini-2024-07-18", reasoning_effort="high"
    )
else:
    parse_client = None


class AnthropicChatClient(BaseLLMClient):
    """Client for interacting with Anthropic's Claude models.

    This class implements the BaseLLMClient interface for Anthropic's Claude models,
    providing functionality for chat completion with support for both standard and
    thinking series models. It handles message construction, response parsing, and
    cost tracking specific to Anthropic's API.

    The class supports two types of API requests:
    1. Thinking series API: Optimized for deep reasoning tasks with streaming support
    2. Basic series API: Standard chat completion for general purposes

    Note that some features like multiple completions (n) and direct response format
    specification are not supported by Anthropic's API and are handled through
    alternative implementations.

    Args:
        api_key (str): Anthropic API key for authentication
        default_model_version (str): Default model version to use for completions
        thinking_model_max_tokens (int): Maximum number of tokens for thinking series models
        basic_model_max_tokens (int): Maximum number of tokens for basic series models
        max_thinking_budget_tokens (int): Maximum budget for thinking series models

    Attributes:
        client (anthropic.Anthropic): Anthropic API client instance
        default_model_version (str): Default model version for completions
        total_cost (float): Total cost incurred from API calls

    Methods:
        parse_chat: Main method for chat completion with Claude models
        _construct_message: Helper method to format messages for API requests
        _update_history: Updates conversation history with new messages
        _chat_thinking_model: Handles requests to thinking series models
        _chat: Handles requests to basic series models
        _get_model_name_from_version: Extracts base model name from version string
        _get_token_count: Calculates token usage from API responses
        _parse_response: Processes and formats API responses

    Reference:
        GitHub: https://github.com/anthropics/anthropic-sdk-python
    """

    def __init__(
        self,
        api_key: str,
        default_model_version: str,
        thinking_model_max_tokens: int,
        basic_model_max_tokens: int,
        max_thinking_budget_tokens: int,
    ) -> None:
        super().__init__()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.default_model_version = default_model_version
        self.thinking_model_max_tokens = thinking_model_max_tokens
        self.basic_model_max_tokens = basic_model_max_tokens
        self.max_thinking_budget_tokens = max_thinking_budget_tokens
        self.total_cost = 0.0

    def _construct_message(
        self,
        user_prompt: dict[str, str],
        message_history: ANTHOROPIC_MESSAGE_HISTORY_TYPE,
        response_format: Type[BaseModel] | None = None,
    ) -> list[MessageParam]:
        if response_format is not None:
            user_content = user_prompt["content"] + dedent(
                """\nProvide a response strictly in below JSON format.
                Do not include any additional commentary or text outside of the JSON object.\n
                {schema_string}
                """.format(
                    schema_string=get_schema_string(response_format)
                )
            )
        else:
            user_content = user_prompt["content"]

        user_message = MessageParam(role="user", content=[{"type": "text", "text": user_content}])
        messages = message_history + [user_message]

        return messages

    def _update_history(
        self,
        user_prompt: dict[str, str],
        message_history: ANTHOROPIC_MESSAGE_HISTORY_TYPE,
        content: str,
    ) -> ANTHOROPIC_MESSAGE_HISTORY_TYPE:
        user_message = MessageParam(role="user", content=user_prompt["content"])
        assistant_message = MessageParam(role="assistant", content=content)
        updated_message_history = message_history + [user_message, assistant_message]

        return updated_message_history

    @traceable(tags=["llm"], run_type="llm")
    def _chat_thinking_model(
        self,
        model_version: str,
        system_prompt: dict[str, str],
        messages: list[MessageParam],
        max_tokens: int,
        max_thinking_tokens: int,
    ) -> list[Message]:
        response = self.client.messages.create(
            model=model_version,
            system=system_prompt.get("content", ""),
            messages=messages,
            max_tokens=max_tokens,
            thinking={"type": "enabled", "budget_tokens": max_thinking_tokens},
            stream=True,
        )

        collected_responses = []
        for chunk in response:
            collected_responses.append(chunk)

        return collected_responses

    @traceable(tags=["llm"], run_type="llm")
    def _chat(
        self,
        model_version: str,
        system_prompt: dict[str, str],
        messages: list[MessageParam],
        max_tokens: int,
        temperature: float,
    ) -> Message:
        response = self.client.messages.create(
            model=model_version,
            system=system_prompt.get("content", ""),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response

    def _get_model_name_from_version(self, model_version: str) -> str:
        return re.sub(r"[-_](\d+|latest)$", "", model_version)

    def _get_token_count(self, response: Message | list[Message]) -> tuple[int, int, int]:
        # [TODO]: Support cached_tokens reffer to the below link.
        # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
        if isinstance(response, list):  # for thinking model
            input_tokens, output_tokens = 0, 0
            for message in response:
                if message.type == "message_start":
                    input_tokens = message.message.usage.input_tokens
                    output_tokens = message.message.usage.output_tokens
                elif message.type == "message_delta":
                    output_tokens = message.usage.output_tokens
        else:  # for standard model
            usage = response.usage
            if usage is None:
                logger.info("Usage information is not found in the response.")
                return 0, 0, 0

            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

        return input_tokens, 0, output_tokens  # Anthropic doesn't provide cached_tokens

    def _parse_response(self, response: Message | list[Message], response_format: Type[BaseModel] | None) -> str:
        try:
            if isinstance(response, list):  # for thinking model
                stream_content = []
                for message in response:
                    if message.type == "content_block_delta" and message.delta.type == "text_delta":
                        stream_content.append(message.delta.text)

                raw_content = "".join(stream_content)
            else:  # for standard model
                raw_content = response.content
                if raw_content is None:
                    raise ValueError("Anthropic response content is None or empty.")

                if isinstance(raw_content, list):
                    raw_content = "".join(block.text for block in raw_content if block.type == "text")

            if parse_client is not None:
                system_prompt, user_prompt = ParseJsonPrompt(raw_content).build()
                content, _, _ = parse_client.parse_chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    message_history=[],
                    n_history=0,
                    n=1,
                    max_tokens=GPT_MAX_TOKENS,
                    response_format=response_format,
                )

            return content
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to parse Anthropic response: {e}")

    def parse_chat(
        self,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE = [],
        n_history: int | None = None,
        n: int | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: Type[BaseModel] | None = None,
        model_version: str | None = None,
        max_retries: int = 3,
        max_thinking_tokens: int | None = None,
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float]:
        """Main method for chat completion with Anthropic's Claude models.

        This method handles chat completion requests, supporting both standard and
        thinking series models. It automatically selects the appropriate API based
        on the model version and includes retry logic with exponential backoff for
        failed requests.

        The method supports two types of API requests:
        1. Thinking series API: Optimized for deep reasoning tasks with streaming
           support and thinking budget control
        2. Basic series API: Standard chat completion for general purposes

        Note that some features like multiple completions (n) and direct response
        format specification are not supported by Anthropic's API and are handled
        through alternative implementations.

        Args:
            system_prompt (dict[str, str]): System prompt containing role and content
                for initializing the chat context
            user_prompt (dict[str, str]): User prompt containing role and content
                for the current request
            message_history (MESSAGE_HISTORY_TYPE, optional): Previous conversation
                history. Defaults to empty list.
            n_history (int | None, optional): Number of previous exchanges to
                include in the context. If None, includes all history. Defaults to None.
            n (int, optional): Number of completions. Note: This parameter
                is not supported by Anthropic's API and is ignored. Defaults to None.
            temperature (float, optional): Controls randomness in generation.
                Higher values (e.g., 0.8) make output more random, lower values
                (e.g., 0.2) make it more deterministic. Defaults to 0.0.
            max_tokens (int | None, optional): Maximum number of tokens to generate
                in the response. Defaults to None.
            response_format (Type[BaseModel] | None, optional): Pydantic model
                defining the expected response structure. If provided, the response
                will be formatted according to this schema using OpenAI's API.
                Defaults to None.
            model_version (str | None, optional): Specific model version to use.
                If None, uses the default model version. Defaults to None.
            max_retries (int, optional): Maximum number of retry attempts for
                failed API calls. Defaults to 3.
            max_thinking_tokens (int | None, optional): Maximum number of tokens to use
                for the thinking process in thinking series models. Defaults to 20000.

        Returns:
            tuple[str, MESSAGE_HISTORY_TYPE, float]: A tuple containing:
                - Generated response content
                - Updated message history including the new exchange
                - Cost of the API call

        Raises:
            ValueError: If there's a validation error in the API process
            ValueError: If all retry attempts fail to get a response from the API

        Note:
            - For response format specification, this method uses OpenAI's API
              to parse the response into the desired format
            - The n parameter is not supported by Anthropic's API and is ignored
        """
        if model_version is None:
            model_version = self.default_model_version

        history = cast(ANTHOROPIC_MESSAGE_HISTORY_TYPE, self._get_last_n_history(message_history, n_history))
        messages = self._construct_message(user_prompt, history, response_format=response_format)
        attempts = 0

        max_tokens = None
        while True:
            try:
                if model_version in ANTHOROPIC_THINKING_SERIES:
                    max_tokens = max_tokens or self.thinking_model_max_tokens
                    response = self._chat_thinking_model(
                        model_version=model_version,
                        system_prompt=system_prompt,
                        messages=messages,
                        max_tokens=max_tokens,
                        max_thinking_tokens=max_thinking_tokens or self.max_thinking_budget_tokens,
                    )
                else:
                    max_tokens = max_tokens or self.basic_model_max_tokens
                    response = self._chat(
                        model_version=model_version,
                        system_prompt=system_prompt,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                content = self._parse_response(response, response_format)
                updated_message_history = self._update_history(
                    user_prompt, cast(ANTHOROPIC_MESSAGE_HISTORY_TYPE, message_history), content
                )
                input_tokens, cached_tokens, output_tokens = self._get_token_count(response)
                cost = self._calculate_cost(
                    input_tokens=input_tokens,
                    cached_tokens=cached_tokens,
                    output_tokens=output_tokens,
                    model_name=self._get_model_name_from_version(model_version),
                )
                self._update_cost(cost)
                return content, updated_message_history, cost
            except ValidationError as e:
                raise ValueError(f"Validation error in messages: {e}")
            except Exception as e:
                logger.info(f"Raise Exception: {e}")
                attempts += 1
                if attempts >= max_retries:
                    raise ValueError(f"Failed to get response from Anthropic after {max_retries} attempts: {e}")

                wait_time = 60 * 2**attempts
                logger.info(f"Retry after {wait_time} seconds...")
                time.sleep(wait_time)
