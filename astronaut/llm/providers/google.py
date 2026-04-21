import re
import time
import traceback
from typing import Any, Type, cast

from google import genai
from loguru import logger
from pydantic import BaseModel

from astronaut.llm.base import BaseLLMClient
from astronaut.schema import GEMINI_MESSAGE_HISTORY_TYPE, MESSAGE_HISTORY_TYPE


class GoogleChatClient(BaseLLMClient):
    """Client for interacting with Google's Gemini models.

    This class implements the BaseLLMClient interface for Google's Gemini models,
    providing functionality for chat completion with support for various model
    configurations. It handles message construction, response parsing, and cost
    tracking specific to Google's API.

    The class supports features like:
    - System instructions
    - Temperature control
    - Multiple completions
    - Response format specification
    - Token usage tracking

    Args:
        api_key (str): Google API key for authentication
        default_model_version (str): Default model version to use for completions
        project_id (str | None, optional): Google Cloud project ID. Defaults to None.

    Attributes:
        client (genai.Client): Google API client instance
        default_model_version (str): Default model version for completions
        total_cost (float): Total cost incurred from API calls

    Methods:
        parse_chat: Main method for chat completion with Gemini models
        _construct_message: Helper method to format messages for API requests
        _update_history: Updates conversation history with new messages
        _get_model_name_from_version: Extracts base model name from version string
        _get_token_count: Calculates token usage from API responses
        _parse_response: Processes and formats API responses

    Reference:
        GitHub: https://github.com/googleapis/python-genai
    """

    def __init__(
        self,
        api_key: str,
        default_model_version: str,
        project_id: str | None = None,
    ) -> None:
        super().__init__()
        self.client = genai.Client(api_key=api_key, project=project_id)
        self.default_model_version = default_model_version
        self.total_cost = 0.0

    def parse_chat(
        self,
        developer_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE = [],
        n_history: int | None = None,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int | None = None,
        response_format: Type[BaseModel] | None = None,
        model_version: str | None = None,
        max_retries: int = 3,
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float]:
        """Main method for chat completion with Google's Gemini models.

        This method handles chat completion requests, supporting various model
        configurations and features. It includes retry logic with exponential
        backoff for failed requests and tracks token usage for cost calculation.

        Args:
            developer_prompt (dict[str, str]): Developer prompt containing role and content
                for initializing the chat context
            user_prompt (dict[str, str]): User prompt containing role and content
                for the current request
            message_history (MESSAGE_HISTORY_TYPE, optional): Previous conversation
                history. Defaults to empty list.
            n_history (int | None, optional): Number of previous exchanges to
                include in the context. If None, includes all history. Defaults to None.
            temperature (float, optional): Controls randomness in generation.
                Higher values (e.g., 0.8) make output more random, lower values
                (e.g., 0.2) make it more deterministic. Defaults to 0.0.
            n (int, optional): Number of completions to generate. Defaults to 1.
            max_tokens (int | None, optional): Maximum number of tokens to generate
                in the response. Defaults to None.
            response_format (Type[BaseModel] | None, optional): Pydantic model
                defining the expected response structure. If provided, the response
                will be formatted according to this schema. Defaults to None.
            model_version (str | None, optional): Specific model version to use.
                If None, uses the default model version. Defaults to None.
            max_retries (int, optional): Maximum number of retry attempts for
                failed API calls. Defaults to 3.

        Returns:
            tuple[str, MESSAGE_HISTORY_TYPE, float]: A tuple containing:
                - Generated response content
                - Updated message history including the new exchange
                - Cost of the API call

        Raises:
            ValueError: If all retry attempts fail to get a response from the API

        Note:
            This method is wrapped with LangSmith tracing for monitoring and debugging.
        """
        if model_version is None:
            model_version = self.default_model_version

        is_deep_research = model_version.startswith("deep-research")
        kwargs: dict[str, Any] = {}

        if is_deep_research:
            history = cast(GEMINI_MESSAGE_HISTORY_TYPE, [])
            messages = (
                developer_prompt.get("content", "")
                + "\n"
                + user_prompt.get("content", "")
            )
            kwargs["input"] = messages
            kwargs["agent"] = model_version
            kwargs["background"] = True
        else:
            history = cast(
                GEMINI_MESSAGE_HISTORY_TYPE,
                self._get_last_n_history(message_history, n_history),
            )
            messages_list = self._construct_message(user_prompt, history)

            generation_config: dict[str, Any] = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                generation_config["max_output_tokens"] = max_tokens
            if response_format is not None:
                kwargs["response_format"] = (
                    response_format.model_json_schema()
                    if hasattr(response_format, "model_json_schema")
                    else response_format.schema()
                )

            kwargs["input"] = messages_list
            kwargs["system_instruction"] = developer_prompt.get("content", "")
            kwargs["model"] = model_version
            kwargs["generation_config"] = generation_config

        attempts = 0
        response = None
        while True:
            try:
                if is_deep_research:
                    response = self._chat_by_research_model(kwargs)
                else:
                    response = self.client.interactions.create(**kwargs)
                break
            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.warning(f"Raise Exception: {e}")
                attempts += 1
                if attempts >= max_retries:
                    raise ValueError(
                        f"Failed to get response from Gemini after {max_retries} attempts: {e}"
                    )

                wait_time = 60 * 2**attempts
                logger.info(f"Retry after {wait_time} seconds...")
                time.sleep(wait_time)

        content = self._parse_response(response)
        updated_message_history = self._update_history(
            user_prompt=user_prompt,
            message_history=history,
            content=content,
        )

        input_tokens, cached_tokens, output_tokens = self._get_token_count(
            response
        )
        cost = self._calculate_cost(
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            model_name=self._get_model_name_from_version(model_version),
        )
        self._update_cost(cost)

        return content, updated_message_history, cost

    def _chat_by_research_model(self, kwargs: dict) -> Any:
        initial_interaction = self.client.interactions.create(**kwargs)

        logger.info("Request is sent to Gemini. Processing in background...")
        while True:
            interaction = self.client.interactions.get(initial_interaction.id)

            if interaction.status == "completed":
                break
            elif interaction.status in ["failed", "cancelled"]:
                raise ValueError(
                    f"Failed to get response from Gemini: {interaction.status}"
                )

            time.sleep(10)

        return interaction

    def _construct_message(
        self,
        user_prompt: dict[str, str],
        message_history: GEMINI_MESSAGE_HISTORY_TYPE,
    ) -> list[dict[str, str]]:
        messages = [{"role": "user", "content": user_prompt["content"]}]
        return message_history + messages

    def _update_history(
        self,
        user_prompt: dict[str, str],
        message_history: GEMINI_MESSAGE_HISTORY_TYPE,
        content: str,
    ) -> GEMINI_MESSAGE_HISTORY_TYPE:
        updated_message_history = message_history + [
            {"role": "user", "content": user_prompt["content"]},
            {"role": "model", "content": content},
        ]
        return updated_message_history

    def _get_model_name_from_version(self, model_version: str) -> str:
        return re.sub(r"(-\d+)+$", "", model_version)

    def _get_token_count(self, response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage_metadata", None) or getattr(
            response, "usage", None
        )
        if usage is None:
            logger.info("Usage information is not found in the response.")
            return 0, 0, 0

        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
            input_tokens = (
                usage_dict.get(
                    "prompt_token_count", usage_dict.get("input_tokens", 0)
                )
                or 0
            )
            cached_tokens = (
                usage_dict.get(
                    "cached_content_token_count",
                    usage_dict.get("cached_tokens", 0),
                )
                or 0
            )
            output_tokens = (
                usage_dict.get(
                    "candidates_token_count", usage_dict.get("output_tokens", 0)
                )
                or 0
            )
        else:
            input_tokens = getattr(
                usage, "prompt_token_count", getattr(usage, "input_tokens", 0)
            )
            cached_tokens = getattr(
                usage,
                "cached_content_token_count",
                getattr(usage, "cached_tokens", 0),
            )
            output_tokens = getattr(
                usage,
                "candidates_token_count",
                getattr(usage, "output_tokens", 0),
            )

        return input_tokens, cached_tokens, output_tokens

    def _parse_response(self, interaction: Any) -> str:
        try:
            content = interaction.outputs[-1].text
            if content is None:
                raise ValueError("Gemini interaction is None.")
            return content
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to parse Gemini interaction: {e}")
