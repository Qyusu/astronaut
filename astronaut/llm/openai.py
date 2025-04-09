import re
import time
from typing import Literal, Optional, Type, cast

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ValidationError

from astronaut.constants import REASONING_SERIES
from astronaut.llm.base import BaseLLMClient
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    MESSAGE_TYPE,
    OPENAI_MESSAGE_HISTORY_TYPE,
)


class OpenAIChatClient(BaseLLMClient):
    """Client for interacting with OpenAI's chat models.

    This class implements the BaseLLMClient interface for OpenAI's chat models,
    providing functionality for chat completion with support for both GPT and
    Reasoning series models. It handles message construction, response parsing,
    and cost tracking specific to OpenAI's API.

    The class supports two types of API requests:
    1. Reasoning series API: Optimized for complex reasoning tasks with specialized
       parameters like reasoning_effort
    2. GPT series API: Standard chat completion with temperature control and
       structured response formats

    Features include:
    - System and developer prompts
    - Temperature control
    - Multiple completions
    - Structured response formats
    - Token usage tracking
    - Automatic retries with exponential backoff

    Args:
        api_key (str): OpenAI API key for authentication
        default_model_version (str): Default model version to use for completions

    Attributes:
        client (OpenAI): OpenAI API client instance
        default_model_version (str): Default model version for completions
        total_cost (float): Total cost incurred from API calls

    Methods:
        parse_chat: Main method for chat completion with OpenAI models
        _construct_message: Helper method to format messages for API requests
        _update_history: Updates conversation history with new messages
        _chat_reasoning_model: Handles requests to reasoning series models
        _chat: Handles requests to GPT series models
        _get_model_name_from_version: Extracts base model name from version string
        _get_token_count: Calculates token usage from API responses
        _parse_response: Processes and formats API responses

    Reference:
        GitHub: https://github.com/openai/openai-python
    """

    def __init__(
        self,
        api_key: str,
        default_model_version: str,
    ) -> None:
        super().__init__()
        self.client = wrap_openai(OpenAI(api_key=api_key))
        self.default_model_version = default_model_version
        self.total_cost = 0.0

    def _construct_message(
        self,
        model_type: str,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: OPENAI_MESSAGE_HISTORY_TYPE,
    ) -> MESSAGE_TYPE:
        if model_type == "reasoning":
            system_message = ChatCompletionDeveloperMessageParam(role="developer", content=system_prompt["content"])
        else:
            system_message = ChatCompletionSystemMessageParam(role="system", content=system_prompt["content"])
        user_message = ChatCompletionUserMessageParam(role="user", content=user_prompt["content"])
        messages = [system_message] + message_history + [user_message]
        return messages

    def _update_history(
        self,
        user_prompt: dict[str, str],
        message_history: OPENAI_MESSAGE_HISTORY_TYPE,
        content: str,
    ) -> OPENAI_MESSAGE_HISTORY_TYPE:
        user_message = ChatCompletionUserMessageParam(role="user", content=user_prompt["content"])
        assistant_message = ChatCompletionAssistantMessageParam(role="assistant", content=content)
        updated_message_history = message_history + [user_message, assistant_message]

        return updated_message_history

    @traceable(tags=["llm"], run_type="llm")
    def _chat_reasoning_model(
        self,
        model_version: str,
        messages: OPENAI_MESSAGE_HISTORY_TYPE,
        n: int,
        max_tokens: Optional[int],
        response_format: Type[BaseModel] | NotGiven,
        reasoning_effort: Literal["low", "medium", "high"],
    ) -> ChatCompletion:
        """Handles chat completion requests using OpenAI's Reasoning series models.

        This method is specifically designed for models optimized for complex reasoning
        tasks, supporting specialized parameters like reasoning_effort to control the
        depth of analysis.

        Args:
            model_version (str): OpenAI model version to use
            messages (OPENAI_MESSAGE_HISTORY_TYPE): List of chat messages including history
            n (int): Number of completions to generate
            max_tokens (Optional[int]): Maximum number of tokens to generate in the response
            response_format (Type[BaseModel] | NotGiven): Expected response format structure
            reasoning_effort (Literal["low", "medium", "high"]): Level of reasoning effort
                to apply during generation

        Returns:
            ChatCompletion: OpenAI Chat Completion object containing the model's response

        Note:
            This method is wrapped with LangSmith tracing for monitoring and debugging.
        """
        completion = self.client.beta.chat.completions.parse(
            model=model_version,
            messages=messages,
            n=n,
            max_completion_tokens=max_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        return completion

    @traceable(tags=["llm"], run_type="llm")
    def _chat(
        self,
        model_version: str,
        messages: OPENAI_MESSAGE_HISTORY_TYPE,
        temperature: float,
        n: int,
        max_tokens: Optional[int],
        response_format: Type[BaseModel] | NotGiven,
    ) -> ChatCompletion:
        """Handles chat completion requests using OpenAI's GPT series models.

        This method provides standard chat completion functionality with support for
        temperature control and structured response formats.

        Args:
            model_version (str): OpenAI model version to use
            messages (OPENAI_MESSAGE_HISTORY_TYPE): List of chat messages including history
            temperature (float): Controls randomness in generation (0.0 to 1.0)
            n (int): Number of completions to generate
            max_tokens (Optional[int]): Maximum number of tokens to generate in the response
            response_format (Type[BaseModel] | NotGiven): Expected response format structure

        Returns:
            ChatCompletion: OpenAI Chat Completion object containing the model's response

        Note:
            This method is wrapped with LangSmith tracing for monitoring and debugging.
        """
        completion = self.client.beta.chat.completions.parse(
            model=model_version,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return completion

    def _get_model_name_from_version(self, model_version: str) -> str:
        return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_version)

    def _get_token_count(self, completion: ChatCompletion) -> tuple[int, int, int]:
        usage = completion.usage
        if usage is None:
            logger.info("Usage information is not found in the completion.")
            return 0, 0, 0

        input_tokens = usage.model_dump().get("prompt_tokens", 0)
        cached_tokens = (usage.model_dump().get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        output_tokens = usage.model_dump().get("completion_tokens", 0)

        return input_tokens, cached_tokens, output_tokens

    def _parse_response(self, completion: ChatCompletion) -> str:
        try:
            content = completion.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI response is None.")
            return content
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to parse OpenAI response: {e}")

    def parse_chat(
        self,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE = [],
        n_history: Optional[int] = None,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[BaseModel]] = None,
        model_version: Optional[str] = None,
        max_retries: int = 3,
        reasoning_effort: Literal["low", "medium", "high"] = "high",
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float]:
        """Main method for chat completion with OpenAI models.

        This method handles chat completion requests, supporting both GPT and Reasoning
        series models. It automatically selects the appropriate API based on the model
        version and includes retry logic with exponential backoff for failed requests.

        The method supports two types of API requests:
        1. Reasoning series API: Uses specialized parameters for complex reasoning tasks
        2. GPT series API: Uses standard parameters for general chat completion

        Args:
            system_prompt (dict[str, str]): System prompt containing role and content
            user_prompt (dict[str, str]): User prompt containing role and content
            message_history (MESSAGE_HISTORY_TYPE, optional): Previous conversation
                history. Defaults to empty list.
            n_history (Optional[int], optional): Number of previous exchanges to include
                in the context. If None, includes all history. Defaults to None.
            temperature (float, optional): Controls randomness in generation.
                Higher values (e.g., 0.8) make output more random, lower values
                (e.g., 0.2) make it more deterministic. Defaults to 0.0.
            n (int, optional): Number of completions to generate. Defaults to 1.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate
                in the response. Defaults to None.
            response_format (Optional[Type[BaseModel]], optional): Pydantic model
                defining the expected response structure. Defaults to None.
            model_version (Optional[str], optional): Specific model version to use.
                If None, uses the default model version. Defaults to None.
            max_retries (int, optional): Maximum number of retry attempts for
                failed API calls. Defaults to 3.
            reasoning_effort (Literal["low", "medium", "high"], optional):
                Level of reasoning effort for Reasoning series models.
                Defaults to "high".

        Returns:
            tuple[str, MESSAGE_HISTORY_TYPE, float]: A tuple containing:
                - Generated response content
                - Updated message history including the new exchange
                - Cost of the API call

        Raises:
            ValueError: If there's a validation error in the API process
            ValueError: If all retry attempts fail to get a response
        """
        if model_version is None:
            model_version = self.default_model_version

        history = cast(OPENAI_MESSAGE_HISTORY_TYPE, self._get_last_n_history(message_history, n_history))
        response_json_format = NOT_GIVEN if response_format is None else response_format
        attempts = 0

        while True:
            try:
                if model_version in REASONING_SERIES:
                    # Use OpenAI O1 series API
                    messages = self._construct_message("reasoning", system_prompt, user_prompt, history)
                    completion = self._chat_reasoning_model(
                        model_version=model_version,
                        messages=messages,
                        n=n,
                        max_tokens=max_tokens,
                        response_format=response_json_format,
                        reasoning_effort=reasoning_effort,
                    )
                else:
                    # Use OpenAI GPT series API
                    messages = self._construct_message("gpt", system_prompt, user_prompt, history)
                    completion = self._chat(
                        model_version=model_version,
                        messages=messages,
                        temperature=temperature,
                        n=n,
                        max_tokens=max_tokens,
                        response_format=response_json_format,
                    )

                content = self._parse_response(completion)
                updated_message_history = self._update_history(
                    user_prompt, cast(OPENAI_MESSAGE_HISTORY_TYPE, message_history), content
                )
                input_tokens, cached_tokens, output_tokens = self._get_token_count(completion)
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
                    raise ValueError(f"Failed to get response from OpenAI after {max_retries} attempts: {e}")

                wait_time = 60 * 2**attempts
                logger.info(f"Retry after {wait_time} seconds...")
                time.sleep(wait_time)
