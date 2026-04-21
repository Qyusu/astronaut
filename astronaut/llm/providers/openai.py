import re
import time
import traceback
from typing import Iterable, Literal, Type, cast

from langsmith.wrappers import wrap_openai
from loguru import logger
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.responses.parsed_response import ParsedResponse
from openai.types.responses.response import Response
from openai.types.responses.response_text_config_param import (
    ResponseTextConfigParam,
)
from openai.types.responses.tool_param import ParseableToolParam, ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning
from pydantic import BaseModel, ValidationError

from astronaut.llm.base import BaseLLMClient
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    OPENAI_MESSAGE_HISTORY_TYPE,
)


class OpenAIChatClient(BaseLLMClient):
    """Client for interacting with OpenAI's chat models.

    This class implements the BaseLLMClient interface for OpenAI's chat models,
    providing functionality for chat completion with support for both GPT and
    Reasoning series models. It handles message construction, response parsing,
    and cost tracking specific to OpenAI's API.

    Features include:
    - Developer and user prompts
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
        _update_history: Updates conversation history with new messages
        _chat_by_responses_api: Handles requests by responses API
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

    def parse_chat(
        self,
        developer_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE = [],
        n_history: int | None = None,
        temperature: float | None = None,
        n: int = 1,
        max_tokens: int | None = None,
        response_format: Type[BaseModel] | None = None,
        model_version: str | None = None,
        max_retries: int = 3,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
        tools: Iterable[ParseableToolParam | ToolParam] | None = None,
        background: bool = False,
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float]:
        """Main method for chat completion with OpenAI models.

        This method handles chat completion requests, supporting both GPT and Reasoning
        series models. It automatically selects the appropriate API based on the model
        version and includes retry logic with exponential backoff for failed requests.

        The method supports three types of API requests:
        1. Responses series API: Uses specialized parameters for complex reasoning tasks. This API is stateful.
        2. Reasoning series API: Uses specialized parameters for complex reasoning tasks. This API is stateless.
        3. GPT series API: Uses standard parameters for general chat completion. This API is stateless.

        Args:
            developer_prompt (dict[str, str]): Developer prompt containing role and content
            user_prompt (dict[str, str]): User prompt containing role and content
            message_history (MESSAGE_HISTORY_TYPE, optional): Previous conversation
                history. Defaults to empty list.
            n_history (int | None, optional): Number of previous history to include
                in the context. If None, includes all history. Defaults to None.
            temperature (float, optional): Controls randomness in generation.
                Higher values (e.g., 0.8) make output more random, lower values
                (e.g., 0.2) make it more deterministic. Defaults to 0.0.
            n (int, optional): Number of completions to generate. Defaults to 1.
            max_tokens (int | None, optional): Maximum number of tokens to generate
                in the response. Defaults to None.
            response_format (Type[BaseModel] | None, optional): Pydantic model
                defining the expected response structure. Defaults to None.
            model_version (str | None, optional): Specific model version to use.
                If None, uses the default model version. Defaults to None.
            max_retries (int, optional): Maximum number of retry attempts for
                failed API calls. Defaults to 3.
            reasoning_effort (Literal["minimum", "low", "medium", "high"], optional):
                Level of reasoning effort for Reasoning series models.
                Defaults to "high".
            verbosity (Literal["low", "medium", "high"], optional):
                Level of verbosity for the response. Defaults to "medium".
            tools (Iterable[ParseableToolParam] | NotGiven, optional):
                Tools to use for the API call. Defaults to NOT_GIVEN.
            background (bool, optional): Whether to run the API call in the background.
                Defaults to False.

        Returns:
            tuple[str, MESSAGE_HISTORY_TYPE, float]: A tuple containing:
                - Generated response content
                - Updated message history including the new exchange
                - Cost of the API call

        Raises:
            ValueError: If there's a validation error in the API process
            ValueError: If all retry attempts fail to get a response
        """
        model_version = model_version or self.default_model_version
        previous_response_id = self._get_previous_response_id(
            cast(OPENAI_MESSAGE_HISTORY_TYPE, message_history), n_history
        )

        attempts = 0
        while True:
            try:
                chat_function = (
                    self._chat_background
                    if background
                    else self._chat_by_responses_api
                )

                chat_kwargs = {
                    "model_version": model_version,
                    "instructions": developer_prompt["content"],
                    "input": user_prompt["content"],
                    "previous_response_id": previous_response_id,
                    "temperature": temperature or NOT_GIVEN,
                    "max_tokens": max_tokens or NOT_GIVEN,
                    "response_format": response_format or NOT_GIVEN,
                    "reasoning": Reasoning(effort=reasoning_effort)
                    if reasoning_effort
                    else NOT_GIVEN,
                    "text": {"verbosity": verbosity}
                    if verbosity
                    else NOT_GIVEN,
                    "tools": tools or NOT_GIVEN,
                }

                raw_response = chat_function(**chat_kwargs)

                content = self._parse_response(raw_response)
                updated_message_history = self._update_history(
                    cast(OPENAI_MESSAGE_HISTORY_TYPE, message_history),
                    raw_response.id,
                )
                input_tokens, cached_tokens, output_tokens = (
                    self._get_token_count(raw_response)
                )
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
                logger.debug(traceback.format_exc())
                logger.warning(f"Raise Exception: {e}")
                attempts += 1
                if attempts >= max_retries:
                    raise ValueError(
                        f"Failed to get response from OpenAI after {max_retries} attempts: {e}"
                    )

                wait_time = 60 * 2**attempts
                logger.info(f"Retry after {wait_time} seconds...")
                time.sleep(wait_time)

    def _chat_by_responses_api(
        self,
        model_version: str,
        instructions: str,
        input: str,
        previous_response_id: str | NotGiven,
        temperature: float | NotGiven,
        max_tokens: int | NotGiven,
        response_format: Type[BaseModel] | NotGiven,
        reasoning: Reasoning | NotGiven,
        text: ResponseTextConfigParam | NotGiven,
        tools: Iterable[ParseableToolParam] | NotGiven,
    ) -> ParsedResponse:
        """Handles chat completion requests using OpenAI's Responses series models.

        This method is specifically designed for models optimized for complex reasoning
        tasks, supporting specialized parameters like reasoning_effort to control the
        depth of analysis.

        Responses API is stateful, but this method is not supporting the stateful feature (not using chat history).

        Args:
            model_version (str): OpenAI model version to use
            instructions (str): Developer prompt containing role and content
            input (str): User prompt containing role and content
            previous_response_id (str): Previous response id
            temperature (float | NotGiven): Temperature parameter
            max_tokens (int | NotGiven): Maximum number of tokens to generate
            response_format (Type[BaseModel] | NotGiven): Expected response format structure
            reasoning (Reasoning | NotGiven): Reasoning parameters
            text (ResponseTextConfigParam | NotGiven): Verbosity of the response
            tools (Iterable[ParseableToolParam] | NotGiven): Tools to use for the API call

        Returns:
            ParsedResponse: OpenAI Parsed Response object containing the model's response

        Note:
            This method is wrapped with LangSmith tracing for monitoring and debugging.
        """
        response = self.client.responses.parse(
            model=model_version,
            instructions=instructions,
            input=input,
            previous_response_id=previous_response_id,
            temperature=temperature,
            max_output_tokens=max_tokens,
            text_format=response_format,
            reasoning=reasoning,
            text=text,
            tools=tools,
            store=True,
        )
        return response

    def _chat_background(
        self,
        model_version: str,
        instructions: str,
        input: str,
        previous_response_id: str | NotGiven,
        temperature: float | NotGiven,
        max_tokens: int | NotGiven,
        response_format: Type[BaseModel] | NotGiven,
        reasoning: Reasoning | NotGiven,
        text: ResponseTextConfigParam | NotGiven,
        tools: Iterable[ToolParam] | NotGiven,
    ) -> Response:
        if response_format is not NOT_GIVEN:
            logger.warning(
                "response_format is not supported in background mode. "
                "It will be ignored."
            )

        response = self.client.responses.create(
            model=model_version,
            instructions=instructions,
            input=input,
            previous_response_id=previous_response_id,
            temperature=temperature,
            max_output_tokens=max_tokens,
            reasoning=reasoning,
            text=text,
            tools=tools,
            background=True,
            store=True,
        )

        logger.info("Request is sent to OpenAI. Processing in background...")
        while response.status in {"queued", "in_progress"}:
            time.sleep(2)
            response = self.client.responses.retrieve(response.id)

        return response

    def _parse_response(self, response: ParsedResponse | Response) -> str:
        try:
            content = response.output[-1].content[0].text  # type: ignore

            if content is None:
                raise ValueError("OpenAI response is None.")
            return content
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to parse OpenAI response: {e}")

    def _get_previous_response_id(
        self,
        message_history: OPENAI_MESSAGE_HISTORY_TYPE,
        n_history: int | None,
    ) -> str | NotGiven:
        if len(message_history) == 0 or n_history == 0:
            return NOT_GIVEN

        return message_history[-1]

    def _update_history(
        self,
        message_history: OPENAI_MESSAGE_HISTORY_TYPE,
        previous_response_id: str,
    ) -> OPENAI_MESSAGE_HISTORY_TYPE:
        return message_history + [previous_response_id]

    def _get_model_name_from_version(self, model_version: str) -> str:
        return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_version)

    def _get_token_count(
        self, response: ParsedResponse | Response
    ) -> tuple[int, int, int]:
        usage = response.usage
        if usage is None:
            logger.info("Usage information is not found in the completion.")
            return 0, 0, 0

        input_tokens = usage.model_dump().get("input_tokens", 0)
        cached_tokens = (
            usage.model_dump().get("input_tokens_details") or {}
        ).get("cached_tokens", 0)
        output_tokens = usage.model_dump().get("output_tokens", 0)

        return input_tokens, cached_tokens, output_tokens
