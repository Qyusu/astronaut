from abc import ABC, abstractmethod
from typing import Literal, Type

from loguru import logger
from pydantic import BaseModel

from astronaut.llm.cost import ChatModelCostTable
from astronaut.schema import MESSAGE_HISTORY_TYPE


class BaseLLMClient(ABC):
    """Abstract base class for Large Language Model (LLM) clients.

    This class provides a common interface for interacting with different LLM providers
    (e.g., OpenAI, Anthropic) and includes functionality for chat completion, message
    history management, and cost tracking. It serves as the foundation for specific
    LLM provider implementations.

    The class tracks the total cost of API calls and provides methods for calculating
    and updating costs based on input, cached, and output tokens.

    Attributes:
        total_cost (float): Total cost incurred from all API calls

    Methods:
        parse_chat: Abstract method for chat completion with various parameters
        _get_last_n_history: Helper method to manage message history
        _calculate_cost: Calculates the cost of API calls based on token usage
        _update_cost: Updates the total cost with the latest API call cost
        get_total_cost: Returns the total cost incurred
    """

    def __init__(self) -> None:
        self.total_cost = 0.0

    @abstractmethod
    def parse_chat(
        self,
        system_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE = [],
        n_history: int | None = None,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int | None = None,
        response_format: Type[BaseModel] | None = None,
        model_version: str | None = None,
        max_retries: int = 3,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        max_thinking_tokens: int | None = None,
        **kwargs: dict,
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float]:
        """Perform chat completion with the LLM.

        This method handles the core chat completion functionality, supporting various
        parameters for controlling the generation process and response format.

        Args:
            system_prompt (dict[str, str]): System prompt containing role and content
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
            n (int, optional): Number of completions to generate.
                Note: Not supported by Anthropic API. Defaults to 1.
            max_tokens (int | None, optional): Maximum number of tokens to
                generate in the response. Defaults to None.
            response_format (Type[BaseModel] | None, optional): Pydantic model
                defining the expected response structure. Defaults to None.
            model_version (str | None, optional): Specific version of the model
                to use. Defaults to None.
            max_retries (int, optional): Maximum number of retry attempts for
                failed API calls. Defaults to 3.
            reasoning_effort (Literal["low", "medium", "high"] | None, optional):
                Level of reasoning effort for OpenAI API. Defaults to None.
            max_thinking_tokens (int | None, optional): Maximum number of tokens
                for thinking process in Anthropic API. Defaults to None.
            **kwargs: Additional provider-specific parameters.

        Returns:
            tuple[str, MESSAGE_HISTORY_TYPE, float]: A tuple containing:
                - Generated response content
                - Updated message history including the new exchange
                - Cost of the API call

        Note:
            The actual implementation of this method should be provided by
            concrete subclasses for specific LLM providers.
        """
        pass

    def _get_last_n_history(self, message_history: MESSAGE_HISTORY_TYPE, n_history: int | None) -> MESSAGE_HISTORY_TYPE:
        if n_history is None:
            # use all history
            history = message_history
        elif n_history == 0:
            history = []
        else:
            # multiply by 2 because the history contains both user and assistant messages
            history = message_history[-2 * n_history :]
        return history

    def _calculate_cost(self, input_tokens: int, cached_tokens: int, output_tokens: int, model_name: str) -> float:
        cost_per_1m_tokens = ChatModelCostTable().get_cost(model_name)
        if cost_per_1m_tokens is None:
            logger.info(f'Model name "{model_name}" is not found in the cost table.')
            return 0.0

        input_cost_per_1m_tokens = cost_per_1m_tokens.input / 10**6
        cached_cost_per_1m_tokens = cost_per_1m_tokens.cached / 10**6 if cost_per_1m_tokens.cached is not None else 0.0
        output_cost_per_1m_tokens = cost_per_1m_tokens.output / 10**6

        non_cached_input_tokens = input_tokens - cached_tokens
        cost = (
            non_cached_input_tokens * input_cost_per_1m_tokens
            + cached_tokens * cached_cost_per_1m_tokens
            + output_tokens * output_cost_per_1m_tokens
        )

        return cost

    def _update_cost(self, cost: float) -> None:
        self.total_cost += cost

    def get_total_cost(self) -> float:
        return self.total_cost
