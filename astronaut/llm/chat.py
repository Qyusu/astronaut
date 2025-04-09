from typing import Optional, Type

from pydantic import BaseModel

from astronaut.llm.anthropic import AnthropicChatClient
from astronaut.llm.google import GoogleChatClient
from astronaut.llm.openai import OpenAIChatClient
from astronaut.schema import MESSAGE_HISTORY_TYPE


class ChatClient:
    """A unified interface for interacting with various Large Language Model (LLM) providers.

    This class provides a consistent interface for working with different LLM platforms
    (OpenAI, Google, Anthropic) while abstracting away platform-specific details.
    It automatically initializes the appropriate client based on the specified platform
    and delegates all chat completion requests to the underlying provider.

    The class supports features like:
    - System and user prompts
    - Message history management
    - Temperature control
    - Multiple completions
    - Response format specification
    - Token usage tracking
    - Automatic retries

    Args:
        platform (str): LLM platform to use ("openai", "google", or "anthropic")
        api_key (str): API key for the specified platform
        default_model_version (str): Default model version to use for completions

    Attributes:
        platform (str): Selected LLM platform
        client (Union[OpenAIChatClient, GoogleChatClient, AnthropicChatClient]):
            Platform-specific client instance
        total_cost (float): Total cost incurred from API calls

    Methods:
        parse_chat: Main method for chat completion with the selected platform
        _initialize_client: Helper method to create the appropriate client instance

    Note:
        Each platform has its own specific features and limitations. For example:
        - Anthropic doesn't support multiple completions (n)
        - Response format specification may be handled differently across platforms
    """

    def __init__(
        self,
        platform: str,
        api_key: str,
        default_model_version: str,
    ) -> None:
        self.platform = platform.lower()
        self.client = self._initialize_client(api_key, default_model_version)
        self.total_cost = 0.0

    def _initialize_client(
        self, api_key: str, default_model_version: str
    ) -> OpenAIChatClient | GoogleChatClient | AnthropicChatClient:
        try:
            if self.platform == "openai":
                return OpenAIChatClient(api_key=api_key, default_model_version=default_model_version)
            elif self.platform == "google":
                return GoogleChatClient(api_key=api_key, default_model_version=default_model_version)
            elif self.platform == "anthropic":
                return AnthropicChatClient(api_key=api_key, default_model_version=default_model_version)
            else:
                raise ValueError(f"Invalid platform: {self.platform}")
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM client: {e}")

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
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float]:
        return self.client.parse_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            message_history=message_history,
            n_history=n_history,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            response_format=response_format,
            model_version=model_version,
            max_retries=max_retries,
        )
