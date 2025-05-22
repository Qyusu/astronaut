from astronaut.llm.base import BaseLLMClient
from astronaut.llm.config import LLMConfig, LLMProvider
from astronaut.llm.providers import (
    AnthropicChatClient,
    GoogleChatClient,
    OpenAIChatClient,
)


class LLMClientFactory:
    """A factory class for creating LLM client instances.

    This class provides a static method to create appropriate LLM client instances
    based on the specified provider configuration. It supports multiple LLM providers
    including OpenAI, Anthropic, and Google.

    Supported Providers:
        - OpenAI: Creates an instance of OpenAIChatClient
        - Anthropic: Creates an instance of AnthropicChatClient
        - Google: Creates an instance of GoogleChatClient

    Methods:
        create: Creates and returns an appropriate LLM client instance based on the
            provided configuration. Raises ValueError if the provider is not supported.
    """

    @staticmethod
    def create(config: LLMConfig) -> BaseLLMClient:
        clients = {
            LLMProvider.OPENAI: OpenAIChatClient,
            LLMProvider.ANTHROPIC: AnthropicChatClient,
            LLMProvider.GOOGLE: GoogleChatClient,
        }

        client_class = clients.get(config.provider)
        if not client_class:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        return client_class(**config.to_dict())
