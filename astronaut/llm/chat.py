from astronaut.llm.config import LLMConfig
from astronaut.llm.factory import LLMClientFactory
from astronaut.llm.models import ChatRequest, ChatResponse


class ChatClient:
    """A class for handling chat interactions with LLM models.

    This class provides functionality to interact with various LLM models through a unified interface.
    It manages the configuration and client creation for different LLM providers, and handles
    the parsing of chat requests and responses.

    Args:
        config (LLMConfig): Configuration object containing LLM provider settings and parameters.

    Attributes:
        config (LLMConfig): The configuration object used for LLM client setup.
        client: The LLM client instance created based on the provided configuration.

    Methods:
        parse_chat: Processes chat requests and returns formatted responses with content,
            message history, and cost information.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = LLMClientFactory.create(config)

    def parse_chat(self, system_prompt: dict[str, str], user_prompt: dict[str, str], **kwargs) -> ChatResponse:
        request_params = ChatRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )

        content, history, cost = self.client.parse_chat(**request_params.model_dump())
        return ChatResponse(content=content, message_history=history, cost=cost)
