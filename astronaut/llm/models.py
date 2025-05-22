from typing import Type

from pydantic import BaseModel

from astronaut.schema import MESSAGE_HISTORY_TYPE


class ChatRequest(BaseModel):
    system_prompt: dict[str, str]
    user_prompt: dict[str, str]
    message_history: MESSAGE_HISTORY_TYPE = []
    n_history: int | None = None
    temperature: float = 0.0
    n: int = 1
    max_tokens: int | None = None
    response_format: Type[BaseModel] | None = None
    model_version: str | None = None
    max_retries: int = 3


class ChatResponse(BaseModel):
    content: str
    message_history: MESSAGE_HISTORY_TYPE
    cost: float


class LLMException(Exception):
    pass


class RateLimitError(LLMException):
    pass


class AuthenticationError(LLMException):
    pass
