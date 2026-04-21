from typing import Iterable, Literal, Type

from openai.types.responses.tool_param import ParseableToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel

from astronaut.schema import MESSAGE_HISTORY_TYPE


class ChatRequest(BaseModel):
    developer_prompt: dict[str, str]
    user_prompt: dict[str, str]
    message_history: MESSAGE_HISTORY_TYPE = []
    n_history: int | None = None
    temperature: float | None = 0.0
    n: int = 1
    max_tokens: int | None = None
    response_format: Type[BaseModel] | None = None
    model_version: str | None = None
    max_retries: int = 3


class OpenAIChatRequest(ChatRequest):
    reasoning_effort: ReasoningEffort | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    tools: Iterable[ParseableToolParam] | None = None
    background: bool = False


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
