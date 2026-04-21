from astronaut.llm.models import (
    AuthenticationError,
    ChatRequest,
    ChatResponse,
    LLMException,
    OpenAIChatRequest,
    RateLimitError,
)


def test_chat_request_defaults() -> None:
    req = ChatRequest(
        developer_prompt={"role": "system", "content": "foo"},
        user_prompt={"role": "user", "content": "bar"},
    )
    assert req.message_history == []
    assert req.n_history is None
    assert req.temperature == 0.0
    assert req.n == 1
    assert req.max_retries == 3


def test_openai_chat_request_defaults() -> None:
    req = OpenAIChatRequest(
        developer_prompt={"role": "system", "content": "foo"},
        user_prompt={"role": "user", "content": "bar"},
    )
    assert req.reasoning_effort is None
    assert req.background is False


def test_chat_response() -> None:
    # MESSAGE_HISTORY_TYPE alias is list[str] | list[MessageParam] | ...
    # We can use simple list of strings for testing if type checking allows or ignore
    history = ["u", "a"]  # type: ignore
    res = ChatResponse(content="hello", message_history=history, cost=0.1)
    assert res.content == "hello"
    assert res.cost == 0.1


def test_exceptions() -> None:
    assert issubclass(RateLimitError, LLMException)
    assert issubclass(AuthenticationError, LLMException)
