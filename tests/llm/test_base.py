from typing import Type

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from astronaut.llm.base import BaseLLMClient
from astronaut.schema import MESSAGE_HISTORY_TYPE


class ConcreteLLMClient(BaseLLMClient):
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
        **kwargs: dict,
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float]:
        return "response", message_history, 0.0


@pytest.fixture
def client() -> ConcreteLLMClient:
    return ConcreteLLMClient()


def test_get_last_n_history_none(client: ConcreteLLMClient) -> None:
    history: MESSAGE_HISTORY_TYPE = ["msg1", "msg2", "msg3"]  # type: ignore
    result = client._get_last_n_history(history, None)
    assert result == history


def test_get_last_n_history_zero(client: ConcreteLLMClient) -> None:
    history: MESSAGE_HISTORY_TYPE = ["msg1", "msg2"]  # type: ignore
    result = client._get_last_n_history(history, 0)
    assert result == []


def test_get_last_n_history_n(client: ConcreteLLMClient) -> None:
    history: MESSAGE_HISTORY_TYPE = ["u1", "a1", "u2", "a2", "u3", "a3"]  # type: ignore

    # n=1 => last 2 messages
    result = client._get_last_n_history(history, 1)
    assert result == ["u3", "a3"]

    # n=2 => last 4 messages
    result = client._get_last_n_history(history, 2)
    assert result == ["u2", "a2", "u3", "a3"]


def test_update_cost(client: ConcreteLLMClient) -> None:
    assert client.get_total_cost() == 0.0
    client._update_cost(1.5)
    assert client.get_total_cost() == 1.5
    client._update_cost(0.5)
    assert client.get_total_cost() == 2.0


def test_calculate_cost(
    mocker: MockerFixture, client: ConcreteLLMClient
) -> None:
    mock_cost_table = mocker.patch("astronaut.llm.base.ChatModelCostTable")
    mock_cost_table_instance = mock_cost_table.return_value
    mock_cost_info = mocker.Mock()
    mock_cost_info.input = 10.0  # $10 per 1M
    mock_cost_info.cached = 5.0  # $5 per 1M
    mock_cost_info.output = 20.0  # $20 per 1M
    mock_cost_table_instance.get_cost.return_value = mock_cost_info

    cost = client._calculate_cost(
        input_tokens=1000,
        cached_tokens=500,
        output_tokens=1000,
        model_name="test-model",
    )

    assert abs(cost - 0.0275) < 1e-9


def test_calculate_cost_model_not_found(
    mocker: MockerFixture, client: ConcreteLLMClient
) -> None:
    mock_cost_table = mocker.patch("astronaut.llm.base.ChatModelCostTable")
    mock_cost_table_instance = mock_cost_table.return_value
    mock_cost_table_instance.get_cost.return_value = None

    cost = client._calculate_cost(100, 0, 100, "unknown")
    assert cost == 0.0
