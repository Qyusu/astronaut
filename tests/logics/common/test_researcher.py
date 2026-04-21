from typing import Any

import pytest
from pytest_mock import MockerFixture

from astronaut.llm import LLMProvider
from astronaut.logics.common.researcher import Researcher
from astronaut.schema import ResearchMode


@pytest.fixture
def mock_client(mocker: MockerFixture) -> Any:
    client = mocker.Mock()
    client.config.provider = LLMProvider.OPENAI
    return client


class TestResearcher:
    def test_init_success(self, mock_client: Any) -> None:
        researcher = Researcher(mock_client, ResearchMode.DEEP)
        assert researcher.model_version == "o3-deep-research-2025-06-26"

    def test_init_invalid_provider(self, mocker: MockerFixture) -> None:
        client = mocker.Mock()
        client.config.provider = LLMProvider.ANTHROPIC
        with pytest.raises(
            ValueError,
            match="Deep-Research mode only supports providers:.*but got: anthropic",
        ):
            Researcher(client, ResearchMode.DEEP)

    def test_init_invalid_mode(self, mock_client: Any) -> None:
        with pytest.raises(
            ValueError,
            match="Unsupported combination: provider=openai, mode=INVALID",
        ):
            Researcher(mock_client, "INVALID")

    def test_research_success(
        self, mock_client: Any, mocker: MockerFixture, tmp_path: Any
    ) -> None:
        researcher = Researcher(mock_client, ResearchMode.LIGHT)

        mock_response = mocker.Mock()
        mock_response.content = "Research Report Content"
        mock_response.cost = 1.0
        mock_client.parse_chat.return_value = mock_response

        report_path = str(tmp_path / "report.md")
        content, cost = researcher.research(
            "Topic", save_report_path=report_path
        )

        assert content == "Research Report Content"
        assert cost == 1.0
        with open(report_path, "r") as f:
            assert f.read() == "Research Report Content"

    def test_load_existing_report_success(self, tmp_path: Any) -> None:
        report_path = tmp_path / "old_report.md"
        report_path.write_text("Old content")

        content, cost = Researcher.load_existing_report(str(report_path))
        assert content == "Old content"
        assert cost == 0.0

    def test_load_existing_report_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Researcher.load_existing_report("/non/existent/path")

    def test_load_existing_report_error(self, mocker: MockerFixture) -> None:
        mocker.patch("builtins.open", side_effect=Exception("Perm error"))
        with pytest.raises(Exception, match="Perm error"):
            Researcher.load_existing_report("path")
