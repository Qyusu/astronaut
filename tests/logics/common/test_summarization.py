from typing import Any

import pytest

from astronaut.logics.common.summarization import PaperSummarizer


@pytest.fixture
def mock_client(mocker: Any) -> Any:
    return mocker.MagicMock()


@pytest.fixture
def summarizer(mock_client: Any) -> PaperSummarizer:
    return PaperSummarizer(
        client=mock_client, model_version="test-model", local_paper_dir="/tmp"
    )


class TestPaperSummarizer:
    def test_summarize_papers_success(
        self, summarizer: PaperSummarizer, mock_client: Any, mocker: Any
    ) -> None:
        query_result = {"matches": [{"metadata": {"document_id": "paper1"}}]}

        mocker.patch.object(
            summarizer, "load_full_text_from_local", return_value="Full text"
        )
        mocker.patch.object(
            summarizer,
            "cut_string_if_over_token_limit",
            side_effect=lambda x, y: x,
        )

        mock_response = mocker.MagicMock()
        mock_response.content = (
            '{"title": "Paper Title", "summary": "Paper Summary"}'
        )
        mock_response.cost = 0.02
        mock_client.parse_chat.return_value = mock_response

        summaries, total_cost = summarizer.summarize_papers(
            query_result, max_paper_tokens=100
        )

        assert len(summaries) == 1
        assert "Paper Title" in summaries[0]
        assert total_cost == 0.02

    def test_summarize_papers_error(
        self, summarizer: PaperSummarizer, mock_client: Any, mocker: Any
    ) -> None:
        query_result = {"matches": [{"metadata": {"document_id": "paper_err"}}]}

        mocker.patch.object(
            summarizer, "load_full_text_from_local", return_value="text"
        )
        mock_client.parse_chat.side_effect = Exception("Summarization error")

        summaries, total_cost = summarizer.summarize_papers(query_result)

        assert len(summaries) == 1
        assert "Not acquired" in summaries[0]
        assert total_cost == 0.0

    def test_summarize_papers_with_tracing(
        self, summarizer: PaperSummarizer, mock_client: Any, mocker: Any
    ) -> None:
        query_result = {"matches": [{"metadata": {"document_id": "paper1"}}]}
        mocker.patch.object(
            summarizer, "load_full_text_from_local", return_value="text"
        )
        mocker.patch.object(
            summarizer, "_summary", return_value=(mocker.MagicMock(), 0.01)
        )

        mock_tree = mocker.MagicMock()
        mock_tracing = mocker.patch(
            "astronaut.logics.common.summarization.tracing_context"
        )
        mock_tracing.return_value.__enter__.return_value = None

        summarizer.summarize_papers(query_result, max_workers=1)

        mocker.patch(
            "astronaut.logics.common.summarization.get_current_run_tree",
            return_value=mock_tree,
        )
        summarizer.summarize_papers(query_result, max_workers=1)

        assert mock_tracing.called

    def test_summarize_papers_parallel_error(
        self, summarizer: PaperSummarizer, mocker: Any
    ) -> None:
        query_result = {"matches": [{"metadata": {"document_id": "p1"}}]}
        mocker.patch.object(
            summarizer,
            "_process_single_paper_with_context",
            side_effect=ValueError("Parallel fail"),
        )

        with pytest.raises(ValueError, match="Parallel fail"):
            summarizer.summarize_papers(query_result)
