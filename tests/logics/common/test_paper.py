from pathlib import Path
from typing import Any

import pytest

from astronaut.logics.common.paper import Paper


@pytest.fixture
def paper_handler(tmp_path: Path) -> Paper:
    return Paper(local_paper_dir=str(tmp_path))


class TestPaper:
    def test_load_full_text_from_local_not_found(
        self, paper_handler: Paper
    ) -> None:
        text = paper_handler.load_full_text_from_local("non_existent_id")
        assert text == ""

    def test_load_full_text_from_local_success(
        self, paper_handler: Paper, mocker: Any, tmp_path: Path
    ) -> None:
        paper_id = "test_paper"
        pdf_path = tmp_path / f"{paper_id}.pdf"
        pdf_path.write_text("dummy pdf content")

        mock_pdf = mocker.MagicMock()
        mock_page = mocker.MagicMock()
        mock_page.extract_text.return_value = "Extracted text content"
        mock_pdf.pages = [mock_page]

        mock_pdf_open = mocker.patch("pdfplumber.open")
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        text = paper_handler.load_full_text_from_local(paper_id)
        assert text == "Extracted text content"
        mock_pdf_open.assert_called_once_with(pdf_path)

    def test_load_full_text_from_local_error(
        self, paper_handler: Paper, mocker: Any, tmp_path: Path
    ) -> None:
        paper_id = "error_paper"
        pdf_path = tmp_path / f"{paper_id}.pdf"
        pdf_path.write_text("dummy")

        mocker.patch("pdfplumber.open", side_effect=Exception("PDF error"))

        text = paper_handler.load_full_text_from_local(paper_id)
        assert text == ""

    def test_cut_string_if_over_token_limit(
        self, paper_handler: Paper, mocker: Any
    ) -> None:
        text = "This is a test string"
        max_tokens = 5

        mock_encoding = mocker.MagicMock()
        # Assume 1 word = 1 token for simplicity in mock
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
        mock_encoding.decode.return_value = "This is a test"

        mocker.patch("tiktoken.encoding_for_model", return_value=mock_encoding)

        result = paper_handler.cut_string_if_over_token_limit(text, max_tokens)
        assert result == "This is a test"
        mock_encoding.encode.assert_called_once_with(text)
        mock_encoding.decode.assert_called_once_with([1, 2, 3, 4, 5])

    def test_cut_string_if_over_token_limit_error(
        self, paper_handler: Paper, mocker: Any
    ) -> None:
        text = "test"
        mocker.patch(
            "tiktoken.encoding_for_model",
            side_effect=Exception("Encoding error"),
        )
        result = paper_handler.cut_string_if_over_token_limit(text, 10)
        assert result == text

    def test_get_paper_path(self, paper_handler: Paper, tmp_path: Path) -> None:
        paper_id = "test_id"
        expected_path = tmp_path / "test_id.pdf"
        assert paper_handler.get_paper_path(paper_id) == expected_path

    def test_paper_exists(self, paper_handler: Paper, tmp_path: Path) -> None:
        paper_id = "exists_id"
        assert not paper_handler.paper_exists(paper_id)

        (tmp_path / f"{paper_id}.pdf").write_text("data")
        assert paper_handler.paper_exists(paper_id)

    def test_paper_exists_value_error(self, mocker: Any) -> None:
        handler = Paper(local_paper_dir="/tmp")
        mocker.patch.object(
            handler, "get_paper_path", side_effect=ValueError("Invalid path")
        )
        assert not handler.paper_exists("any_id")
