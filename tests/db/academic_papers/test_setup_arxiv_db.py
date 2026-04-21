import pytest
from pytest_mock import MockerFixture

from astronaut.db.academic_papers.reference import AcademicPaper
from astronaut.db.academic_papers.setup_arxiv_db import ArxivPaperDB


@pytest.fixture
def mock_arxiv_db(mocker: MockerFixture) -> ArxivPaperDB:
    mocker.patch("astronaut.db.academic_papers.setup_arxiv_db.EmbeddingClient")
    mocker.patch("astronaut.db.academic_papers.setup_arxiv_db.PineconeClient")
    mocker.patch("astronaut.db.academic_papers.setup_arxiv_db.SearchReference")
    mocker.patch("astronaut.configs.settings.EMBEDDING_MODEL_VERSION", "v1")
    mocker.patch("astronaut.configs.settings.PINECONE_API_KEY", "key")

    return ArxivPaperDB(
        index_name="test-index",
        chunk_size=100,
        max_results_per_request=10,
        init_db=False,
    )


def test_setup(mocker: MockerFixture, mock_arxiv_db: ArxivPaperDB) -> None:
    mock_arxiv_db.db.create_index = mocker.Mock()  # type: ignore
    mock_arxiv_db.setup()
    mock_arxiv_db.db.create_index.assert_called_once()


def test_get_all_paper_in_date_range(
    mocker: MockerFixture, mock_arxiv_db: ArxivPaperDB
) -> None:
    paper1 = AcademicPaper(
        id="1",
        title="t1",
        authors=["a1"],
        abstract="abs1",
        published_date="d1",
        published_cite="Arxiv",
        pdf_url="url1",
    )
    mock_arxiv_db.searcher.search_arxiv.return_value = [paper1]  # type: ignore

    mock_arxiv_db.max_results_per_request = 10
    mock_arxiv_db.searcher.search_arxiv.side_effect = [  # type: ignore
        [
            paper1
        ],  # First call returns 1 paper (less than max 10), so it should break
    ]

    papers = mock_arxiv_db.get_all_paper_in_date_range(
        "query", None, "20200101", "20200102"
    )

    assert len(papers) == 1
    assert papers[0] == paper1


def test_pdf_to_text(
    mocker: MockerFixture, mock_arxiv_db: ArxivPaperDB
) -> None:
    mock_pdfplumber = mocker.patch(
        "astronaut.db.academic_papers.setup_arxiv_db.pdfplumber"
    )
    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = "page text"
    mock_pdf = mocker.Mock()
    mock_pdf.pages = [mock_page]
    mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

    text = mock_arxiv_db.pdf_to_text("path/to.pdf")

    assert text == "page text"


def test_download_pdf_to_text(
    mocker: MockerFixture, mock_arxiv_db: ArxivPaperDB
) -> None:
    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("builtins.open", mocker.mock_open())
    mock_requests = mocker.patch(
        "astronaut.db.academic_papers.setup_arxiv_db.requests"
    )
    mock_requests.get.return_value.content = b"pdf content"

    mocker.patch.object(mock_arxiv_db, "pdf_to_text", return_value="text")

    text = mock_arxiv_db.download_pdf_to_text("url", "path")

    assert text == "text"
    mock_requests.get.assert_called_once_with("url")


def test_upsert_paper(
    mocker: MockerFixture, mock_arxiv_db: ArxivPaperDB
) -> None:
    paper = AcademicPaper(
        id="1",
        title="t1",
        authors=["a1"],
        abstract="abs1",
        published_date="d1",
        published_cite="Arxiv",
        pdf_url="url",
    )

    mocker.patch.object(
        mock_arxiv_db, "get_all_paper_in_date_range", side_effect=[[paper], []]
    )
    mocker.patch.object(
        mock_arxiv_db, "download_pdf_to_text", return_value="text"
    )
    mock_arxiv_db.db.upsert = mocker.Mock()  # type: ignore
    mock_arxiv_db.db.total_cost = 0.0  # type: ignore

    mock_arxiv_db.upsert_paper(
        category=None,
        keywords=["k1"],
        start_date="20200101",
        end_date="20200105",
        delta_days=2,
    )

    mock_arxiv_db.db.upsert.assert_called()
