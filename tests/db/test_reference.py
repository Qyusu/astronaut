from xml.etree.ElementTree import Element, tostring

import pytest
from pytest_mock import MockFixture

from astronaut.db.reference import SearchReference


def mock_arxiv_response() -> str:
    """Create a mocked XML response from the arXiv API."""
    root = Element("feed", xmlns="http://www.w3.org/2005/Atom")

    entry = Element("entry")
    title = Element("title")
    title.text = "Test Paper Title"
    entry.append(title)

    author = Element("author")
    name = Element("name")
    name.text = "Author Name"
    author.append(name)
    entry.append(author)

    summary = Element("summary")
    summary.text = "This is an abstract."
    entry.append(summary)

    published = Element("published")
    published.text = "2024-01-01T00:00:00Z"
    entry.append(published)

    link = Element("link", type="application/pdf", href="http://arxiv.org/pdf/test.pdf")
    entry.append(link)

    root.append(entry)
    return tostring(root, encoding="unicode")


@pytest.fixture
def search_reference() -> "SearchReference":
    return SearchReference()


def test_build_search_query(search_reference: SearchReference) -> None:
    # Single keyword
    assert search_reference.build_search_query("quantum", None, None, None, None) == 'all:"quantum"'
    # Multiple keywords with AND
    assert (
        search_reference.build_search_query(["quantum computing", "machine learning"], "AND", None, None, None)
        == 'all:"quantum computing" AND all:"machine learning"'
    )
    # Category and single keyword
    assert (
        search_reference.build_search_query(["quantum computing"], "AND", "cs.AI", None, None)
        == 'cat:cs.AI AND (all:"quantum computing")'
    )
    # Category and keywords
    assert (
        search_reference.build_search_query(["quantum computing", "machine learning"], "OR", "cs.AI", None, None)
        == 'cat:cs.AI AND (all:"quantum computing" OR all:"machine learning")'
    )
    # Date range
    assert (
        search_reference.build_search_query(["quantum computing"], "AND", None, "2022-01-01", "2022-12-31")
        == 'all:"quantum computing" AND submittedDate:[2022-01-01 TO 2022-12-31]'
    )
    # Invalid operator
    with pytest.raises(ValueError):
        search_reference.build_search_query(["quantum", "computing"], "INVALID", None, None, None)


def test_get_text_or_default(search_reference: SearchReference) -> None:
    elements = Element("elements")
    element = Element("test")
    element.text = "Sample Text"
    elements.append(element)
    assert search_reference.get_text_or_default(elements, "test") == "Sample Text"
    assert search_reference.get_text_or_default(elements, "empty") == "Not Exist"


def test_get_url_or_default(search_reference: SearchReference) -> None:
    elements = Element("elements")
    element = Element("link", href="http://example.com")
    elements.append(element)
    assert search_reference.get_url_or_default(elements, "link") == "http://example.com"
    assert search_reference.get_url_or_default(elements, "empty") == "Not Exist"


def test_search_arxiv(mocker: MockFixture, search_reference: SearchReference) -> None:
    mock_response = mocker.patch("requests.get")
    mock_response.return_value.status_code = 200
    mock_response.return_value.text = mock_arxiv_response()

    query = "quantum computing"
    result = search_reference.search_arxiv(query, start_idx=0, max_results=1)

    # Verify the results
    assert len(result) == 1
    paper = result[0]
    assert paper.title == "Test Paper Title"
    assert paper.authors == ["Author Name"]
    assert paper.abstract == "This is an abstract."
    assert paper.published_date == "2024-01-01T00:00:00Z"
    assert paper.pdf_url == "http://arxiv.org/pdf/test.pdf"

    # Ensure the mock was called with the correct URL and parameters
    mock_response.assert_called_once_with(
        "http://export.arxiv.org/api/query",
        params={"search_query": 'all:"quantum computing"', "start": 0, "max_results": 1},
    )


def test_search_arxiv_with_retry(mocker: MockFixture, search_reference: SearchReference) -> None:
    """Test the retry mechanism of search_arxiv."""
    mock_response = mocker.patch("requests.get")
    # First call raises ConnectionError, second call succeeds
    mock_response.side_effect = [
        ConnectionError("Connection failed"),
        mocker.Mock(status_code=200, text=mock_arxiv_response()),
    ]

    result = search_reference.search_arxiv("quantum computing")
    assert len(result) == 1
    assert mock_response.call_count == 2


def test_search_arxiv_error_response(mocker: MockFixture, search_reference: SearchReference) -> None:
    """Test handling of non-200 response status."""
    mock_response = mocker.patch("requests.get")
    # Always return 404 status to trigger all retries
    mock_response.return_value.status_code = 404

    with pytest.raises(Exception) as exc_info:
        search_reference.search_arxiv("quantum computing")
    assert "Failed to fetch data after 3 retries." in str(exc_info.value)
    assert mock_response.call_count == 3  # Verify that it tried 3 times


def test_search_arxiv_complex_query(mocker: MockFixture, search_reference: SearchReference) -> None:
    """Test search with multiple filters."""
    mock_response = mocker.patch("requests.get")
    mock_response.return_value.status_code = 200
    mock_response.return_value.text = mock_arxiv_response()

    result = search_reference.search_arxiv(
        query=["quantum computing", "machine learning"],
        query_operator="AND",
        category="cs.AI",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    expected_query = (
        'cat:cs.AI AND (all:"quantum computing" AND all:"machine learning") '
        "AND submittedDate:[2024-01-01 TO 2024-12-31]"
    )
    mock_response.assert_called_once_with(
        "http://export.arxiv.org/api/query",
        params={"search_query": expected_query, "start": 0, "max_results": 10},
    )


def test_search_arxiv_multiple_authors(mocker: MockFixture, search_reference: SearchReference) -> None:
    """Test parsing of multiple authors."""

    def mock_multiple_authors_response() -> str:
        root = Element("feed", xmlns="http://www.w3.org/2005/Atom")
        entry = Element("entry")

        title = Element("title")
        title.text = "Test Paper"
        entry.append(title)

        # Add multiple authors
        authors = ["Author One", "Author Two", "Author Three"]
        for author_name in authors:
            author = Element("author")
            name = Element("name")
            name.text = author_name
            author.append(name)
            entry.append(author)

        summary = Element("summary")
        summary.text = "Abstract"
        entry.append(summary)

        published = Element("published")
        published.text = "2024-01-01T00:00:00Z"
        entry.append(published)

        link = Element("link", type="application/pdf", href="http://arxiv.org/pdf/test.pdf")
        entry.append(link)

        root.append(entry)
        return tostring(root, encoding="unicode")

    mock_response = mocker.patch("requests.get")
    mock_response.return_value.status_code = 200
    mock_response.return_value.text = mock_multiple_authors_response()

    result = search_reference.search_arxiv("quantum computing")
    assert len(result) == 1
    assert len(result[0].authors) == 3
    assert result[0].authors == ["Author One", "Author Two", "Author Three"]


def test_search_arxiv_missing_fields(mocker: MockFixture, search_reference: SearchReference) -> None:
    """Test handling of missing optional fields in response."""

    def mock_missing_fields_response() -> str:
        root = Element("feed", xmlns="http://www.w3.org/2005/Atom")
        entry = Element("entry")

        # Only include required fields
        title = Element("title")
        title.text = "Test Paper"
        entry.append(title)

        author = Element("author")
        name = Element("name")
        name.text = "Author Name"
        author.append(name)
        entry.append(author)

        root.append(entry)
        return tostring(root, encoding="unicode")

    mock_response = mocker.patch("requests.get")
    mock_response.return_value.status_code = 200
    mock_response.return_value.text = mock_missing_fields_response()

    result = search_reference.search_arxiv("quantum computing")
    assert len(result) == 1
    assert result[0].abstract == "Not Exist"
    assert result[0].pdf_url == "Not Exist"
