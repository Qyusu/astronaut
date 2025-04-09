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
    assert search_reference.build_search_query("quantum", None, None, None, None) == '"quantum"'
    # Multiple keywords with AND
    assert (
        search_reference.build_search_query(["quantum computing", "machine learning"], "AND", None, None, None)
        == '"quantum computing" AND "machine learning"'
    )
    # Category and single keyword
    assert (
        search_reference.build_search_query(["quantum computing"], "AND", "cs.AI", None, None)
        == 'cat:cs.AI AND ("quantum computing")'
    )
    # Category and keywords
    assert (
        search_reference.build_search_query(["quantum computing", "machine learning"], "OR", "cs.AI", None, None)
        == 'cat:cs.AI AND ("quantum computing" OR "machine learning")'
    )
    # Date range
    assert (
        search_reference.build_search_query(["quantum computing"], "AND", None, "2022-01-01", "2022-12-31")
        == '"quantum computing" AND submittedDate:[2022-01-01 TO 2022-12-31]'
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
        params={"search_query": '"quantum computing"', "start": 0, "max_results": 1},
    )
