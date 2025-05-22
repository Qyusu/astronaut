import time
import xml.etree.ElementTree as ET
from textwrap import dedent

import requests
from langsmith import traceable
from loguru import logger
from pydantic import BaseModel

ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"
ARXIV_API_URL = "http://export.arxiv.org/api/query"


class AcademicPaper(BaseModel):
    """A data model representing an academic paper with its metadata.

    This class stores and manages information about academic papers, including
    their identification, authorship, content, and publication details.

    Attributes:
        id (str): Unique identifier for the paper
        title (str): Title of the paper
        authors (list[str]): List of authors' names
        abstract (str): Abstract of the paper
        published_date (str): Date when the paper was published
        published_cite (str): Publication venue or source
        pdf_url (str): URL to the PDF version of the paper

    Methods:
        __str__: Returns a formatted string representation of the paper
    """

    id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: str
    published_cite: str
    pdf_url: str

    def __str__(self) -> str:
        return dedent(
            f"""
            Title: {self.title}
            Authors: {", ".join(self.authors)}
            Abstract: {self.abstract}
            Published Date: {self.published_date}
            Published Cite: {self.published_cite}
            PDF URL: {self.pdf_url}
            --------------------------------
            """
        )


class SearchReference:
    """A class for searching and retrieving academic papers from arXiv.

    This class provides functionality to search academic papers on arXiv,
    build search queries with various filters, and parse the results into
    AcademicPaper objects. It includes retry mechanisms for robust API calls.

    Available search scopes:
        - all: Search in all fields
        - ti/title: Search in title
        - au/author: Search in author names
        - abs/abstract: Search in abstract
        - co/comment: Search in comments
        - jr/journal-ref: Search in journal references
        - cat/subject: Search in categories

    Methods:
        build_search_query: Constructs a search query string with optional filters
        get_text_or_default: Safely extracts text content from XML elements
        get_url_or_default: Safely extracts URL attributes from XML elements
        request_with_retry: Makes HTTP requests with retry logic
        search_arxiv: Performs paper searches on arXiv and returns results
    """

    SEARCH_SCOPES = [
        "all",
        "ti",
        "title",
        "au",
        "author",
        "abs",
        "abstract",
        "co",
        "comment",
        "jr",
        "journal-ref",
        "cat",
        "subject",
    ]

    def build_search_query(
        self,
        keywords: str | list[str],
        operator: str | None,
        category: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> str:
        if isinstance(keywords, list) and (operator not in ["AND", "OR"]):
            raise ValueError("Invalid operator. Please use 'AND' or 'OR'.")

        def add_search_scope(keyword: str) -> str:
            parts = keyword.split(":", 1)
            if len(parts) == 2 and parts[0].lower() in self.SEARCH_SCOPES:
                return f'"{keyword}"'
            return f'all:"{keyword}"'

        if isinstance(keywords, list):
            search_query = f" {operator} ".join(add_search_scope(keyword) for keyword in keywords)
        else:
            search_query = add_search_scope(keywords)

        if category is not None:
            search_query = f"cat:{category} AND ({search_query})"

        if start_date and end_date:
            search_query = f"{search_query} AND submittedDate:[{start_date} TO {end_date}]"

        return search_query

    def get_text_or_default(self, element: ET.Element, tag: str, default: str = "Not Exist") -> str:
        found = element.find(tag)
        return found.text if found is not None and found.text else default

    def get_url_or_default(self, element: ET.Element, tag: str, default: str = "Not Exist") -> str:
        found = element.find(tag)
        return found.attrib["href"] if found is not None and found.attrib["href"] else default

    def request_with_retry(self, url: str, params: dict, max_retry: int = 3) -> requests.Response:
        for i in range(max_retry):
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    return response
            except ConnectionError as e:
                logger.warning(f"Failed to connect: {e}. Retry {i+1}/{max_retry}")
                time.sleep(10)
            except Exception as e:
                logger.warning(f"Failed to fetch data: {e}. Retry {i+1}/{max_retry}")

        raise Exception(f"Failed to fetch data after {max_retry} retries.")

    @traceable(tags=["retreival", "arxiv"])
    def search_arxiv(
        self,
        query: str | list[str],
        start_idx: int = 0,
        max_results: int = 10,
        query_operator: str | None = None,
        category: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[AcademicPaper]:
        # fetch data from arXiv API
        search_query = self.build_search_query(query, query_operator, category, start_date, end_date)
        logger.info(f"Searching papers from arXiv: {search_query}")
        params = {
            "search_query": search_query,
            "start": start_idx,
            "max_results": max_results,
        }
        response = self.request_with_retry(ARXIV_API_URL, params)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.status_code}")

        # parse response result
        root = ET.fromstring(response.text)
        papers = []
        for entry in root.findall(f"{ATOM_NAMESPACE}entry"):
            arxiv_id = self.get_text_or_default(entry, f"{ATOM_NAMESPACE}id").split("/")[-1]
            title = self.get_text_or_default(entry, f"{ATOM_NAMESPACE}title").strip()
            authors = [
                self.get_text_or_default(author, f"{ATOM_NAMESPACE}name")
                for author in entry.findall(f"{ATOM_NAMESPACE}author")
            ]
            publish_date = self.get_text_or_default(entry, f"{ATOM_NAMESPACE}published")
            abstract = self.get_text_or_default(entry, f"{ATOM_NAMESPACE}summary")
            pdf_url = self.get_url_or_default(entry, f"{ATOM_NAMESPACE}link[@type='application/pdf']")
            papers.append(
                AcademicPaper(
                    id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published_date=publish_date,
                    published_cite="Arxiv",
                    pdf_url=pdf_url,
                )
            )
        logger.info(f"Found {len(papers)} papers.")

        return papers
