import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pdfplumber
import requests
from loguru import logger

from astronaut.configs import settings
from astronaut.constants import PAPER_PDF_DIRC
from astronaut.db.client import PineconeClient
from astronaut.db.reference import AcademicPaper, SearchReference
from astronaut.llm.embedding import EmbeddingClient

# wait time for arXiv API
ARXIV_API_WAIT_TIME = 5


class ArxivPaperDB:
    """A class for managing and storing arXiv papers in a vector database.

    This class provides functionality to search, download, and store arXiv papers
    in a vector database (Pinecone) with text embeddings. It handles paper retrieval,
    PDF processing, and database operations with chunking and embedding capabilities.

    Args:
        index_name (str): Name of the Pinecone index to use
        chunk_size (int): Size of text chunks for embedding
        max_results_per_request (int): Maximum number of papers to fetch per API request
        init_db (bool): Whether to initialize/clear the database before operations

    Attributes:
        index_name (str): Name of the current index
        chunk_size (int): Size of text chunks for embedding
        max_results_per_request (int): Maximum papers per API request
        init_db (bool): Database initialization flag
        searcher (SearchReference): Instance for arXiv paper search
        client (EmbeddingClient): Client for generating text embeddings
        db (PineconeClient): Client for vector database operations

    Methods:
        setup: Initializes the vector database
        get_all_paper_in_date_range: Fetches all papers within a date range
        pdf_to_text: Converts PDF file to text
        download_pdf_to_text: Downloads PDF and converts to text
        add_days: Adds days to a date string
        upsert_paper: Upserts papers to the vector database
    """

    def __init__(self, index_name: str, chunk_size: int, max_results_per_request: int, init_db: bool) -> None:
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.max_results_per_request = max_results_per_request
        self.init_db = init_db
        self.searcher = SearchReference()

        self.client = EmbeddingClient(
            platform=settings.EMBEDDING_PLATFORM,
            api_key=settings.OPENAI_API_KEY if settings.EMBEDDING_PLATFORM == "openai" else "",
            embeddings_model_version=settings.EMBEDDING_MODEL_VERSION,
        )
        if settings.PINECONE_API_KEY is not None:
            self.db = PineconeClient(
                api_key=settings.PINECONE_API_KEY, index_name=self.index_name, embed_client=self.client
            )
        else:
            raise ValueError("PINECONE_API_KEY is not set.")

    def setup(self) -> None:
        if self.init_db:
            self.db.delete_index(self.index_name)

        self.db.create_index(
            dimension=settings.EMBEDDING_DIM,
            metric="cosine",
        )

    def get_all_paper_in_date_range(
        self, query: str, category: str | None, start_date: str, end_date: str
    ) -> list[AcademicPaper]:
        paper_list = []
        current_idx = 0
        while True:
            start_time = time.time()
            papers = self.searcher.search_arxiv(
                query=query,
                category=category,
                start_idx=current_idx,
                max_results=self.max_results_per_request,
                start_date=start_date,
                end_date=end_date,
            )
            paper_list.extend(papers)
            current_idx += len(papers)

            if len(papers) < self.max_results_per_request:
                # no more papers
                break

            # wait for the next request to the arXiv API
            elapsed_time = time.time() - start_time
            if elapsed_time < ARXIV_API_WAIT_TIME:
                time.sleep(ARXIV_API_WAIT_TIME - elapsed_time)

        return paper_list

    def pdf_to_text(self, pdf_path: str) -> str:
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)

        return "\n".join(text_content)

    def download_pdf_to_text(self, pdf_url: str, save_path: str) -> str:
        if Path(save_path).exists():
            return self.pdf_to_text(save_path)
        else:
            response = requests.get(pdf_url)
            with open(save_path, "wb") as f:
                f.write(response.content)

        return self.pdf_to_text(save_path)

    def add_days(self, date_str: str, delta_days: int) -> str:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        new_date_obj = date_obj + timedelta(days=delta_days)
        return new_date_obj.strftime("%Y%m%d")

    def upsert_paper(
        self, category: str | None, keywords: list[str], start_date: str, end_date: str, delta_days: int = 30
    ) -> None:
        total_paper_num = 0
        keyword_paper_counts = {keyword: 0 for keyword in keywords}

        for keyword in keywords:
            # reset dates for each keyword
            next_start_date = start_date
            next_end_date = self.add_days(start_date, delta_days)
            while True:
                papers = self.get_all_paper_in_date_range(
                    query=keyword, category=category, start_date=next_start_date, end_date=next_end_date
                )
                for paper in papers:
                    try:
                        text = self.download_pdf_to_text(
                            paper.pdf_url, f"{PAPER_PDF_DIRC}/{self.index_name}/{paper.id}.pdf"
                        )
                        self.db.upsert(
                            document_id=paper.id,
                            text=text,
                            chunk_size=self.chunk_size,
                            metadata={
                                "document_id": paper.id,
                                "pdf_url": paper.pdf_url,
                                "abstract": paper.abstract,
                                "published_date": paper.published_date,
                                "keyword": keyword,
                            },
                            chunk_method="size",
                            allow_update=False,
                        )
                    except Exception as e:
                        logger.error(f"Failed to upsert paper: {paper.id}. Skipped this file. Error: {e}")

                keyword_paper_counts[keyword] += len(papers)
                total_paper_num += len(papers)

                logger.info(
                    f"Upserted papers for keyword '{keyword}' from {next_start_date} to {next_end_date}. "
                    f"Papers for this keyword: {keyword_paper_counts[keyword]}, Total papers: {total_paper_num}"
                )

                next_start_date = next_end_date
                next_end_date = self.add_days(next_start_date, delta_days)

                if end_date < next_start_date:
                    # all papers are fetched for this keyword
                    break

        # Print summary for each keyword
        logger.info("\nSummary of papers by keyword:")
        for keyword, count in keyword_paper_counts.items():
            logger.info(f"'{keyword}': {count} papers")

        logger.info(
            f"\nDone upserting papers from arXiv. Total number of papers: {total_paper_num}."
            f"Total cost: ${self.db.total_cost:.2f}"
        )


@click.command()
@click.option("--chunk_size", type=int, default=1024, required=False, help="The size of the chunk.")
@click.option(
    "--max_results_per_request", type=int, default=50, required=False, help="The maximum number of results per request."
)
@click.option("--init_db", type=bool, default=False, required=False, help="Initialize the database.")
@click.option("--category", type=str, default=None, required=False, help="The category of the arXiv paper.")
@click.option(
    "--keywords",
    type=str,
    default="Quantum Machine Learning",
    required=False,
    help="The keywords of the arXiv paper. Multiple keywords can be specified with comma separation.",
    callback=lambda ctx, param, value: [k.strip() for k in value.split(",")],
)
@click.option(
    "--date_range",
    type=str,
    default="20200101-20241231",
    required=False,
    help="The date range of the arXiv paper. Format: 'YYYYMMDD-YYYYMMDD'",
)
def setup_arxiv_db(
    chunk_size: int,
    max_results_per_request: int,
    init_db: bool,
    category: str | None,
    keywords: list[str],
    date_range: str,
) -> None:
    if settings.ARXIV_INDEX_NAME is None:
        raise ValueError("ARXIV_INDEX_NAME is not set.")

    start_date, end_date = date_range.split("-")
    arxiv_db = ArxivPaperDB(
        index_name=settings.ARXIV_INDEX_NAME,
        chunk_size=chunk_size,
        max_results_per_request=max_results_per_request,
        init_db=init_db,
    )
    arxiv_db.setup()
    arxiv_db.upsert_paper(category=category, keywords=keywords, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    setup_arxiv_db()
