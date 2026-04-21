import contextvars
import functools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, tracing_context
from loguru import logger
from openai.types.shared.reasoning_effort import ReasoningEffort
from pinecone import QueryResponse

import astronaut.constants as exp_const
from astronaut.llm import ChatClient
from astronaut.logics.common.paper import Paper
from astronaut.prompts.common import SummaryPaperPrompt
from astronaut.schema import SummaryPaperResult


class PaperSummarizer(Paper):
    """
    Summarizes academic papers from local PDF files using language models.

    This class loads the full text of academic papers from local PDF files and generates summaries
    using a language model. It supports batch processing, token length limitation, and customizable
    summary length.

    Args:
        client (ChatClient): Client for interacting with the language model.
        model_version (str): Version of the language model to use.
        local_paper_dir (str | None): Path to the local directory containing paper PDFs
            (default: from settings).

    Attributes:
        MAX_PAPER_SUMMARY_WORDS (int): Maximum number of words for each paper summary.
        client (ChatClient): Language model client.
        model_version (str): Model version.
        local_paper_dir (str | None): Directory containing local PDFs.

    Methods:
        summarize_papers(result, max_paper_tokens=None):
            Loads the full text of papers specified in result from local PDFs and generates summaries.
            Optionally limits the number of tokens per paper with max_paper_tokens.
            Returns a list of summaries and the total cost.
    """

    MAX_PAPER_SUMMARY_WORDS = 1000

    def __init__(
        self,
        client: ChatClient,
        model_version: str,
        local_paper_dir: str,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        super().__init__(local_paper_dir)
        self.client = client
        self.model_version = model_version
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    @traceable(tags=exp_const.LANGSMITH_SUMMARY_PAPER_TAGS)
    def summarize_papers(
        self,
        result: QueryResponse,
        max_paper_tokens: int | None = None,
        max_workers: int | None = None,
    ) -> tuple[list[str], float]:
        """
        Summarize papers from local PDF files using parallel processing.

        Args:
            result: The result from the Pinecone query.
            max_paper_tokens: The maximum number of tokens to summarize the related papers.
            max_workers: Maximum number of threads to use for parallel processing.
                        If None, will use the default from ThreadPoolExecutor.

        Returns:
            A tuple containing the list of summaries and the total cost.
        """
        papers = []
        summary_cost = 0.0

        # Get current langsmith run tree for tracing context propagation
        current_langsmith_run_tree = get_current_run_tree()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for match in result["matches"]:
                ctx = contextvars.copy_context()
                future = executor.submit(
                    ctx.run,
                    functools.partial(
                        self._process_single_paper_with_context,
                        match,
                        max_paper_tokens,
                        current_langsmith_run_tree,
                    ),
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    summary, cost = future.result()
                    papers.append(summary)
                    summary_cost += cost
                except Exception as exc:
                    logger.error(f"Paper summarization failed: {exc}")
                    raise exc

        return papers, summary_cost

    def _process_single_paper_with_context(
        self,
        match_data: dict,
        max_paper_tokens: int | None = None,
        parent_run_tree: Any | None = None,
    ) -> tuple[str, float]:
        """
        Process a single paper for summarization with proper tracing context.

        Args:
            match_data: Single match data from Pinecone query result.
            max_paper_tokens: The maximum number of tokens to summarize the paper.
            parent_run_tree: Parent run tree for tracing context.

        Returns:
            A tuple containing the summary string and the cost.
        """
        if parent_run_tree:
            with tracing_context(parent=parent_run_tree):
                return self._process_single_paper(match_data, max_paper_tokens)
        else:
            return self._process_single_paper(match_data, max_paper_tokens)

    def _process_single_paper(
        self, match_data: dict, max_paper_tokens: int | None = None
    ) -> tuple[str, float]:
        """
        Process a single paper for summarization.

        Args:
            match_data: Single match data from Pinecone query result.
            max_paper_tokens: The maximum number of tokens to summarize the paper.

        Returns:
            A tuple containing the summary string and the cost.
        """
        paper = self.load_full_text_from_local(
            match_data["metadata"]["document_id"]
        )
        paper = (
            self.cut_string_if_over_token_limit(paper, max_paper_tokens)
            if max_paper_tokens is not None
            else paper
        )
        summary, cost = self._summary(
            paper_content=paper,
            max_summary_words=self.MAX_PAPER_SUMMARY_WORDS,
        )
        return str(summary), cost

    def _summary(
        self, paper_content: str, max_summary_words: int
    ) -> tuple[SummaryPaperResult, float]:
        developer_prompt, user_prompt = SummaryPaperPrompt(
            paper_content, max_summary_words
        ).build()
        logger.info("Paper summarization...")

        try:
            response = self.client.parse_chat(
                developer_prompt=developer_prompt,
                user_prompt=user_prompt,
                message_history=[],
                n_history=0,
                response_format=SummaryPaperResult,
                reasoning=self.reasoning_effort,
                verbosity=self.verbosity,
                model_version=self.model_version,
            )
            logger.info("Paper summarization done.")

            result = json.loads(response.content)
            logger.debug(f"Paper summarization result: {result}")
            summary_result = SummaryPaperResult(**result)
            return summary_result, response.cost
        except Exception as e:
            logger.error(f"Error parsing paper summarization result: {e}")
            summary_result = SummaryPaperResult.create_error_fallback()

            return summary_result, 0.0
