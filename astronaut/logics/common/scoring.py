import json
from textwrap import dedent
from typing import Literal

from langsmith import traceable
from loguru import logger
from openai.types.shared.reasoning_effort import ReasoningEffort
from pinecone import QueryResponse

import astronaut.constants as exp_const
from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.prompts.common import ScoringIdeaPrompt
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    GeneratedIdea,
    IdeaScore,
    Score,
    ScoringResult,
)


class ScoringIdea:
    """
    Scores generated ideas using language models and related academic papers.

    This class provides functionality to score generated ideas with the help of language models.
    It can automatically retrieve related works from an arXiv database (via Pinecone),
    supports multi-round scoring, and handles both standard and reasoning-based model versions.

    Args:
        llm_client (ChatClient): Client for interacting with the language model.
        paper_db_client (PineconeClient | None): Client for querying the arXiv database (can be None).
        model_version (str): Version of the language model to use.

    Attributes:
        llm_client (ChatClient): Language model client.
        paper_db_client (PineconeClient | None): arXiv database client.
        model_version (str): Model version.
        is_o1_series (bool): Whether the model is a reasoning (O1) series.

    Methods:
        scoring_with_related_works(idea, max_scoring_round=3, max_paper_per_query=1, score_histories=""):
            Scores an idea using related works from the arXiv database, with multi-round support.
            Returns a tuple of (ScoringResult, total_cost).
        _scoring(...):
            Internal method for scoring an idea with a given related work and message history.
        _format_fetch_paper(result):
            Formats the fetched paper information for scoring prompts.
    """

    def __init__(
        self,
        llm_client: ChatClient,
        paper_db_client: PineconeClient | None,
        model_version: str,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.paper_db_client = paper_db_client
        self.model_version = model_version
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    @traceable(tags=exp_const.LANGSMITH_SCORING_IDEA_TAGS)
    def scoring_with_related_works(
        self,
        idea: GeneratedIdea,
        max_scoring_round: int = exp_const.DEFAULT_MAX_SCORING_ROUND,
        max_paper_per_query: int = 1,
        score_histories: str = "",
    ) -> tuple[ScoringResult, float]:
        """Score an idea with related works from arXiv database.

        Args:
            idea: Idea to be scored
            max_scoring_round: Maximum number of scoring rounds
            max_paper_per_query: Maximum number of papers per query
            score_histories: History of previous scores

        Returns:
            Tuple of (ScoringResult, total_cost)
        """
        if max_scoring_round <= 0:
            return self._no_scoring_process()

        total_cost = 0.0
        key_sentences = list(idea.key_sentences)
        history: MESSAGE_HISTORY_TYPE = []
        seen_papers: list[str] = []

        for i in range(1, max_scoring_round + 1):
            if self.paper_db_client is not None:
                # Get related work information
                related_works: list[str] = []
                for query in key_sentences:
                    result = self.paper_db_client.query(
                        query,
                        top_k=max_paper_per_query,
                        metadata_filter={"id": {"$nin": seen_papers}},
                    )
                    papers = self._format_fetch_paper(result)
                    related_works.extend(papers)
                    seen_papers.extend([r["id"] for r in result["matches"]])
                related_works_str = "\n--------------".join(related_works)
            else:
                related_works_str = exp_const.NOT_PROVIDED_INFORMATION

            # Evaluate the generated idea and assign the score
            logger.info(f"Idea scoring round ({i}/{max_scoring_round})")
            scoring_result, history, cost = self._scoring(
                idea=idea,
                related_work=related_works_str,
                message_history=history,
                n_history=None,
                round=i,
                max_round=max_scoring_round,
                score_histories=score_histories,
            )
            total_cost += cost

            if scoring_result.is_lack_information:
                logger.info(
                    "Lack of information. Retry to fetch related work information."
                )
                key_sentences = list(scoring_result.additional_key_sentences)
            else:
                break

        return scoring_result, total_cost

    def _scoring(
        self,
        idea: GeneratedIdea,
        related_work: str,
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: int | None,
        round: int,
        max_round: int,
        score_histories: str = "",
    ) -> tuple[ScoringResult, MESSAGE_HISTORY_TYPE, float]:
        developer_prompt, user_prompt = ScoringIdeaPrompt(
            score_histories=score_histories
        ).build(
            idea=idea.explanation,
            related_work=related_work,
            round=round,
            max_round=max_round,
        )

        logger.info("Scoring Idea...")
        logger.debug(
            f"developer_prompt: {developer_prompt}, user_prompt: {user_prompt}"
        )
        response = self.llm_client.parse_chat(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=message_history,
            n_history=n_history,
            temperature=0.0,
            response_format=ScoringResult,
            reasoning=self.reasoning_effort,
            verbosity=self.verbosity,
            model_version=self.model_version,
        )

        result = json.loads(response.content)
        score_str = ", ".join(
            [
                f"{k}={v.get('score', 0.0)}"
                for k, v in result.get("score", {}).items()
            ]
        )
        logger.debug(f"Scoring result: {result}")
        logger.info(
            f"Idea scoring is done and result is parsed as JSON. Score: {score_str}"
        )

        return ScoringResult(**result), response.message_history, response.cost

    def _format_fetch_paper(self, result: QueryResponse) -> list[str]:
        papers = []
        for r in result["matches"]:
            paper = dedent(
                """
                Paper Id: {id}
                Abstract: {abstract}
                Releted Chunk: {chunk}
                """
            ).format(
                id=r["id"],
                abstract=r["metadata"]["abstract"],
                chunk=r["metadata"]["chunk_text"],
            )
            papers.append(paper)
        return papers

    def _no_scoring_process(self) -> tuple[ScoringResult, float]:
        logger.info("Scoring process is skipped.")
        return ScoringResult(
            score=IdeaScore(
                originality=Score(score=None, reason=""),
                feasibility=Score(score=None, reason=""),
                versatility=Score(score=None, reason=""),
            ),
            is_lack_information=False,
            additional_key_sentences=[],
        ), 0.0
