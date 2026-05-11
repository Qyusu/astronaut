import json
from typing import Literal

from langsmith import traceable
from loguru import logger
from openai.types.shared.reasoning_effort import ReasoningEffort

from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.logics.common.paper import Paper
from astronaut.prompts.common import AskExpertPrompt
from astronaut.schema import ExpertResult


class PaperExpert(Paper):
    def __init__(
        self,
        llm_client: ChatClient,
        paper_db_client: PineconeClient,
        model_version: str,
        local_paper_dir: str,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        super().__init__(local_paper_dir)
        self.llm_client = llm_client
        self.paper_db_client = paper_db_client
        self.model_version = model_version
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    @traceable(tags=["expert", "paper"])
    def request_expert_response(
        self,
        expert_role: str,
        question: str,
        top_k: int = 3,
        max_paper_tokens: int | None = None,
    ) -> tuple[ExpertResult, float]:
        papers = []
        result = self.paper_db_client.query(question, top_k=top_k)
        for r in result["matches"]:
            paper = self.load_full_text_from_local(r["metadata"]["document_id"])
            paper = (
                self.cut_string_if_over_token_limit(paper, max_paper_tokens)
                if max_paper_tokens is not None
                else paper
            )
            papers.append(paper)

        return self._ask_expert(expert_role, question, papers)

    def _ask_expert(
        self, expert_role: str, question: str, papers: list[str]
    ) -> tuple[ExpertResult, float]:
        paper_str = "\n\n".join(
            [f"### Paper {i + 1}\n{paper}" for i, paper in enumerate(papers)]
        )
        developer_prompt, user_prompt = AskExpertPrompt(
            expert_role, question, paper_str
        ).build()
        logger.info(f'Asking expert "{expert_role}"...')

        try:
            response = self.llm_client.parse_chat(
                developer_prompt=developer_prompt,
                user_prompt=user_prompt,
                message_history=[],
                n_history=0,
                response_format=ExpertResult,
                reasoning=self.reasoning_effort,
                verbosity=self.verbosity,
                model_version=self.model_version,
            )
            logger.info("Expert response done.")

            result = json.loads(response.content)
            logger.trace(f"Expert response result: {result}")
            expert_result = ExpertResult(**result)
            return expert_result, response.cost
        except Exception as e:
            logger.error(f"Error parsing expert response: {e}")
            expert_result = ExpertResult.create_error_fallback(
                expert_role=expert_role,
                question=question,
            )

            return expert_result, 0.0
