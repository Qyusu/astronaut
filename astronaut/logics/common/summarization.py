import json

from langsmith import traceable
from loguru import logger

from astronaut.constants import GPT_MAX_TOKENS
from astronaut.llm import ChatClient
from astronaut.prompts import SummaryPaperPrompt
from astronaut.schema import SummaryPaperResult


class SummaryPaper:
    """A class for summarizing academic papers using language models.

    This class provides functionality to summarize academic papers using language models.
    It supports both standard and reasoning-based model versions, with appropriate parsing
    and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use

    Methods:
        summary: Summarizes academic papers
    """

    def __init__(self, client: ChatClient, model_version: str) -> None:
        self.client = client
        self.model_version = model_version

    @traceable(tags=["summary", "paper"])
    def summary(self, paper_content: str, max_summary_words: int) -> tuple[str, float]:
        logger.info("Summarize Paper...")
        system_prompt, user_prompt = SummaryPaperPrompt(paper_content, max_summary_words).build()
        response = self.client.parse_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=0,
            n=1,
            max_tokens=GPT_MAX_TOKENS,
            # response_format=SummaryPaperResult,
            model_version=self.model_version,
        )

        # result = json.loads(content)
        logger.info("Summarize Paper is done.")

        return response.content, response.cost
