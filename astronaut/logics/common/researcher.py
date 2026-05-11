from langsmith import traceable
from loguru import logger

from astronaut.llm import ChatClient
from astronaut.prompts.common import ResearcherDeveloperPrompt
from astronaut.schema import ResearchMode

DEEP_RESEARCH_MODEL_VERSION = "o3-deep-research-2025-06-26"
LIGHT_RESEARCH_MODEL_VERSION = "o4-mini-deep-research-2025-06-26"


class Researcher:
    def __init__(self, client: ChatClient, research_mode: ResearchMode) -> None:
        self.client = client
        self.research_mode = research_mode
        self.model_version = self._set_model_version()

    def _set_model_version(self) -> str:
        if self.research_mode == ResearchMode.DEEP:
            return DEEP_RESEARCH_MODEL_VERSION
        elif self.research_mode == ResearchMode.LIGHT:
            return LIGHT_RESEARCH_MODEL_VERSION
        else:
            raise ValueError(f"Invalid research mode: {self.research_mode}")

    @traceable(tags=["deep-research", "idea"])
    def research(
        self,
        user_prompt: str,
        save_report_path: str | None = None,
    ) -> tuple[str, float]:
        logger.info(f'Running deep research with model "{self.model_version}"')
        response = self.client.parse_chat(
            developer_prompt=ResearcherDeveloperPrompt().get_prompt(),
            user_prompt={
                "role": "user",
                "content": user_prompt,
            },
            message_history=[],
            n_history=0,
            response_format=None,
            model_version=self.model_version,
            tools=[
                {"type": "web_search_preview"},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
            background=True,
        )
        logger.info("Deep research done")
        research_report = response.content

        if save_report_path:
            self._save_research_report(research_report, save_report_path)

        return research_report, response.cost

    @staticmethod
    def load_existing_report(report_path: str) -> tuple[str, float]:
        """Load existing research report from markdown file"""
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = f.read()
            logger.info(f"Loaded existing research report from {report_path}")
            return report, 0.0
        except FileNotFoundError:
            logger.error(f"Research report not found: {report_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading research report: {e}")
            raise

    def _save_research_report(
        self, research_report: str, save_report_path: str
    ) -> None:
        with open(save_report_path, "w", encoding="utf-8") as f:
            f.write(research_report)
        logger.info(f"Research report saved to {save_report_path}")
