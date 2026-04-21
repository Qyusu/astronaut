from langsmith import traceable
from loguru import logger

import astronaut.constants as exp_const
from astronaut.llm import ChatClient, LLMProvider
from astronaut.prompts.common import ResearcherDeveloperPrompt
from astronaut.schema import ResearchMode

OPENAI_DEEP_RESEARCH_MODEL_VERSION = "o3-deep-research-2025-06-26"
OPENAI_LIGHT_RESEARCH_MODEL_VERSION = "o4-mini-deep-research-2025-06-26"
GOOGLE_RESEARCH_MODEL_VERSION = "deep-research-pro-preview-12-2025"


class Researcher:
    def __init__(self, client: ChatClient, research_mode: ResearchMode) -> None:
        self.client = client
        self.research_mode = research_mode
        self._check_supported_provider()
        self.model_version = self._set_model_version()

    def _check_supported_provider(self) -> None:
        supported_providers = {LLMProvider.OPENAI, LLMProvider.GOOGLE}
        provider = self.client.config.provider
        if provider not in supported_providers:
            allowed = ", ".join(p.value for p in supported_providers)
            raise ValueError(
                f"Deep-Research mode only supports providers: {allowed}, "
                f"but got: {provider.value}"
            )

    def _set_model_version(self) -> str:
        provider = self.client.config.provider
        mode = self.research_mode

        match (provider, mode):
            case (LLMProvider.OPENAI, ResearchMode.DEEP):
                return OPENAI_DEEP_RESEARCH_MODEL_VERSION
            case (LLMProvider.OPENAI, ResearchMode.LIGHT):
                return OPENAI_LIGHT_RESEARCH_MODEL_VERSION
            case (LLMProvider.GOOGLE, _):
                return GOOGLE_RESEARCH_MODEL_VERSION
            case _:
                raise ValueError(
                    f"Unsupported combination: provider={provider.value}, mode={mode}"
                )

    @traceable(tags=exp_const.LANGSMITH_DEEP_RESEARCH_IDEA_TAGS)
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
            use_interactions_api=True,
        )
        logger.info("Deep research done")

        research_report = response.content
        cost = response.cost

        if save_report_path:
            self._save_research_report(research_report, save_report_path)

        return research_report, cost

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
