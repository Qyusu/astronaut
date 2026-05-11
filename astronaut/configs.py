import os
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from astronaut.constants import (
    DEFAULT_FILTERING_STEPS,
    DEFAULT_MAX_IDEA_NUM,
    DEFAULT_MAX_REFLECTION_ROUND,
    DEFAULT_MAX_SCORING_ROUND,
    DEFAULT_MAX_SUGGESTION_NUM,
    DEFAULT_MAX_TRIAL_NUM,
    MODULE_HOME,
)
from astronaut.schema import ReflectionType, ResearchMode

ENV_NONE_PATTERN = ["None", "NONE", "none", ""]
LOG_LEVEL = Literal[
    "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
]
MODEL_TYPE = Literal["quantum_kernel", "vqe"]


class ExperimentSettings(BaseModel):
    model_type: MODEL_TYPE = Field(
        description="Specify the model type from 'quantum_kernel' or 'vqe'."
    )
    experiment_name: str = Field(
        description="Specify the name of the experiment."
    )
    desc: str = Field(
        defalut="Automatically generated quantum circuit.",
        description="Specify the description of the experiment.",
    )
    max_trial_num: int = Field(
        default=DEFAULT_MAX_TRIAL_NUM,
        description="Specify the maximum number of trials to run.",
    )
    max_idea_num: int = Field(
        default=DEFAULT_MAX_IDEA_NUM,
        description="Specify the maximum number of ideas to generate at one trial in the generation component.",
    )
    max_suggestion_num: int = Field(
        default=DEFAULT_MAX_SUGGESTION_NUM,
        description="Specify the maximum number of suggestions during idea review.",
    )
    filtering_steps: list[int] = Field(
        default=DEFAULT_FILTERING_STEPS,
        description="Specify the number of ideas for each steps of filtering.",
    )
    research_mode: ResearchMode = Field(
        default=ResearchMode.NONE,
        description="Specify the type of deep research mode for seed idea generation (NONE, DEEP, LIGHT).",
    )
    existing_research_report_path: str | None = Field(
        default=None,
        description="Optional path to existing research report markdown file. If provided, the research step will be skipped and this file will be used instead.",
    )
    reflection_type: ReflectionType = Field(
        default=ReflectionType.NONE,
        description="Specify the type of reflection to use (NONE, SELF, DEBATE).",
    )
    max_reflection_round: int = Field(
        default=DEFAULT_MAX_REFLECTION_ROUND,
        description="Specify the maximum number of reflection rounds during idea refinement.",
    )
    max_scoring_round: int = Field(
        default=DEFAULT_MAX_SCORING_ROUND,
        description="Specify the maximum number of scoring rounds during idea evaluation.",
    )
    log_level: LOG_LEVEL = Field(
        default="DEBUG",
        description="Specify the log level for the log file (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL).",
    )
    console_log_level: LOG_LEVEL = Field(
        default="INFO",
        description="Specify the log level for console output (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL).",
    )

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentSettings":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f"{MODULE_HOME}/.env",
        env_file_encoding="utf-8",
    )

    # API key setting
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Chat setting
    CHAT_PLATFORM: str = "openai"
    DEFAULT_LIGHT_MODEL_VERSION: str = "gpt-5-mini-2025-08-07"
    DEFAULT_MODEL_VERSION: str = "gpt-5-2025-08-07"
    FILTERING_MODEL_VERSION: str = ""
    IDEA_MODEL_VERSION: str = ""
    SCORING_MODEL_VERSION: str = ""
    SUMMARY_MODEL_VERSION: str = ""
    EXPERT_MODEL_VERSION: str = ""
    REFLECTION_MODEL_VERSION: str = ""
    REFERENCE_EXTRACTION_MODEL_VERSION: str = ""
    CODE_MODEL_VERSION: str = ""
    VALIDATION_MODEL_VERSION: str = ""
    REVIEW_MODEL_VERSION: str = ""

    # Embedding setting
    EMBEDDING_PLATFORM: str = "openai"
    EMBEDDING_MODEL_VERSION: str | None = None
    EMBEDDING_DIM: int = 1536

    # Pinecone setting
    PINECONE_API_KEY: str | None = None
    PENNLYLANE_INDEX_NAME: str | None = None
    ARXIV_INDEX_NAME: str | None = None

    # LangSmith setting
    LANGCHAIN_TRACING_V2: str = ""
    LANGCHAIN_ENDPOINT: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = ""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.FILTERING_MODEL_VERSION:
            self.FILTERING_MODEL_VERSION = self.DEFAULT_MODEL_VERSION
        if not self.IDEA_MODEL_VERSION:
            self.IDEA_MODEL_VERSION = self.DEFAULT_MODEL_VERSION
        if not self.SCORING_MODEL_VERSION:
            self.SCORING_MODEL_VERSION = self.DEFAULT_LIGHT_MODEL_VERSION
        if not self.SUMMARY_MODEL_VERSION:
            self.SUMMARY_MODEL_VERSION = self.DEFAULT_LIGHT_MODEL_VERSION
        if not self.EXPERT_MODEL_VERSION:
            self.EXPERT_MODEL_VERSION = self.DEFAULT_LIGHT_MODEL_VERSION
        if not self.REFLECTION_MODEL_VERSION:
            self.REFLECTION_MODEL_VERSION = self.DEFAULT_MODEL_VERSION
        if not self.REFERENCE_EXTRACTION_MODEL_VERSION:
            self.REFERENCE_EXTRACTION_MODEL_VERSION = (
                self.DEFAULT_LIGHT_MODEL_VERSION
            )
        if not self.CODE_MODEL_VERSION:
            self.CODE_MODEL_VERSION = self.DEFAULT_MODEL_VERSION
        if not self.VALIDATION_MODEL_VERSION:
            self.VALIDATION_MODEL_VERSION = self.DEFAULT_LIGHT_MODEL_VERSION
        if not self.REVIEW_MODEL_VERSION:
            self.REVIEW_MODEL_VERSION = self.DEFAULT_MODEL_VERSION

        if (
            not self.EMBEDDING_MODEL_VERSION
            or self.EMBEDDING_MODEL_VERSION in ENV_NONE_PATTERN
        ):
            self.EMBEDDING_MODEL_VERSION = None

        if (
            not self.PINECONE_API_KEY
            or self.PINECONE_API_KEY in ENV_NONE_PATTERN
        ):
            self.PINECONE_API_KEY = None

        if (
            not self.PENNLYLANE_INDEX_NAME
            or self.PENNLYLANE_INDEX_NAME in ENV_NONE_PATTERN
        ):
            self.PENNLYLANE_INDEX_NAME = None

        if (
            not self.ARXIV_INDEX_NAME
            or self.ARXIV_INDEX_NAME in ENV_NONE_PATTERN
        ):
            self.ARXIV_INDEX_NAME = None


settings = Settings()

# set langsmith environment variables as environment variables
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
