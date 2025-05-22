import os

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_NONE_PATTERN = ["None", "NONE", "none", ""]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
    )

    # API key setting
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Chat setting
    CHAT_PLATFORM: str = "openai"
    DEFAULT_MODEL_VERSION: str = "gpt-4.1-2025-04-14"
    DEFAULT_REASONING_MODEL_VERSION: str = "o4-mini-2025-04-16"
    IDEA_MODEL_VERSION: str = ""
    SCORING_MODEL_VERSION: str = ""
    SUMMARY_MODEL_VERSION: str = ""
    REFLECTION_MODEL_VERSION: str = ""
    CODE_MODEL_VERSION: str = ""
    VALIDATION_MODEL_VERSION: str = ""
    REVIEW_MODEL_VERSION: str = ""
    PARSER_MODEL_VERSION: str = ""

    # Embedding setting
    EMBEDDING_PLATFORM: str = "openai"
    EMBEDDING_MODEL_VERSION: str | None = None
    EMBEDDING_DIM: int = 1536

    # Pinecone setting
    PINECONE_API_KEY: str | None = None
    PENNLYLANE_INDEX_NAME: str | None = None
    ARXIV_INDEX_NAME: str | None = None

    # Local paper setting
    LOCAL_PAPER_DIR: str | None = None

    # LangSmith setting
    LANGCHAIN_TRACING_V2: str = ""
    LANGCHAIN_ENDPOINT: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = ""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.IDEA_MODEL_VERSION:
            self.IDEA_MODEL_VERSION = self.DEFAULT_REASONING_MODEL_VERSION
        if not self.SCORING_MODEL_VERSION:
            self.SCORING_MODEL_VERSION = self.DEFAULT_MODEL_VERSION
        if not self.SUMMARY_MODEL_VERSION:
            self.SUMMARY_MODEL_VERSION = self.DEFAULT_MODEL_VERSION
        if not self.REFLECTION_MODEL_VERSION:
            self.REFLECTION_MODEL_VERSION = self.DEFAULT_REASONING_MODEL_VERSION
        if not self.CODE_MODEL_VERSION:
            self.CODE_MODEL_VERSION = self.DEFAULT_REASONING_MODEL_VERSION
        if not self.VALIDATION_MODEL_VERSION:
            self.VALIDATION_MODEL_VERSION = self.DEFAULT_MODEL_VERSION
        if not self.REVIEW_MODEL_VERSION:
            self.REVIEW_MODEL_VERSION = self.DEFAULT_REASONING_MODEL_VERSION
        if not self.PARSER_MODEL_VERSION:
            self.PARSER_MODEL_VERSION = self.DEFAULT_MODEL_VERSION

        if not self.EMBEDDING_MODEL_VERSION or self.EMBEDDING_MODEL_VERSION in ENV_NONE_PATTERN:
            self.EMBEDDING_MODEL_VERSION = None

        if not self.PINECONE_API_KEY or self.PINECONE_API_KEY in ENV_NONE_PATTERN:
            self.PINECONE_API_KEY = None

        if not self.PENNLYLANE_INDEX_NAME or self.PENNLYLANE_INDEX_NAME in ENV_NONE_PATTERN:
            self.PENNLYLANE_INDEX_NAME = None

        if not self.ARXIV_INDEX_NAME or self.ARXIV_INDEX_NAME in ENV_NONE_PATTERN:
            self.ARXIV_INDEX_NAME = None

        if not self.LOCAL_PAPER_DIR or self.LOCAL_PAPER_DIR in ENV_NONE_PATTERN:
            self.LOCAL_PAPER_DIR = None


settings = Settings()

# set langsmith environment variables as environment variables
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
