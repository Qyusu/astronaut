import os

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    DEFAULT_MODEL_VERSION: str = "gpt-4o-mini"
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
    EMBEDDING_MODEL_VERSION: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536

    # Pinecone setting
    PINECONE_API_KEY: str = ""
    PENNLYLANE_INDEX_NAME: str = ""
    ARXIV_INDEX_NAME: str = ""

    # Local paper setting
    LOCAL_PAPER_DIR: str = ""

    # LangSmith setting
    LANGCHAIN_TRACING_V2: str = ""
    LANGCHAIN_ENDPOINT: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = ""


settings = Settings()

# set langsmith environment variables as environment variables
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
