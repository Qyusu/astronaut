from astronaut.llm.config import (
    AnthropicConfig,
    GoogleConfig,
    LLMProvider,
    OpenAIConfig,
)


def test_config_to_dict_excludes_provider() -> None:
    config = OpenAIConfig(api_key="key", default_model_version="gpt-4")
    d = config.to_dict()
    assert "provider" not in d
    assert d["api_key"] == "key"
    assert d["default_model_version"] == "gpt-4"


def test_openai_config_defaults() -> None:
    config = OpenAIConfig(api_key="key", default_model_version="v1")
    assert config.provider == LLMProvider.OPENAI


def test_anthropic_config_defaults() -> None:
    config = AnthropicConfig(api_key="key", default_model_version="v1")
    assert config.provider == LLMProvider.ANTHROPIC
    assert config.thinking_model_max_tokens == 64000
    assert config.basic_model_max_tokens == 8192


def test_google_config_defaults() -> None:
    config = GoogleConfig(api_key="key", default_model_version="v1")
    assert config.provider == LLMProvider.GOOGLE
    assert config.project_id is None
