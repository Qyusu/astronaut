from astronaut.llm.cost import (
    ChatModelCostPer1MToken,
    ChatModelCostTable,
    EmbeddingModelCostPer1MToken,
    EmbeddingModelCostTable,
)


def test_chat_model_cost_per_1m_token() -> None:
    """Test ChatModelCostPer1MToken model."""
    cost = ChatModelCostPer1MToken(input=1.0, cached=0.5, output=2.0)
    assert cost.input == 1.0
    assert cost.cached == 0.5
    assert cost.output == 2.0

    cost_without_cached = ChatModelCostPer1MToken(input=1.0, output=2.0)
    assert cost_without_cached.input == 1.0
    assert cost_without_cached.cached is None
    assert cost_without_cached.output == 2.0


def test_chat_model_cost_table_get_cost() -> None:
    """Test getting cost for specific models."""
    cost_table = ChatModelCostTable()

    # Test existing models
    gpt4o_cost = cost_table.get_cost("gpt-4o")
    assert gpt4o_cost is not None
    assert gpt4o_cost.input == 2.5
    assert gpt4o_cost.cached == 1.25
    assert gpt4o_cost.output == 10.0

    gemini_cost = cost_table.get_cost("gemini-2.0-flash")
    assert gemini_cost is not None
    assert gemini_cost.input == 0.1
    assert gemini_cost.cached == 0.025
    assert gemini_cost.output == 0.4

    # Test non-existent model
    assert cost_table.get_cost("non-existent-model") is None


def test_chat_model_cost_table_list_models() -> None:
    """Test listing all available models."""
    cost_table = ChatModelCostTable()
    models = cost_table.list_models()

    assert len(models) > 0
    assert "gpt-4o" in models
    assert "gemini-2.0-flash" in models
    assert "claude-3-5-haiku" in models


def test_embedding_model_cost_per_1m_token() -> None:
    """Test EmbeddingModelCostPer1MToken model."""
    cost = EmbeddingModelCostPer1MToken(input=0.1)
    assert cost.input == 0.1


def test_embedding_model_cost_table_get_cost() -> None:
    """Test getting cost for specific embedding models."""
    cost_table = EmbeddingModelCostTable()

    # Test existing models
    small_cost = cost_table.get_cost("text-embedding-3-small")
    assert small_cost is not None
    assert small_cost.input == 0.020

    large_cost = cost_table.get_cost("text-embedding-3-large")
    assert large_cost is not None
    assert large_cost.input == 0.130

    # Test non-existent model
    assert cost_table.get_cost("non-existent-model") is None


def test_embedding_model_cost_table_list_models() -> None:
    """Test listing all available embedding models."""
    cost_table = EmbeddingModelCostTable()
    models = cost_table.list_models()

    assert len(models) > 0
    assert "text-embedding-3-small" in models
    assert "text-embedding-3-large" in models
    assert "text-embedding-ada-002" in models
