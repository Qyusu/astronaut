from pydantic import BaseModel


class ChatModelCostPer1MToken(BaseModel):
    input: float
    cached: float | None = None
    output: float


class ChatModelCostTable(BaseModel):
    costs: dict[str, ChatModelCostPer1MToken] = {
        "gpt-4o-mini": ChatModelCostPer1MToken(input=0.15, cached=0.075, output=0.6),
        "gpt-4o": ChatModelCostPer1MToken(input=2.5, cached=1.25, output=10.0),
        "gpt-4.5-preview": ChatModelCostPer1MToken(input=75.0, cached=37.5, output=150.0),
        "o1-mini": ChatModelCostPer1MToken(input=3.0, cached=0.55, output=12.0),
        "o1-preview": ChatModelCostPer1MToken(input=15.0, output=60.0),
        "o1": ChatModelCostPer1MToken(input=15.0, cached=7.5, output=60.0),
        "o1-pro": ChatModelCostPer1MToken(input=150.0, output=600.0),
        "o3-mini": ChatModelCostPer1MToken(input=1.1, cached=0.55, output=4.4),
        "gemini-1.5-flash-8b": ChatModelCostPer1MToken(input=0.0375, cached=0.01, output=0.15),
        "gemini-1.5-flash": ChatModelCostPer1MToken(input=0.075, cached=0.01875, output=0.3),
        "gemini-1.5-pro": ChatModelCostPer1MToken(input=1.25, cached=0.3125, output=5.00),
        "gemini-2.0-flash-lite": ChatModelCostPer1MToken(input=0.075, output=0.3),
        "gemini-2.0-flash": ChatModelCostPer1MToken(input=0.1, cached=0.025, output=0.4),
        "gemini-2.0-pro-exp": ChatModelCostPer1MToken(input=0.0, output=0.0),
        "gemini-2.5-pro-exp": ChatModelCostPer1MToken(input=0.0, output=0.0),
        "claude-3-opus": ChatModelCostPer1MToken(input=15.0, cached=1.5, output=75.0),
        "claude-3-haiku": ChatModelCostPer1MToken(input=0.25, output=1.25),
        "claude-3-5-haiku": ChatModelCostPer1MToken(input=0.8, cached=0.08, output=4.0),
        "claude-3-5-sonnet": ChatModelCostPer1MToken(input=3.0, output=15.0),
        "claude-3-7-sonnet": ChatModelCostPer1MToken(input=3.0, cached=0.3, output=15.0),
    }

    def get_cost(self, model_name: str) -> ChatModelCostPer1MToken | None:
        return self.costs.get(model_name, None)

    def list_models(self) -> list[str]:
        return list(self.costs.keys())


class EmbeddingModelCostPer1MToken(BaseModel):
    input: float


class EmbeddingModelCostTable(BaseModel):
    costs: dict[str, EmbeddingModelCostPer1MToken] = {
        "text-embedding-3-small": EmbeddingModelCostPer1MToken(input=0.020),
        "text-embedding-3-large": EmbeddingModelCostPer1MToken(input=0.130),
        "text-embedding-ada-002": EmbeddingModelCostPer1MToken(input=0.100),
    }

    def get_cost(self, model_name: str) -> EmbeddingModelCostPer1MToken | None:
        return self.costs.get(model_name, None)

    def list_models(self) -> list[str]:
        return list(self.costs.keys())
