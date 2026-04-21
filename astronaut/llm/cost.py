from pydantic import BaseModel


class ChatModelCostPer1MToken(BaseModel):
    input: float
    cached: float | None = None
    output: float


class ChatModelCostTable(BaseModel):
    costs: dict[str, ChatModelCostPer1MToken] = {
        "gpt-4.1-nano": ChatModelCostPer1MToken(input=0.1, cached=0.025, output=0.4),
        "gpt-4.1-mini": ChatModelCostPer1MToken(input=0.4, cached=0.1, output=1.6),
        "gpt-4.1": ChatModelCostPer1MToken(input=2.0, cached=0.5, output=8.0),
        "gpt-5-chat-latest": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "gpt-5-nano": ChatModelCostPer1MToken(input=0.05, cached=0.005, output=0.4),
        "gpt-5-mini": ChatModelCostPer1MToken(input=0.25, cached=0.025, output=2.0),
        "gpt-5": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "gpt-5-pro": ChatModelCostPer1MToken(input=15.0, output=120.0),
        "gpt-5-codex": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "gpt-5.1": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "gpt-5.1-codex": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "gpt-5.1-codex-max": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "gpt-5.2": ChatModelCostPer1MToken(input=1.75, cached=0.175, output=14.0),
        "gpt-5.2-pro": ChatModelCostPer1MToken(input=21.0, output=168.0),
        "gpt-5.2-codex": ChatModelCostPer1MToken(input=1.75, cached=0.175, output=14.0),
        "gpt-5.3-codex": ChatModelCostPer1MToken(input=1.75, cached=0.175, output=14.0),
        "gpt-5.4-nano": ChatModelCostPer1MToken(input=0.20, cached=0.02, output=1.25),
        "gpt-5.4-mini": ChatModelCostPer1MToken(input=0.75, cached=0.075, output=4.5),
        "gpt-5.4": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "o1": ChatModelCostPer1MToken(input=15.0, cached=7.5, output=60.0),
        "o1-pro": ChatModelCostPer1MToken(input=150.0, output=600.0),
        "o3-mini": ChatModelCostPer1MToken(input=1.1, cached=0.55, output=4.4),
        "o3": ChatModelCostPer1MToken(input=2.0, cached=0.5, output=8.0),
        "o3-pro": ChatModelCostPer1MToken(input=20.0, output=80.0),
        "o3-deep-research": ChatModelCostPer1MToken(input=10.0, cached=2.5, output=40.0),
        "o4-mini": ChatModelCostPer1MToken(input=1.1, cached=0.275, output=4.4),
        "o4-mini-deep-research": ChatModelCostPer1MToken(input=2.0, cached=0.5, output=8.0),
        "gemini-2.0-flash-lite": ChatModelCostPer1MToken(input=0.075, output=0.3),
        "gemini-2.0-flash": ChatModelCostPer1MToken(input=0.1, cached=0.025, output=0.4),
        "gemini-2.5-flash-lite": ChatModelCostPer1MToken(input=0.1, cached=0.01, output=0.4),
        "gemini-2.5-flash": ChatModelCostPer1MToken(input=0.3, cached=0.03, output=2.5),
        "gemini-2.5-pro": ChatModelCostPer1MToken(input=1.25, cached=0.125, output=10.0),
        "gemini-3-flash-preview": ChatModelCostPer1MToken(input=0.0, cached=0.0, output=0.0),
        "gemini-3.1-pro-preview": ChatModelCostPer1MToken(input=0.0, cached=0.0, output=0.0),
        "gemini-3-deep-research-pro-preview": ChatModelCostPer1MToken(input=0.0, cached=0.0, output=0.0),
        "claude-opus-4": ChatModelCostPer1MToken(input=15.0, cached=1.5, output=75.0),
        "claude-opus-4-1": ChatModelCostPer1MToken(input=15.0, cached=1.5, output=75.0),
        "claude-opus-4-5": ChatModelCostPer1MToken(input=5.0, cached=0.5, output=25.0),
        "claude-opus-4-6": ChatModelCostPer1MToken(input=5.0, cached=0.5, output=25.0),
        "claude-haiku-4-5": ChatModelCostPer1MToken(input=1.0, cached=0.1, output=5.0),
        "claude-sonnet-4": ChatModelCostPer1MToken(input=3.0, cached=0.3, output=15.0),
        "claude-sonnet-4-5": ChatModelCostPer1MToken(input=3.0, cached=0.3, output=15.0),
        "claude-sonnet-4-6": ChatModelCostPer1MToken(input=3.0, cached=0.3, output=15.0),
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
