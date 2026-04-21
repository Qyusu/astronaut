from typing import Literal

from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel


class CodeValidationParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "minimal"
    verbosity: Literal["low", "medium", "high"] | None = "medium"


class ExpertParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "minimal"
    verbosity: Literal["low", "medium", "high"] | None = "low"


class GenerateCodeParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "xhigh"
    verbosity: Literal["low", "medium", "high"] | None = "medium"


class GenerateIdeaParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "xhigh"
    verbosity: Literal["low", "medium", "high"] | None = "high"


class PaperSummarizationParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "minimal"
    verbosity: Literal["low", "medium", "high"] | None = "low"


class PennyLaneReferenceParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "minimal"
    verbosity: Literal["low", "medium", "high"] | None = "low"


class ReflectionParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "xhigh"
    verbosity: Literal["low", "medium", "high"] | None = "high"


class ReviewParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "xhigh"
    verbosity: Literal["low", "medium", "high"] | None = "high"


class ScoringParams(BaseModel):
    reasoning_effort: ReasoningEffort | None = "minimal"
    verbosity: Literal["low", "medium", "high"] | None = "low"
