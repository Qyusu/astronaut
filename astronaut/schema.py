import json
from enum import Enum
from textwrap import dedent
from typing import Any, Type, TypeAlias

import pandas as pd
from anthropic.types import MessageParam
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from astronaut.constants import COMPLETED

OPENAI_MESSAGE_HISTORY_TYPE: TypeAlias = list[str]
GEMINI_MESSAGE_HISTORY_TYPE: TypeAlias = list[str]
ANTHOROPIC_MESSAGE_HISTORY_TYPE: TypeAlias = list[MessageParam]
MESSAGE_HISTORY_TYPE: TypeAlias = (
    OPENAI_MESSAGE_HISTORY_TYPE
    | GEMINI_MESSAGE_HISTORY_TYPE
    | ANTHOROPIC_MESSAGE_HISTORY_TYPE
)


def get_schema_string(schema: Type[BaseModel]) -> str:
    def _extract_descriptions_with_type(schema: dict) -> dict:
        def _resolve_ref(ref: str, definitions: dict) -> dict:
            """Expand $ref and get the target definition"""
            ref_name = ref.split("/")[-1]
            return definitions.get(ref_name, {})

        def _traverse(properties: dict, definitions: dict) -> dict:
            """recursive processing of "properties" in schema"""
            result: dict[str, Any] = {}
            for key, value in properties.items():
                type_str = f"({value.get('type', 'object')})"

                if "$ref" in value:
                    ref_schema = _resolve_ref(value["$ref"], definitions)
                    if "description" in value:
                        result[key] = f"{value['description']} {type_str}"
                    result[key] = _traverse(
                        ref_schema.get("properties", {}), definitions
                    )

                elif "items" in value and "$ref" in value["items"]:
                    # if the value is a list of objects, add two items for the list
                    ref_schema = _resolve_ref(
                        value["items"]["$ref"], definitions
                    )
                    list_item = _traverse(
                        ref_schema.get("properties", {}), definitions
                    )
                    result[key] = [list_item, list_item]

                elif "properties" in value:
                    result[key] = _traverse(value["properties"], definitions)

                elif "description" in value:
                    result[key] = f"{value['description']} {type_str}"

            return result

        definitions = schema.get("$defs", {})
        return _traverse(schema.get("properties", {}), definitions)

    schema_dict = schema.model_json_schema()
    formatted_schema = _extract_descriptions_with_type(schema_dict)
    safe_schema = (
        json.dumps(formatted_schema, indent=2, ensure_ascii=False)
        .replace("{", "{{")
        .replace("}", "}}")
    )

    return safe_schema


class MessageHistory(BaseModel):
    review: MESSAGE_HISTORY_TYPE = Field(
        default_factory=list, description="The message history for review."
    )
    idea: MESSAGE_HISTORY_TYPE = Field(
        default_factory=list, description="The message history for idea."
    )
    code: MESSAGE_HISTORY_TYPE = Field(
        default_factory=list, description="The message history for code."
    )


class MessageHistoryNum(BaseModel):
    review: int | None = Field(
        default=None,
        description="Maximum number of message history for review.",
    )
    idea: int | None = Field(
        default=None, description="Maximum number of message history for idea."
    )
    code: int | None = Field(
        default=None, description="Maximum number of message history for code."
    )


class Score(BaseModel):
    score: float | None = Field(
        default=None, description="The score of viewpoint."
    )
    reason: str = Field(default="", description="The reason of the score.")


class IdeaScore(BaseModel):
    originality: Score = Field(
        default=Score(), description="The score and reason of originality."
    )
    feasibility: Score = Field(
        default=Score(), description="The score and reason of feasibility."
    )
    versatility: Score = Field(
        default=Score(), description="The score and reason of versatility."
    )

    def __str__(self) -> str:
        if (
            self.originality.score is None
            or self.feasibility.score is None
            or self.versatility.score is None
        ):
            return "The scoring process is not conducted yet."

        template = dedent(
            """
            - Originality: {originality}
            - Feasibility: {feasibility}
            - Versatility: {versatility}
            """
        )
        return template.format(
            originality=self.originality.score,
            feasibility=self.feasibility.score,
            versatility=self.versatility.score,
        )


class ScoringResult(BaseModel):
    score: IdeaScore = Field(..., description="The score of idea.")
    is_lack_information: bool = Field(
        ...,
        description="Whether the external information is lack for scoring idea.",
    )
    additional_key_sentences: list[str] = Field(
        ...,
        description="The additional key sentences for requesting more external information.",
    )


class SummaryPaperResult(BaseModel):
    title: str = Field(..., description="The title of the paper.")
    summary: str = Field(..., description="The summary of the paper.")

    def __str__(self) -> str:
        template = dedent(
            """
            Title: {title}
            Summary: {summary}
            """
        )
        return template.format(title=self.title, summary=self.summary)

    @classmethod
    def create_error_fallback(cls) -> "SummaryPaperResult":
        return cls(
            title="Not acquired.",
            summary="Paper summarization could not be acquired due to parsing error.",
        )


class ExpertResult(BaseModel):
    expert_role: str = Field(..., description="The role of the expert.")
    question: str = Field(..., description="The question of the expert.")
    answer: str = Field(..., description="The answer of the expert.")

    @classmethod
    def create_error_fallback(
        cls, expert_role: str, question: str
    ) -> "ExpertResult":
        return cls(
            expert_role=expert_role,
            question=question,
            answer="Expert answer could not be acquired due to parsing error.",
        )


class ReviewIdeaResult(BaseModel):
    keep_points: list[str] = Field(
        ..., description="The points to keep in the next idea."
    )
    suggestions: list[str] = Field(
        ..., description="The suggestions for the next idea."
    )

    def suggestions_list_str(self) -> str:
        if self.suggestions[0] == COMPLETED:
            return COMPLETED
        else:
            return "- " + "\n- ".join(self.suggestions)

    def review_comment(self) -> str:
        if self.suggestions[0] == COMPLETED:
            return COMPLETED
        else:
            keep_points_str = "- " + "\n- ".join(self.keep_points)
            suggestions_str = "- " + "\n- ".join(self.suggestions)
            return f"#### Keep Points:\n{keep_points_str}\n\n#### Suggestions:\n{suggestions_str}"


class GeneratedIdea(BaseModel):
    name: str = Field(..., description="The name of the generated idea.")
    summary: str = Field(..., description="The summary of the generated idea.")
    explanation: str = Field(
        ..., description="The detail explanation of the generated idea."
    )
    formula: str = Field(..., description="The formula of the generated idea.")
    key_sentences: list[str] = Field(
        ..., description="The key sentences expressing the generated idea."
    )

    def __str__(self) -> str:
        template = dedent(
            """
            ## name
            {name}

            ## summary
            {summary}

            ## explanation
            {explanation}

            ## formula
            {formula}

            ## key_sentences
            {key_sentences}
            """
        )
        return template.format(
            name=self.name,
            summary=self.summary,
            explanation=self.explanation,
            formula=self.formula,
            key_sentences="\n".join(self.key_sentences),
        )

    def get_string_for_code_generation(self) -> str:
        template = dedent(
            """
            ### Name
            {name}

            ### Explanation
            {explanation}

            ### Formula
            {formula}
            """
        )
        return template.format(
            name=self.name,
            explanation=self.explanation,
            formula=self.formula,
        )


class GeneratedIdeaResult(BaseModel):
    results: list[GeneratedIdea] = Field(
        ..., description="The results of the idea generation."
    )

    def details_str(self) -> str:
        template = dedent(
            """
            ## Idea {i}
            ### Name
            {name}

            ### Explanation
            {explanation}

            ### Formula
            {formula}
            """
        )

        details = []
        for i, result in enumerate(self.results):
            idea_str = template.format(
                i=i + 1,
                name=result.name,
                explanation=result.explanation,
                formula=result.formula,
            )
            details.append(idea_str)

        return "\n".join(details)


class GeneratedImpl(BaseModel):
    class_name: str = Field(
        ..., description="The name of the implemented class."
    )
    code: str = Field(..., description="The code of the implemented class.")

    @field_validator("code")
    def remove_code_wrapping(cls, value: str) -> str:
        """Remove wrapping strings like ```python\n and \n``` from code."""
        if value.startswith("```python\n") and value.endswith("\n```"):
            return value[len("```python\n") : -len("\n```")]
        return value


class GeneratedImplResult(BaseModel):
    results: list[GeneratedImpl] = Field(
        ..., description="The results of the code generation."
    )


class GeneratedResult(BaseModel):
    idea: GeneratedIdea = Field(..., description="The generated idea.")
    score: IdeaScore | None = Field(
        ..., description="The score of the generated idea."
    )
    implement: GeneratedImpl = Field(
        ..., description="The implementation of the generated idea."
    )


class GeneratedResults(BaseModel):
    results: list[GeneratedResult] = Field(
        ..., description="The results of the idea generation."
    )


class PennyLaneReferenceResult(BaseModel):
    method_names: list[str] = Field(
        ..., description="The method names of the PennyLane reference."
    )


class ResearchMode(str, Enum):
    """Research mode configuration."""

    NONE = "none"
    DEEP = "deep"
    LIGHT = "light"


class ReflectionType(str, Enum):
    """Reflection type configuration."""

    NONE = "none"
    SELF_REFLECTION = "self"
    DEBATE_REFLECTION = "debate"


class ReflectionConfig(BaseModel):
    """Configuration for reflection behavior."""

    type: ReflectionType = Field(
        default=ReflectionType.NONE, description="Type of reflection to use"
    )
    max_rounds: int = Field(
        default=3, description="Maximum number of reflection rounds"
    )


class SelfReflectIdeaResult(BaseModel):
    result: GeneratedIdea = Field(
        ..., description="The result of the idea reflection."
    )
    is_completed: bool = Field(
        ..., description="Whether the reflection is completed."
    )

    @classmethod
    def create_error_fallback(
        cls, idea: GeneratedIdea
    ) -> "SelfReflectIdeaResult":
        return cls(
            result=idea,
            is_completed=True,  # Stop reflection on error
        )


class DebateAdvocateResponse(BaseModel):
    feedback_id: int = Field(..., description="The id of the feedback.")
    action: str = Field(
        ...,
        description="The action to take for the feedback. 'accept' or 'reject'.",
    )
    reason: str = Field(..., description="The reason for the action.")

    def __str__(self) -> str:
        template = dedent(
            """
            Feedback ID: {feedback_id}
            Action: {action}
            Reason: {reason}
            """
        )
        return template.format(
            feedback_id=self.feedback_id,
            action=self.action,
            reason=self.reason,
        )

    @classmethod
    def create_error_fallback(cls) -> "DebateAdvocateResponse":
        return cls(
            feedback_id=0,
            action="reject",
            reason="Failed to parse LLM response",
        )


class DebateAdvocateIdeaResult(BaseModel):
    idea: GeneratedIdea = Field(
        ..., description="The idea to be refined by feedback."
    )
    responses: list[DebateAdvocateResponse] = Field(
        ..., description="The responses to the feedback."
    )

    @classmethod
    def create_error_fallback(
        cls, idea: GeneratedIdea
    ) -> "DebateAdvocateIdeaResult":
        return cls(
            idea=idea,
            responses=[DebateAdvocateResponse.create_error_fallback()],
        )


class DebateCriticFeedback(BaseModel):
    feedback_id: int = Field(..., description="The id of the feedback.")
    comment: str = Field(..., description="The comment of the feedback.")
    reason: str = Field(..., description="The reason of the feedback.")

    def __str__(self) -> str:
        template = dedent(
            """
            Feedback ID: {feedback_id}
            Comment: {comment}
            Reason: {reason}
            """
        )
        return template.format(
            feedback_id=self.feedback_id,
            comment=self.comment,
            reason=self.reason,
        )

    @classmethod
    def create_error_fallback(cls) -> "DebateCriticFeedback":
        return cls(
            feedback_id=0,
            comment="Failed to parse LLM response",
            reason="Error occurred during critic reflection",
        )


class DebateCriticIdeaResult(BaseModel):
    question: str = Field(
        ..., description="The question for asking to the expert."
    )
    expert_role: str = Field(..., description="The role of the expert.")
    feedback: list[DebateCriticFeedback] = Field(
        ..., description="The feedback from the critic."
    )

    @computed_field
    @property
    def is_completed(self) -> bool:
        """Whether the reflection is completed. True when feedback list is empty."""
        return len(self.feedback) == 0

    @classmethod
    def create_error_fallback(cls) -> "DebateCriticIdeaResult":
        return cls(
            question="",
            expert_role="",
            feedback=[DebateCriticFeedback.create_error_fallback()],
        )


class ModelVersions(BaseModel):
    default: str = Field(..., description="The default model version.")
    filtering: str = Field(
        ..., description="The model version for idea filtering."
    )
    idea: str = Field(..., description="The model version for idea generation.")
    scoring: str = Field(..., description="The model version for scoring idea.")
    summary: str = Field(
        ..., description="The model version for summarizing paper."
    )
    expert: str = Field(..., description="The model version for asking expert.")
    reflection: str = Field(
        ..., description="The model version for reflection."
    )
    reference: str = Field(
        ..., description="The model version for reference extraction."
    )
    code: str = Field(..., description="The model version for code generation.")
    validation: str = Field(
        ..., description="The model version for code validation."
    )
    review: str = Field(..., description="The model version for idea review.")

    def model_post_init(self, __context: dict[str, Any]) -> None:
        for field in self.model_fields:
            value = getattr(self, field)
            if value == "":
                setattr(self, field, self.default)


class RunContext(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={pd.DataFrame: lambda df: df.to_dict(orient="records")},
    )

    model_type: str = Field(..., description="The model type.")
    gen_config_dirc: str = Field(
        ..., description="The directory path for generated config."
    )
    gen_code_dirc: str = Field(
        ..., description="The directory path for generated code."
    )
    local_paper_dir: str = Field(
        ..., description="The directory path for local academic papers."
    )
    model_versions: ModelVersions = Field(
        ..., description="The model versions for each task."
    )
    n_qubits: int = Field(..., description="The number of device qubits.")
    max_trial_num: int = Field(..., description="The maximum number of trial.")
    max_idea_num: int = Field(..., description="The maximum number of ideas.")
    max_suggestion_num: int = Field(
        ..., description="The maximum number of suggestions."
    )
    max_scoring_round: int = Field(
        ..., description="The maximum number of scoring rounds."
    )
    reflection_config: ReflectionConfig = Field(
        default_factory=ReflectionConfig,
        description="Configuration for reflection behavior",
    )
    best_idea_abstract: str = Field(
        ..., description="The abstract of the best idea."
    )
    last_code: str = Field(
        ..., description="The code of the last implementation."
    )
    total_cost: float = Field(..., description="The total API usage cost")
    need_idea_review: bool = Field(
        ..., description="Whether the idea needs review."
    )
    review_comment: str = Field(
        ..., description="The review comment for the idea."
    )
    last_trial_results: str = Field(..., description="The last trial results.")
    score_histories: str = Field(..., description="The history of scores.")
    message_history: MessageHistory = Field(
        ..., description="The message history."
    )
    n_message_history: MessageHistoryNum = Field(
        ..., description="The number of message history for consideration."
    )
    eval_result_df: pd.DataFrame = Field(
        ..., description="The evaluation result dataframe."
    )
