from astronaut.prompts.common.code_fixer import FixCodePrompt
from astronaut.prompts.common.expert import AskExpertPrompt
from astronaut.prompts.common.parse import ParseJsonPrompt
from astronaut.prompts.common.pennylane_reference import RelatedReferencePrompt
from astronaut.prompts.common.reflection import (
    DebateAdvocateIdeaPrompt,
    DebateCriticIdeaPrompt,
    SelfReflectionIdeaPrompt,
)
from astronaut.prompts.common.researcher import ResearcherDeveloperPrompt
from astronaut.prompts.common.scoring import ScoringIdeaPrompt
from astronaut.prompts.common.summarization import SummaryPaperPrompt

__all__ = [
    "FixCodePrompt",
    "AskExpertPrompt",
    "ParseJsonPrompt",
    "RelatedReferencePrompt",
    "DebateAdvocateIdeaPrompt",
    "DebateCriticIdeaPrompt",
    "SelfReflectionIdeaPrompt",
    "ResearcherDeveloperPrompt",
    "ScoringIdeaPrompt",
    "SummaryPaperPrompt",
]
