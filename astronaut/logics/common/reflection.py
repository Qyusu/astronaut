import json
from abc import ABC, abstractmethod
from typing import Callable, Literal, TypeVar

from langsmith import traceable
from loguru import logger
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel

from astronaut.constants import (
    NOT_PROVIDED_INFORMATION,
)
from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.logics.common.expert import PaperExpert
from astronaut.logics.common.summarization import PaperSummarizer
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    DebateAdvocateIdeaResult,
    DebateCriticIdeaResult,
    GeneratedIdea,
    IdeaScore,
    ReflectionConfig,
    ReflectionType,
    SelfReflectIdeaResult,
)

MAX_PAPER_PER_QUERY = 3
MAX_PAPER_CONTENT_TOKENS = 300000
MAX_CRITIC_ROUND = 2

T = TypeVar("T", bound=BaseModel)


class ReflectionStrategy(ABC):
    """Abstract base class for reflection strategies."""

    def __init__(
        self,
        reflector: "ReflectIdea",
        config: ReflectionConfig,
    ) -> None:
        self.reflector = reflector
        self.config = config

    @abstractmethod
    def should_reflect(self, idea: GeneratedIdea, score: IdeaScore) -> bool:
        """Determine if reflection should be performed."""
        pass

    @abstractmethod
    def reflect(
        self,
        prompt_builder_fn: Callable[..., tuple[dict[str, str], dict[str, str]]],
        n_idea_history: int | None,
        seed_idea: GeneratedIdea,
        seed_score: IdeaScore,
        critic_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ]
        | None = None,
        **kwargs,
    ) -> tuple[GeneratedIdea, float]:
        """Perform reflection based on the strategy."""
        pass


class NoReflectionStrategy(ReflectionStrategy):
    """Strategy that performs no reflection."""

    def should_reflect(self, idea: GeneratedIdea, score: IdeaScore) -> bool:
        return False

    def reflect(
        self,
        prompt_builder_fn: Callable[..., tuple[dict[str, str], dict[str, str]]],
        n_idea_history: int | None,
        seed_idea: GeneratedIdea,
        seed_score: IdeaScore,
        critic_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ]
        | None = None,
        **kwargs,
    ) -> tuple[GeneratedIdea, float]:
        return seed_idea, 0.0


class SelfReflectionStrategy(ReflectionStrategy):
    """Strategy that performs self-reflection."""

    def should_reflect(self, idea: GeneratedIdea, score: IdeaScore) -> bool:
        return self.config.max_rounds > 0

    def reflect(
        self,
        prompt_builder_fn: Callable[..., tuple[dict[str, str], dict[str, str]]],
        n_idea_history: int | None,
        seed_idea: GeneratedIdea,
        seed_score: IdeaScore,
        critic_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ]
        | None = None,
        **kwargs,
    ) -> tuple[GeneratedIdea, float]:
        return self.reflector.reflect_multi_round(
            prompt_builder_fn=prompt_builder_fn,
            n_idea_history=n_idea_history,
            seed_idea=seed_idea,
            seed_score=seed_score,
            max_reflection_round=self.config.max_rounds,
            **kwargs,
        )


class DebateReflectionStrategy(ReflectionStrategy):
    """Strategy that performs debate-based reflection."""

    def __init__(
        self,
        reflector: "ReflectIdea",
        config: ReflectionConfig,
    ) -> None:
        super().__init__(reflector, config)
        if reflector.expert_model is None:
            raise ValueError(
                "DebateReflectionStrategy requires expert model. Please set academic perper database at .env file."
            )

    def should_reflect(self, idea: GeneratedIdea, score: IdeaScore) -> bool:
        return self.config.max_rounds > 0

    def reflect(
        self,
        prompt_builder_fn: Callable[..., tuple[dict[str, str], dict[str, str]]],
        n_idea_history: int | None,
        seed_idea: GeneratedIdea,
        seed_score: IdeaScore,
        critic_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ]
        | None = None,
        **kwargs,
    ) -> tuple[GeneratedIdea, float]:
        if critic_prompt_builder_fn is None:
            raise ValueError(
                "DebateReflectionStrategy requires 'critic_prompt_builder_fn' argument"
            )

        return self.reflector.reflect_debate(
            advocate_prompt_builder_fn=prompt_builder_fn,
            critic_prompt_builder_fn=critic_prompt_builder_fn,
            n_idea_history=n_idea_history,
            seed_idea=seed_idea,
            seed_score=seed_score,
            max_reflection_round=self.config.max_rounds,
            **kwargs,
        )


def create_reflection_strategy(
    reflector: "ReflectIdea", config: ReflectionConfig
) -> ReflectionStrategy:
    """Factory function to create appropriate reflection strategy."""
    if config.type == ReflectionType.NONE:
        return NoReflectionStrategy(reflector, config)
    elif config.type == ReflectionType.SELF_REFLECTION:
        return SelfReflectionStrategy(reflector, config)
    elif config.type == ReflectionType.DEBATE_REFLECTION:
        return DebateReflectionStrategy(reflector, config)
    else:
        raise ValueError(f"Unknown reflection type: {config.type}")


class ReflectIdea:
    """A class for reflecting on and improving generated feature map ideas.

    This class provides functionality to reflect on and improve generated feature map
    ideas using language models. It supports both standard and reasoning-based model
    versions, with appropriate parsing and response formatting.

    Args:
        llm_client (ChatClient): Client for interacting with the language model
        model_versions (ModelVersions): Model versions to use
        paper_db_client (PineconeClient): Client for interacting with the arXiv database
        paper_summarizer (PaperSummarizer): Summarizer for the related papers
        max_paper_per_query (int): The maximum number of related papers to fetch per query
        max_paper_content_tokens (int): The maximum number of tokens to summarize the related papers

    Methods:
        reflect: Reflects on and improves generated feature map ideas
        reflect_multi_round: Reflects on and improves generated feature map ideas over multiple rounds
        reflect_with_strategy: Reflects using a specified strategy
    """

    def __init__(
        self,
        llm_client: ChatClient,
        model_version: str,
        paper_db_client: PineconeClient | None,
        paper_summarizer: PaperSummarizer,
        expert_model: PaperExpert | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        max_paper_per_query: int = MAX_PAPER_PER_QUERY,
        max_paper_content_tokens: int = MAX_PAPER_CONTENT_TOKENS,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model_version = model_version
        self.paper_db_client = paper_db_client
        self.paper_summarizer = paper_summarizer
        self.expert_model = expert_model
        self.max_paper_per_query = max_paper_per_query
        self.max_paper_content_tokens = max_paper_content_tokens
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    @traceable(tags=["self-reflection", "idea"])
    def reflect_multi_round(
        self,
        prompt_builder_fn: Callable[..., tuple[dict[str, str], dict[str, str]]],
        n_idea_history: int | None,
        seed_idea: GeneratedIdea,
        seed_score: IdeaScore,
        max_reflection_round: int,
    ) -> tuple[GeneratedIdea, float]:
        """Reflect an idea over multiple rounds, using a prompt builder and optional paper summarization.

        Args:
            prompt_builder_fn: A function that builds the system and user prompts for each reflection round.
            n_idea_history: The number of previous idea messages to include in the reflection history.
            seed_idea: The initial idea to reflect on.
            seed_score: The initial score of the idea.
            max_reflection_round: The maximum number of reflection rounds to perform.

        Returns:
            A tuple containing the final idea and the total cost of the reflection process.
        """
        total_cost = 0.0
        final_idea = seed_idea
        final_score = seed_score
        reflection_message_history = []

        for i in range(1, max_reflection_round + 1):
            logger.info(
                f"Idea self-reflection round ({i}/{max_reflection_round})"
            )
            related_works_str, related_works_cost = self._get_related_works(
                final_idea.summary
            )
            total_cost += related_works_cost

            developer_prompt, user_prompt = prompt_builder_fn(
                current_reflection_round=i,
                max_reflection_round=max_reflection_round,
                previous_idea=str(final_idea),
                previous_score=str(final_score),
                related_work=related_works_str,
            )
            reflected_idea, reflection_message_history, cost = self._reflect(
                developer_prompt=developer_prompt,
                user_prompt=user_prompt,
                message_history=reflection_message_history,
                n_history=n_idea_history,
                response_format=SelfReflectIdeaResult,
            )
            total_cost += cost

            reflected_idea = (
                reflected_idea
                or SelfReflectIdeaResult.create_error_fallback(final_idea)
            )

            # update final idea status
            final_idea = reflected_idea.result

            if reflected_idea.is_completed:
                logger.info("Idea reflection process done.")
                break

        return final_idea, total_cost

    @traceable(tags=["debate-reflection", "idea"])
    def reflect_debate(
        self,
        advocate_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ],
        critic_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ],
        n_idea_history: int | None,
        seed_idea: GeneratedIdea,
        seed_score: IdeaScore,
        max_reflection_round: int,
    ) -> tuple[GeneratedIdea, float]:
        """Reflect an idea using debate-based approach with advocate and critic perspectives.

        Args:
            advocate_prompt_builder_fn: Function that builds prompts for the advocate (supporting) perspective
            critic_prompt_builder_fn: Function that builds prompts for the critic (challenging) perspective
            n_idea_history: The number of previous idea messages to include in the reflection history
            seed_idea: The initial idea to reflect on
            seed_score: The initial score of the idea
            max_reflection_round: The maximum number of debate rounds to perform

        Returns:
            A tuple containing the final idea and the total cost of the debate process
        """
        total_cost = 0.0
        refined_idea = seed_idea
        advocate_response = (
            "This is the first round then no advocate response is provided."
        )
        advocate_history = []
        critic_history = []

        for current_reflection_round in range(1, max_reflection_round + 1):
            logger.info(
                f"Idea debate reflection round ({current_reflection_round}/{max_reflection_round})"
            )

            # Critic perspective
            critical_feedback, critic_history, critic_cost, is_completed = (
                self._critic(
                    critic_prompt_builder_fn=critic_prompt_builder_fn,
                    current_reflection_round=current_reflection_round,
                    max_reflection_round=max_reflection_round,
                    refined_idea=refined_idea,
                    initial_score=str(seed_score),
                    advocate_response=advocate_response,
                    critic_history=critic_history,
                    n_idea_history=n_idea_history,
                )
            )
            total_cost += critic_cost

            if is_completed:
                logger.info("Debate reflection process completed.")
                break

            # Advocate perspective
            refined_idea, advocate_response, advocate_history, advocate_cost = (
                self._advocate(
                    advocate_prompt_builder_fn=advocate_prompt_builder_fn,
                    current_reflection_round=current_reflection_round,
                    max_reflection_round=max_reflection_round,
                    refined_idea=refined_idea,
                    critical_feedback=critical_feedback,
                    advocate_history=advocate_history,
                    n_idea_history=n_idea_history,
                )
            )
            total_cost += advocate_cost

        return refined_idea, total_cost

    @traceable(tags=["critic", "idea"])
    def _critic(
        self,
        critic_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ],
        current_reflection_round: int,
        max_reflection_round: int,
        refined_idea: GeneratedIdea,
        initial_score: str,
        advocate_response: str,
        critic_history: MESSAGE_HISTORY_TYPE,
        n_idea_history: int | None,
        max_critic_round: int = MAX_CRITIC_ROUND,
    ) -> tuple[str, MESSAGE_HISTORY_TYPE, float, bool]:
        """Run the critic perspective for a debate round.

        Args:
            critic_prompt_builder_fn: Function that builds prompts for the critic perspective
            current_reflection_round: Current round number
            max_reflection_round: Maximum number of reflection rounds
            refined_idea: Current refined idea to critique
            initial_score: The initial score of the idea
            advocate_response: Previous advocate response
            critic_history: Message history for the critic
            n_idea_history: Number of previous idea messages to include in history
            max_critic_round: Maximum number of critic rounds

        Returns:
            A tuple containing:
            - critical_feedback: The critical feedback as a string
            - critic_history: Updated message history for the critic
            - critic_cost: Cost of the critic operation
            - is_completed: Whether the debate process is completed
        """
        if self.expert_model is None:
            raise ValueError("Debate reflection requires expert model.")

        logger.info(
            f"Running critic perspective for round {current_reflection_round}"
        )
        total_cost = 0.0

        # initial related works get from idea summary
        additional_information, related_works_cost = self._get_related_works(
            refined_idea.summary
        )
        total_cost += related_works_cost
        additional_information_query = None

        for _ in range(max_critic_round):
            critic_system, critic_user = critic_prompt_builder_fn(
                current_reflection_round=current_reflection_round,
                max_reflection_round=max_reflection_round,
                previous_idea=str(refined_idea),
                previous_score=initial_score,
                advocate_response=advocate_response,
                additional_information=additional_information,
                additional_information_query=additional_information_query,
            )

            critic_result, critic_history, critic_cost = self._reflect(
                developer_prompt=critic_system,
                user_prompt=critic_user,
                message_history=critic_history,
                n_history=n_idea_history,
                response_format=DebateCriticIdeaResult,
            )
            total_cost += critic_cost
            if critic_result is None:
                # If raise json parsing error, forced stop the current critic round
                critic_result = DebateCriticIdeaResult.create_error_fallback()
                break

            if critic_result.question:
                expert_response, expert_cost = (
                    self.expert_model.request_expert_response(
                        expert_role=critic_result.expert_role,
                        question=critic_result.question,
                        top_k=MAX_PAPER_PER_QUERY,
                        max_paper_tokens=int(
                            MAX_PAPER_CONTENT_TOKENS / MAX_PAPER_PER_QUERY
                        ),
                    )
                )
                additional_information = f"*{expert_response.expert_role}* answer: {expert_response.answer}"
                total_cost += expert_cost
            else:
                break

        critical_feedback = "--------------\n".join(
            [str(feedback) for feedback in critic_result.feedback]
        )

        return (
            critical_feedback,
            critic_history,
            total_cost,
            critic_result.is_completed,
        )

    @traceable(tags=["advocate", "idea"])
    def _advocate(
        self,
        advocate_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ],
        current_reflection_round: int,
        max_reflection_round: int,
        refined_idea: GeneratedIdea,
        critical_feedback: str,
        advocate_history: MESSAGE_HISTORY_TYPE,
        n_idea_history: int | None,
    ) -> tuple[GeneratedIdea, str, MESSAGE_HISTORY_TYPE, float]:
        """Run the advocate perspective for a debate round.

        Args:
            advocate_prompt_builder_fn: Function that builds prompts for the advocate perspective
            current_reflection_round: Current round number
            max_reflection_round: Maximum number of reflection rounds
            refined_idea: Current refined idea to improve
            critical_feedback: Critical feedback from the critic
            advocate_history: Message history for the advocate
            n_idea_history: Number of previous idea messages to include in history

        Returns:
            A tuple containing:
            - refined_idea: The refined idea
            - advocate_response: The advocate response as a string
            - advocate_history: Updated message history for the advocate
            - advocate_cost: Cost of the advocate operation
        """
        logger.info(
            f"Running advocate perspective for round {current_reflection_round}"
        )

        advocate_system, advocate_user = advocate_prompt_builder_fn(
            current_reflection_round=current_reflection_round,
            max_reflection_round=max_reflection_round,
            previous_idea=str(refined_idea),
            critical_feedback=critical_feedback,
        )

        advocate_result, advocate_history, advocate_cost = self._reflect(
            developer_prompt=advocate_system,
            user_prompt=advocate_user,
            message_history=advocate_history,
            n_history=n_idea_history,
            response_format=DebateAdvocateIdeaResult,
        )
        advocate_result = (
            advocate_result
            or DebateAdvocateIdeaResult.create_error_fallback(refined_idea)
        )

        refined_idea = advocate_result.idea
        advocate_response = "--------------\n".join(
            [str(response) for response in advocate_result.responses]
        )

        return refined_idea, advocate_response, advocate_history, advocate_cost

    def _reflect(
        self,
        developer_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        response_format: type[T],
        n_history: int | None,
    ) -> tuple[T | None, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Reflecting idea...")
        logger.debug(
            f"developer_prompt: {developer_prompt}, user_prompt: {user_prompt}"
        )

        try:
            response = self.llm_client.parse_chat(
                developer_prompt=developer_prompt,
                user_prompt=user_prompt,
                message_history=message_history,
                n_history=n_history,
                response_format=response_format,
                reasoning=self.reasoning_effort,
                verbosity=self.verbosity,
                model_version=self.model_version,
            )

            result = json.loads(response.content)
            logger.debug(f"Idea reflection result: {result}")
            logger.info("Idea reflection done and result parsed as JSON.")

            return (
                response_format(**result),
                response.message_history,
                response.cost,
            )
        except Exception as e:
            # If raise json parsing error, return None and fallback in the caller
            logger.error(f"Error reflecting idea: {e}")
            return None, message_history, 0.0

    def _get_related_works(self, query: str) -> tuple[str, float]:
        if self.paper_db_client is not None:
            result = self.paper_db_client.query(
                query, top_k=self.max_paper_per_query
            )
            papers, summary_cost = self.paper_summarizer.summarize_papers(
                result, self.max_paper_content_tokens
            )
            return "\n--------------".join(papers), summary_cost
        else:
            return NOT_PROVIDED_INFORMATION, 0.0

    def reflect_with_strategy(
        self,
        strategy: ReflectionStrategy,
        prompt_builder_fn: Callable[..., tuple[dict[str, str], dict[str, str]]],
        n_idea_history: int | None,
        idea: GeneratedIdea,
        score: IdeaScore,
        critic_prompt_builder_fn: Callable[
            ..., tuple[dict[str, str], dict[str, str]]
        ]
        | None = None,
        **kwargs,
    ) -> tuple[GeneratedIdea, float]:
        """Reflect an idea using the provided strategy.

        Args:
            strategy: The reflection strategy to use
            prompt_builder_fn: A function that builds the system and user prompts for each reflection round
            n_idea_history: The number of previous idea messages to include in the reflection history
            idea: The idea to reflect on
            score: The score of the idea
            critic_prompt_builder_fn: Optional critic prompt builder for debate reflection strategy
            **kwargs: Additional arguments to pass to the strategy

        Returns:
            A tuple containing the final idea and the total cost of the reflection process
        """
        if not strategy.should_reflect(idea, score):
            logger.info("Strategy determined no reflection is needed.")
            return idea, 0.0

        logger.info(
            f'Performing reflection using "{strategy.__class__.__name__}"'
        )
        return strategy.reflect(
            prompt_builder_fn=prompt_builder_fn,
            n_idea_history=n_idea_history,
            seed_idea=idea,
            seed_score=score,
            critic_prompt_builder_fn=critic_prompt_builder_fn,
            **kwargs,
        )
