from typing import Any

from langsmith import traceable

from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.logics.common import (
    GenerateCode,
    GenerateIdea,
    PaperExpert,
    PaperSummarizer,
    ReflectIdea,
    ScoringIdea,
)
from astronaut.logics.common.experiments import format_batch_results
from astronaut.logics.common.reflection import (
    DebateReflectionStrategy,
    NoReflectionStrategy,
    ReflectionStrategy,
    SelfReflectionStrategy,
    create_reflection_strategy,
)
from astronaut.params import (
    ExpertParams,
    GenerateCodeParams,
    GenerateIdeaParams,
    PaperSummarizationParams,
    ReflectionParams,
    ScoringParams,
)
from astronaut.prompts.common import (
    DebateAdvocateIdeaPrompt,
    DebateCriticIdeaPrompt,
    SelfReflectionIdeaPrompt,
)
from astronaut.prompts.quantum_kernel import (
    FEATURE_MAP_CONSTRAINTS,
    REFLECTION_EXPERTISE,
    FeedbackBasedFeatureMapIdeaPrompt,
    GenerateFeatureMapCodePrompt,
)
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    GeneratedIdea,
    GeneratedIdeaResult,
    GeneratedImplResult,
    GeneratedResults,
    MessageHistory,
    RunContext,
    ScoringResult,
)


@traceable(tags=["generation"])
def generate_feature_map(
    llm_client: ChatClient,
    arxiv_db_client: PineconeClient | None,
    trial_num: int,
    context: RunContext,
) -> tuple[GeneratedResults, MessageHistory, float]:
    generate_cost = 0.0

    # Generate feature map idea
    first_generated_idea, idea_message_history, cost = (
        generate_feature_map_ideas(llm_client, trial_num, context)
    )
    generate_cost += cost

    # Initialize instances for scoring, summarization and reflection
    idea_scorer = ScoringIdea(
        llm_client=llm_client,
        paper_db_client=arxiv_db_client,
        model_version=context.model_versions.scoring,
        reasoning_effort=ScoringParams().reasoning_effort,
        verbosity=ScoringParams().verbosity,
    )
    paper_summarizer = PaperSummarizer(
        client=llm_client,
        model_version=context.model_versions.summary,
        local_paper_dir=context.local_paper_dir,
        reasoning_effort=PaperSummarizationParams().reasoning_effort,
        verbosity=PaperSummarizationParams().verbosity,
    )
    if arxiv_db_client is not None:
        expert_model = PaperExpert(
            llm_client=llm_client,
            paper_db_client=arxiv_db_client,
            model_version=context.model_versions.expert,
            local_paper_dir=context.local_paper_dir,
            reasoning_effort=ExpertParams().reasoning_effort,
            verbosity=ExpertParams().verbosity,
        )
    else:
        expert_model = None

    reflector = ReflectIdea(
        llm_client=llm_client,
        model_version=context.model_versions.reflection,
        paper_db_client=arxiv_db_client,
        paper_summarizer=paper_summarizer,
        expert_model=expert_model,
        reasoning_effort=ReflectionParams().reasoning_effort,
        verbosity=ReflectionParams().verbosity,
    )

    finalized_ideas = []
    finalized_scores = []
    for idea in first_generated_idea.results:
        # Scoring the initial idea
        first_generated_idea_score, cost = (
            idea_scorer.scoring_with_related_works(
                idea=idea,
                max_scoring_round=context.max_scoring_round,
                score_histories=context.score_histories,
            )
        )

        # Reflect the generated idea using strategy pattern
        reflection_strategy = create_reflection_strategy(
            reflector, context.reflection_config
        )
        final_idea, cost = reflect_feature_map_idea_with_strategy(
            reflector=reflector,
            strategy=reflection_strategy,
            context=context,
            idea=idea,
            idea_score=first_generated_idea_score,
        )
        generate_cost += cost

        # Scoring the reflected idea (only if reflection actually occurred)
        if final_idea != idea:  # Reflection occurred
            final_idea_score, cost = idea_scorer.scoring_with_related_works(
                idea=final_idea,
                max_scoring_round=context.max_scoring_round,
                score_histories=context.score_histories,
            )
            generate_cost += cost
        else:
            final_idea_score = first_generated_idea_score

        finalized_ideas.append(final_idea)
        finalized_scores.append(final_idea_score.score)

    # Generate feature map codes by finalized ideas
    generate_code_result, code_message_history, cost = (
        generate_feature_map_codes(
            llm_client=llm_client,
            code_model_version=context.model_versions.code,
            reference_model_version=context.model_versions.reference,
            last_code=context.last_code,
            finalized_ideas=finalized_ideas,
            message_history=context.message_history.code,
            n_code_history=context.n_message_history.code,
        )
    )
    generate_cost += cost

    # Update the message history
    updated_message_history = MessageHistory(
        review=context.message_history.review,
        idea=idea_message_history,
        code=code_message_history,
    )

    # Format the result
    finalized_results = format_batch_results(
        finalized_ideas, finalized_scores, generate_code_result.results
    )

    return finalized_results, updated_message_history, generate_cost


def generate_feature_map_ideas(
    llm_client: ChatClient, trial_num: int, context: RunContext
) -> tuple[GeneratedIdeaResult, MESSAGE_HISTORY_TYPE, float]:
    """
    Generate feature map ideas using the LLM client and context.
    Returns: (GeneratedIdeaResult, MESSAGE_HISTORY_TYPE, cost)
    """
    idea_generator = GenerateIdea(
        client=llm_client,
        model_version=context.model_versions.idea,
        reasoning_effort=GenerateIdeaParams().reasoning_effort,
        verbosity=GenerateIdeaParams().verbosity,
    )
    developer_prompt, user_prompt = FeedbackBasedFeatureMapIdeaPrompt().build(
        current_trial_num=trial_num,
        max_trial_num=context.max_trial_num,
        idea_num=context.max_idea_num,
        review_comment=str(context.review_comment),
        seed_ideas=str(context.last_trial_results) if trial_num == 1 else None,
    )
    first_generated_idea, idea_map_message_history, cost = (
        idea_generator.generate(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=context.message_history.idea,
            n_history=context.n_message_history.idea,
        )
    )
    return first_generated_idea, idea_map_message_history, cost


def create_reflection_prompt_builders(
    strategy: ReflectionStrategy,
) -> tuple[Any, Any]:
    """Create appropriate prompt builders based on strategy type."""

    if isinstance(strategy, SelfReflectionStrategy):

        def advocate_builder(
            **kwargs: Any,
        ) -> tuple[dict[str, str], dict[str, str]]:
            return SelfReflectionIdeaPrompt(
                expertise=REFLECTION_EXPERTISE,
                constraints=FEATURE_MAP_CONSTRAINTS,
            ).build(**kwargs)

        return advocate_builder, None

    elif isinstance(strategy, DebateReflectionStrategy):

        def advocate_builder(
            **kwargs: Any,
        ) -> tuple[dict[str, str], dict[str, str]]:
            return DebateAdvocateIdeaPrompt(
                expertise=REFLECTION_EXPERTISE,
                constraints=FEATURE_MAP_CONSTRAINTS,
            ).build(**kwargs)

        def critic_builder(
            **kwargs: Any,
        ) -> tuple[dict[str, str], dict[str, str]]:
            return DebateCriticIdeaPrompt(
                expertise=REFLECTION_EXPERTISE,
                constraints=FEATURE_MAP_CONSTRAINTS,
            ).build(**kwargs)

        return advocate_builder, critic_builder

    elif isinstance(strategy, NoReflectionStrategy):

        def dummy_builder(
            **kwargs: Any,
        ) -> tuple[dict[str, str], dict[str, str]]:
            return {}, {}

        return dummy_builder, None

    else:
        raise ValueError(f"Unknown strategy type: {type(strategy)}")


def reflect_feature_map_idea_with_strategy(
    reflector: ReflectIdea,
    strategy: ReflectionStrategy,
    context: RunContext,
    idea: GeneratedIdea,
    idea_score: ScoringResult,
) -> tuple[GeneratedIdea, float]:
    """
    Reflect the generated feature map idea using the specified strategy.
    Returns: (final_idea, cost)
    """

    # Create appropriate prompt builders based on strategy
    advocate_builder, critic_builder = create_reflection_prompt_builders(
        strategy
    )

    final_idea, cost = reflector.reflect_with_strategy(
        strategy=strategy,
        prompt_builder_fn=advocate_builder,
        n_idea_history=context.n_message_history.idea,
        idea=idea,
        score=idea_score.score,
        critic_prompt_builder_fn=critic_builder,
    )
    return final_idea, cost


def generate_feature_map_codes(
    llm_client: ChatClient,
    code_model_version: str,
    reference_model_version: str,
    last_code: str,
    finalized_ideas: list[GeneratedIdea],
    message_history,
    n_code_history,
) -> tuple["GeneratedImplResult", MESSAGE_HISTORY_TYPE, float]:
    """
    Generate feature map codes for the finalized ideas.
    Returns: (GeneratedImplResult, code_message_history, cost)
    """

    def code_prompt_builder(
        last_code: str,
        idea_result: GeneratedIdea,
        pennylane_reference: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        prompt = GenerateFeatureMapCodePrompt(
            code=last_code,
            idea=idea_result.get_string_for_code_generation(),
            pennylane_reference=pennylane_reference,
        )
        return prompt.build()

    code_generator = GenerateCode(
        client=llm_client,
        code_model_version=code_model_version,
        reference_model_version=reference_model_version,
        reasoning_effort=GenerateCodeParams().reasoning_effort,
        verbosity=GenerateCodeParams().verbosity,
    )
    generate_code_result, code_message_history, cost = (
        code_generator.generate_codes(
            prompt_builder_fn=code_prompt_builder,
            last_code=last_code,
            finalized_idea=GeneratedIdeaResult(results=finalized_ideas),
            message_history=message_history,
            n_code_history=n_code_history,
        )
    )
    return generate_code_result, code_message_history, cost
