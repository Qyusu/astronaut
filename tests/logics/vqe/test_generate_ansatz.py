from typing import Any

import pytest

import astronaut.logics.common as common_logics
import astronaut.schema as base_schema
from astronaut.logics.vqe.generate_ansatz import (
    create_reflection_prompt_builders,
    generate_ansatz,
    generate_ansatz_codes,
    generate_ansatz_ideas,
    reflect_ansatz_idea_with_strategy,
)


def test_create_reflection_prompt_builders(mocker: Any) -> None:
    mock_reflector = mocker.MagicMock()
    config = base_schema.ReflectionConfig()

    # SelfReflection
    strategy_self = common_logics.SelfReflectionStrategy(
        reflector=mock_reflector, config=config
    )
    advocate, critic = create_reflection_prompt_builders(strategy_self)
    assert advocate is not None
    assert critic is None

    # Debate
    # DebateReflectionStrategy requires expert_model in reflector
    mock_reflector.expert_model = mocker.MagicMock()
    strategy_debate = common_logics.DebateReflectionStrategy(
        reflector=mock_reflector, config=config
    )
    advocate, critic = create_reflection_prompt_builders(strategy_debate)
    assert advocate is not None
    assert critic is not None

    # NoReflection
    strategy_none = common_logics.NoReflectionStrategy(
        reflector=mock_reflector, config=config
    )
    advocate, critic = create_reflection_prompt_builders(strategy_none)
    assert advocate is not None
    assert critic is None


def test_generate_ansatz_ideas(
    mocker: Any, mock_llm_client: Any, run_context: base_schema.RunContext
) -> None:
    mock_gen_idea = mocker.patch(
        "astronaut.logics.common.GenerateIdea", autospec=True
    )
    instance = mock_gen_idea.return_value
    instance.generate.return_value = (
        base_schema.GeneratedIdeaResult(results=[]),
        [],
        0.1,
    )

    result, history, cost = generate_ansatz_ideas(
        llm_client=mock_llm_client,
        trial_num=1,
        context=run_context,
        target_molecule_names=["H2"],
    )

    assert cost == 0.1
    assert instance.generate.called


def test_generate_ansatz_codes(mocker: Any, mock_llm_client: Any) -> None:
    mock_gen_code = mocker.patch(
        "astronaut.logics.common.GenerateCode", autospec=True
    )
    instance = mock_gen_code.return_value
    instance.generate_codes.return_value = (
        base_schema.GeneratedImplResult(results=[]),
        [],
        0.2,
    )

    result, history, cost = generate_ansatz_codes(
        llm_client=mock_llm_client,
        code_model_version="test-model",
        reference_model_version="test-ref",
        last_code="print(1)",
        finalized_ideas=[],
        message_history=[],
        n_code_history=None,
    )

    assert cost == 0.2
    assert instance.generate_codes.called


def test_reflect_ansatz_idea_with_strategy(
    mocker: Any, run_context: base_schema.RunContext
) -> None:
    mock_reflector = mocker.MagicMock(spec=common_logics.ReflectIdea)
    strategy = common_logics.SelfReflectionStrategy(
        reflector=mock_reflector, config=run_context.reflection_config
    )
    idea = base_schema.GeneratedIdea(
        name="Idea",
        summary="S",
        explanation="E",
        formula="F",
        key_sentences=["K"],
    )
    score = base_schema.ScoringResult(
        score=base_schema.IdeaScore(
            originality=base_schema.Score(score=4.0, reason="R"),
            feasibility=base_schema.Score(score=4.0, reason="R"),
            versatility=base_schema.Score(score=4.0, reason="R"),
        ),
        is_lack_information=False,
        additional_key_sentences=[],
    )

    mock_reflector.reflect_with_strategy.return_value = (idea, 0.3)

    final_idea, cost = reflect_ansatz_idea_with_strategy(
        reflector=mock_reflector,
        strategy=strategy,
        context=run_context,
        idea=idea,
        idea_score=score,
    )

    assert final_idea == idea
    assert cost == 0.3
    assert mock_reflector.reflect_with_strategy.called


def test_generate_ansatz_orchestration(
    mocker: Any, mock_llm_client: Any, run_context: base_schema.RunContext
) -> None:
    # Patching PennyLaneReference to avoid CI error
    mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
    # Mocking all dependencies for full orchestration test
    idea_result = base_schema.GeneratedIdeaResult(
        results=[
            base_schema.GeneratedIdea(
                name="Idea",
                summary="S",
                explanation="E",
                formula="F",
                key_sentences=["K"],
            )
        ]
    )
    mocker.patch(
        "astronaut.logics.vqe.generate_ansatz.generate_ansatz_ideas",
        return_value=(idea_result, [], 0.1),
    )

    mock_scorer = mocker.patch(
        "astronaut.logics.common.ScoringIdea", autospec=True
    ).return_value
    mock_scorer.scoring_with_related_works.return_value = (
        base_schema.ScoringResult(
            score=base_schema.IdeaScore(
                originality=base_schema.Score(score=4.0, reason="R"),
                feasibility=base_schema.Score(score=4.0, reason="R"),
                versatility=base_schema.Score(score=4.0, reason="R"),
            ),
            is_lack_information=False,
            additional_key_sentences=[],
        ),
        0.2,
    )

    mocker.patch(
        "astronaut.logics.vqe.generate_ansatz.reflect_ansatz_idea_with_strategy",
        return_value=(idea_result.results[0], 0.3),
    )

    code_result = base_schema.GeneratedImplResult(
        results=[base_schema.GeneratedImpl(class_name="Ansatz", code="code")]
    )
    mocker.patch(
        "astronaut.logics.vqe.generate_ansatz.generate_ansatz_codes",
        return_value=(code_result, [], 0.4),
    )

    dataset_configs: dict[str, Any] = {"H2": {}}

    results, message_history, cost = generate_ansatz(
        llm_client=mock_llm_client,
        arxiv_db_client=None,
        trial_num=1,
        context=run_context,
        dataset_configs=dataset_configs,
    )

    # 0.1 (idea) + 0.2 (score) + 0.3 (reflect) + 0.4 (code) = 1.0
    assert cost == pytest.approx(1.0)
    assert len(results.results) == 1
    assert results.results[0].idea.name == "Idea"
    assert results.results[0].implement.class_name == "Ansatz"
