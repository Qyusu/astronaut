from typing import Any

import pytest

from astronaut.logics.common.reflection import (
    DebateReflectionStrategy,
    NoReflectionStrategy,
    SelfReflectionStrategy,
    create_reflection_strategy,
)
from astronaut.schema import (
    GeneratedIdea,
    IdeaScore,
    ReflectionConfig,
    ReflectionType,
    Score,
)


@pytest.fixture
def mock_reflector(mocker: Any) -> Any:
    return mocker.MagicMock()


@pytest.fixture
def idea() -> GeneratedIdea:
    return GeneratedIdea(
        name="Test Idea",
        summary="Summary",
        explanation="Explanation",
        formula="Formula",
        key_sentences=["Key 1"],
    )


@pytest.fixture
def score() -> IdeaScore:
    return IdeaScore(
        originality=Score(score=0.8, reason="Good"),
        feasibility=Score(score=0.7, reason="Okay"),
        versatility=Score(score=0.9, reason="Great"),
    )


class TestReflectionStrategies:
    def test_create_no_reflection(self, mock_reflector: Any) -> None:
        config = ReflectionConfig(type=ReflectionType.NONE)
        strategy = create_reflection_strategy(mock_reflector, config)
        assert isinstance(strategy, NoReflectionStrategy)

    def test_create_self_reflection(self, mock_reflector: Any) -> None:
        config = ReflectionConfig(type=ReflectionType.SELF_REFLECTION)
        strategy = create_reflection_strategy(mock_reflector, config)
        assert isinstance(strategy, SelfReflectionStrategy)

    def test_create_debate_reflection(self, mock_reflector: Any) -> None:
        config = ReflectionConfig(type=ReflectionType.DEBATE_REFLECTION)
        strategy = create_reflection_strategy(mock_reflector, config)
        assert isinstance(strategy, DebateReflectionStrategy)

    def test_no_reflection_should_reflect(
        self, mock_reflector: Any, idea: GeneratedIdea, score: IdeaScore
    ) -> None:
        config = ReflectionConfig(type=ReflectionType.NONE)
        strategy = NoReflectionStrategy(mock_reflector, config)
        assert strategy.should_reflect(idea, score) is False

    def test_no_reflection_reflect_returns_seed(
        self,
        mocker: Any,
        mock_reflector: Any,
        idea: GeneratedIdea,
        score: IdeaScore,
    ) -> None:
        config = ReflectionConfig(type=ReflectionType.NONE)
        strategy = NoReflectionStrategy(mock_reflector, config)
        prompt_builder = mocker.MagicMock()

        result_idea, cost = strategy.reflect(
            prompt_builder_fn=prompt_builder,
            n_idea_history=None,
            seed_idea=idea,
            seed_score=score,
        )

        assert result_idea == idea
        assert cost == 0.0

    def test_self_reflection_should_reflect(
        self, mock_reflector: Any, idea: GeneratedIdea, score: IdeaScore
    ) -> None:
        config = ReflectionConfig(type=ReflectionType.SELF_REFLECTION)
        strategy = SelfReflectionStrategy(mock_reflector, config)
        assert strategy.should_reflect(idea, score) is True

    def test_debate_reflection_should_reflect(
        self, mock_reflector: Any, idea: GeneratedIdea, score: IdeaScore
    ) -> None:
        config = ReflectionConfig(type=ReflectionType.DEBATE_REFLECTION)
        strategy = DebateReflectionStrategy(mock_reflector, config)
        assert strategy.should_reflect(idea, score) is True

    def test_self_reflection_reflect(
        self,
        mocker: Any,
        mock_reflector: Any,
        idea: GeneratedIdea,
        score: IdeaScore,
    ) -> None:
        config = ReflectionConfig(type=ReflectionType.SELF_REFLECTION)
        strategy = SelfReflectionStrategy(mock_reflector, config)

        mock_reflector.reflect_multi_round.return_value = (
            GeneratedIdea(
                name="Improved",
                summary="...",
                explanation="...",
                formula="...",
                key_sentences=[],
            ),
            1.5,
        )

        prompt_builder = mocker.MagicMock()
        result_idea, cost = strategy.reflect(
            prompt_builder_fn=prompt_builder,
            n_idea_history=None,
            seed_idea=idea,
            seed_score=score,
        )

        assert result_idea.name == "Improved"
        assert cost == 1.5
        assert mock_reflector.reflect_multi_round.called
