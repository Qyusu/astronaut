from textwrap import dedent

from astronaut.schema import IdeaScore, ModelVersions, Score


class TestIdeaScore:
    def test_str(self) -> None:
        idea_score = IdeaScore(
            originality=Score(score=5.0, reason="good"),
            feasibility=Score(score=6.2, reason="good"),
            versatility=Score(score=2.0, reason="bad"),
        )
        assert str(idea_score) == dedent(
            """
            - Originality: 5.0
            - Feasibility: 6.2
            - Versatility: 2.0
            """
        )

    def test_is_improved(self) -> None:
        # case1: diff_score = 4.0 (idea going up)
        idea_score = IdeaScore(
            originality=Score(score=5.0, reason="good"),
            feasibility=Score(score=6.2, reason="good"),
            versatility=Score(score=2.0, reason="bad"),
        )
        prev_score = IdeaScore(
            originality=Score(score=3.0, reason="good"),
            feasibility=Score(score=5.2, reason="good"),
            versatility=Score(score=1.0, reason="bad"),
        )

        # improved case
        assert idea_score.is_improved(prev_score, threshold=3.9)  # border case
        assert idea_score.is_improved(prev_score, threshold=0.0)
        assert idea_score.is_improved(prev_score, threshold=-30.0)  # min threshold

        # not improved case
        assert not idea_score.is_improved(prev_score, threshold=4.0)  # border case
        assert not idea_score.is_improved(prev_score, threshold=30.0)  # max threshold

        # case2: diff_score = -4.0 (idea going down)
        idea_score = IdeaScore(
            originality=Score(score=3.0, reason="good"),
            feasibility=Score(score=5.2, reason="good"),
            versatility=Score(score=1.0, reason="bad"),
        )
        prev_score = IdeaScore(
            originality=Score(score=5.0, reason="good"),
            feasibility=Score(score=6.2, reason="good"),
            versatility=Score(score=2.0, reason="bad"),
        )

        # improved case
        assert idea_score.is_improved(prev_score, threshold=-4.1)  # border case
        assert idea_score.is_improved(prev_score, threshold=-30.0)  # min threshold

        # not improved case
        assert not idea_score.is_improved(prev_score, threshold=4.0)  # border case
        assert not idea_score.is_improved(prev_score, threshold=0.0)  # border case
        assert not idea_score.is_improved(prev_score, threshold=30.0)  # max threshold


def test_default_values() -> None:
    model_versions = ModelVersions(
        default="gpt-4o-2024-11-20",
        idea="",
        scoring="",
        summary="",
        reflection="",
        code="",
        validation="",
        review="",
        parser="",
    )
    assert model_versions.default == "gpt-4o-2024-11-20"
    assert model_versions.idea == "gpt-4o-2024-11-20"
    assert model_versions.scoring == "gpt-4o-2024-11-20"
    assert model_versions.summary == "gpt-4o-2024-11-20"
    assert model_versions.reflection == "gpt-4o-2024-11-20"
    assert model_versions.code == "gpt-4o-2024-11-20"
    assert model_versions.validation == "gpt-4o-2024-11-20"
    assert model_versions.review == "gpt-4o-2024-11-20"
    assert model_versions.parser == "gpt-4o-2024-11-20"
