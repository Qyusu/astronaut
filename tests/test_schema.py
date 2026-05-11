from textwrap import dedent

from astronaut.schema import IdeaScore, ModelVersions, Score


class TestIdeaScore:
    def test_str(self) -> None:
        # test for default values
        idea_score = IdeaScore()
        assert str(idea_score) == "The scoring process is not conducted yet."

        # test for scoring process is not conducted
        idea_score = IdeaScore(
            originality=Score(score=None, reason=""),
            feasibility=Score(score=None, reason=""),
            versatility=Score(score=None, reason=""),
        )
        assert str(idea_score) == "The scoring process is not conducted yet."

        # test for scoring process is conducted
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


def test_default_values() -> None:
    model_versions = ModelVersions(
        default="gpt-5-2025-08-07",
        filtering="",
        idea="",
        scoring="",
        summary="",
        expert="",
        reflection="",
        reference="",
        code="",
        validation="",
        review="",
    )
    assert model_versions.default == "gpt-5-2025-08-07"
    assert model_versions.filtering == "gpt-5-2025-08-07"
    assert model_versions.idea == "gpt-5-2025-08-07"
    assert model_versions.scoring == "gpt-5-2025-08-07"
    assert model_versions.summary == "gpt-5-2025-08-07"
    assert model_versions.expert == "gpt-5-2025-08-07"
    assert model_versions.reflection == "gpt-5-2025-08-07"
    assert model_versions.reference == "gpt-5-2025-08-07"
    assert model_versions.code == "gpt-5-2025-08-07"
    assert model_versions.validation == "gpt-5-2025-08-07"
    assert model_versions.review == "gpt-5-2025-08-07"
