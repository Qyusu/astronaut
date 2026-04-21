from textwrap import dedent

import astronaut.constants as exp_const
from astronaut.schema import (
    DebateAdvocateIdeaResult,
    DebateAdvocateResponse,
    DebateCriticFeedback,
    DebateCriticIdeaResult,
    ExpertResult,
    GeneratedIdea,
    GeneratedIdeaResult,
    GeneratedImpl,
    IdeaScore,
    ModelVersions,
    ReflectionConfig,
    ReflectionType,
    ReviewIdeaResult,
    Score,
    SelfReflectIdeaResult,
    SummaryPaperResult,
)


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


class TestSummaryPaperResult:
    def test_str(self) -> None:
        res = SummaryPaperResult(title="T", summary="S")
        s = str(res)
        assert "Title: T" in s
        assert "Summary: S" in s

    def test_create_error_fallback(self) -> None:
        res = SummaryPaperResult.create_error_fallback()
        assert res.title == "Not acquired."
        assert "due to parsing error" in res.summary


class TestExpertResult:
    def test_create_error_fallback(self) -> None:
        res = ExpertResult.create_error_fallback(
            expert_role="Role", question="Q"
        )
        assert res.expert_role == "Role"
        assert res.question == "Q"
        assert "due to parsing error" in res.answer


class TestReviewIdeaResult:
    def test_review_comment_not_completed(self) -> None:
        res = ReviewIdeaResult(keep_points=["K1"], suggestions=["S1"])
        assert not res.is_completed
        comment = res.review_comment()
        assert "#### Keep Points:\n- K1" in comment
        assert "#### Suggestions:\n- S1" in comment

    def test_review_comment_completed(self) -> None:
        res = ReviewIdeaResult(
            keep_points=["K1"], suggestions=[exp_const.COMPLETED]
        )
        assert res.is_completed
        assert res.review_comment() == exp_const.COMPLETED

    def test_empty_lists(self) -> None:
        res = ReviewIdeaResult(keep_points=[], suggestions=[])
        assert not res.is_completed
        comment = res.review_comment()
        assert "No keep points." in comment
        assert "No suggestions." in comment


class TestGeneratedIdea:
    def test_str(self) -> None:
        idea = GeneratedIdea(
            name="Test Name",
            summary="Test Summary",
            explanation="Test Explanation",
            formula="Test Formula",
            key_sentences=["Key 1", "Key 2"],
        )
        s = str(idea)
        assert "## name\nTest Name" in s
        assert "## summary\nTest Summary" in s
        assert "## explanation\nTest Explanation" in s
        assert "## formula\nTest Formula" in s
        assert "## key_sentences\nKey 1\nKey 2" in s

    def test_get_string_for_code_generation(self) -> None:
        idea = GeneratedIdea(
            name="Test Name",
            summary="Test Summary",
            explanation="Test Explanation",
            formula="Test Formula",
            key_sentences=["Key 1"],
        )
        s = idea.get_string_for_code_generation()
        assert "### Name\nTest Name" in s
        assert "Test Summary" not in s
        assert "### Explanation\nTest Explanation" in s
        assert "### Formula\nTest Formula" in s
        assert "Key 1" not in s


class TestGeneratedIdeaResult:
    def test_details_str(self) -> None:
        idea1 = GeneratedIdea(
            name="N1",
            summary="S1",
            explanation="E1",
            formula="F1",
            key_sentences=["K1"],
        )
        idea2 = GeneratedIdea(
            name="N2",
            summary="S2",
            explanation="E2",
            formula="F2",
            key_sentences=["K2"],
        )
        res = GeneratedIdeaResult(results=[idea1, idea2])
        s = res.details_str()
        assert "## Idea 1" in s
        assert "### Name\nN1" in s
        assert "S1" not in s
        assert "### Explanation\nE1" in s
        assert "### Formula\nF1" in s
        assert "K1" not in s
        assert "## Idea 2" in s
        assert "### Name\nN2" in s
        assert "S2" not in s
        assert "### Explanation\nE2" in s
        assert "### Formula\nF2" in s
        assert "K2" not in s


class TestGeneratedImpl:
    def test_remove_code_wrapping(self) -> None:
        # Wrapper removal
        code_with_wrap = "```python\ndef foo(): pass\n```"
        impl = GeneratedImpl(class_name="C", code=code_with_wrap)
        assert impl.code == "def foo(): pass"

        # No wrapper
        code_no_wrap = "def bar(): pass"
        impl2 = GeneratedImpl(class_name="C", code=code_no_wrap)
        assert impl2.code == "def bar(): pass"


def test_reflection_config_default() -> None:
    config = ReflectionConfig()
    assert config.type == ReflectionType.NONE
    assert config.max_rounds == 3


class TestSelfReflectIdeaResult:
    def test_create_error_fallback(self) -> None:
        idea = GeneratedIdea(
            name="N",
            summary="S",
            explanation="E",
            formula="F",
            key_sentences=["K"],
        )
        res = SelfReflectIdeaResult.create_error_fallback(idea)
        assert res.result == idea
        assert res.is_completed is True


class TestDebateAdvocateResponse:
    def test_str(self) -> None:
        res = DebateAdvocateResponse(feedback_id=1, action="accept", reason="R")
        s = str(res)
        assert "Feedback ID: 1" in s
        assert "Action: accept" in s
        assert "Reason: R" in s

    def test_create_error_fallback(self) -> None:
        res = DebateAdvocateResponse.create_error_fallback()
        assert res.feedback_id == 0
        assert res.action == "reject"
        assert "Failed to parse" in res.reason


class TestDebateAdvocateIdeaResult:
    def test_create_error_fallback(self) -> None:
        idea = GeneratedIdea(
            name="N",
            summary="S",
            explanation="E",
            formula="F",
            key_sentences=["K"],
        )
        res = DebateAdvocateIdeaResult.create_error_fallback(idea)
        assert res.idea == idea
        assert len(res.responses) == 1
        assert res.responses[0].feedback_id == 0


class TestDebateCriticFeedback:
    def test_str(self) -> None:
        fb = DebateCriticFeedback(feedback_id=1, comment="C", reason="R")
        s = str(fb)
        assert "Feedback ID: 1" in s
        assert "Comment: C" in s
        assert "Reason: R" in s

    def test_create_error_fallback(self) -> None:
        fb = DebateCriticFeedback.create_error_fallback()
        assert fb.feedback_id == 0
        assert "Failed to parse" in fb.comment


class TestDebateCriticIdeaResult:
    def test_is_completed(self) -> None:
        res_empty = DebateCriticIdeaResult(
            question="Q", expert_role="E", feedback=[]
        )
        assert res_empty.is_completed is True

        fb = DebateCriticFeedback(feedback_id=1, comment="C", reason="R")
        res_filled = DebateCriticIdeaResult(
            question="Q", expert_role="E", feedback=[fb]
        )
        assert res_filled.is_completed is False

    def test_create_error_fallback(self) -> None:
        res = DebateCriticIdeaResult.create_error_fallback()
        assert res.question == ""
        assert res.expert_role == ""
        assert len(res.feedback) == 1
        assert res.feedback[0].feedback_id == 0


def test_default_values() -> None:
    model_versions = ModelVersions(
        default="gpt-5-2025-08-07",
        expanding="",
        idea="",
        scoring="",
        summary="",
        expert="",
        reflection="",
        reference="",
        code="",
        review="",
    )
    assert model_versions.default == "gpt-5-2025-08-07"
    assert model_versions.expanding == "gpt-5-2025-08-07"
    assert model_versions.idea == "gpt-5-2025-08-07"
    assert model_versions.scoring == "gpt-5-2025-08-07"
    assert model_versions.summary == "gpt-5-2025-08-07"
    assert model_versions.expert == "gpt-5-2025-08-07"
    assert model_versions.reflection == "gpt-5-2025-08-07"
    assert model_versions.reference == "gpt-5-2025-08-07"
    assert model_versions.code == "gpt-5-2025-08-07"
    assert model_versions.review == "gpt-5-2025-08-07"
