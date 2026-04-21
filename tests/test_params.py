from astronaut.params import (
    CodeValidationParams,
    ExpertParams,
    GenerateCodeParams,
    GenerateIdeaParams,
    PaperSummarizationParams,
    PennyLaneReferenceParams,
    ReflectionParams,
    ReviewParams,
    ScoringParams,
)


def test_code_validation_params_defaults() -> None:
    params = CodeValidationParams()
    assert params.reasoning_effort == "minimal"
    assert params.verbosity == "medium"


def test_expert_params_defaults() -> None:
    params = ExpertParams()
    assert params.reasoning_effort == "minimal"
    assert params.verbosity == "low"


def test_generate_code_params_defaults() -> None:
    params = GenerateCodeParams()
    assert params.reasoning_effort == "xhigh"
    assert params.verbosity == "medium"


def test_generate_idea_params_defaults() -> None:
    params = GenerateIdeaParams()
    assert params.reasoning_effort == "xhigh"
    assert params.verbosity == "high"


def test_paper_summarization_params_defaults() -> None:
    params = PaperSummarizationParams()
    assert params.reasoning_effort == "minimal"
    assert params.verbosity == "low"


def test_pennylane_reference_params_defaults() -> None:
    params = PennyLaneReferenceParams()
    assert params.reasoning_effort == "minimal"
    assert params.verbosity == "low"


def test_reflection_params_defaults() -> None:
    params = ReflectionParams()
    assert params.reasoning_effort == "xhigh"
    assert params.verbosity == "high"


def test_review_params_defaults() -> None:
    params = ReviewParams()
    assert params.reasoning_effort == "xhigh"
    assert params.verbosity == "high"


def test_scoring_params_defaults() -> None:
    params = ScoringParams()
    assert params.reasoning_effort == "minimal"
    assert params.verbosity == "low"
