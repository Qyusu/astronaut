import pandas as pd

from astronaut.logics.common.experiments import utils
from astronaut.logics.common.experiments.utils import (
    get_best_result_id,
    update_run_context,
    setup_generated_dircs,
    initialize_run_context_base,
    get_best_idea_abstract,
    update_score_histories,
    format_batch_results
)
import astronaut.schema as base_schema
from astronaut.schema import RunContext
from pytest_mock import MockerFixture


def test_update_run_context(run_context: RunContext) -> None:
    updated_context = update_run_context(
        run_context,
        eval_result_df=pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "aggregated_cost": [0.1, 0.05, 0.2]}),
    )

    assert updated_context.eval_result_df.equals(
        pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "aggregated_cost": [0.1, 0.05, 0.2]})
    )


def test_get_best_result_id() -> None:
    # maximize accuracy (Quantum Kernel)
    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "accuracy_validation": [0.5, 0.35, 0.9]})
    best_result_id = get_best_result_id(result_df, "accuracy_validation", minimize=False)
    assert best_result_id == "1_3"

    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "accuracy_validation": [0.0, 0.5, 0.0]})
    best_result_id = get_best_result_id(result_df, "accuracy_validation", minimize=False)
    assert best_result_id == "1_2"

    # minimize cost (VQE)
    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "aggregated_cost": [0.1, 0.05, 0.2]})
    best_result_id = get_best_result_id(result_df, "aggregated_cost", minimize=True)
    assert best_result_id == "1_2"

    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "aggregated_cost": [0.1, float("inf"), float("inf")]})
    best_result_id = get_best_result_id(result_df, "aggregated_cost", minimize=True)
    assert best_result_id == "1_1"


def test_setup_generated_dircs(mocker: MockerFixture) -> None:
    mock_exists = mocker.patch("os.path.exists", return_value=False)
    mock_makedirs = mocker.patch("os.makedirs")

    gen_config_dirc, gen_code_dirc = setup_generated_dircs("test_exp")

    assert "test_exp" in gen_config_dirc
    assert "test_exp" in gen_code_dirc
    assert mock_makedirs.call_count == 2
    mock_makedirs.assert_any_call(gen_config_dirc)
    mock_makedirs.assert_any_call(gen_code_dirc)


def test_initialize_run_context_base(mocker: MockerFixture) -> None:
    mock_setup_dircs = mocker.patch("astronaut.logics.common.experiments.utils.setup_generated_dircs", return_value=("/config", "/code"))
    mock_experiment = mocker.Mock()
    mock_experiment.name = "test_exp"

    context = initialize_run_context_base(
        experiment=mock_experiment,
        max_trial_num=5,
        max_idea_num=3,
        max_suggestion_num=2,
        max_scoring_round=1,
        reflection_type=base_schema.ReflectionType.NONE,
        max_reflection_round=2,
        seed_code="print('hello')",
    )

    assert context["gen_config_dirc"] == "/config"
    assert context["gen_code_dirc"] == "/code"
    assert context["max_trial_num"] == 5
    assert context["last_code"] == "print('hello')"
    assert isinstance(context["message_history"], base_schema.MessageHistory)
    assert context["reflection_config"].type == base_schema.ReflectionType.NONE


def test_get_best_idea_abstract(mocker: MockerFixture) -> None:
    mock_idea = mocker.Mock()
    mock_idea.explanation = "Optimize circuit depth."
    mock_idea.formula = "min(depth)"

    idea = base_schema.GeneratedIdea(
        name="test",
        summary="sum",
        explanation="Optimize circuit depth.",
        formula="min(depth)",
        key_sentences=["key1"]
    )
    result = base_schema.GeneratedResult(
        idea=idea,
        score=None,
        implement=base_schema.GeneratedImpl(code="code", class_name="Test")
    )

    abstract = get_best_idea_abstract(result)

    assert "### Idea Explanation:\nOptimize circuit depth." in abstract
    assert "### Idea Mathematical Expression:\nmin(depth)" in abstract


def test_update_score_histories() -> None:
    initial_history = "Initial history."
    idea = base_schema.GeneratedIdea(
        name="Algo1",
        summary="Uses ansatz X",
        explanation="exp",
        formula="f",
        key_sentences=["k"]
    )
    score = base_schema.IdeaScore(
        originality=base_schema.Score(score=0.9, reason="O"),
        feasibility=base_schema.Score(score=0.8, reason="F"),
        versatility=base_schema.Score(score=0.7, reason="V")
    )
    implement = base_schema.GeneratedImpl(code="c", class_name="C")

    results = base_schema.GeneratedResults(results=[
        base_schema.GeneratedResult(idea=idea, score=score, implement=implement)
    ])

    updated_history = update_score_histories(initial_history, results)

    assert initial_history in updated_history
    assert "Algo1" in updated_history
    assert "Uses ansatz X" in updated_history
    assert "0.9" in updated_history


def test_format_batch_results() -> None:
    idea = base_schema.GeneratedIdea(
        name="Algo1",
        summary="summary",
        explanation="exp",
        formula="f",
        key_sentences=["k"]
    )
    score = base_schema.IdeaScore(
        originality=base_schema.Score(score=0.9, reason="O"),
        feasibility=base_schema.Score(score=0.8, reason="F"),
        versatility=base_schema.Score(score=0.7, reason="V")
    )
    implement = base_schema.GeneratedImpl(code="c", class_name="C")

    formatted = format_batch_results([idea], [score], [implement])

    assert len(formatted.results) == 1
    assert formatted.results[0].idea == idea
    assert formatted.results[0].score == score
    assert formatted.results[0].implement == implement
