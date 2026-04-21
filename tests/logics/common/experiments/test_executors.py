import pandas as pd
import pytest
from typing import Any
from astronaut.logics.common.experiments.executors import TrialExecutor, ExpandingExecutor, BaselineExecutor
from astronaut.logics.common.experiments.dependencies import ExperimentDependencies
import astronaut.schema as base_schema
from astronaut.logics.common.experiments.runner import BaseIdeaProcessor
from pytest_mock import MockerFixture

@pytest.fixture
def mock_dependencies(mocker: MockerFixture) -> Any:
    deps = mocker.Mock(spec=ExperimentDependencies)
    deps.llm_client = mocker.Mock()
    deps.arxiv_db_client = mocker.Mock()
    deps.qml_db_client = mocker.Mock()
    return deps

@pytest.fixture
def mock_context(mocker: MockerFixture) -> Any:
    context = mocker.Mock(spec=base_schema.RunContext)
    context.message_history = base_schema.MessageHistory()
    context.gen_code_dirc = "/tmp"
    context.model_versions = mocker.Mock()
    context.model_versions.expanding = "v_exp"
    context.model_versions.code = "v_code"
    context.model_versions.reference = "v_ref"
    context.last_code = "last_code"
    # Allow model_copy to return self for simplicity in some tests, or a new mock
    context.model_copy.return_value = context
    return context

@pytest.fixture
def mock_idea_processor(mocker: MockerFixture) -> None:
    processor = mocker.Mock(spec=BaseIdeaProcessor)
    return processor

class TestTrialExecutor:
    def test_execute_trial(self, mock_dependencies: Any, mock_context: Any, mock_idea_processor: Any, mocker: MockerFixture) -> None:
        trial_num = 1

        generated_idea = base_schema.GeneratedIdea(
            name="idea1", summary="sum", explanation="exp", formula="f", key_sentences=["k"]
        )
        generated_results = base_schema.GeneratedResults(
            results=[
                base_schema.GeneratedResult(
                    idea=generated_idea, score=None,
                    implement=base_schema.GeneratedImpl(code="c", class_name="C")
                )
            ]
        )
        message_history = base_schema.MessageHistory(review=["r"], idea=["i"])
        cost = 0.5

        mock_idea_generator_fn = mocker.Mock(return_value=(generated_results, message_history, cost))

        validated_result = generated_results.results[0]
        agg_result_series = pd.Series({"id": "1_1", "score": 0.9})
        result_df = pd.DataFrame([{"id": "1_1", "raw": 1}])
        proc_cost = 0.1

        mock_idea_processor.process_idea.return_value = (
            validated_result, agg_result_series, result_df, proc_cost
        )

        executor = TrialExecutor(
            dependencies=mock_dependencies,
            context=mock_context,
            idea_processor=mock_idea_processor,
            idea_generator_fn=mock_idea_generator_fn
        )
        res, trial_df, all_idea_df, total_cost, new_ctx = executor.execute_trial(trial_num)

        assert res.results[0] == validated_result
        assert not trial_df.empty
        assert not all_idea_df.empty
        assert total_cost == cost + proc_cost

        mock_idea_generator_fn.assert_called_once()
        mock_idea_processor.process_idea.assert_called_once()

    def test_update_context(self, mock_dependencies: Any, mock_context: Any, mock_idea_processor: Any, mocker: MockerFixture) -> None:
        executor = TrialExecutor(
            dependencies=mock_dependencies,
            context=mock_context,
            idea_processor=mock_idea_processor,
            idea_generator_fn=mocker.Mock()
        )

        new_context = mocker.Mock(spec=base_schema.RunContext)
        executor.update_context(new_context)

        assert executor.context == new_context
        mock_idea_processor.update_context.assert_called_with(new_context)


class TestExpandingExecutor:
    def test_init_raises_value_error(self, mock_dependencies: Any, mock_context: Any, mock_idea_processor: Any, mocker: MockerFixture) -> None:
        with pytest.raises(ValueError, match="Expanding steps must be a list of at least two integers"):
            ExpandingExecutor(
                dependencies=mock_dependencies,
                context=mock_context,
                expanding_steps=[1],
                idea_processor=mock_idea_processor,
                research_based_prompt_factory=mocker.Mock(),
                expanding_prompt_factory=mocker.Mock(),
                code_generator_fn=mocker.Mock(),
                feedback_formatter_fn=mocker.Mock(),
                seed_phase_id_prefix="seed",
                expanding_phase_id_prefix="exp"
            )

    def test_execute_idea_expanding(self, mock_dependencies: Any, mock_context: Any, mock_idea_processor: Any, mocker: MockerFixture) -> None:
        mock_gen_idea_cls = mocker.patch("astronaut.logics.common.experiments.executors.common_logics.GenerateIdea")
        mock_gen_idea_instance = mock_gen_idea_cls.return_value

        generated_idea = base_schema.GeneratedIdea(
            name="idea1", summary="sum", explanation="exp", formula="f", key_sentences=["k"]
        )
        generated_results = base_schema.GeneratedIdeaResult(results=[generated_idea])
        mock_gen_idea_instance.generate.return_value = (generated_results, None, 1.0)

        mock_code_generator_fn = mocker.Mock()
        gen_codes = base_schema.GeneratedImplResult(
            results=[base_schema.GeneratedImpl(code="c", class_name="C")]
        )
        mock_code_generator_fn.return_value = (gen_codes, [], 0.5)

        validated_result = base_schema.GeneratedResult(
            idea=generated_idea, score=None, implement=gen_codes.results[0]
        )
        agg_series = pd.Series({"val": 1})
        raw_df = pd.DataFrame([{"val": 1}])
        mock_idea_processor.process_idea.return_value = (validated_result, agg_series, raw_df, 0.2)

        executor = ExpandingExecutor(
            dependencies=mock_dependencies,
            context=mock_context,
            expanding_steps=[1, 2],
            idea_processor=mock_idea_processor,
            research_based_prompt_factory=mocker.Mock(return_value=("p1", "p2")),
            expanding_prompt_factory=mocker.Mock(return_value=("p3", "p4")),
            code_generator_fn=mock_code_generator_fn,
            feedback_formatter_fn=mocker.Mock(return_value="Report"),
            seed_phase_id_prefix="seed",
            expanding_phase_id_prefix="exp"
        )

        ideas, report, cost = executor.execute_idea_expanding("Research Report")

        assert len(ideas.results) == 1 # Final expanded ideas
        assert "Report" in report
        assert cost > 0


class TestBaselineExecutor:
    def test_execute(self, mocker: MockerFixture) -> None:
        mock_experiment = mocker.Mock()
        mock_experiment.run.return_value = (None, "Result")

        mock_config_updater = mocker.Mock()
        mock_log_result = mocker.Mock()

        mocker.patch("builtins.open", mocker.mock_open(read_data="key: value"))
        mocker.patch("yaml.safe_load", return_value={"key": "value"})

        mocker.patch("astronaut.logics.common.experiments.executors.ExperimentConfig", return_value="MockConfig")

        executor = BaselineExecutor(
            experiment=mock_experiment,
            dataset_configs={"dataset1": {"k": "v"}},
            config_updater_fn=mock_config_updater,
            log_result_fn=mock_log_result
        )

        executor.execute("dummy_path.yaml")

        mock_config_updater.assert_called_once()
        mock_experiment.run.assert_called_once()
